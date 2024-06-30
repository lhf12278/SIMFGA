from __future__ import print_function
import argparse
import sys
import time
import torch
from tqdm import tqdm
from os.path import join
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tools.eval_metrics import eval_sysu, eval_regdb, evaluate,evaluate_bupt
from model.model_main import embed_net
from tools.utils import *
from tools.loss import OriTripletLoss, KLDivLoss
from tensorboardX import SummaryWriter
# from data.data_manager import VCM
# from data.data_loader import VideoDataset_train, VideoDataset_test
from data.dataloader import *
from data.ChannelAug import ChannelRandomErasing, ChannelExchange
from tools.loss import WeightedRegularizedTriplet, CrossEntropyLabelSmooth
import setproctitle
setproctitle.setproctitle('zuozhigang')

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='BUPTCampus', help='dataset name: VCM(Video Cross-modal)')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=10, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/buptc_log/', type=str,
                    help='log save path')
parser.add_argument('--num_workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=128, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=256, type=int,
                    metavar='imgh', help='img height')
# parser.add_argument('--batch-size', default=4 , type=int,
#                     metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.2, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.8, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--seed', default=1234, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument("--cmt_depth", type=int, default=3, help="cross modal transformer self attn layers")
parser.add_argument('--T', default=7, type=float, help='temperature')

# buptc
parser.add_argument('--data_root', type=str, default='/media/lele/c/zuozhigang/datasets/BUPTCampus/DATA/')
# /home/ps/D/data-1/zzg/DATA

parser.add_argument('--seq_lenth', type=int, default=10)


parser.add_argument('--fake', action='store_true', default=False)

parser.add_argument('--test_sampler', type=str,
                                 default='ConsistentModalitySampler', help='None for no shuffle')
parser.add_argument('--test_bs', type=int, default=64)
parser.add_argument('--test_frame_sample', type=str, default='uniform')
parser.add_argument('--distance', type=str, default='euclidean')
parser.add_argument('--max_rank', type=int, default=20)

parser.add_argument('--train_frame_sample', type=str, default='random')
parser.add_argument('--train_sampler', type=str,default='RandomIdentitySampler', help='None for shuffle')
parser.add_argument('--train_bs', type=int, default=8)
parser.add_argument('--train_sampler_nc', type=int, default=2)
parser.add_argument('--train_sampler_nt', type=int, default=1)
parser.add_argument('--random_flip', action='store_false', default=False)

parser.add_argument('--auxiliary_sampler', type=str, default='RandomCameraSampler',
                                 help='None for shuffle, or RandomIdentitySampler/RandomCameraSampler')
parser.add_argument('--auxiliary_sampler_nc', type=int, default=2)
parser.add_argument('--auxiliary_sampler_nt', type=int, default=1)  # 1 or 2
parser.add_argument('--a', default=0.5, type=int, help='hyper parameter')
parser.add_argument('--b', default=0.5, type=int, help='hyper parameter')
parser.add_argument('--c', default=0.5, type=int, help='hyper parameter')
parser.add_argument('--d', default=1, type=int, help='hyper parameter')
# torch.backends.cudnn.enabled = False

args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True

log_path = args.log_path + 'BUPTC_log/'
test_mode = [1, 2]

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

# log file name
suffix = args.dataset
suffix = suffix + '_drop_{}_{}_{}_lr_{}_seed_{}'.format(args.drop, args.num_pos, args.train_bs, args.lr, args.seed)
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


best_acc_t2v = 0  # best test accuracy
best_acc_v2t = 0
best_map_acc_t2v = 0  # best test accuracy
best_map_acc_v2t = 0
start_epoch = 0
wG = 0
end = time.time()

print('==> Loading data..')
# Data loading code


if args.dataset == 'BUPTCampus':
    queryloader, _ = get_dataloader(args, 'query', True)
    galleryloader, _ = get_dataloader(args, 'gallery', True)
    trainloader, n_class = get_dataloader(args, 'train', True)

print('==> Building model..')
if args.method == 'agw':
    net = embed_net(args, n_class, no_local='on', gm_pool='on', arch=args.arch)
else:
    net = embed_net(args, n_class, no_local='on', gm_pool='on', arch=args.arch)
net.to(device)

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
if args.method == 'base':
    criterion1 = CrossEntropyLabelSmooth(num_classes=n_class)
else:
    criterion1 = nn.CrossEntropyLoss()
if args.method == 'base':
    criterion2 = WeightedRegularizedTriplet()
else:
    loader_batch = args.train_bs * 2
    criterion2 = OriTripletLoss(batch_size=loader_batch, margin=args.margin)

criterion3 = KLDivLoss()


criterion1.to(device)
criterion2.to(device)
criterion3.to(device)


# optimizer
if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck0.parameters())) + list(map(id, net.bottleneck1.parameters()))+list(map(id, net.bottleneck2.parameters()))+list(map(id, net.bottleneck3.parameters())) + list(map(id, net.bottleneck4.parameters())) \
                     + list(map(id, net.classifier0.parameters())) + list(map(id, net.classifier1.parameters()))+list(map(id, net.classifier2.parameters()))+list(map(id, net.classifier3.parameters())) +list(map(id, net.classifier4.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_P = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck0.parameters(), 'lr': args.lr},
        {'params': net.bottleneck1.parameters(), 'lr': args.lr},
        {'params': net.bottleneck2.parameters(), 'lr': args.lr},
        {'params': net.bottleneck3.parameters(), 'lr': args.lr},
        {'params': net.bottleneck4.parameters(), 'lr': args.lr},
        {'params': net.classifier0.parameters(), 'lr': args.lr},
        {'params': net.classifier1.parameters(), 'lr': args.lr},
        {'params': net.classifier2.parameters(), 'lr': args.lr},
        {'params': net.classifier3.parameters(), 'lr': args.lr},
        {'params': net.classifier4.parameters(), 'lr': args.lr},
        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


def adjust_learning_rate(optimizer_P, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 35:
        lr = args.lr
    elif 35 <= epoch < 80:
        lr = args.lr * 0.1
    elif epoch >= 80:
        lr = args.lr * 0.01

    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    return lr

def compute_ide_loss(logits_list, pids):
    avg_ide_loss = 0
    avg_logits = 0
    part_num = 6
    for i in range(part_num):
        logits_i = logits_list[i]
        avg_logits += 1.0 / float(part_num) * logits_i
        ide_loss_i = criterion1(logits_i, pids)
        avg_ide_loss += 1.0 / float(part_num) * ide_loss_i
    return avg_ide_loss, avg_logits

def compute_part_loss(logits_list, labels_list):
    avg_ide_loss = 0
    avg_logits = 0
    part_num = 3
    for i in range(part_num):
        logits_i = logits_list[i]
        labels_list_i = labels_list[i]
        avg_logits += 1.0 / float(part_num) * logits_i
        ide_loss_i = criterion1(logits_i, labels_list_i)
        avg_ide_loss += 1.0 / float(part_num) * ide_loss_i
    return avg_ide_loss, avg_logits

def train(epoch, wG):
    # adjust learning rate
    current_lr = adjust_learning_rate(optimizer_P, epoch)
    train_loss = AverageMeter()
    id_loss0 = AverageMeter()
    tri_loss0 = AverageMeter()
    loss_ide0 = AverageMeter()

    id_loss1 = AverageMeter()
    tri_loss1 = AverageMeter()
    id_loss2 = AverageMeter()
    tri_loss2 = AverageMeter()
    id_loss3 = AverageMeter()
    tri_loss3 = AverageMeter()
    id_loss_a = AverageMeter()
    part_loss = AverageMeter()
    kl_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.train()
    end = time.time()

    for batch_idx, (imgs_rgb, pids_rgb, camid_rgb, imgs_ir, pids_ir, camid_ir) in enumerate(trainloader):
        input1 = imgs_rgb
        input2 = imgs_ir
        label1 = pids_rgb
        label2 = pids_ir
        labels = torch.cat((label1, label2), dim=0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        label1 = Variable(label1.cuda())
        label2 = Variable(label2.cuda())
        labels = Variable(labels.cuda())

        labels_list = []
        head_labels = Variable(torch.ones(loader_batch).long().cuda())
        body_labels = Variable(torch.zeros(loader_batch).long().cuda())
        leg_labels = Variable(2 * torch.ones(loader_batch).long().cuda())
        labels_list.append(head_labels)
        labels_list.append(body_labels)
        labels_list.append(leg_labels)

        data_time.update(time.time() - end)

        feat, out0, logits_list, en_feature, en_feature_p, feat1, feat1_p, de_feature, de_feature_p, spatial_feat, spatial_feat_p, logits_list1= net(input1, input2, seq_len=args.seq_lenth)

        de_feature_p1, de_feature_p2 = de_feature_p.chunk(2, 0)
        loss_id0 = criterion1(out0, labels)
        loss_id1 = criterion1(en_feature_p, labels)
        loss_id2 = criterion1(de_feature_p, labels)
        loss_id3 = criterion1(spatial_feat_p, labels)
        loss_id_a = criterion1(feat1_p, label1) + criterion1(feat1_p, label2)
        ide_loss0, avg_logits = compute_ide_loss(logits_list, labels)
        loss_part, avg_logits1 = compute_part_loss(logits_list1, labels_list)

        loss_tri0, batch_acc0 = criterion2(feat, labels)
        loss_tri1, batch_acc1 = criterion2(en_feature, labels)
        loss_tri2, batch_acc2 = criterion2(de_feature, labels)
        loss_tri3, batch_acc3 = criterion2(spatial_feat, labels)
        loss_kl = criterion3(F.softmax(de_feature_p1 / args.T, dim=1), F.softmax(feat1_p / args.T, dim=1)) + criterion3(F.softmax(de_feature_p2 / args.T, dim=1), F.softmax(feat1_p / args.T, dim=1))

        correct += (batch_acc0 / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)
        #loss function
        # if epoch <= 40:
        #     loss = loss_id0 + loss_tri0 + ide_loss0 + loss_part
        # # # elif 30 < epoch <= 40:
        # # # loss = (loss_tri1 + loss_id1 + loss_id_a) + (loss_id0 + loss_tri0 + ide_loss0)
        # elif epoch > 40:

        # loss = (loss_tri1 + loss_id1 + loss_id_a) + (loss_id0 + loss_tri0 + ide_loss0 + loss_part) + (loss_id2 + loss_tri2 + loss_id3 + loss_tri3) + loss_kl
        # loss = 0.5*(loss_tri1 + loss_id1 + loss_id_a) + 0.5* (loss_id0 + loss_tri0 + ide_loss0 + loss_part) + 0.5*(loss_id2 + loss_tri2 + loss_id3 + loss_tri3) + loss_kl
        # loss = 0.3*(loss_tri1 + loss_id1 + loss_id_a) + (loss_id0 + loss_tri0 + ide_loss0 + loss_part) + 0.2*(loss_id2 + loss_tri2 + loss_id3 + loss_tri3) + 0.5*loss_kl
        # loss = 0.5 * (loss_tri1 + loss_id1 + loss_id_a) + 0.5 *(loss_id0 + loss_tri0 + ide_loss0 + loss_part) + 0.5 * (loss_id2 + loss_tri2 + loss_id3 + loss_tri3) + loss_kl
        loss = args.a * (loss_tri1 + loss_id1 + loss_id_a) + args.b * (loss_id0 + loss_tri0 + ide_loss0 + loss_part) + args.c * (loss_id2 + loss_tri2 + loss_id3 + loss_tri3) + args.d * loss_kl

        optimizer_P.zero_grad()
        loss.backward()
        optimizer_P.step()

        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss0.update(loss_id0.item(), 2 * input1.size(0))
        loss_ide0.update(ide_loss0.item(), 2 * input1.size(0))
        tri_loss0.update(loss_tri0.item(), 2 * input1.size(0))
        id_loss1.update(loss_id1.item(), 2 * input1.size(0))
        tri_loss1.update(loss_tri1.item(), 2 * input1.size(0))
        id_loss2.update(loss_id2.item(), 2 * input1.size(0))
        tri_loss2.update(loss_tri2.item(), 2 * input1.size(0))
        id_loss3.update(loss_id3.item(), 2 * input1.size(0))
        tri_loss3.update(loss_tri3.item(), 2 * input1.size(0))
        id_loss_a.update(loss_id_a.item(), 2 * input1.size(0))
        part_loss.update(loss_part.item(), 2 * input1.size(0))
        kl_loss.update(loss_kl.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 100 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'ideLoss: {loss_ide.val:.4f} ({loss_ide.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                   epoch, batch_idx, len(trainloader), current_lr,
                   100. * correct / total, batch_time=batch_time,
                   train_loss = train_loss, id_loss = id_loss0, tri_loss = tri_loss0, loss_ide=loss_ide0))
    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss0.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss0.avg, epoch)
    writer.add_scalar('loss_ide', loss_ide0.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    return 1. / (1. + train_loss.avg)

def test_BUPTCampus(model, dataloader_query, dataloader_gallery, show=False, save_dir='', return_all=False, postfix=''):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    print('========== Testing ==========')
    model.eval()
    with torch.no_grad():
        # query
        print('Extracting Query Feature...')
        query_feats, query_pids, query_modals, query_cids = [], [], [], []
        for batch_idx, (imgs, pids, cids, modals) in enumerate(dataloader_query):
            imgs = Variable(imgs.cuda())
            modal = modals[0]
            if modal == 0:
                # test_mode = [1, 2]
                feats = net(imgs, imgs, test_mode[0], seq_len=args.seq_lenth)

            elif modal == 1:
                feats = net(imgs, imgs, test_mode[1], seq_len=args.seq_lenth)
            else:
                continue
            query_feats.append(feats)
            query_pids.append(pids)
            query_cids.append(cids)
            query_modals.append(modal.repeat(pids.size()))
        query_feats = torch.cat(query_feats, dim=0)  # [Nq, C]
        query_pids = torch.cat(query_pids, dim=0)  # [Nq,]
        query_modals = torch.cat(query_modals, dim=0)
        query_cids = torch.cat(query_cids, dim=0)

        # gallery
        print('Extracting Gallery Feature...')
        gallery_feats, gallery_pids, gallery_modals, gallery_cids = [], [], [], []
        for batch_idx, (imgs, pids, cids, modals) in enumerate(dataloader_gallery):
            imgs = Variable(imgs.cuda())
            modal = modals[0]
            assert modals.eq(modal).all()
            if modal == 0:
                feats = net(imgs, imgs, test_mode[0], seq_len=args.seq_lenth)

            elif modal == 1:
                feats = net(imgs, imgs, test_mode[1], seq_len=args.seq_lenth)
            else:
                continue
            gallery_feats.append(feats)
            gallery_pids.append(pids)
            gallery_cids.append(cids)
            gallery_modals.append(modal.repeat(pids.size()))
        gallery_feats = torch.cat(gallery_feats, dim=0)  # [Ng, C]
        gallery_pids = torch.cat(gallery_pids, dim=0)  # [Ng,]
        gallery_modals = torch.cat(gallery_modals, dim=0)
        gallery_cids = torch.cat(gallery_cids, dim=0)

        # save
        if save_dir:
            torch.save(query_feats, join(save_dir, f'query_feats{postfix}.pth'))
            torch.save(query_pids, join(save_dir, 'query_pids.pth'))
            torch.save(query_modals, join(save_dir, 'query_modals.pth'))
            torch.save(query_cids, join(save_dir, 'query_cids.pth'))
            torch.save(gallery_feats, join(save_dir, f'gallery_feats{postfix}.pth'))
            torch.save(gallery_pids, join(save_dir, 'gallery_pids.pth'))
            torch.save(gallery_modals, join(save_dir, 'gallery_modals.pth'))
            torch.save(gallery_cids, join(save_dir, 'gallery_cids.pth'))
    # # distance
    if args.distance == 'cosine':
        distance = 1 - query_feats @ gallery_feats.T
    else:
        distance = euclidean_dist(query_feats, gallery_feats)

    CMC, MAP = [], []
    # evaluate (intra/inter-modality)
    for q_modal in (0, 1):
        for g_modal in (0, 1):
            q_mask = query_modals == q_modal
            g_mask = gallery_modals == g_modal
            tmp_distance = distance[q_mask, :][:, g_mask]
            tmp_qid = query_pids[q_mask]
            tmp_gid = gallery_pids[g_mask]
            tmp_cmc, tmp_ap = evaluate_bupt(tmp_distance, tmp_qid, tmp_gid, args)
            CMC.append(tmp_cmc * 100)
            MAP.append(tmp_ap * 100)
            if show:
                print_metrics(
                    tmp_cmc, tmp_ap,
                    prefix='{:<3}->{:<3}:  '.format(MODALITY_[q_modal], MODALITY_[g_modal])
                )

    # evaluate (omni-modality)
    cmc, ap = evaluate_bupt(distance, query_pids, gallery_pids, args)

    CMC.append(cmc * 100)
    MAP.append(ap * 100)

    if show:
        print_metrics(cmc, ap, prefix='AllModal:  ')

    del query_feats, query_pids, query_modals, gallery_feats, gallery_pids, gallery_modals, distance

    if return_all:
        return CMC, MAP
    else:
        return cmc * 100, ap * 100

# training
print('==> Start Training...')

for epoch in range(start_epoch, 201 - start_epoch):

    print('==> Preparing Data Loader...')

    # training
    wG = train(epoch, wG)

    if epoch >= 0 and epoch % 10 == 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)

        # testing
        if args.dataset == 'BUPTCampus':

            CMC, MAP = test_BUPTCampus(net, queryloader, galleryloader, show=True, return_all=True)
            writer.add_scalar('Eval/mAP(%)', MAP[-1], epoch)
            writer.add_scalar('Eval/Rank1(%)', CMC[-1][0], epoch)
            writer.add_scalar('Eval/Rank5(%)', CMC[-1][4], epoch)
            writer.add_scalar('Eval/Rank10(%)', CMC[-1][9], epoch)

            MODE = ['RGB->RGB', 'RGB->IR ', 'IR->RGB ', 'IR->IR  ', 'AllModal']

            log_info = 'Epoch:[{}/{}]'.format(epoch, 200)
            for i, mode in enumerate(MODE):
                log_info += '\n\t{}:  mAP:{:.2f}% Rank1:{:.2f}% Rank5:{:.2f}% Rank10:{:.2f}% Rank20:{:.2f}%' \
                    .format(mode, MAP[i], CMC[i][0], CMC[i][4], CMC[i][9], CMC[i][19])
                print(
                    '\n\t{}:  mAP:{:.2f}% Rank1:{:.2f}% Rank5:{:.2f}% Rank10:{:.2f}% Rank20:{:.2f}%' \
                        .format(mode, MAP[i], CMC[i][0], CMC[i][4], CMC[i][9], CMC[i][19])
                    , file=test_log_file)

                if CMC[2][0] > best_acc_t2v:
                    best_acc_t2v = CMC[2][0]
                    best_epoch = epoch
                    state_dict = {
                        'net': net.state_dict(),
                        'optimizer': optimizer_P,
                        'epoch': epoch
                    }
                    torch.save(state_dict, checkpoint_path + suffix + 'rank1_t2v_best.t')

                if CMC[1][0] > best_acc_v2t:
                    best_acc_v2t = CMC[1][0]
                    best_epoch = epoch
                    state_dict = {
                        'net': net.state_dict(),
                        'optimizer': optimizer_P,
                        'epoch': epoch
                    }

                    torch.save(state_dict, checkpoint_path + suffix + 'rank1_v2t_best.t')

                if MAP[2] > best_map_acc_t2v:
                    best_map_acc_t2v = MAP[2]
                    best_epoch = epoch
                    state = {
                        'net': net.state_dict(),
                        'optimizer': optimizer_P,
                        'epoch': epoch,
                    }
                    torch.save(state, checkpoint_path + suffix + 'map_t2v_best.t')

                if MAP[1] > best_map_acc_v2t:
                    best_map_acc_v2t = MAP[1]
                    best_epoch = epoch
                    state = {
                        'net': net.state_dict(),
                        'optimizer': optimizer_P,
                        'epoch': epoch,
                    }
                    torch.save(state, checkpoint_path + suffix + 'map_v2t_best.t')
            test_log_file.flush()


