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
parser.add_argument('--resume', '-r', default='BUPTCampus_drop_0.2_4_8_lr_0.01_seed_1234map_best.t', type=str,
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

parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.2, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.6, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--seed', default=1234, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='2', type=str,
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
parser.add_argument('--distance', type=str, default='cosine')
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

print('==> Resuming from checkpoint..')
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

print('==> Start Testing...')
# testing
if args.dataset == 'BUPTCampus':
    CMC, MAP = test_BUPTCampus(net, queryloader, galleryloader, show=True, return_all=True)
    MODE = ['RGB->RGB', 'RGB->IR ', 'IR->RGB ', 'IR->IR  ', 'AllModal']
    for i, mode in enumerate(MODE):
        print(
            '\n\t{}:  mAP:{:.2f}% Rank1:{:.2f}% Rank5:{:.2f}% Rank10:{:.2f}% Rank20:{:.2f}%' \
                .format(mode, MAP[i], CMC[i][0], CMC[i][4], CMC[i][9], CMC[i][19])
            , file=test_log_file)


