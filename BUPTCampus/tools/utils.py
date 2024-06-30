import os
from collections import defaultdict
import numbers
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import scipy.io as scio
import torch
from torchvision import transforms
from data.ChannelAug import ChannelExchange,ChannelRandomErasing

MODALITY = {'RGB/IR': -1, 'RGB': 0, 'IR': 1}
MODALITY_ = {-1:'All', 0: 'RGB', 1: 'IR'}
CAMERA = {'LS3': 0, 'G25': 1, 'CQ1': 2, 'W4': 3, 'TSG1': 4, 'TSG2': 5}

def print_metrics(cmc, ap, prefix=''):
    print(
        '{}mAP: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}.'
            .format(prefix, ap, cmc[0], cmc[4], cmc[9], cmc[19])
    )


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.T, beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist.cpu().numpy()



def get_transform(opt, mode):
    if mode == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((opt.img_h, opt.img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((opt.img_h, opt.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ChannelRandomErasing(probability=0.5),
            # ChannelExchange(gray=2),
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((opt.img_h, opt.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise RuntimeError('Error transformation mode.')



def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
    """

    def __init__(self, train_thermal_label, train_color_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        # print('uni_label')
        # print(uni_label)
        self.n_classes = len(uni_label)

        N = np.minimum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos))):
            # print('aaa')
            # print(int(N/(batchSize*num_pos)))
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            batch_idx = list(batch_idx)
            for i in range(batchSize):
                # print('i')
                # print(i)
                # print(batch_idx)
                # print(len(batch_idx))
                # print(len(color_pos))
                # print(len(thermal_pos))
                # print(uni_label[-1])
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                    # index = np.hstack((index1, index2))
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
                    # index = np.hstack((index, sample_color))
                    # index = np.hstack((index, sample_thermal))
        self.N = int(N / (batchSize * num_pos)) * (batchSize * num_pos)
        print("index1.shape = {}".format(index1.shape))
        print("index2.shape = {}".format(index2.shape))
        # print("index.shape = {}".format(index.shape))
        print("index1 = {}".format(index1))
        print("index2 = {}".format(index2))
        # print("index = {}".format(index))
        print("N = {}".format(self.N))
        print("return iter {}".format(np.arange(len(index1))))
        self.index1 = index1
        self.index2 = index2
        # self.index = list(index)
        # print("index = {}".format(index))
        # self.N  = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))
        # return iter(self.index2)

    def __len__(self):
        return self.N
    
class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad