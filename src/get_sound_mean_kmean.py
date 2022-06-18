from __future__ import print_function, absolute_import, division

import os
import time
import math
import datetime
import argparse
import os.path as path
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.soundmnist import SoundMNIST 
from models.soundlenet5 import SoundLenet5mean
from models.lenet5 import LeNet5
from models.snet import SNetPluse
from utils.misc import save_ckpt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    # general parm
    parser.add_argument('-i', '--image_root', default = '../data/mnist/', type = str, help='data root' )
    parser.add_argument('-s', '--sound_root', default = '../data/sound_450/', type = str, help='data root' )
    parser.add_argument('--checkpoint', default='./save/sound_mean/new/', type=str, help='checkpoint directory')
    parser.add_argument('--per_class_num', default = 15, type = int, help='per_class_num' )
    parser.add_argument('--n_clusters', default = 10, type = int, help='n_clusters' )
    parser.add_argument('--snapshot', default=1, type=int, help='save models for every # epochs (default: 1)')


    parser.add_argument('--soundmnist_model_path', default='./save/soundmnist/new/15/2022-04-04T09:34:24.967372/', 
                        type=str, help='pre-trained sound mnist model')
    parser.add_argument('--soundmnist_model_name', default='epoch_30_test_accuracy_87.55555555555556_train_accuracy_100.0_best_model.path.tar', 
                        type=str, help='pre-trained sound mnist model')
    
    # model related parm
    parser.add_argument('-b', '--batch_size', default = 32, type = int, help='batch size' )
    parser.add_argument('-e', '--epochs', default = 1, type = int, help='num of epoch' )
    parser.add_argument('--lr', default = 1e-3, type = float, help='initial learning rate' )
    parser.add_argument('--vis_device', default='0', type=str, help='set visiable device')

    args = parser.parse_args()

    return args

def main(args):
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.vis_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SoundMNIST(img_root=args.image_root,sound_root=args.sound_root, per_class_num=args.per_class_num, train=True)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers=1, pin_memory=True)

    #creat model 
    print('==> model creating....')
    extractor1 = LeNet5().to(device)
    extractor2 = SNetPluse().to(device)
    
    image_sound_extractor = SoundLenet5mean(extractor1, extractor2, extractor_grad=False).to(device)
    
    ckpt_image_sound = torch.load(path.join(args.soundmnist_model_path, args.soundmnist_model_name))
    image_sound_extractor.load_state_dict(ckpt_image_sound['state_dict'])


    ckpt_dir_path = path.join(args.checkpoint)
    if not path.exists(ckpt_dir_path):
        os.makedirs(ckpt_dir_path)
        print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

    
    final_mean = list()

    print('start training')
    for epoch in range(args.epochs):
        # Train for one epoch
        sound_mean = torch.randn(1, 320).to(device)
        sound_mean = train(train_loader, image_sound_extractor, sound_mean, device)
        final_mean.append(sound_mean[1:,:])
    
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(final_mean[0].cpu().numpy())
    new_mean = kmeans.cluster_centers_

    save_path = path.join(ckpt_dir_path, 'sound_mean_' + str(args.per_class_num*10)+'.npy')
    print(new_mean.shape)
    np.save(save_path, new_mean)
    print('done save!!!')



def train(train_loader, model, sound_mean,device):
    ''' train one epoch'''

    torch.set_grad_enabled(False)
    model.eval()

    for i, batch in enumerate(train_loader):
        images = batch[0].to(device)
        sounds = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs, feature = model(images, sound=sounds, mode='two')
        sound_mean = torch.cat([sound_mean, feature], dim=0)

    return sound_mean







if __name__ == '__main__':
    main(parse_args())