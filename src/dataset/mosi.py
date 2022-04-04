import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


import numpy as np
import pandas as pd
import os
import cv2 
import random
from PIL import Image
import pickle

import math
import sys
sys.path.append("../")
import h5py
import pandas as pd
import json


class CMUREG(torch.utils.data.Dataset):
    """  soundmnist dataset for meta-learning"""

    def __init__(self, data_root, text_ratio=0.1, split='train'):

        self.data_root = data_root
        self.split = split
        self.text_ratio = text_ratio



        h5f = h5py.File(os.path.join(self.data_root + 'X_train.h5'), 'r')
        x_train = h5f['data'][:]
        h5f.close()
        h5f = h5py.File(os.path.join(self.data_root + 'y_train.h5'), 'r')
        y_train = h5f['data'][:]
        h5f.close()
        h5f = h5py.File(os.path.join(self.data_root + 'X_valid.h5'), 'r')
        x_valid = h5f['data'][:]
        h5f.close()
        h5f = h5py.File(os.path.join(self.data_root + 'y_valid.h5'), 'r')
        y_valid = h5f['data'][:]
        h5f.close()
        h5f = h5py.File(os.path.join(self.data_root + 'X_test.h5'), 'r')
        x_test = h5f['data'][:]
        h5f.close()
        h5f = h5py.File(os.path.join(self.data_root + 'y_test.h5'), 'r')
        y_test = h5f['data'][:]
        h5f.close()
        ad = 5
        td = 300
        train_audio_list, train_visual_list, train_text_list, train_labels_list = x_train[:, :, td:td + ad], x_train[:, :, td + ad:], x_train[:, :, :td], y_train
        val_audio_list, val_visual_list, val_text_list, val_labels_list = x_valid[:, :, td:td + ad], x_valid[:, :, td + ad:], x_valid[:, :, :td], y_valid
        test_audio_list, test_visual_list, test_text_list, test_labels_list = x_test[:, :, td:td + ad], x_test[:, :, td + ad:], x_test[:, :, :td], y_test
        
        num_of_train = int(len(train_audio_list) * text_ratio)

        if self.split == 'train':
            self.train_audio_list = train_audio_list[0:num_of_train]
            self.train_visual_list = train_visual_list[0:num_of_train]
            self.train_text_list = train_text_list[0:num_of_train]
            self.train_labels_list = train_labels_list[0:num_of_train]
            
        elif self.split == 'val':
            self.val_audio_list = val_audio_list
            self.val_visual_list = val_visual_list
            self.val_text_list = val_text_list
            self.val_labels_list = val_labels_list
        
        elif self.split == 'test':
            self.test_audio_list = test_audio_list
            self.test_visual_list = test_visual_list
            self.test_text_list = test_text_list
            self.test_labels_list = test_labels_list

        else:
            raise ValueError('No such split: %s' % self.split)

    def get_length(self):

        if self.split == 'train':
            length = len(self.train_labels_list)

        elif self.split == 'val':
            length = len(self.val_labels_list)

        else:
            length = len(self.test_labels_list)

        return length 


    def __len__(self):
        """ Returns size of the dataset
        returns:
            int - number of samples in the dataset
        """
        return self.get_length()

    def __getitem__(self, index):
        """ get image and label  """
        if self.split == 'train':
            audio = self.train_audio_list[index]
            visual = self.train_visual_list[index]
            text = self.train_text_list[index]
            label = self.train_labels_list[index]

            audio = torch.tensor(audio).float()
            visual =  torch.tensor(visual).float()
            text = torch.tensor(text).float()
            label = torch.tensor(label).float()
            label_o = torch.zeros(1).float()
            if label>=0:
                label_o=0 # greater than zero means positive
            else:
                label_o=1 # less than zero means negative
            return audio, visual, text, label

        elif self.split == 'val':
            audio = self.val_audio_list[index]
            visual = self.val_visual_list[index]
            text = self.val_text_list[index]
            label = self.val_labels_list[index]

            audio = torch.tensor(audio).float()
            visual =  torch.tensor(visual).float()
            text = torch.tensor(text).float()
            label = torch.tensor(label).float()
            label_o = torch.zeros(1).float()
            if label>=0:
                label_o=0 # greater than zero means positive
            else:
                label_o=1 # less than zero means negative
            return audio, visual, text, label

        elif self.split == 'test':
            audio = self.test_audio_list[index]
            visual = self.test_visual_list[index]
            text = self.test_text_list[index]
            label = self.test_labels_list[index]

            audio = torch.tensor(audio).float()
            visual =  torch.tensor(visual).float()
            text = torch.tensor(text).float()
            label = torch.tensor(label).float()
            label_o = torch.zeros(1).float()
            if label>=0:
                label_o=0 # greater than zero means positive
            else:
                label_o=1 # less than zero means negative
            return audio, visual, text, label
        else:
            raise ValueError('No such split: %s' % self.meta_split)
if __name__ == '__main__':
    from PIL import Image
    import torch
    import pickle
    data_root = '../data/MOSI/'
    dataset = CMUREG(data_root, text_ratio=1.0, split='train')
    # sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 5, shuffle = False, num_workers=2, pin_memory=True)
    print(len(dataset))
