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
import math
import sys
sys.path.append("../")
from utils.wav2mfcc import wav2mfcc 


class MetaTrSouMNIST(torch.utils.data.Dataset):
	"""  soundmnist dataset for meta-learning"""

	def __init__(self, img_root, sound_root, per_class_num=50, meta_split='mtr'):

		self.img_root = img_root
		self.sound_root = sound_root

		self.meta_split = meta_split
		self.per_class_num = per_class_num

		if self.meta_split == 'mtr':
			self.meta_train_image_list = self.get_meta_train_image_list(self.img_root,self.per_class_num)
			
		elif self.meta_split == 'mval':
			self.meta_val_image_list = self.get_meta_val_image_list(self.img_root,self.per_class_num)
			self.meta_val_sound_list = self.get_meta_val_sound_list(self.sound_root,self.per_class_num)

		else:
			raise ValueError('No such split: %s' % self.meta_split)


	def get_meta_train_image_list(self, img_root, per_class_num):
		tr_img_root = os.path.join(img_root+'train/')
		tr_number_list = sorted(os.listdir(tr_img_root))
		meta_train_image_list = list() # image for training 
		for i in tr_number_list:
			images = sorted(os.listdir(os.path.join(tr_img_root+i)))
			for j in range(per_class_num, 105):
			# for j in range(105):
				path = os.path.join(tr_img_root+i+'/'+images[j])
				meta_train_image_list.append(path)

		return meta_train_image_list

	def get_meta_val_image_list(self, root, per_class_num):
		te_root = os.path.join(root+'train/')
		te_number_list = sorted(os.listdir(te_root))
		meta_val_img_list = list() # image for training 
		for i in te_number_list:
			images = sorted(os.listdir(os.path.join(te_root+i)))
			for j in range(per_class_num):
				path = os.path.join(te_root+i+'/'+images[j])
				meta_val_img_list.append(path)

		return meta_val_img_list

	def get_meta_val_sound_list(self, root, per_class_num):
		tr_root = os.path.join(root+'train/')
		tr_number_list = sorted(os.listdir(tr_root))
		meta_val_sound_list = list() # image for training 
		for i in tr_number_list:
		    jack_list = list()
		    nico_list = list()
		    theo_list = list()
		    sounds = sorted(os.listdir(tr_root+ i))
		    for j in sounds:
		        if j.split('_')[1] == 'jackson':
		            if len(jack_list) < int(per_class_num/3):
		                jack_list.append(os.path.join(tr_root+i+'/'+j))
		        elif j.split('_')[1] == 'nicolas':
		            if len(nico_list) < int(per_class_num/3):
		                nico_list.append(os.path.join(tr_root+i+'/'+j))
		        else:
		            if len(theo_list) < int(per_class_num/3):
		                theo_list.append(os.path.join(tr_root+i+'/'+j))
		    temp_list = jack_list + nico_list + theo_list
		    meta_val_sound_list += temp_list

		return meta_val_sound_list


	def get_length(self):

		if self.meta_split == 'mtr':
			length = len(self.meta_train_image_list)
		else:
			length = len(self.meta_val_image_list)

		return length 


	def __len__(self):
		""" Returns size of the dataset
		returns:
			int - number of samples in the dataset
		"""
		return self.get_length()

	def __getitem__(self, index):
		""" get image and label  """
		transformations = transforms.Compose([transforms.ToTensor(),
											  transforms.Normalize([0.5], [0.5])])
		if self.meta_split == 'mtr':
			image_path = self.meta_train_image_list[index]
			image_label = int(image_path.split('/')[-2])

			img = Image.open(image_path)
			im = transformations(img)
			label = torch.tensor(image_label).long()
			return im, label

		elif self.meta_split == 'mval':
			image_path = self.meta_val_image_list[index]
			image_label = int(image_path.split('/')[-2])
			sound_path = self.meta_val_sound_list[index]
			sound_label = int(sound_path.split('/')[-2])

			img = Image.open(image_path)
			sound = np.asarray(wav2mfcc(sound_path))
			assert image_label==sound_label

			im = transformations(img)
			sound = transformations(sound)
			label = torch.tensor(image_label).long()
			return im, sound, label
		else:
			raise ValueError('No such split: %s' % self.meta_split)



if __name__ == '__main__':
	from PIL import Image
	import torch
	img_root = '../data/mnist/'
	sound_root = '../data/sound_450/'

	dataset = MetaTrSouMNIST(img_root, sound_root, per_class_num=105, meta_split='mtr')
	sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
	loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=1, pin_memory=True,sampler=sampler)
	
	print(len(dataset))

