import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


import numpy as np
import pandas as pd
import os
import cv2 
import random
# import scipy.io as scio
from PIL import Image
import math
import sys
sys.path.append("../")
from utils.wav2mfcc import wav2mfcc 


class Sound(torch.utils.data.Dataset):
	"""  soundmnist dataset """

	def __init__(self, sound_root, per_class_num=100,train=True):

		self.sound_root = sound_root

		self.train = train
		self.per_class_num = per_class_num

		if self.train:
			# self.train_img_list = self.get_image_train_list(self.img_root,self.per_class_num)
			self.train_sound_list = self.get_sound_train_list(self.sound_root,self.per_class_num)

		else:
			# self.test_img_list = self.get_image_test_list(self.img_root,self.per_class_num)
			self.test_sound_list = self.get_sound_test_list(self.sound_root,self.per_class_num)


	def get_sound_train_list(self, root, per_class_num):
		tr_root = os.path.join(root+'train/')
		tr_number_list = sorted(os.listdir(tr_root))
		train_sound_list = list() # image for training 
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
		    train_sound_list += temp_list

		return train_sound_list

	def get_sound_test_list(self, root, per_class_num):
		te_root = os.path.join(root+'test/')
		te_number_list = sorted(os.listdir(te_root))
		test_sound_list = list() # image for training 
		for i in te_number_list:
			sounds = sorted(os.listdir(os.path.join(te_root+i)))
			for j in sounds:
				path = os.path.join(te_root+i+'/' + j)
				test_sound_list.append(path)

		return test_sound_list



	def get_length(self):

		if self.train:
			length = len(self.train_sound_list)
		else:
			length = len(self.test_sound_list)

		return length 


	def __len__(self):
		""" Returns size of the dataset
		returns:
			int - number of samples in the dataset
		"""
		return self.get_length()

	def __getitem__(self, index):
		""" get image and label  """
		if self.train:
			sound_path = self.train_sound_list[index]
			sound_label = int(sound_path.split('/')[-2])
		else:
			sound_path = self.test_sound_list[index]
			sound_label = int(sound_path.split('/')[-2])

		transformations = transforms.Compose([transforms.ToTensor(),
											  transforms.Normalize([0.5], [0.5])])
		sound = wav2mfcc(sound_path)
		sd = transformations(sound)
		label = torch.tensor(sound_label).long()
		return sd, label






if __name__ == '__main__':
	from PIL import Image
	import torch
	sound_root = '../data/sound_450/'
	dataset = Sound(sound_root,per_class_num=3, train=True)
	print(len(dataset))
	
