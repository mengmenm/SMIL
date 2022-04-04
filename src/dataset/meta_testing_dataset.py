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


class MetaTeSouMNIST(torch.utils.data.Dataset):
	"""  soundmnist dataset for meta-learning"""

	def __init__(self, img_root):

		self.img_root = img_root
		# self.sound_root = sound_root

		# self.meta_split = meta_split
		# self.per_class_num = per_class_num

		self.meta_testing_image_list = self.get_meta_testing_image_list(self.img_root)


	def get_meta_testing_image_list(self, img_root):
		tr_img_root = os.path.join(img_root+'test/')
		tr_number_list = sorted(os.listdir(tr_img_root))
		meta_testing_image_list = list() # image for training 
		for i in tr_number_list:
			images = sorted(os.listdir(os.path.join(tr_img_root+i)))
			# for j in range(len(images)):
			for j in range(45):
				path = os.path.join(tr_img_root+i+'/'+images[j])
				meta_testing_image_list.append(path)

		return meta_testing_image_list


	def get_length(self):

		length = len(self.meta_testing_image_list)

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

		image_path = self.meta_testing_image_list[index]
		image_label = int(image_path.split('/')[-2])

		img = Image.open(image_path)
		im = transformations(img)
		label = torch.tensor(image_label).long()
		return im, label




if __name__ == '__main__':
	from PIL import Image
	import torch
	img_root = './data/mnist/'

	dataset = MetaTeSouMNIST(img_root)
	# sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
	# loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=1, pin_memory=True,sampler=sampler)
	print(len(dataset))
