from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image


class dataload_fc(Dataset):
	def __init__(self,data_path,label_file):
		list_dir = os.listdir(data_path)
		total_pickles = len(list_dir)
		self.data = []
		for pickle in total_pickles:
			with open(os.path.join(data_path,pickle),'rb') as file:
				d = pickle.load(file)
			self.data.append(d)
			self.data = torch.concat(self.data,dim=3)
		#load labels
		with open(label_file,'r') as file:
			self.labs = file.readlines()  #remove '\n'
		self.len = len(self.data)
		with open('../labelPickle.pickle', 'r') as file:
			self.labelDic = pickle.load(file)

	def __getitem__(self,index):
		data = self.data[index]
		lab = self.labls[index]
		label = self.labelDic[lab]

		return data, label
	def __len__(self):
		return self.len

			se