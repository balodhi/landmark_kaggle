from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import itertools
from sklearn.preprocessing import normalize
import csv


class dataload_val(Dataset):
    def __init__(self, data_path, label_path, label_dict_path, data_type, totalFiles, transform=None):
        self.model_sets = ['3', '11', '21', '31', '41', '51', '61', '71', '81', '91', '101', '201', '301', '401', '501',
                           '601', '701', '801', '901', '1001', '1k']
        #self.model_sets = ['3', '11']
        # self.model_sets = ['101','1001']
        list_dir = os.listdir(data_path)
        totalImages = 0
        self.transform = transform
        self.data_path = data_path
        self.totalFiles = totalFiles

        #read the label file for the modelset
        lines = []
        labels = []
        with open(label_path) as csvfile:
            readcsv = csv.reader(csvfile,delimiter=',')
            for row in readcsv:

                lines.append(row[0])
                labels.append(row[1])

        totalImages = len(lines)
        self.lines = lines
        self.labels = labels


        # load label dictionary

        with open(label_dict_path, 'rb') as file:
             self.label_dict = pickle.load(file,encoding='latin1')
            #self.label_dict = pickle.load(file)

        self.len = totalImages
        # print(self.len,self.labels[40000],"adsfadsf")

        # read all pickles pickle from all model sets and add it to self.data
        self.data = []

        for model in self.model_sets:
            data = []
            for i in range(0, 6):
                file_name = os.path.join(self.data_path, (str(model) + '_output'),
                                         (str(model) + '_output' + str(i) + '.pickle'))
                with open(file_name, 'rb') as file:
                    file.seek(0)
                    a = pickle.load(file)
                    data.append(a)
            a = np.concatenate((data), axis=0)
            #print(len(a),"!!!!!")
            self.data.append(a)

        self.data = np.concatenate((self.data), axis=1)
        print(len(self.lines))


    def __getitem__(self, index):
        line = self.lines[index]

        label_index = self.labels[index]
        #print(str(label_index))
        label = self.label_dict[str(label_index)]
        data = self.data[int(line)]

        return data, label

    def __len__(self):
        return self.len