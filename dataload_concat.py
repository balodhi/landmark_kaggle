from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import itertools
from sklearn.preprocessing import normalize


class dataload_concat(Dataset):
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

        labs = []
        lab_dir = os.listdir(label_path)
        npklfiles = 0
        for file in lab_dir:
            if file.endswith('.pickle'):
                npklfiles += 1

        # load all labels
        label_files_list = range(0, npklfiles)
        for file_id in label_files_list:
            file_name = os.path.join(label_path, (data_type + '_labels_' + str(file_id) + '.pickle'))
            with open(file_name, 'rb') as file:
                file.seek(0)
                a = pickle.load(file)
                labs.append(a)
                totalImages += len(a)

        self.labels = list(itertools.chain.from_iterable(labs))
        # load label dictionary

        with open(label_dict_path, 'rb') as file:
             self.label_dict = pickle.load(file,encoding='latin1')
            #self.label_dict = pickle.load(file)

        self.len = totalImages
        # print(self.len,self.labels[40000],"adsfadsf")

        # read the first pickle from all model sets and add it to self.data
        self.data = []
        for model in self.model_sets:
            file_name = os.path.join(data_path, (str(model) + '_output'), (str(model) + '_output0.pickle'))

            with open(file_name, 'rb') as file:
                file.seek(0)
                a = pickle.load(file)
                self.data.append(a)
        self.data = np.concatenate((self.data), axis=1)
        # print(len(self.data))
        self.pickle_idx = 0

    def __getitem__(self, index):
        # print(index)
        d_size = 20000
        idx = index // d_size
        if idx == self.pickle_idx:
            data = self.data[index - d_size * idx]
            #data = normalize(data[:, np.newaxis], norm='l1', axis=0).ravel()
        else:

            self.pickle_idx += 1
            if self.pickle_idx > self.totalFiles:
                print('reading again zero file:', self.pickle_idx)
                self.data = []
                for model in self.model_sets:
                    file_name = os.path.join(self.data_path, (str(model) + '_output'), (str(model) + '_output0.pickle'))

                    with open(file_name, 'rb') as file:
                        file.seek(0)
                        a = pickle.load(file)
                        self.data.append(a)
                self.data = np.concatenate((self.data), axis=1)
                self.pickle_idx = 0
            else:
                print('reading new file:', self.pickle_idx)
                self.data = []
                for model in self.model_sets:
                    file_name = os.path.join(self.data_path, (str(model) + '_output'),
                                             (str(model) + '_output' + str(self.pickle_idx) + '.pickle'))

                    with open(file_name, 'rb') as file:
                        file.seek(0)
                        a = pickle.load(file)
                        self.data.append(a)
                self.data = np.concatenate((self.data), axis=1)
            data = self.data[0]
            #data = normalize(data[:, np.newaxis], norm='l1', axis=0).ravel()


        # label
        label_index = self.labels[index]
        label = self.label_dict[str(label_index)]
        #print(len(data))

        return data, label

    def __len__(self):
        return self.len