from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image


class DataloadFiles(Dataset):
    def __init__(self, data_path,labelfile, transform=None):

        list_dir = os.listdir(data_path)
        totalImages = len(list_dir)
        self.transform = transform
        self.data_path = data_path

        with open(labelfile,'r') as labelfile:
            labs = labelfile.readlines()
        self.len = totalImages
        self.labels = self.makelabels(labs)
    def __getitem__(self, index):
        image_name = os.path.join(self.data_path, (str(index)+'.jpg'))
        image = Image.open(image_name)
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = np.array(image)

        return img,label
    def __len__(self):
        return self.len

    def makelabels(self, lab):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(lab)
        return le.transform(lab)


    def nClasses(self):
        return len(np.unique(self.labels))

    def datasetSize(self):
        return self.len
