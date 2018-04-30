from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
from PIL import Image

class dataload(Dataset):
    def __init__(self,data_path,transform=None):
        #it accepts the directory as input and read all the pickle files
        #data_path is the directory path of pickle files
        data=[]
        list_dir = os.listdir(data_path)
        list_dir.sort()
        list_dir = sorted(list_dir)
        self.transform = transform
        for file in list_dir:
            
            if file.endswith('.pickle'):
                with open(os.path.join(data_path,file), 'rb') as pickleFile:
                    pickleFile.seek(0)
                    a = pickle.load(pickleFile)
                    data.append(a)
        
        labs = []
        self.images=[]
        
        self.transform = transform
        for picklefile in data:
            for point in picklefile:
                labs.append(point[0])
                self.images.append(point[1])
                #print(point[0])

        self.labels=self.makelabels(labs)
        self.len = len(self.labels)
        #self.nClases = len(np.unique(self.labels))
            
    def __getitem__(self,index):
        
        if self.transform is not None:
            img = self.transform(self.images[index])
        else:
            
            img= np.array(self.images[index])
            
        return img, self.labels[index]
    
    def __len__(self):
        return self.len
    def makelabels(self,lab):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(lab)
        return le.transform(lab)
    def nClasses(self):
        return len(np.unique(self.labels))
    def datasetSize(self):
        return self.len

    
    
    
class dataload_test(Dataset):
    def __init__(self,data_path,transform=None):
        #it accepts the directory as input and read all the pickle files
        #data_path is the directory path of pickle files
        data=[]
        list_dir = os.listdir(data_path)
        list_dir.sort()
        list_dir = sorted(list_dir)
        self.transform = transform
        for file in list_dir:
            
            if file.endswith('.pickle'):
                with open(os.path.join(data_path,file), 'rb') as pickleFile:
                    pickleFile.seek(0)
                    a = pickle.load(pickleFile)
                    data.append(a)
        
        labs = []
        self.images=[]
        
        self.transform = transform
        for picklefile in data:
            for point in picklefile:
                labs.append(point[0])
                self.images.append(point[1])
                #print(point[0])

        self.labels=labs
        self.len = len(self.labels)
        #self.nClases = len(np.unique(self.labels))
            
    def __getitem__(self,index):
        
        if self.transform is not None:
            img = self.transform(self.images[index])
        else:
            
            img= np.array(self.images[index])
            
        return img, self.labels[index]
    
    def __len__(self):
        return self.len
    def nClasses(self):
        return len(np.unique(self.labels))
    def datasetSize(self):
        return self.len

    
    
class dataload_csv(Dataset):
    def __init__(self, data_path, labelfile, transform=None):

        list_dir = os.listdir(data_path)
        totalImages = len(list_dir)
        self.transform = transform
        self.data_path = data_path

        with open(labelfile,'r') as labelfile:
            labs = labelfile.readlines()
        self.len = totalImages
        self.labels=labs
    def __getitem__(self, index):
        image_name = os.path.join(self.data_path, (str(index)+'.png'))
        image = Image.open(image_name)
        label = self.labels[index][:-1]
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = np.array(image)

        return img,label
    def __len__(self):
        return self.len
    def nClasses(self):
        return len(np.unique(self.labels))

    def datasetSize(self):
        return self.len