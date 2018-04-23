from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import pickle

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
