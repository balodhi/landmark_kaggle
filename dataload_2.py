from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import pickle

class dataload_2(Dataset):
    def __init__(self, data_path, transform=None):
        # it accepts the directory as input and read all the pickle files
        # data_path is the directory path of pickle files
       # data = []
        list_dir = os.listdir(data_path)
        #list_dir.sort()
        #list_dir = sorted(list_dir)
        totalImages = 0
        self.transform = transform
        self.data_path = data_path
        labs = []
        npklfiles = 0
        for file in list_dir:
            if file.endswith('.pickle'):
                npklfiles += 1
        self.fileslist = range(0, npklfiles)
        self.images = []
        self.labels = []
        for file in self.fileslist:
            with open(os.path.join(data_path, (str(file)+'.pickle')), 'rb') as pickleFile:
                pickleFile.seek(0)
                a = pickle.load(pickleFile)
                totalImages += len(a)
                for im in a:
                    labs.append(im[0])
        self.transform = transform
        self.len = totalImages
        #read the first pickle and add it to self.images
        with open(os.path.join(data_path, ('0.pickle')), 'rb') as pickleFile:
            pickleFile.seek(0)
            a = pickle.load(pickleFile)
        for img in a:
            self.images.append(img[1])
        #print(self.len, 'adfasdf', str(totalImages))
        self.labels = self.makelabels(labs)
        #self.nClases = len(np.unique(self.labels))
        self.pickle_idx= 0

    def __getitem__(self, index):
        #print('requested index is:',str(index))
        idx = index // 1000
        if idx == self.pickle_idx:
            image = self.images[index - 1000 * idx]
        else:
            self.pickle_idx += 1
            with open(os.path.join(self.data_path, (str(self.pickle_idx)+'.pickle')), 'rb') as pickleFile:
                pickleFile.seek(0)
                a = pickle.load(pickleFile)
            self.images = []
            for img in a:
                self.images.append(img[1])
            image = self.images[0]


        #print('calculated file id is:',str(fileid), 'and image id is: ',str(imageid))
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