from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
from PIL import Image
import itertools
from sklearn.preprocessing import normalize
import csv
from tqdm import tqdm

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




class dataload_val(Dataset):
    def __init__(self, data_path, label_path, label_dict_path, data_type, totalFiles, transform=None):
        self.model_sets = ['3', '11', '21', '31', '41', '51', '61', '71', '81', '91', '101', '201', '301', '401', '501',
                           '601', '701', '801', '901', '1001', '1k']
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
            self.label_dict = pickle.load(file)

        self.len = totalImages

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
            self.data.append(a)

        self.data = np.concatenate((self.data), axis=1)
        print(len(self.lines))


    def __getitem__(self, index):
        line = self.lines[index]

        label_index = self.labels[index]
        label = self.label_dict[str(label_index)]
        data = self.data[int(line)]

        return data, label

    def __len__(self):
        return self.len



class dataload_concat_2(Dataset):
    def __init__(self,data_path,label_path,label_dict_path,data_type,transform=None):
        self.model_sets = ['3', '11', '21', '31', '41', '51', '61', '71', '81', '91', '101', '201', '301', '401', '501', '601', '701', '801', '901', '1001', '1k']
        self.model_sets = ['3', '11']
        #self.model_sets = ['101','1001']
        list_dir = os.listdir(data_path)
        totalImages = 0
        self.transform = transform
        self.data_path = data_path

        labs = []
        lab_dir = os.listdir(label_path)
        npklfiles = 0
        for file in lab_dir:
            if file.endswith('.pickle'):
                npklfiles += 1

        #load all labels
        label_files_list = range(0, npklfiles)
        for file_id in label_files_list:
            file_name = os.path.join(label_path,(data_type+'_labels_'+str(file_id)+'.pickle'))
            with open(file_name,'rb') as file:
                file.seek(0)
                a = pickle.load(file)
                labs.append(a)
                totalImages += len(a)
        
        self.labels = list(itertools.chain.from_iterable(labs))
        #load label dictionary
        
        with open(label_dict_path,'rb') as file:
            #self.label_dict = pickle.load(file,encoding='latin1')
            self.label_dict = pickle.load(file)

        self.len = totalImages
        #print(self.len,self.labels[40000],"adsfadsf")

        #read the first pickle from all model sets and add it to self.data
        self.data = []
        for model in self.model_sets:
            file_name = os.path.join(data_path,(str(model)+'_output'),(str(model)+'_output0.pickle'))

            with open(file_name,'rb') as file:
                file.seek(0)
                a = pickle.load(file)
                self.data.append(a)
        self.data = np.concatenate((self.data),axis=1)
        #print(len(self.data))
        self.pickle_idx = 0

    def __getitem__(self,index):
        #print(index)
        d_size = 20000
        idx = index // d_size
        if idx == self.pickle_idx:
            data = self.data[index - d_size * idx]
        else:
            
            self.pickle_idx += 1
            print('reading new file:', self.pickle_idx)
            self.data = []
            for model in self.model_sets:
                file_name = os.path.join(self.data_path,(str(model)+'_output'),(str(model)+'_output'+str(self.pickle_idx)+'.pickle'))

                with open(file_name,'rb') as file:
                    file.seek(0)
                    a = pickle.load(file)
                    self.data.append(a)
            self.data = np.concatenate((self.data),axis=1)
            data = self.data[0]

        #label
        label_index = self.labels[index]
        label = self.label_dict[str(label_index)]
        #print len(self.label_dict)

        return data,label
    def __len__(self):
        return self.len



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
             #self.label_dict = pickle.load(file,encoding='latin1')
            self.label_dict = pickle.load(file)

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

class dataload_test(Dataset):
    def __init__(self, data_path, transform=None):
        imagefiles =  os.listdir(data_path)
        self.data = []

        files = range(0,len(imagefiles))
        for id in tqdm(files):
            image = str(id)+".pickle"
            filename = os.path.join(data_path, image)
            with open(os.path.join(data_path,filename), 'rb') as picklefile:
                images = pickle.load(picklefile)
            for im in images:
                self.data.append(im)
            #im.fp.close()
        self.len = len(self.data)
        #self.len = 11
        self.transform = transform

    def __getitem__(self, index):

        img =  self.data[index]
        if self.transform is not None:

            img = self.transform(img)
        img = np.array(img)
       # print("adsfasdfasdfadsf")
       # print(img.shape,"@@@@@@@")

        return img

    def __len__(self):
        return self.len
class dataload_test_concat(Dataset):
    def __init__(self, data_path, skip_files, transform=None):
        self.model_sets = ['3', '11', '21', '31', '41', '51', '61', '71', '81', '91', '101', '201', '301', '401', '501',
                           '601', '701', '801', '901', '1001', '1k']

        self.transform = transform
        self.data_path = data_path
        self.skip_files = skip_files

        self.data = []

        for model in self.model_sets:
            file_name = os.path.join(self.data_path, (str(model) + '_output.pickle'))
            with open(file_name, 'rb') as file:
                file.seek(0)
                a = pickle.load(file)
                self.data.append(a)
        self.data = np.concatenate((self.data), axis=2)
        self.len = len(self.data) + len(self.skip_files)
        self.skip = 0

    def __getitem__(self, index):

        if index in self.skip_files:
            print("image in the skip list")
            self.skip += 1
            return -1

        img = self.data[index - self.skip]
        if self.transform is not None:
            img = self.transform(img)
        img = np.array(img)
        return img

    def __len__(self):
        return self.len
