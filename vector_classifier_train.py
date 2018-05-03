import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import dataloader_concat as dt
import torch.utils.data as data
import tools


class FC_Classifier(nn.Module):
    def __init__(self):
        super(FC_Classifier, self).__init__()
        self.fc_1 = nn.Linear(6583, 650)
        self.fc_2 = nn.Linear(650, 658)
        self.fc_3 = nn.Linear(658, 14951)
        
    def forward(self, x):
        
        out = self.fc_1(x)
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out

def train(dataloader____, model, criterion, optimizer):
    
    correct_cnt = 0
    for idx, (vec, label) in enumerate(dataloader____):


        #torch_vec = torch.from_numpy(vec)
        torch_vec = vec
        torch_vec = Variable(torch_vec)
        #torch_vec = torch_vec.unsqueeze(0)
        
        #labels = torch.LongTensor(1)
        #labels.fill_(label)
        aaa = label
        label = Variable(label)
        
        
        #outputs = cnn(images)
        outputs = model(torch_vec.cuda())
        loss = criterion(outputs, label.cuda())
        
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
        


        smax = nn.Softmax()
        smax_out = smax(outputs)

        print label.shape
        prec1, prec1 = tools.Accuracy(smax_out.data, aaa, topk=(1, 1))

        print prec1, '!!!!'
        '''
        prediect_label = np.argmax(smax_out.data, axis=1)
        #if prediect_label == label.data.numpy():
        #    correct_cnt += 1
        #accuracy = correct_cnt * 100 / len(dataloader____)
        
        
        if idx % 100 == 0:
            print (idx, ':', prediect_label, label.data.numpy(), '---->', (loss.data).cpu().numpy())
        '''
    print ('Train Accuracy :', accuracy)    

    
    
def val(dataloader____, model, criterion, optimizer):
    
    model.eval()
    correct_cnt = 0
    for idx, (vec, label) in enumerate(dataloader____):
        
        #torch_vec = torch.from_numpy(vec)
        torch_vec = Variable(torch_vec)
        #torch_vec = torch_vec.unsqueeze(0)
        
        #labels = torch.LongTensor(1)
        #label.fill_(label)
        label = Variable(label)
        
        
        #outputs = cnn(images)
        outputs = model(torch_vec.cuda())
        loss = criterion(outputs, label.cuda())
        
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
        smax = nn.Softmax()
        smax_out = smax(outputs)
        prediect_label = np.argmax(smax_out.data)
        print smax_out.data, '!!!'
        print prediect_label, '!!!'
        
        #if prediect_label == label:
        #    correct_cnt += 1
        #accuracy = correct_cnt * 100 / len(train_All_key_list)
        
        
        
        if idx % 100 == 0:
            print (idx, ':', prediect_label, label.data.numpy(), '---->', correct)
    print ('Validation Accuracy :', accuracy)

                   
                   
                   
def run():

    data_path = '/media/hwejin/SSD_1/DATA/val/data'
    label_path = '/media/hwejin/SSD_1/DATA/val/label'
    label_dict_path = '/media/hwejin/SSD_1/DATA/val/encoded_label.pickle'
    data_type = 'val'

    data_loader = dt.dataload_concat(data_path, label_path, label_dict_path, data_type)
    train_loader = data.DataLoader(data_loader, batch_size=3,
                                shuffle=False,drop_last=False)


    
    Classifier = FC_Classifier()
    Classifier = torch.nn.DataParallel(Classifier).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(Classifier.parameters(), lr=0.00001)

    val_loader = train_loader
    epochs = 100
    
    for epoch in range(epochs):
        train(train_loader, Classifier, criterion, optimizer)
        val(train_loader, Classifier, criterion)
                   
run()