import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable




class FC_Classifier(nn.Module):
    def __init__(self):
        super(FC_Classifier, self).__init__()
        self.fc_1 = nn.Linear(12 * 1024, 1024*6)
        self.fc_2 = nn.Linear(6 * 1024, 1024)
        self.fc_3 = nn.Linear(1024, 2)
        
    def forward(self, x):
        
        out = self.fc_1(x)
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out

def train(dataloader____, model, criterion, optimizer):
    
    correct_cnt = 0
    for idx, (vec, label) in enumerate(train_All_key_list):
        
        torch_vec = torch.from_numpy(vec)
        torch_vec = Variable(torch_vec)
        torch_vec = torch_vec.unsqueeze(0)
        
        labels = torch.LongTensor(1)
        label.fill_(label)
        label = Variable(label)
        
        
        #outputs = cnn(images)
        outputs = model(torch_vec.cuda())
        loss = criterion(outputs, label.cuda())
        
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
        smax = nn.Softmax()
        smax_out = smax(outputs)[0]
        prediect_label = np.argmax(smax_out.data)
        
        
        if prediect_label == label:
            correct_cnt += 1
        accuracy = correct_cnt * 100 / len(train_All_key_list)
        
        
        
        if idx % 100 == 0:
            print (idx, ':', prediect_label, label, '---->', (loss.data).cpu().numpy())
    print ('Train Accuracy :', accuracy)    

    
    
def val(dataloader____, model, criterion, optimizer):
    
    model.eval()
    correct_cnt = 0
    for idx, (vec, label) in enumerate(dataloader____):
        
        torch_vec = torch.from_numpy(vec)
        torch_vec = Variable(torch_vec)
        torch_vec = torch_vec.unsqueeze(0)
        
        labels = torch.LongTensor(1)
        label.fill_(label)
        label = Variable(label)
        
        
        #outputs = cnn(images)
        outputs = model(torch_vec.cuda())
        loss = criterion(outputs, label.cuda())
        
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
        smax = nn.Softmax()
        smax_out = smax(outputs)[0]
        prediect_label = np.argmax(smax_out.data)
        
        
        if prediect_label == label:
            correct_cnt += 1
        accuracy = correct_cnt * 100 / len(train_All_key_list)
        
        
        
        if idx % 100 == 0:
            print (idx, ':', prediect_label, label, '---->', correct)
    print ('Validation Accuracy :', accuracy)

                   
                   
                   
def run():
        
    Classifier = FC_Classifier()
    Classifier = torch.nn.DataParallel(Classifier).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(Classifier.parameters(), lr=0.00001)
    
    train_loader = 0
    val_loader = 0
    epochs = 100
    
    for epoch in range(epochs):
        train(train_loader, Classifier, criterion, optimizer)
        val(train_loader, Classifier, criterion)
                   