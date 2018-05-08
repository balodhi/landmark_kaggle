import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
#import dataload_concat as dt
import sys
sys.path.append('..')

import Tools.dataload as dt
import Tools.tools

from torch.optim import lr_scheduler
# Hyper Parameters
input_size = 14951
hidden_size = 8000
num_classes = 14951
num_epochs = 50
batch_size = 300
learning_rate = 0.001

train_dataset = dt.dataload_concat('/hdd1/data_set/train/', '/hdd1/data_set/train/train', '/hdd1/data_set/encoded_label.pickle', 'train',55)
val_dataset = dt.dataload_concat('/hdd1/data_set/val/', '/hdd1/data_set/val/val', '/hdd1/data_set/encoded_label.pickle', 'val',5)
#val_dataset = dt.dataload_concat('/media/hwejin/SSD_1/DATA/temp_pickles', '/media/hwejin/SSD_1/DATA/temp_pickles/val', '/media/hwejin/SSD_1/DATA/temp_pickles/encoded_label.pickle', 'val',5)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)



# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size*2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        #out = self.relu(out)
        out = self.fc2(out)
        #out = self.fc4(out)
        #out = self.relu(out)
        out = self.fc3(out)

        return out



def train(data_loader, net, criterion, optimizer, scheduler, epoch):
    losses = tools.AverageMeter()
    acc = tools.AverageMeter()
    scheduler.step()
    net.train(True)
    loop = len(data_loader)



    for i, (images, labels) in enumerate(data_loader):

        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 14951)).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prec1, prec1 = tools.Accuracy(outputs, labels, topk=(1, 1))

        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))


        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
            print('Accuracy : ', (prec1.cpu().data.numpy()[0]))
            print('Accuracy (All): ', acc.avg.cpu().data.numpy()[0])
            print('Losses (All): ', losses.avg)
            print()


def val(data_loader, net, criterion, scheduler, epoch):
    net.train(False)
    losses = tools.AverageMeter()
    acc = tools.AverageMeter()

    net.eval()
    loop = len(data_loader)
    for i, (images, labels) in enumerate(data_loader):
        #images, labels = data_loader[i]
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 14951)).cuda()
        labels = Variable(labels).cuda()



        outputs = net(images)
        loss = criterion(outputs, labels)

        prec1, prec1 = tools.Accuracy(outputs, labels, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))


        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(val_dataset) // batch_size, loss.data[0]))
            print('Accuracy : ', (prec1.cpu().data.numpy()[0]))
            print('Accuracy (All): ', acc.avg.cpu().data.numpy()[0])
            print('Losses (All): ', losses.avg)
            print()
    return acc.avg


def run():
    net = Net(input_size, hidden_size, num_classes)
    net.cuda()

    # Loss and Optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    for epoch in range(num_epochs):
        print(len(train_loader),"!!!!")
        train(train_loader, net, criterion, optimizer, scheduler, epoch)
        print("validation")
        acc = val(val_loader, net, criterion, scheduler, epoch)

    # Save the Model
    torch.save(net.state_dict(), 'model2.pkl')

run()

'''

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 14951)).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prec1, prec1 = tools.Accuracy(outputs, labels, topk=(1, 1))




        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
            print((prec1.cpu().data.numpy()), '!!!!')
'''

'''
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
'''
