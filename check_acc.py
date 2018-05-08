import torch
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import dataload_val as dt
import tools
import dataload_concat as dt2

from torch.optim import lr_scheduler
import os


input_size = 14951
hidden_size = 8000
num_classes = 14951
batch_size = 1000

#train_dataset = dt.dataload_val('/hdd1/data_set/data_set/train/', '/hdd1/data_set/data_set/train/train', '/hdd1/data_set/data_set/encoded_label.pickle', 'train',55)
val_dataset = dt.dataload_val('/hdd1/data_set/val/', '/hdd1/data_set/sorted/sorted_val_11.txt', '/hdd1/data_set/encoded_label.pickle', 'val',5)
#val_dataset = dt2.dataload_concat('/hdd1/data_set/val/', '/hdd1/data_set/val/val', '/hdd1/data_set/encoded_label.pickle', 'val',5)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

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

def load_model(checkpoint_path,model):
    print("loading the model")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    else:
        print("no checkpoint found")
    return model

def val(data_loader, net, criterion):
    #net.train(False)
    losses = tools.AverageMeter()
    acc = tools.AverageMeter()

    net.eval()
    loop = len(data_loader)
    for i, (images, labels) in enumerate(data_loader):
        #images, labels = data_loader[i]
        # Convert torch tensor to Variable
        #print(images)
        #print (labels,"!!!")
        images = Variable(images.view(-1, 14951)).cuda()
        labels = Variable(labels).cuda()



        outputs = net(images)
        loss = criterion(outputs, labels)

        prec1, prec1 = tools.Accuracy(outputs, labels, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))


        #if (i + 1) % 10 == 0:
        print('Step [%d], Loss: %.4f'
                  % (i + 1, loss.data[0]))
        print('Accuracy : ', (prec1.cpu().data.numpy()[0]))
        print('Accuracy (All): ', acc.avg.cpu().data.numpy()[0])
        print('Losses (All): ', losses.avg)
        print()

    return acc.avg

def run():
    net = Net(input_size, hidden_size, num_classes)
    net = load_model('./model2.pkl',net)
    net.cuda()

    # Loss and Optimizer

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print("validation")
    acc = val(val_loader, net, criterion)


run()
