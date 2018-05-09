import sys
sys.path.append('..')

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import pickle


import Net
import Tools.dataload as dt
import Tools.tools as tools

input_size = 14951
hidden_size = 8000
num_classes = 14951
batch_size = 1


skip_list = [2651, 33645, 41595, 53389, 91057, 97840, 117013]
'''
#val_dataset = dt.dataload_val('/hdd1/data_set/val/', '/hdd1/data_set/sorted/sorted_val_11.txt', '/hdd1/data_set/encoded_label.pickle', 'val',5)
val_dataset = dt.dataload_val('/media/hwejin/SSD_1/DATA/temp_pickles', 
	'/media/hwejin/SSD_1/DATA/temp_pickles/sorted/sorted_val_11.txt', 
	'/media/hwejin/SSD_1/DATA/temp_pickles/encoded_label.pickle', 'val',5)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

'''



test_dataset = dt.dataload_test_concat('../../vectors/test', skip_list)
'''
test_dataset = dt.dataload_val('/media/hwejin/SSD_1/DATA/temp_pickles', 
	'/media/hwejin/SSD_1/DATA/temp_pickles/sorted/sorted_val_11.txt', 
	'/media/hwejin/SSD_1/DATA/temp_pickles/encoded_label.pickle', 'val',5)
'''
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
print (len(test_dataset))

#f = open('/media/hwejin/SSD_1/DATA/temp_pickles/encoded_label.pickle')
f = open('/hdd1/data_set/encoded_label.pickle','rb')
encoded_label = pickle.load(f,encoding='latin1')
#encoded_label = pickle.load(f)
f.close()




def __sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

def __softmax(x):                                        
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def __label_convert(label_dict, label):
    #converted_label_dict = dict((v,k) for k,v in label_dict.iteritems())
    converted_label_dict = dict((v,k) for k,v in label_dict.items())
    return converted_label_dict[label]

def load_model(checkpoint_path,model):
    print("loading the model")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    else:
        print("no checkpoint found")
    return model

def val(data_loader, net, criterion):
    losses = tools.AverageMeter()
    acc = tools.AverageMeter()

    net.eval()
    loop = len(data_loader)
    for i, (images, labels) in enumerate(data_loader):
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

def test(data_loader, net, criterion):
	losses = tools.AverageMeter()
	acc = tools.AverageMeter()
	net.eval()
	loop = len(data_loader)
	f = open('/hdd1/data_set/prediction.csv', 'w')
	f.write('id,landmarks\n')
	for i, (images) in enumerate(data_loader):
		if len(images.shape) == 1:
			f.write("'"+str(idx) +"'"+ ', \n')
		else:
			images = Variable(images.view(-1, 14951)).cuda()
			#labels = Variable(labels).cuda()
			outputs = net(images)
			numpy_output = outputs.cpu().data.numpy()
			for idx in range(numpy_output.shape[0]):
				softmax_value = __softmax(numpy_output[idx])
				max_value = np.max(softmax_value)
				max_label = np.argmax(softmax_value)
				converted_label = __label_convert(encoded_label, max_label)
				f.write("'"+str(idx) +"'"+ ','+ str(converted_label) + ' ' + str(max_value) + '\n')
	f.close()
	return 1



def validation_run():
    net = Net.Net(input_size, hidden_size, num_classes)
    net = load_model('../../landmark/model2.pkl',net)
    #net = load_model('/media/hwejin/SSD_1/DATA/temp_pickles/model2.pkl',net)
    net.cuda()
    criterion = nn.CrossEntropyLoss()

    print("validation")
    acc = val(val_loader, net, criterion)


def test_run():
    print("test")
    net = Net.Net(input_size, hidden_size, num_classes)
    net = load_model('../../landmark/model2.pkl',net)
    #net = load_model('/media/hwejin/SSD_1/DATA/temp_pickles/model2.pkl',net)
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    acc = test(test_loader, net, criterion)
    
#validation_run()
test_run()
