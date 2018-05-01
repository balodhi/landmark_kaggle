import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import models
import dataload_files as dataload
import path as path_cfg
import numpy as np
import pickle


def data_caller(test_dir, label_file):
    test_data = dataload.DataloadFiles(test_dir, label_file, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))

    test_loader = data.DataLoader(test_data, batch_size=test_batch_size,
                                  shuffle=False, drop_last=False)
    return test_loader


def test(test_loader):
    total_vec_list = []

    for model in modelset:
        checkpoint_path = os.path.join(check_point_dir, 'resnet18_' + model + '.pth.tar')
        CNN_model, _, _, _ = models.test_model_loader(checkpoint_path, model_name, dropouts=True)

        # switch to evaluate mode
        CNN_model.eval()

        vec_list = []
        for i, (img, label) in enumerate(test_loader):
            label = label[0]
            img = Variable(img).cuda()

            output = CNN_model(img)
            vec = output.data.cpu().numpy()[0]
            vec_list.append(vec)
            #print(vec)
            #break

        # write all output to the file
        with open((str(model) + '_output.pickle'), 'wb') as file:
            pickle.dump(vec_list, file)

test_batch_size = 1
model_name_list = ['resnet18']
mean = [0.4606, 0.4737, 0.4678]
std = [0.0143, 0.0170, 0.0235]
test_dir = '../11/AugmentedTrainingSet/'
label_dir = './t.csv'
check_point_dir = '../weights/checkpoints/weights/'
modelset = ['3', '11', '21', '31', '41', '51', '61', '71', '81', '91', '101', '201', '301', '401', '501', '601', '701',
            '801', '901', '1001']
for idx, model_name in enumerate(model_name_list):
    test_loader = data_caller(test_dir, label_dir)
    test(test_loader)
'''
f = open('vectors.pickle', 'w')
pickle.dump(info_dict, f)
f.close()
'''








'''
        if len(total_vec_list) < 1:
            for idx, (img, label) in enumerate(test_loader):
                total_vec_list.append((label[0], vec_list[idx]))
        else:
            for idx, (img, label) in enumerate(test_loader):
                mergedlist = np.append(total_vec_list[idx][1], vec_list[idx])
                total_vec_list[idx] = ((total_vec_list[idx][0], mergedlist))
'''

def ____data_caller(data_type):
    test_dir = os.path.join(path_cfg.data_root_path, data_type)

    test_data = dataload_bilal.dataload_test(test_dir, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))

    test_loader = data.DataLoader(test_data, batch_size=test_batch_size,
                                  shuffle=False, drop_last=False)
    return test_loader

