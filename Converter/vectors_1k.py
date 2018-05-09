import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
sys.path.append('..')
import os
import numpy as np
import pickle
from tqdm import tqdm



import CNN.models
import Tools.dataload as dataload
import Tools.path as path_cfg


def data_caller(test_dir, label_file):
    test_data = dataload.DataloadFiles(test_dir, label_file, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))

    test_loader = data.DataLoader(test_data, batch_size=test_batch_size,
                                  shuffle=False, drop_last=False)
    return test_loader


def test(test_loader,save_folder):
    total_vec_list = []

    for model in modelset:
        print('using model: ',str(model))
        checkpoint_path = os.path.join(check_point_dir, 'resnet18_' + model + '.pth.tar')
        CNN_model, _, _, _ = models.test_model_loader(checkpoint_path, model_name, dropouts=True)

        # switch to evaluate mode
        CNN_model.eval()
        print('length: ',str(len(test_loader)))

        vec_list = []
        for data in tqdm(test_loader):
            img, label = data
            label = label[0]
            img = Variable(img).cuda(0)

            output = CNN_model(img)
            vec = output.data.cpu().numpy()
            vec_list.append(vec)
            #print(vec)
            #break

        # write all output to the file
        with open(os.path.join(save_folder, (str(model) + '_output.pickle')), 'wb') as file:
            pickle.dump(vec_list, file)

test_batch_size = 128 
model_name_list = ['resnet18']
mean = [0.4606, 0.4737, 0.4678]
std = [0.0143, 0.0170, 0.0235]
test_dir1 = '../val/'
#test_dir2 = '../nfsdrive/landmark/FullTrainingSet/val/'
label_dir1 = '../val_labels.txt'
#label_dir1 = '../nfsdrive/landmark/FullTrainingSet/val_labels.txt'

#check_point_dir = '../weights/checkpoints/weights/'
check_point_dir = '../'
#modelset = ['3', '11', '21', '31', '41', '51', '61' ]
modelset = ['1k']
for idx, model_name in enumerate(model_name_list):
    test_loader1 = data_caller(test_dir1, label_dir1)
    test(test_loader1, '../vectors/val/')

 #   test_loader2 = data_caller(test_dir2, label_dir2)
 #   test(test_loader2, '../vectors/val/')
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

