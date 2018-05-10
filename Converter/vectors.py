import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable



import sys
sys.path.append('../')
import os
import numpy as np
import pickle
from tqdm import tqdm

import CNN.models
import Tools.dataload as dataload
import Tools.path as path_cfg

test_batch_size = 128 
model_name_list = ['resnet18']
mean = [0.4606, 0.4737, 0.4678]
std = [0.0143, 0.0170, 0.0235]
train_list_path = '/media/hwejin/SSD_1/Code/Github/landmark_kaggle/vec_/val_set.pickle'
check_point_dir = '/media/hwejin/SSD_1/DATA/landmark/checkpoints/withdropout'
out_path = '../vectors/train/'
modelset = ['3', '11', '21', '31', '41', '51', '61', '71', '81', '91', '101', '201', '301', '401', '501', '601', '701', '801', '901', '1001', '1k']
modelset = ['3']

def __data_caller(train_list_path):
	f = open(train_list_path)
	train_list = pickle.load(f)
	f.close()

	pred_data = dataload.Dataload_vector(train_list, transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)]))
	pred_loader = data.DataLoader(pred_data, batch_size=test_batch_size,shuffle=False, drop_last=False)
	return pred_loader


def run(test_loader,save_folder):
    total_vec_list = []

    for model in modelset:
        print('using model: ',str(model))
        checkpoint_path = os.path.join(check_point_dir, 'resnet18_' + model + '_rew.pth.tar')
        CNN_model, _, _, _ = CNN.models.test_model_loader(checkpoint_path, model_name, dropouts=True)

        # switch to evaluate mode
        CNN_model.eval()
        print('length: ',str(len(test_loader)))

        vec_list = []
        for data in tqdm(test_loader):
        	img, label, key, linenumber, image_path = data
        	img = Variable(img).cuda(0)
        	output = CNN_model(img)
        	vec = output.data.cpu().numpy()
        	vec_list.append((vec, label, key, linenumber, image_path))

        # write all output to the file
        with open(os.path.join(save_folder, (str(model) + '_output.pickle')), 'wb') as file:
            pickle.dump(vec_list, file)


for idx, model_name in enumerate(model_name_list):
    pred_loader = __data_caller(train_list_path)
    run(pred_loader, out_path)
