import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from EnsembleDropout import *

import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.parallel

import sys
sys.path.append('../')

import os
import Tools.path as path_cfg



from torch.optim import lr_scheduler

def model_setter(model_name, learning_rate=0.001, output_size=2, usePretrained=True, isTest=False, dropouts=True):

    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        if dropouts:
            en = EnsembleDropout()
            model.fc = nn.Sequential(en, nn.Linear(num_ftrs, output_size))
        else:
            model.fc = nn.Linear(num_ftrs, output_size)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_size)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_size)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_size)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_size)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=usePretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_size)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=usePretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_size)
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=usePretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_size)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=usePretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_size)

    model = torch.nn.DataParallel(model).cuda()
    if not isTest:
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        optimizer = 0
        criterion = 0      
        scheduler = 0
    return model, optimizer, criterion, scheduler




def save_checkpoint(state, is_best, model_name):
    filename = os.path.join(path_cfg.snapshot_root_path, model_name + '.pth.tar')
    torch.save(state, filename)

def load_checkpoint(model, model_name):
    path = os.path.join(path_cfg.snapshot_root_path, model_name + '.pth.tar')
    if os.path.exists(path):
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print ('    best accr : ', best_prec1)
    else:
        print ('no check point')
    return model

def __pretrained_model_converter(model, new_output_size, dropouts):
    if dropouts:
        num_ftrs = model.module.fc[1].in_features
        en = EnsembleDropout()
        model.module.fc = nn.Sequential(en, nn.Linear(num_ftrs, new_output_size)).cuda()
    else:
        num_ftrs = model.module.fc.in_features
        model.module.fc = nn.Linear(num_ftrs, new_output_size).cuda()
    return model

def rollingWeightLoader(checkpoint_path, model_name, learning_rate, new_num_of_class, dropouts=True):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if dropouts:
            num_of_class = checkpoint['state_dict']['module.fc.1.weight'].shape[0]
        else:
            num_of_class = checkpoint['state_dict']['module.fc.weight'].shape[0]
        CNN_model, CNN_optimizer, CNN_criterion, CNN_scheduler = model_setter(model_name, 
                                                                          learning_rate, 
                                                                          output_size=num_of_class, dropouts=dropouts)

        CNN_model.load_state_dict(checkpoint['state_dict'])

        # This module gonna change last output size for prediction
        # Because the number of rolling data classes and training data classes are different.
        __pretrained_model_converter(CNN_model, new_num_of_class, dropouts)

        return CNN_model, CNN_optimizer, CNN_criterion, CNN_scheduler

def test_model_loader(checkpoint_path, model_name, dropouts=False):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if dropouts:
            num_of_class = checkpoint['state_dict']['module.fc.1.weight'].shape[0]
        else:
            num_of_class = checkpoint['state_dict']['module.fc.weight'].shape[0]
        CNN_model, CNN_optimizer, CNN_criterion, CNN_scheduler = model_setter(model_name, 
                                                                                     output_size=num_of_class,
                                                                                     isTest=True, 
                                                                                     dropouts=dropouts)

        CNN_model.load_state_dict(checkpoint['state_dict'])

        return CNN_model, CNN_optimizer, CNN_criterion, CNN_scheduler
