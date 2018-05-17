import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as ds
import torchvision.models as torchmodels
from torch.autograd import Variable

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn.parallel
import torch.nn as nn
import os
import numpy as np
import time
import sys
import argparse

import sys
sys.path.append('../')
import models
import Tools.tools as tools
import Tools.dataload as dataload_landmark
import Tools.path as path_cfg
import Tools.normalize as nml_cfg

model_name_list = ['resnet18']



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = tools.AverageMeter()
    losses = tools.AverageMeter()
    acc = tools.AverageMeter()    
    end = time.time()
    
    for i, (img, label) in enumerate(train_loader):


        target = label.cuda(async=True)
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        output = model(img)
        loss = criterion(output,label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec1 = tools.Accuracy(output.data, target, topk=(1, 1))
        losses.update(loss.data[0], img.size(0))
        acc.update(prec1[0], img.size(0))        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()        
        if i % 50 == 0:
            print('\tEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, loss=losses, acc=acc))
    return acc.avg     
            
def val(val_loader, model, criterion):
    batch_time = tools.AverageMeter()
    losses = tools.AverageMeter()
    acc = tools.AverageMeter()    
    end = time.time()
    
    
    model.eval()
    for i, (img, label) in enumerate(val_loader):
    

        target = label.cuda(async=True)
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        
        
        output = model(img)
        loss = criterion(output,label)
        
        
        
        # measure accuracy and record loss
        prec1, prec1 = tools.Accuracy(output.data, target, topk=(1, 1))
        losses.update(loss.data[0], img.size(0))
        acc.update(prec1[0], img.size(0))
        
        
        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()  
        
        
        if i % 100 == 0:
            print('\tVal: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))
            
            
    print(' * Accuracy {acc.avg:.3f}'.format(acc=acc))

    return acc.avg



def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    
    parser.add_argument('--train_batch_size',    help='Size of the batches for train.', default=128, type=int)
    parser.add_argument('--val_batch_size',      help='Size of the batches for validation.', default=128, type=int)
    parser.add_argument('--validation',          help='Do Validation.', type=tools.str2bool, nargs='?',const=True, default=False)

    parser.add_argument('--learning_rate',       help='Start learning rate.', type=float, default=0.0005)
    parser.add_argument('--epochs',              help='Number of epochs to train.', type=int, default=10)

    parser.add_argument('--keep_train',          help='Keep training on previouse snapshot.', type=tools.str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--pretrain_imagenet',   help='Use pretrained weight on Imagenet.', type=tools.str2bool, nargs='?',const=True, default=True)

    parser.add_argument('--data_type',           help='Which data do you want to train.', type=str, default='landmark_data_single')
    parser.add_argument('--dropouts',            help='Apply multiple dropouts', type=tools.str2bool, nargs='?',const=False, default=False)
    parser.add_argument('--weighted_loss',       help='Apply weighted loss', type=tools.str2bool, nargs='?',const=True, default=False)
    return parser.parse_args(args)



def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    print ('--------------Arguments----------------')
    print ('data_type : ', args.data_type)
    print ('learning_rate : ', args.learning_rate)
    print ('validation : ', args.validation)
    print ('epochs : ', args.epochs)
    print ('keep_train : ', args.keep_train)
    print ('pretrain_imagenet : ', args.pretrain_imagenet)
    print ('train_batch_size : ', args.train_batch_size)
    print ('val_batch_size : ', args.val_batch_size)   
    print ('dropouts : ', args.dropouts)    
    print ('weighted_loss : ', args.weighted_loss)  
    print ('---------------------------------------')
    mean = nml_cfg.mean
    std = nml_cfg.std

    
    # Make snapshot directory
    tools.directoryMake(path_cfg.snapshot_root_path)

    train_dir = os.path.join(path_cfg.data_root_path, 'train', args.data_type)
    train_dir = os.path.join(path_cfg.data_root_path, args.data_type)
    val_dir = os.path.join(path_cfg.data_root_path, 'val', args.data_type)        
	


    # Make Train data_loader
    train_data = ds.ImageFolder(train_dir, transforms.Compose([
                transforms.Resize(299),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
                ]))
    print train_data.classes
    num_of_class = len(os.listdir(train_dir))
    train_loader = data.DataLoader(train_data, batch_size=args.train_batch_size,
                                shuffle=True,drop_last=False)

    # Make Validation data_loader
    if args.validation:
        val_data = ds.ImageFolder(val_dir, transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
                ]))
        num_of_val_class = len(os.listdir(val_dir))
        val_loader = data.DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False,drop_last=False)        

    print ('----------------Data-------------------')
    print ('num_of_class : ', num_of_class)
    print ('num_of_images : ', len(train_data))
    print ('---------------------------------------\n\n')




    # Make Weight
    weight = make_weight(train_dir, args.weighted_loss)


    for model_idx, model_name in enumerate(model_name_list):
        

        save_model_name = model_name +'_'+ args.data_type
        CNN_model, CNN_optimizer, CNN_criterion, CNN_scheduler = model_setter(
            model_name, 
            weight,
            learning_rate=args.learning_rate, 
            output_size=num_of_class,
            usePretrained=args.pretrain_imagenet,
            dropouts=args.dropouts)
        if args.keep_train:
            CNN_model = models.load_checkpoint(CNN_model, save_model_name)



        best_prec = 0
        for epoch in range(args.epochs):
            prec = train(train_loader, CNN_model, CNN_criterion, CNN_optimizer, epoch)
            if args.validation:
                prec = val(val_loader, CNN_model, CNN_criterion)

            # Learning rate scheduler 
            CNN_scheduler.step()
            # Model weight will be saved based on it's validation performance
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            models.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': CNN_model.state_dict(),
                'best_prec1': best_prec,
            }, is_best
            , save_model_name)
        
        print ('Best Performance : ', best_prec) 
        print ('\n\n\n')



def model_setter(model_name, weight, learning_rate=0.001, output_size=2, usePretrained=True, isTest=False, dropouts=True):

    
    if model_name == 'resnet18':
        model = torchmodels.resnet18(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        if dropouts:
            en = EnsembleDropout()
            model.fc = nn.Sequential(en, nn.Linear(num_ftrs, output_size))
        else:
            model.fc = nn.Linear(num_ftrs, output_size)
    elif model_name == 'resnet34':
        model = torchmodels.resnet34(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        if dropouts:
            en = EnsembleDropout()
            model.fc = nn.Sequential(en, nn.Linear(num_ftrs, output_size))
        else:
            model.fc = nn.Linear(num_ftrs, output_size)
    elif model_name == 'resnet50':
        model = torchmodels.resnet50(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        if dropouts:
            en = EnsembleDropout()
            model.fc = nn.Sequential(en, nn.Linear(num_ftrs, output_size))
        else:
            model.fc = nn.Linear(num_ftrs, output_size)
    elif model_name == 'resnet101':
        model = torchmodels.resnet101(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        if dropouts:
            en = EnsembleDropout()
            model.fc = nn.Sequential(en, nn.Linear(num_ftrs, output_size))
        else:
            model.fc = nn.Linear(num_ftrs, output_size)
    elif model_name == 'resnet152':
        model = torchmodels.resnet152(pretrained=usePretrained)
        num_ftrs = model.fc.in_features
        if dropouts:
            en = EnsembleDropout()
            model.fc = nn.Sequential(en, nn.Linear(num_ftrs, output_size))
        else:
            model.fc = nn.Linear(num_ftrs, output_size)

    model = torch.nn.DataParallel(model).cuda()
    if not isTest:
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss(weight=weight).cuda()
    else:
        optimizer = 0
        criterion = 0      
        scheduler = 0
    return model, optimizer, criterion, scheduler


def make_weight(path, doMake=True):
    root_dir = path
    listdir = os.listdir(root_dir)
    weight = np.ones(len(listdir))
    if doMake:
        sum_all = 0
        for idx, filename in enumerate(listdir):
            cnt =  len(os.listdir(os.path.join(root_dir, str(filename))))
            sum_all += cnt
            weight[idx] = cnt
        weight = 1. - (weight / sum_all)
    weight.astype(float)
    weight = torch.from_numpy(weight).float().cuda()
    return weight




if __name__ == '__main__':
    main()
