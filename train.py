import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.parallel

import os
import numpy as np
import time
import sys
import argparse

import models
import tools
import dataload as dataload_bilal
import path as path_cfg

model_name_list = ['resnet18']



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = tools.AverageMeter()
    losses = tools.AverageMeter()
    acc = tools.AverageMeter()    
    end = time.time()
    
    
    for i, (img,label) in enumerate(train_loader):
        target = label.cuda(async=True)
        
        
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        
        
        output = model(img)
        loss = criterion(output,label)
        
        
        
        # measure accuracy and record loss
        prec1, prec1 = tools.Accuracy(output.data, target, topk=(1, 1))
        losses.update(loss.data[0], img.size(0))
        acc.update(prec1[0], img.size(0))
        
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()        
        if i % 50 == 0:
            print('\tEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, loss=losses, acc=acc))
            
            
            
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
    parser.add_argument('--val_batch_size',      help='Size of the batches for validation.', default=8, type=int)
    parser.add_argument('--learning_rate',       help='Start learning rate.', type=float, default=0.0002)
    parser.add_argument('--rolling_effect',      help='Applying rolling effect.', action='store_const', const=True, default=True)
    parser.add_argument('--rolling_data_type',   help='Which data want to use for rolling effect.', type=str, default='TEST')
    parser.add_argument('--keep_train',          help='Keep training on previouse snapshot.', action='store_const', const=True, default=True)
    parser.add_argument('--pretrain_imagenet',   help='Use pretrained weight on Imagenet.', action='store_const', const=True, default=True)
    parser.add_argument('--epochs',              help='Number of epochs to train.', type=int, default=1)
    parser.add_argument('--data_type',           help='Which data do you want to train.', type=str, default='TEST')

    return parser.parse_args(args)



def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    print args.rolling_effect
    # Make snapshot directory
    tools.directoryMake(path_cfg.snapshot_root_path)
    
    # Make Train, Val data_loader
    train_dir = os.path.join(path_cfg.data_root_path, args.data_type)
    train_data = dataload_bilal.dataload(train_dir, transforms.Compose([
                #transforms.Resize(224),
                #transforms.RandomSizedCrop(224),
                #transforms.Resize(299),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]))
    train_loader = data.DataLoader(train_data, batch_size=args.train_batch_size,
                                shuffle=True,drop_last=False)

    val_dir = os.path.join(path_cfg.data_root_path, args.data_type)
    val_data = dataload_bilal.dataload(val_dir, transforms.Compose([
                #transforms.Resize(224),
                #transforms.RandomSizedCrop(224),
                #transforms.Resize(299),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]))
    val_loader = data.DataLoader(val_data, batch_size=args.val_batch_size,
                                shuffle=False,drop_last=False)

    
    
    
    # Finally we got num_of_classes
    num_of_class = train_data.nClasses()



    for model_idx, model_name in enumerate(model_name_list):
        # To apply rolling effect
        rolling_weight_path = os.path.join(path_cfg.snapshot_root_path, model_name +'_'+ args.rolling_data_type + '.pth.tar')
        if args.rolling_effect and os.path.exists(rolling_weight_path):
            # Load model weight trained on rolling data
            CNN_model, CNN_optimizer, CNN_criterion, CNN_scheduler = models.rollingWeightLoader(args.data_type, 
                                                                                            model_name, 
                                                                                            args.learning_rate)
            # This module gonna change last output size for prediction
            # Because the number of rolling data classes and training data classes are different.
            CNN_model = models.pretrained_model_converter(CNN_model, num_of_class)
        else: 
            # Scratch Model
            CNN_model, CNN_optimizer, CNN_criterion, CNN_scheduler = models.model_setter(model_name, 
                                                                              learning_rate=args.learning_rate, 
                                                                              output_size=num_of_class,
                                                                              usePretrained=args.pretrain_imagenet)
            
            print 'Scratch model'
            # keep training on previouse epoch.
            if args.keep_train:
                print 'Keep training on previouse weight'
                checkpoint_path = os.path.join(path_cfg.snapshot_root_path, model_name +'_'+ args.data_type + '.pth.tar')
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)
                    CNN_model.load_state_dict(checkpoint['state_dict'])
    
    
    
        best_prec1 = 0
        for epoch in range(args.epochs):
            train(train_loader, CNN_model, CNN_criterion, CNN_optimizer, epoch)
            prec1 = val(val_loader, CNN_model, CNN_criterion)
            
            
            # Learning rate scheduler 
            CNN_scheduler.step()
            print '    lr : ', CNN_scheduler.get_lr()
            
            
            # Model weight will be saved based on it's validation performance
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            models.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': CNN_model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best
            , model_name +'_'+ args.data_type)
        
        print 'Best Performance : ', best_prec1




if __name__ == '__main__':
    main()
