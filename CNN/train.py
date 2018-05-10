import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as ds
from torch.autograd import Variable
import torch.nn.parallel

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
    
    
    for i, (img,label, og_label, _, _) in enumerate(train_loader):
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

    parser.add_argument('--learning_rate',       help='Start learning rate.', type=float, default=0.0002)
    parser.add_argument('--epochs',              help='Number of epochs to train.', type=int, default=20)


    parser.add_argument('--rolling_weight_path', help='Which data want to use for rolling effect.', type=str, default='701')
    parser.add_argument('--rolling_effect',      help='Applying rolling effect.', type=tools.str2bool, nargs='?',const=True, default=True)


    parser.add_argument('--keep_train',          help='Keep training on previouse snapshot.', type=tools.str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--pretrain_imagenet',   help='Use pretrained weight on Imagenet.', type=tools.str2bool, nargs='?',const=True, default=True)

    parser.add_argument('--data_type',           help='Which data do you want to train.', type=str, default='701')
    parser.add_argument('--dropouts',            help='Apply multiple dropouts', type=tools.str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--shuffle_pickle',      help='Apply shuffle when make pickles', type=tools.str2bool, nargs='?',const=True, default=False)
    parser.add_argument('--remove_pickle',       help='Remove pikles(train, val) after training.', type=tools.str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--train_val_set_dir', 	 help='Where is the train_set, val_set files?', type=str, default='../../landmark_data/csv')

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
    print ('rolling_effect : ', args.rolling_effect)
    print ('rolling_weight_path : ', args.rolling_weight_path)
    print ('keep_train : ', args.keep_train)
    print ('pretrain_imagenet : ', args.pretrain_imagenet)
    print ('train_batch_size : ', args.train_batch_size)
    print ('val_batch_size : ', args.val_batch_size)
    print ('shuffle_pickle : ', args.shuffle_pickle)
    print ('remove_pickle : ', args.remove_pickle)    
    print ('dropouts : ', args.dropouts)    
    print ('---------------------------------------')
    mean = nml_cfg.mean
    std = nml_cfg.std





    
    # Make snapshot directory
    tools.directoryMake(path_cfg.snapshot_root_path)
    
    
    # Divide pickle file and make new file and set train, val path seperatly.
    if args.validation:
        train_dir, val_dir = tools.divideDataset(os.path.join(path_cfg.data_root_path, args.data_type), args.shuffle_pickle)
    else:
        train_dir = os.path.join(path_cfg.data_root_path, args.data_type)
        val_dir = os.path.join(path_cfg.data_root_path, args.data_type)        
	


    # Make Train, Val data_loader
    train_data = dataload_landmark.Dataload_CNN(train_dir, args.train_val_set_dir, transforms.Compose([
                transforms.Resize(299),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
                ]))



    train_loader = data.DataLoader(train_data, batch_size=args.train_batch_size,
                                shuffle=True,drop_last=False)
    '''
    val_loader = data.DataLoader(val_data, batch_size=args.val_batch_size,
                                shuffle=False,drop_last=False)

    '''
    


    num_of_class = train_data.nClasses()
    print ('----------------Data-------------------')
    print ('num_of_class : ', num_of_class)
    print ('num_of_images : ', len(train_data))
    print ('---------------------------------------\n\n')

    for model_idx, model_name in enumerate(model_name_list):
        
        if args.rolling_effect:
            save_model_name = model_name +'_'+ args.data_type + '_rew'
        else:
            save_model_name = model_name +'_'+ args.data_type

        # To apply rolling effect
        rw_path = os.path.join(path_cfg.snapshot_root_path, args.rolling_weight_path)
        if args.rolling_effect and os.path.exists(rw_path):
            print ('Rolling Effect is applied.')
            # Load model weight trained on rolling data
            CNN_model, CNN_optimizer, CNN_criterion, CNN_scheduler = models.rollingWeightLoader(rw_path, 
                                                                                            model_name, 
                                                                                            args.learning_rate,
                                                                                            num_of_class,
                                                                                            args.dropouts)
        else: 
            print ('Rolling Effect is not applied.' )
            # Scratch Model
            CNN_model, CNN_optimizer, CNN_criterion, CNN_scheduler = models.model_setter(model_name, 
                                                                              learning_rate=args.learning_rate, 
                                                                              output_size=num_of_class,
                                                                              usePretrained=args.pretrain_imagenet,dropouts=args.dropouts)
            
            print ('Scratch model')
            # keep training on previouse epoch.
            if args.keep_train:
                checkpoint_path = os.path.join(path_cfg.snapshot_root_path, save_model_name + '.pth.tar')
                if os.path.exists(checkpoint_path):
                    print ('Keep training on previouse epoch')
                    checkpoint = torch.load(checkpoint_path)
                    CNN_model.load_state_dict(checkpoint['state_dict'])
    
    
    
        best_prec1 = 0
        for epoch in range(args.epochs):
            prec_train = train(train_loader, CNN_model, CNN_criterion, CNN_optimizer, epoch)
            #prec_val = val(val_loader, CNN_model, CNN_criterion)
            
            
            # Learning rate scheduler 
            CNN_scheduler.step()
            
            prec1 = prec_train
            # Model weight will be saved based on it's validation performance
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            models.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': CNN_model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best
            , save_model_name)
        
        print ('Best Performance : ', best_prec1) 
        print ('\n\n\n')
        if args.validation and args.remove_pickle:
            tools.remove_files(train_dir)
            tools.remove_files(val_dir)


if __name__ == '__main__':
    main()



    '''
    train_data = ds.ImageFolder('../11/AugmentedTrainingSet/' , transforms.Compose([
           #transforms.Resize(256),
           transforms.RandomResizedCrop(224),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           ]))
	'''
