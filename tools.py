import os
import pickle
from random import shuffle

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
        
def Accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

    
    
    
def directoryMake(path):
    if not os.path.exists(path):
        os.mkdir(path)
    

def divideDataset(data_path):
    train_path = os.path.join(data_path, 'train')  
    val_path = os.path.join(data_path, 'val')    

    if os.path.exists(os.path.join(data_path, 'train', 'train.pickle')) and os.path.exists(os.path.join(data_path, 'val', 'val.pickle')):
        print 'Training and Validation files already exists.'
        return train_path, val_path


    print 'Now training and validation sets are producing. Wait..'
    data=[]
    list_dir = os.listdir(data_path)
    list_dir.sort()
    list_dir = sorted(list_dir)
    
    
    pickle_dict_All = {}

    for file in list_dir:
        if file.endswith('.pickle'):
            with open(os.path.join(data_path,file), 'rb') as pickleFile:
                pickleFile.seek(0)
                data = pickle.load(pickleFile)
            
            for (label, image) in data:
                label = int(label)
                if label in pickle_dict_All:

                    pickle_dict_All[label].append(image)
                else:
                    pickle_dict_All[label] = [image]
                    
    pickle_train_list = []
    pickle_val_list = []
    for key in pickle_dict_All:
        shuffle(pickle_dict_All[key])


        for i in range(len(pickle_dict_All[key])):
            if i < len(pickle_dict_All[key]) * 0.9:
                pickle_train_list.append((key, pickle_dict_All[key][i]))
            else:
                pickle_val_list.append((key, pickle_dict_All[key][i]))


                
         
                    
    directoryMake(train_path)
    directoryMake(val_path)
    with open(os.path.join(data_path, 'train', 'train.pickle'), 'wb') as f:
        pickle.dump(pickle_train_list, f)
    with open(os.path.join(data_path, 'val', 'val.pickle'), 'wb') as f:
        pickle.dump(pickle_val_list, f)                    
                  
    return train_path, val_path
    


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return False
    
