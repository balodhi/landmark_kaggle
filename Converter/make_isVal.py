import os
import pickle

f = open('/media/hwejin/SSD_1/DATA/landmark/data/train.csv','r')
lines = f.readlines()
f.close()

all_dict = {}

for idx, line in enumerate(lines):
    if idx == 0:
        continue
    else:
        cut = line[:-1].split(',')
        key = cut[0][1:-1]
        url = cut[1][1:-1]
        label = cut[2]
        
        if label in all_dict.keys():
            all_dict[label].append(key)
        else:
            all_dict[label] = [key]
            
            
            
val_dict = {}

for label in all_dict:
    if len(all_dict[label]) < 2:
        for idx, key in enumerate(all_dict[label]):
            val_dict[key] = True
    elif len(all_dict[label]) < 11:
        for idx, key in enumerate(all_dict[label]):
            if idx == 0:
                val_dict[key] = True
            else:
                val_dict[key] = False
    else:
        for idx, key in enumerate(all_dict[label]):
            if idx < len(all_dict[label]) * 0.1:
                val_dict[key] = True
            else:
                val_dict[key] = False   
                
f = open('isVal.pkl', 'w')
pickle.dump(val_dict, f)
f.close()
