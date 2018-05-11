import os
import copy
import shutil
from PIL import Image
import pickle
import copy
import numpy as np
from random import shuffle
import numpy as np

#where the og data is ?
og_path = '/media/hwejin/HDD_1/KAGGLE/combine_landmark')


#root path
out_root_path = '/media/hwejin/HDD_1/KAGGLE/'



out_pickle_path = os.path.join(out_root_path, 'pickle')
if not os.path.exists(out_pickle_path):
    os.makedirs(out_pickle_path)
    
    
out_csv_path = os.path.join(out_root_path, 'csv')
if not os.path.exists(out_csv_path):
    os.makedirs(out_csv_path)



f = open('/media/hwejin/SSD_1/DATA/landmark/data/train.csv','r')
lines = f.readlines()
f.close()


all_file_dict = {}

for idx, line in enumerate(lines[:10]):
    if idx == 0:
        continue
    else:
        cut = line[:-1].split(',')
        idx = idx - 1
        path = os.path.join(og_path, str(idx) + '.jpg')
        temp_dict = {}
        temp_dict['KEY'] = cut[0][1:-1]
        temp_dict['URL'] = cut[1][1:-1]
        temp_dict['PATH'] = path
        temp_dict['LINENUMBER'] = idx
                
        if cut[2] in all_file_dict.keys():
            all_file_dict[cut[2]].append(temp_dict)
        else:
            temp_list = []
            temp_list.append(temp_dict)
            all_file_dict[cut[2]] = temp_list



split_list = ['3', '11', '21', '31', '41', '51', '61', '71', '81', '91', 
              '101', '201', '301', '401', '501', '601', '701', '801', '901', '1001', '100000000000']                
file_dict = copy.deepcopy(all_file_dict)

for split in split_list:
    split_path = os.path.join(out_pickle_path, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path) 
    image_list = []
    save_cnt = 0
    
    del_key_list = []
    for dict_idx, key in enumerate(file_dict):
        if len(file_dict[key]) < int(split):
            for idx, info in enumerate(file_dict[key]):
                try:
                    img = Image.open(info['PATH'])
                    image_list.append((info, img, key))
                    line_num = info['LINENUMBER']
                except Exception as e:
                    print (info, e)
                    #dummy_img = np.zeros([224,224,3],dtype=np.uint8)
                    #dummy_img.fill(255)
                    #dummy_img = Image.fromarray(np.uint8(dummy_img))
                    
                    #image_list.append((info, dummy_img, key))
                    #line_num = info['LINENUMBER']
                    
                    
                    
                    
                if len(image_list) == 1000:
                    f = open(os.path.join(split_path, str(save_cnt) + '.pickle'), 'wb')
                    pickle.dump(image_list, f)
                    f.close()
                    save_cnt += 1
                    image_list = []
                    
                    
            del_key_list.append(key)

    for key in del_key_list:
        file_dict.pop(key)  
    del_key_list = []
    
    if len(image_list) > 0 :
        f = open(os.path.join(split_path, str(save_cnt) + '.pickle'), 'wb')
        pickle.dump(image_list, f)
        f.close()
        save_cnt += 1
        image_list = []



file_dict = copy.deepcopy(all_file_dict)

train_list = []
val_list = []

train_label = []
val_label = []


del_key_list = []
for dict_idx, key in enumerate(file_dict):
    if len(file_dict[key]) < 2:
        for idx, info in enumerate(file_dict[key]):
            info['HOWMANYIMAGES'] = len(file_dict[key])
            train_list.append((info, key))
            val_list.append((info, key))
                
                
            if key not in train_label:
                train_label.append(key)
            if key not in val_label:
                val_label.append(key)
                
                
                
                
                
        del_key_list.append(key)
    elif len(file_dict[key]) < 11:
        for idx, info in enumerate(file_dict[key]):
            info['HOWMANYIMAGES'] = len(file_dict[key])
            if idx == len(file_dict[key]) - 1:
                val_list.append((info, key))
            else:
                train_list.append((info, key))
                
                
                
                
            if key not in train_label:
                train_label.append(key)
            if key not in val_label:
                val_label.append(key)
                    
                    
                    
        del_key_list.append(key)
    else:
        if len(file_dict[key]) < int(split):
            for idx, info in enumerate(file_dict[key]):
                info['HOWMANYIMAGES'] = len(file_dict[key])
                if idx < len(file_dict[key]) * 0.9:
                    train_list.append((info, key))
                else:
                    val_list.append((info, key))
                    
                    
            if key not in train_label:
                train_label.append(key)
            if key not in val_label:
                val_label.append(key)
            del_key_list.append(key)
            
for key in del_key_list:
    file_dict.pop(key)
        
shuffle(train_list)
shuffle(val_list)


f = open(os.path.join(out_csv_path,'train_set.pickle'), 'wb')
pickle.dump(train_list, f)
f.close()
        
f = open(os.path.join(out_csv_path,'val_set.pickle'), 'wb')
pickle.dump(val_list, f)
f.close()

