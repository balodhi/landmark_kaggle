import os
import pickle
import copy



root_path = '/media/hwejin/HDD_1/KAGGLE/combine_landmark'
out_path = '/media/hwejin/SSD_1/DATA'
f = open('/media/hwejin/SSD_1/DATA/temp_pickles/train.csv','r')
lines = f.readlines()
f.close()


all_file_dict = {}

for idx, line in enumerate(lines):
    if idx == 0:
        continue
    else:
        cut = line[:-1].split(',')
        idx = idx - 1
        path = os.path.join(root_path, str(idx) + '.jpg')
        if os.path.isfile(path):
            if os.path.getsize(path) > 0:
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

train_list = []
val_list = []
for split in split_list:
    
    
    del_key_list = []
    for dict_idx, key in enumerate(file_dict):
        if len(file_dict[key]) < 2:
            for idx, info in enumerate(file_dict[key]):
                info['HOWMANYIMAGES'] = len(file_dict[key])
                train_list.append((info, key))
                val_list.append((info, key))
            del_key_list.append(key)
        elif len(file_dict[key]) < 11:
            for idx, info in enumerate(file_dict[key]):
                info['HOWMANYIMAGES'] = len(file_dict[key])
                if idx == len(file_dict[key]) - 1:
                    val_list.append((info, key))
                else:
                    train_list.append((info, key))
            del_key_list.append(key)
        else:
            if len(file_dict[key]) < int(split):
                for idx, info in enumerate(file_dict[key]):
                    info['HOWMANYIMAGES'] = len(file_dict[key])
                    if idx < len(file_dict[key]) * 0.9:
                        train_list.append((info, key))
                    else:
                        val_list.append((info, key))
                del_key_list.append(key)

    for key in del_key_list:
        file_dict.pop(key)
        
        
f = open(os.path.join(out_path,'train_set.pickle'), 'w')
pickle.dump(train_list, f)
f.close()
        
f = open(os.path.join(out_path,'val_set.pickle'), 'w')
pickle.dump(val_list, f)
f.close()