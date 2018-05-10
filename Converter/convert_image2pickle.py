import os
import copy
import shutil
from PIL import Image
import pickle
import copy

#root path
root_path = '/media/hwejin/HDD_1/KAGGLE/combine_landmark'
list_dir = os.listdir(root_path)

#where to save
save_path = 'save_'


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
split_list = ['3']

file_dict = copy.deepcopy(all_file_dict)
if not os.path.exists(save_path):
    os.makedirs(save_path)



for split in split_list:
    now_path = os.path.join(save_path, split)
    if not os.path.exists(now_path):
        os.makedirs(now_path) 
    image_list = []
    save_cnt = 0
    
    del_key_list = []
    
    for dict_idx, key in enumerate(file_dict):

        if len(file_dict[key]) < int(split):
            
            for idx, info in enumerate(file_dict[key]):
                try:
                    #check can this file be opened by PIL
                    img = Image.open(info['PATH'])
                    img = img.resize((224,224), Image.ANTIALIAS)
                    image_list.append((info, img, key))
                    
                except Exception as e:
                    print (info['PATH'], e)
                if len(image_list) == 1000:
                    f = open(os.path.join(now_path, str(save_cnt) + '.pickle'), 'wb')
                    pickle.dump(image_list, f)
                    f.close()
                    save_cnt += 1
                    image_list = []
                    
                    
            del_key_list.append(key)

    for key in del_key_list:
        file_dict.pop(key)  
        
    del_key_list = []
    
    if len(image_list) > 0 :
        f = open(os.path.join(now_path, str(save_cnt) + '.pickle'), 'wb')
        pickle.dump(image_list, f)
        f.close()
        save_cnt += 1
        image_list = []
