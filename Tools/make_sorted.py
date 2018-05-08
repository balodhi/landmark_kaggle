import pickle
import os

f = open('/media/hwejin/SSD_1/DATA/temp_pickles/train.csv', 'r')

#val_labels.txt
all_lines = f.readlines()
f.close()
print len(all_lines)


all_dict = {}

for all_line in all_lines[1:]:
    cut = all_line[:-1].split(',')
    label = cut[2]
    
    if label in all_dict.keys():
        all_dict[label].append(0)
    else:
        all_dict[label] = [0]
        
        
root = '/media/hwejin/SSD_1/DATA/temp_pickles/val'
file_list = os.listdir(root)
total_num = len(file_list)

all_list = []
for idx in range(6):
    f = open(os.path.join(root, 'val_labels_' + str(idx) + '.pickle'))
    #f = open(os.path.join(root, 'val_labels_0.pickle'))
    a = pickle.load(f)
    for i in a:
        all_list.append(i)
    f.close()
        
        
        
        
f = open('/media/hwejin/SSD_1/DATA/temp_pickles/sorted/sorted_val.txt', 'w')
split_list = ['3', '11', '21', '31', '41', '51', '61', '71', '81', '91', '101',
              '201', '301', '401', '501', '601', '701', '801', '901', '1001', '1k']
for idx, line in enumerate(all_list):
    
    line_number = idx
    label = line
     
    out = ''
    label_type_index = -1
    if len(all_dict[str(label)]) < 3:
        out = '3'
        label_type_index = 0
    elif len(all_dict[str(label)]) < 11:
        out = '11'
        label_type_index = 1
    elif len(all_dict[str(label)]) < 21:
        out = '21'
        label_type_index = 2
    elif len(all_dict[str(label)]) < 31:
        out = '31'
        label_type_index = 3
    elif len(all_dict[str(label)]) < 41:
        out = '41'
        label_type_index = 4
    elif len(all_dict[str(label)]) < 51:
        out = '51'
        label_type_index = 5
    elif len(all_dict[str(label)]) < 61:
        out = '61'
        label_type_index = 6
    elif len(all_dict[str(label)]) < 71:
        out = '71'
        label_type_index = 7
    elif len(all_dict[str(label)]) < 81:
        out = '81'
        label_type_index = 8
    elif len(all_dict[str(label)]) < 91:
        out = '91'
        label_type_index = 9
    elif len(all_dict[str(label)]) < 101:
        out = '101'
        label_type_index = 10
    elif len(all_dict[str(label)]) < 201:
        out = '201'
        label_type_index = 11
    elif len(all_dict[str(label)]) < 301:
        out = '301'
        label_type_index = 12
    elif len(all_dict[str(label)]) < 401:
        out = '401'
        label_type_index = 13
    elif len(all_dict[str(label)]) < 501:
        out = '501'
        label_type_index = 14
    elif len(all_dict[str(label)]) < 601:
        out = '601'
        label_type_index = 15
    elif len(all_dict[str(label)]) < 701:
        out = '701'
        label_type_index = 16
    elif len(all_dict[str(label)]) < 801:
        out = '801'
        label_type_index = 17
    elif len(all_dict[str(label)]) < 901:
        out = '901'
        label_type_index = 18
    elif len(all_dict[str(label)]) < 1001:
        out = '1001'  
        label_type_index = 19
    else:
        out = '1000k' 
        label_type_index = 20
        
        
    label_type = out  
    one_line = str(line_number) + ',' + str(label) + ',' + str(label_type) + '\n'
    f.write(one_line)
    
    f_label = open('/media/hwejin/SSD_1/DATA/temp_pickles/sorted/sorted_val_' + split_list[label_type_index] + '.txt', 'a')
    f_label.write(one_line)
    f_label.close()
    
    
    
f.close()