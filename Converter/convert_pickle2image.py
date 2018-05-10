import os
import pickle
from PIL import Image


root_path = '../save_'
out_path = '../here_'

folder_list = os.listdir(root_path)
for folder in folder_list:
    folder_path = os.path.join(root_path, folder)
    files_list = os.listdir(folder_path)
    for files in files_list:
        file_path = os.path.join(folder_path, files)
        
        f = open(file_path)
        pkl = pickle.load(f)
        print len(pkl)
        print pkl[0]
        
        for info in pkl:
            img = info[1]
            line_num = info[0]['LINENUMBER']
            img.convert('RGB').save(os.path.join(out_path, str(line_num) + '.jpg'))