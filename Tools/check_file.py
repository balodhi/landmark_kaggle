import pickle
from tqdm import tqdm
import path
import os
skip=False
rootpath= path.data_root_path
dirlist=os.listdir(rootpath)
print(dirlist)
total_folders = len(dirlist)

for folder in dirlist:
    dir2 = os.listdir(os.path.join(rootpath,folder))
    #print((dir2))
    print("processing ",folder)

    nfiles = range(0,len(dir2))
    for cnt in tqdm(nfiles):
        file = dir2[cnt]
        #print(file)
        
        if file.endswith('.pickle'):
            try:
                with open(os.path.join(rootpath,folder,file), 'rb') as pickleFile:
                    pickleFile.seek(0)
                    a = pickle.load(pickleFile)
               
            except Exception as e:
                print(e)
                print("#"*30,folder," is not readable so skipping it.")
                skip=True
                break
    if skip:
        skip=False
        continue
                
    print("#"*30,folder," is readable")
