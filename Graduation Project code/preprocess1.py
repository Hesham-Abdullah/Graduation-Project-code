# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:27:20 2019

@author: Workstation1
"""

import os
from random import sample
from shutil import copyfile
source_directory = os.path.join('full_dataV1')

class_names = os.listdir(source_directory) 

os.mkdir(os.path.join('data', 'preprocessed'))

preprocessed_dir = os.path.join('data', 'preprocessed')

text = []
text_file_train = []
text_file_test = []
count = 0
split_size = 0.8

for name in class_names:
    
    os.mkdir(os.path.join(preprocessed_dir, name))
    print('[!]',name, 'is added!')
    src = os.path.join(source_directory, name)    
    temp = []
    
    for idx, video in enumerate(os.listdir(src)):
        vid_name = name+'_'+str(idx + 1)+'.mp4'
        copyfile(os.path.join(src, video), os.path.join(preprocessed_dir, name, vid_name))
        print('[+]',vid_name, 'is coppied')
        text.append(name+'/'+vid_name)
        temp.append(name+'/'+vid_name)
        count += 1
    train_size = int(split_size * len(temp))
    text_file_train += sample(temp, train_size)
    text_file_test += [x for x in temp if x not in text_file_train]
    
        
with open(os.path.join('data' ,'dataset', 'trainlist01.txt'), 'w') as f:
    for item in text_file_train:
        f.write("%s\n" % item)    
    
with open(os.path.join('data' ,'dataset','testlist01.txt'), 'w') as f:
    for item in text_file_test:
        f.write("%s\n" % item)         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
