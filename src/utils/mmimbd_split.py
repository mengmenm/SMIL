import os
from shutil import copy
import json
# As per README:
# All files of iteration 0-4 move to testsing-spectrograms
# All files of iteration 5-49 move to training-spectrograms

def separate(source, split_path):
    
    num_list = sorted(os.listdir(source))
    # print(num_list)
    split = '/home/mengma/Desktop/nips2020/data/mmimdb/split.json'
    with open(split) as f:
      data = json.load(f)

    train_list = data['train']
    dev_list  = data['dev']
    test_list = data['test']
    # split train dataset
    for i in train_list:
        copy(source + i + '.jpeg', '../data/mmimdb/train/')
        # copy(source + i + '.json', '/home/mengma/Desktop/nips2020/data/mmimdb/train/')

    for i in dev_list:
        copy(source + i + '.jpeg', '..0/data/mmimdb/val/')
        # copy(source + i + '.json', '/home/mengma/Desktop/nips2020/data/mmimdb/val/')

    for i in test_list:
        copy(source + i + '.jpeg', '../data/mmimdb/test/')
        # copy(source + i + '.json', '/home/mengma/Desktop/nips2020/data/mmimdb/test/')
        

separate('../data/mmimdb/dataset/', '../data/mmimdb/split.json')