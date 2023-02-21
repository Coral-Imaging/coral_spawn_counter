#! /usr/bin/en python3

"""_summary_
Take two datasets, annotated in yolo format and combine them
"""

import os
import shutil

# location of two datasets that need combining
dataset0_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/20211_amtenuis_100'
dataset1_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/20211_amtenuis_900'

# location of end result/final dataset as a folder
dataset_out_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/20211_amtenuis_1000'

"""
we expect the following format for each dataset folder:    
'dataset_dir'
    'images' - location of image files (jpg/png)
    'obj_train_data' - location of all training image annotations (text file/image with same name.txt)
    'obj.data' - number of classes, name of training text file, file that has class names
    'obj.names' - names of each class in order
    'train.txt' - list of image names for training
    (and soon val.txt/test.txt)
"""

