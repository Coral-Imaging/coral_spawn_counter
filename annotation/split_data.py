#! /usr/bin/env python3

"""
script to split yolov5 dataset into training/validation/testing sets
splits the images
splits the image annotations
splits the text files
into their respective folders/files
creates yml file for running yolov5

note: does not erase the old files, simply create anew

input: 
- train/val/test ratio (from 0-1, must add up to 1, can be 0)
- image folder name that has all the images (imagename.jpg/png)
- image metadata folder name that has all the annotations (has all the imagename.txt files) 
- location of train.txt file - lists all the image names in the image metadata folder

output:
- location of all the folders for things split up
- yml file of data for training
"""

import os
import shutil
import random
from sklearn.model_selection import train_test_split


def clean_dirs(target_img_dir, target_meta_dir, target_ann_file):
    if os.path.isdir(target_img_dir):
        shutil.rmtree(target_img_dir)
    os.makedirs(target_img_dir)
    
    if os.path.isdir(target_meta_dir):
        shutil.rmtree(target_meta_dir)
    os.makedirs(target_meta_dir)

    if os.path.exists(target_ann_file):
        os.remove(target_ann_file)
    
# Inputs
# ==================================================================================================

# train/val/test ratio
train_ratio = 0.7
val_ratio = 0.15
# test_ratio is remainder
test_ratio = 1.0 - train_ratio - val_ratio

if train_ratio > 1 or train_ratio < 0:
    ValueError(train_ratio,f'train_ratio must 0 < train_ratio <= 1, train_ratio = {train_ratio}')
if val_ratio < 0 or val_ratio >= 1:
    ValueError(val_ratio, f'val_ratio must be 0 < val_ratio < 1, val_ratio = {val_ratio}')
if test_ratio < 0 or test_ratio >= 1:
    ValueError(test_ratio, f'0 < test_ratio < 1, test_ratio = {test_ratio}')
if not ((train_ratio + val_ratio + test_ratio) == 1):
    ValueError(train_ratio, 'sum of train/val/test ratios must equal 1')

# image folder with all images
img_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/images'

# image metadata folder
meta_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/obj_train_data'

# annotation file with all the filenames:
ann_file = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/annotations.txt'


# Declaring output file names/locations
# ==================================================================================================

# training:
# TODO does not delete existing sets, so would simply just add more files into the mix - should delete existing sets, if not empty - with user input
img_train_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/img_train'
meta_train_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/train'
ann_train_file = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/train.txt'
clean_dirs(img_train_dir, meta_train_dir, ann_train_file)

# validation
if val_ratio > 0:
    img_val_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/img_val'
    meta_val_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/val'
    ann_val_file = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/val.txt'
    clean_dirs(img_val_dir, meta_val_dir, ann_val_file)

# testing
if test_ratio > 0:
    img_test_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/img_test'
    meta_test_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/test'
    ann_test_file = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/test.txt'
    clean_dirs(img_test_dir, meta_test_dir, ann_test_file)


# ==================================================================================================

# find out how many images in original folder
# using ratios, divy up images, favouring training images and whole numbers, ensure all images are used up
# TODO check input images/folders are consistent?

img_list = sorted(os.listdir(img_dir))

n_img = len(img_list)
n_train = round(n_img * train_ratio)

if val_ratio == 0:
    n_val = 0
elif test_ratio == 0:
    n_val = n_img - n_train
else:
    n_val = round(n_img * val_ratio)

if test_ratio == 0:
    n_test = 0
else:
    n_test = n_img - n_train - n_val

print(f'total images: {n_img}')
print(f'n_train = {n_train}')
print(f'n_val = {n_val}')
print(f'n_test = {n_test}')

# randomly split the images (might use pytorch random split of images?)
train_val_filenames, test_filenames = train_test_split(img_list, test_size=test_ratio, random_state=42) # hopefully works with 0 as test_ratio?
train_filenames, val_filenames = train_test_split(train_val_filenames, test_size=val_ratio, random_state=42)

# sanity check:
print(f'length of train_filenames = {len(train_filenames)}')
print(f'length of val_filenames = {len(val_filenames)}')
print(f'length of test_filenames = {len(test_filenames)}')

# for each list of images
# generate relevant folders, copy images into folders
# generate list of images .txt file
# generate/copy all annotation .txt files into relevant folders

# TODO turn this into a function!
# copy images
for fname in train_filenames:
    shutil.copyfile(os.path.join(img_dir, fname),os.path.join(img_train_dir, fname))

# copy annotations
for fname in train_filenames:
    ann_file = fname[:-4] + '.txt'
    shutil.copyfile(os.path.join(meta_dir, ann_file),os.path.join(meta_train_dir, ann_file))

# write the text file of all the image names
with open(ann_train_file, 'w') as f:
    for fname in train_filenames:
        f.write(fname)
    


# for fname in val_filenames:
#     shutil.copyfile(os.path.join(img_dir, fname),os.path.join(img_val_dir, fname))

# for fname in test_filenames:
#     shutil.copyfile(os.path.join(img_dir, fname),os.path.join(img_test_dir, fname))



print('done')

import code
code.interact(local=dict(globals(), **locals()))