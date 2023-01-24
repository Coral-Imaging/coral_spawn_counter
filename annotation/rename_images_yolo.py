#! /usr/bin/env python3

# rename images by adding cslics_id to front of image names from yolo annotations format

# 1. changing all the filenames in train.txt
# 2. changing all the filenames in obj_train_data


import os
import shutil

# location of renamed images (has cslics##_ in front of image name):
cslics_img_dir = '/home/agkelpie/Data/202211_atenuis_100/images'

# location of train.txt
data_train_file = '/home/agkelpie/Data/202211_atenuis_100/metadata/train.txt'

# location of obj_train_data files
yolo_annotation_files = '/home/agkelpie/Data/202211_atenuis_100/metadata/obj_train_data' 
# yolo_annotation_files_new = '/home/agkelpie/Data/202211_atenuis_100/metadata/obj_train_data_new'
# os.makedirs(yolo_annotation_files_new, exist_ok=True)

# get all filenames in cslics_img_dir
img_names_new = sorted(os.listdir(cslics_img_dir))

# create a dictionary with old names: new names
img_dict = {}
for img_name in img_names_new:
    img_dict[img_name[9:]] = img_name


# =========================================
# 1. changing all the filenames in train.txt

# output file name:
data_train_dir, train_filename = os.path.split(data_train_file)
data_train_file_mod = os.path.join(data_train_dir, 'train_mod.txt')

# open file, read all the lines
with open(data_train_file, 'r') as file:
    lines = file.readlines()
        
# replace lines
for idx, line in enumerate(lines):
    # find the old image name in train.txt
    file_dir, img_name = os.path.split(line)

    # find the matching image name from img_names_new/cslcis_img_dir
    img_name_new = img_dict[img_name.strip()]

    # replace lines, and add newline character at end
    lines[idx] = os.path.join(file_dir, img_name_new + '\n')

# write to output file
with open(data_train_file_mod, 'w') as file:
    file.writelines(lines)
                                     

# =========================================
# 2. changing all the filenames in obj_train_data

# get list of all filenames in the folder
names_to_replace = sorted(os.listdir(yolo_annotation_files))

for name in names_to_replace:
    img_name_to_replace = name[:-4] + '.png'
    new_name = img_dict[img_name_to_replace][:-4] + '.txt'
    shutil.move(os.path.join(yolo_annotation_files, name),
                os.path.join(yolo_annotation_files, new_name))
    

print('done')

import code
code.interact(local=dict(globals(), **locals()))
        