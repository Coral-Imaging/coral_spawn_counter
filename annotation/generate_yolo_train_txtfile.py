#! /usr/bin/env python3

# generate train.txt file for yolo annotation format to cvat
# this is just a text file with all the image names in it and their location

import os

img_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_900/images_jpg'
img_list = sorted(os.listdir(img_dir))

data_dir = 'obj_train_data'
output_file = 'train.txt'
metadata_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_900/data'

lines = []
for img_name in img_list:
    lines.append(os.path.join(data_dir, img_name + '\n'))

with open(os.path.join(metadata_dir, output_file), 'w') as file:
    file.writelines(lines)
file.close()