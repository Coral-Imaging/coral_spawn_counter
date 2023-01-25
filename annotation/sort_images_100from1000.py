#! /usr/bin/env python3

# sort images, from the 1000 image folder,
# find pre-labelled 100 images from img_dir100
# move those 100 to another folder
# do this to make labelling easier for asanalytics

import os
import shutil

# image directory of 100 labelled images
img_dir_100 = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_100/images'

# image directory of 1000 labelled images
img_dir_1000 = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/images'

# output image directory of 900 labelled images
img_dir_output = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_900/images'
os.makedirs(img_dir_output, exist_ok=True)
# TODO clear out directory if not empty?

# NOTE: not all 100 images may appear in the 1000, since some of the 1000 were removed for class balancing

img_list_100 = sorted(os.listdir(img_dir_100))
img_list_1000 = sorted(os.listdir(img_dir_1000))

# copy the 1000 images into img_dir_output
print('copying images to new folder')
shutil.copytree(img_dir_1000, img_dir_output, dirs_exist_ok=True)

# for each image in img_list_100, if we find the corresponding image in img_list_1000, then delete it
for i, img_name in enumerate(img_list_100):
    if img_name in img_list_1000:
        print(f'{i}/{len(img_list_100)-1}: removing {img_name}')
        os.remove(os.path.join(img_dir_output, img_name))
    else:
        print(f'{i}/{len(img_list_100)-1}: unable to find {img_name}')

print('done')

import code
code.interact(local=dict(globals(), **locals()))
