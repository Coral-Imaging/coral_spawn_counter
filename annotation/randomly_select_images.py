#! /usr/bin/env python3

# list all images in 'images' folder
# randomly remove all until reaching certain number

import os
import shutil
import random

# assume nimg is greater than number of images in the folder
nimg = 250
print(f'nimg = {nimg}')
img_dir = 'images'
img_set = set(os.listdir(img_dir))

if len(img_set) < nimg:
    print('ERROR: nimg > img_set')
    exit()

img_rng = random.sample(img_set, nimg)

# take those image names and then move them into a new folder:
img_dir_new = 'images_reduced'
os.makedirs(img_dir_new, exist_ok=True)

for i, img_name in enumerate(img_rng):
    print(f'img {i}/{len(img_rng)}')
    shutil.copy(os.path.join(img_dir, img_name), 
                os.path.join(img_dir_new, img_name))

print('done')