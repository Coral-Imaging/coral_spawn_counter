#! /usr/bin/env python3

# convert all images in folder from png to jpeg 
# (ideally preserve metadata, but not a must)

import os
import PIL.Image as PIL_Image
from PIL.PngImagePlugin import PngInfo
from PIL.JpegImagePlugin import JpegImageFile

# image directory
img_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_900/images'
print(f'converting .png images from {img_dir}')

img_dir_new = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_900/images_jpg'
os.makedirs(img_dir_new, exist_ok=True)
print(f'saving .jpg images in {img_dir_new}')

# list all files in img_dir
img_list = sorted(os.listdir(img_dir))

for i, img_name in enumerate(img_list):
    print(f'{i}/{len(img_list)}: {img_name}')
    img_png = PIL_Image.open(os.path.join(img_dir, img_name))
    img_jpg = img_png.convert('RGB')
    img_jpg.save(os.path.join(img_dir_new, img_name[:-4] + '.jpg'))

print('done')
