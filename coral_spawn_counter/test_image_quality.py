#!/usr/bin/env python3

# test script to see if saving image when reducing quality helps save file size

# sample image
# save it normally
# compress/reduce quality
# then save
# manually compare the two sizes



import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt

print('test_image_quality.py')

img_pattern = '*.jpg'
img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/images'
img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))

save_dir = '.'

i = 0

img_name = img_list[0]
img = cv.imread(img_name) # BGR format



basename = os.path.basename(img_name).rsplit('.', 1)[0]

# test script for image pattern, saving it as a folder
import re
pattern = r'cslics\d+_(\d{8})'
match = re.search(pattern, basename)
print(match)
date_str = match.group(1)
print(date_str)

save_name = os.path.join(save_dir, basename + '_normal.jpg')
cv.imwrite(save_name, img)


save_name2 = os.path.join(save_dir, basename + '_q75.jpg')
quality = 75
encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
cv.imwrite(os.path.join(save_dir, save_name2), img, encode_param)

# manually compare
# as expected, greatly reduces the filesize, almost by half
print('done')