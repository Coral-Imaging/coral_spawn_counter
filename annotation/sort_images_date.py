#! /usr/bin/env/python3

# given folder
# import images and json files
# sort them by date

import os
import shutil
import glob

# dir = '/media/dorian/cslics_ssd/images/detections_surface/detection_textfiles'
# file_list = sorted(glob.glob(os.path.join(dir, '*.t*'))) # .jpg, .json

# dir = '/media/dorian/cslics_ssd/images/detections_surface/detection_images'
# file_list = sorted(glob.glob(os.path.join(dir, '*.j*'))) # .jpg, .json

# copied over to ssd
# img_dir = '/home/dorian/Data/cslics_2023_datasets/unsorted/cslics06/images'
# save_dir = '/home/dorian/Data/cslics_2023_datasets/sorted/cslics06/images/'

# on T2D hard drive, could be slow... but still faster than copying everything back and forth
# img_dir = '/media/dorian/T2D/cslics_2023_datasets/unsorted/cslics09/images'
# save_dir = '/media/dorian/T2D/cslics_2023_datasets/sorted/cslics09/images/'



# img_dir = '/media/dorian/T2D/cslics_2023_datasets/unsorted/cslics09/images/detections_surface/detection_images'
# save_dir = '/media/dorian/T2D/cslics_2023_datasets/sorted/cslics09/images/detections_surface/detection_images'


# file_list = sorted(glob.glob(os.path.join(img_dir, '*.j*'))) # .jpg, .json

img_dir = '/media/dorian/T2D/cslics_2023_datasets/unsorted/cslics09/images/detections_surface/detection_textfiles'
save_dir = '/media/dorian/T2D/cslics_2023_datasets/sorted/cslics09/images/detections_surface/detection_textfiles'

file_list = sorted(glob.glob(os.path.join(img_dir, '*.t*'))) # .txt


print(f'img_dir: {img_dir}')
print(f'num files: {len(file_list)}')

# for each file
# read in date string
# cslics01_20231104_015959_504237_img.json/.jpg

date_list = []
# img_name = img_list[0]
for i, fname in enumerate(file_list):
    basename = os.path.basename(fname)
    datestr = basename[9:17]
    if datestr not in date_list:
        print(f'new img_name date: {datestr}')
        date_list.append(datestr)
        os.makedirs(os.path.join(save_dir, datestr), exist_ok=True)
    
    print(f'moving: {i}/{len(file_list)}: {fname}')
    shutil.move(fname, os.path.join(save_dir, datestr, basename))
    
    # import code
    # code.interact(local=dict(globals(), **locals()))


print('done')
   

