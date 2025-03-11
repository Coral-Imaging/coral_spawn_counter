#!/usr/bin/env python3

"""
quick and dirty script to test resolution for two different datasets to ensure icam-540 is okay selection wrt resolution changes
"""

import os
import glob
from pathlib import Path
from ultralytics import YOLO
import cv2 as cv
import shutil

# specified model
model_name = '/home/dtsai/Data/cslics_datasets/models/cslics_subsurface_20250205_640p_yolov8n.pt'
model = YOLO(model_name)

# target dataset, labelled
data_file = '/home/dtsai/Code/cslics/coral_spawn_counter/resolution_test/test.yaml'
data_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/100000009c23b5af/split/images/test' # for now, must match data_file validation set

# run detector on dataset
metrics = model.val(data=data_file)

# evaluate F1 score (or some other metric) across dataset
# metrics.results_dict["metrics/precision(B)"]
# metrics.results_dict["metrics/recall(A)"]
print(f'normal sized images: map50 = {metrics.results_dict["metrics/mAP50(B)"]}')


# resize images to specified resolution
def copy_labels_dir(label_list, out_dir, label_dir='train/labels'):
    # copy laels from glob list to directory
    for label_name in label_list:
        destination = os.path.join(out_dir, label_dir,os.path.basename(label_name))
        if not os.path.lexists(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(label_name, destination)
        
# TODO should read data_file to get relevant directories, but shortcut - we just copy-paste the directory here
print(f'data_dir = {data_dir}')

# target image size
# res_horizontal = 3840
res_vertical = 2160
image_size = [res_vertical]

image_list = glob.glob(os.path.join(data_dir,'*.jpg'))
print(f'length of image_list = {len(image_list)}')

# get aspect ratio, assume all images in same dir have same aspect ratio
if len(image_list) > 1:
    image = cv.imread(image_list[0])
    height, width, chan = image.shape
    ar = height/width
else:
    ValueError('no images in image_list')

# output directory
resize_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/100000009c23b5af/resize/images/test'
os.makedirs(resize_dir, exist_ok=True)
print(f'out_dir = {resize_dir}')

label_list = glob.glob(os.path.join('/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/100000009c23b5af/split/labels/test','*.txt'))
label_out_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/100000009c23b5af/resize/'
copy_labels_dir(label_list, label_out_dir, label_dir='labels/test')

# clear out files in directory before?

# loop resize over images in list
for image_name in image_list:

    image = cv.imread(image_name)
    print(f'reading image = {image_name}')

    # resize
    height_r = int(res_vertical)
    width_r = int(res_vertical / ar)
    image_r = cv.resize(image, (width_r, height_r), interpolation=cv.INTER_LINEAR)

    # save image (bgr)
    image_basename = os.path.basename(image_name)
    save_file = os.path.join(resize_dir, image_basename)
    print(f'saving image = {save_file}')
    cv.imwrite(save_file, image_r)    

    

# rerun evaluation
resize_data = '/home/dtsai/Code/cslics/coral_spawn_counter/resolution_test/resize.yaml'

# compare two scores, ideally should be very comparable
resize_model = YOLO(model_name)
metrics_resize = resize_model.val(data=resize_data)

# metrics_resize.results_dict["metrics/precision(B)"]
# metrics_resize.results_dict["metrics/recall(A)"]
print(f'normal sized images: map50 = {metrics.results_dict["metrics/mAP50(B)"]}')
print(f'normal sized images: map50 = {metrics_resize.results_dict["metrics/mAP50(B)"]}')



print('done')



import code
code.interact(local=dict(globals(), **locals()))