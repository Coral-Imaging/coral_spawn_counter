#! /usr/bin/env python3

"""
create spawn count table given images in a folder
- apply circle detection and create spawn count table
- read metadata from iamges, add to table
"""

import cv2 as cv
import os
import pandas as pd
from pprint import *


from coral_spawn_counter.Image import Image

# directories
root_dir = '/home/cslics/cslics_ws/src/rrap-downloader/cslics_data'
hostnames = ['cslics03'] # TODO automatically grab hostnames in root_dir
img_folder = 'images'
table_name = 'spawn_counts.csv'
img_detections = 'detections'
metadata_folder = 'metadata'


# for each host, grab all images, process them (count spawn), read metadata, save table
for host in hostnames:
    print(host)
    img_dir = os.path.join(root_dir, host, img_folder)
    img_list = os.listdir(img_dir)
    img_list.sort() # sort list so latest info goes at the end of the table
    # pprint(img_list)

    # TODO read in the existing spawn_count file, and only do detections on images
    # that have not yet been processed

    # create directories
    det_dir = os.path.join(root_dir, host, img_detections)
    meta_dir = os.path.join(root_dir, host, metadata_folder)
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    imgs = []

    for img_name in img_list:

        img = Image(os.path.join(img_dir, img_name))

        img.count_spawn()
        img.save_detection_img(save_dir=os.path.join(root_dir, host, img_detections))
        
        print(f'{img_name}: {img.count}')
        imgs.append(img)

    # unpackage columns of data for spawn count table
    cam_id = [int(img.metadata['camera_index']) for img in imgs]
    cap_time = [str(img.metadata['capture_time']) for img in imgs]
    spawn_count = [img.count for img in imgs]

    # save as dataframe
    df = pd.DataFrame({"camera_id": cam_id,
                      "image_name": img_list,
                      "capture_time": cap_time,
                      "count": spawn_count})
    
    df.to_csv(os.path.join(root_dir, host, metadata_folder, table_name))
    
print('done')