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

# hostnames = ['cslics02', 'cslics04'] # TODO automatically grab hostnames in root_dir
hostnames = os.listdir(root_dir) # we assume a folder structure as shown below

img_folder = 'images'
table_name = 'spawn_counts.csv'
img_detections = 'detections'
metadata_folder = 'metadata'

FORCE_REDO = False

# detection parameters for far focus cslics: cslics2:
det_param_far = {'blur': 5,
                'dp': 0.7,
                'minDist': 10,
                'param1': 20,
                'param2': 20,
                'maxRadius': 20,
                'minRadius': 5}

# detection parameters for near focus cslics: cslics04
det_param_close = {'blur': 5,
                'dp': 0.7,
                'minDist': 30,
                'param1': 75,
                'param2': 20,
                'maxRadius': 40,
                'minRadius': 10}

det_param_wide = det_param_close # no detection parameters for wide FOV yet

host_det_param = {"cslics01": det_param_close,
              "cslics02": det_param_far,
              "cslics03": det_param_far,
              "cslics04": det_param_close,
              "cslics06": det_param_wide,
              "cslics07": det_param_wide
             }

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

    # read in current spawn table if it already exists (so we don't reprocess images)
    spawn_table_file = os.path.join(root_dir, host, metadata_folder, table_name)
    if os.path.exists(spawn_table_file):
        # print('spawn_counts.csv already exists, reading it to find which images are already processed')
        df0 = pd.read_csv(spawn_table_file)
        img_list0 = list(df0['image_name'])
    else:
        img_list0 = []

    imgs = []

    for img_name in img_list:

        if not FORCE_REDO:
            if img_name in img_list0:
                print(f'Skipping {img_name}')
                continue

        print(f'Processing {img_name}')
        img = Image(os.path.join(img_dir, img_name))

        img.count_spawn(det_param=host_det_param[host])
        img.save_detection_img(save_dir=os.path.join(root_dir, host, img_detections))
        
        print(f'{img_name}: {img.count}')
        imgs.append(img)

    if len(imgs) == 0:
        # no changes, so no new file to write
        continue

    # unpackage columns of data for spawn count table
    cam_id = [int(img.metadata['camera_index']) for img in imgs]
    cap_time = [str(img.metadata['capture_time']) for img in imgs]
    spawn_count = [img.count for img in imgs]
    img_list_new = [img.img_basename for img in imgs]

    # import code
    # code.interact(local=dict(globals(), **locals()))
    # save as dataframe
    df = pd.DataFrame({"camera_id": cam_id,
                    "image_name": img_list_new,
                    "capture_time": cap_time,
                    "count": spawn_count})

    if len(img_list0) == 0:
        df.to_csv(spawn_table_file, mode='w')
    else:    
        df.to_csv(spawn_table_file, mode='a')
    
print('done')
