#! /usr/bin/env python3

"""
create spawn count table given images in a folder
- apply circle detection and create spawn count table
- read metadata from iamges, add to table
"""

import cv2 as cv
import sys
import os
import pandas as pd
import json
from pprint import *
import shutil
import PIL.Image as PIL_Image

from coral_spawn_counter.CoralImage import CoralImage

img_folder = 'images'
table_name = 'spawn_counts.csv'
img_detections = 'detections'
metadata_folder = 'metadata'

FORCE_REDO = True

# read circle detection parameters from file:
det_param_path = '/media/agkelpie/cslics_ssd/2022_NovSpawning/20221112_AMaggieTenuis/cslics04/metadata'
det_param_file = 'circ_det_param.json'
with open(os.path.join(det_param_path, det_param_file), 'r') as f:
    det_param = json.load(f)
pprint(det_param)

# for each host, grab all images, process them (count spawn), read metadata, save table
# for host in hostnames:
    
def spawn_table(host):
    print(host)
    img_dir = os.path.join(root_dir, host, img_folder)
    img_list = os.listdir(img_dir)
    img_list.sort() # sort list so latest info goes at the end of the table
    # pprint(img_list)

    print(f'number of images to process: {len(img_list)}')
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
    phases = []

    # evaluate how many there are to do. If # is greater than 60, then don't do  because we'll get overlapping processes?

    for img_name in img_list:

        if not FORCE_REDO:
            if img_name in img_list0:
                print(f'Skipping {img_name}')
                continue

        print(f'Processing {img_name}')
        
        try: 
            # NOTE sometimes image connection/writing gets interrupted, and thus an image ends up in the data folder
            # that is incomplete image, and in this situation, we want to skip the image

            cimg = CoralImage(os.path.join(img_dir, img_name))

            # HACK because wide lens is not suited for circle detection at the moment
            # if host in wide_lens:
            #     cimg.count = 0
            #     shutil.copy(os.path.join(img_dir, img_name), os.path.join(det_dir, img_name))
            # else:
                # only open/save single image at a time to reduce memory requirements
                # (previously, each img was saved with Image)
            img = PIL_Image.open(os.path.join(img_dir, img_name))
            cimg.count_spawn(img, det_param=det_param)
            cimg.save_detection_img(img, save_dir=os.path.join(root_dir, host, img_detections))

            if 'phase' in cimg.metadata:
                phases.append(cimg.metadata['phase'])
            else:
                phases.append('n/a')
            
            print(f'{img_name}: {cimg.count}')
            imgs.append(cimg)

        except Exception as e:
            print(e)
            print(f'Error processing image {img_name} - thus skipping')


    if len(imgs) == 0:
        # no changes, so no new file to write
        return False

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
                    "count": spawn_count,
                    "phase": phases,})

    if len(img_list0) == 0:
        df.to_csv(spawn_table_file, mode='w', index=False)
    else:    
        df.to_csv(spawn_table_file, mode='a', header=False, index=False)
        
    return True


if __name__ == "__main__":

    # directories
    # root_dir = '/home/cslics/cslics_ws/src/rrap-downloader/cslics_data'
    # root_dir = '/home/cslics/Pictures/cslics_data_Nov15_test'
    root_dir = '/media/agkelpie/cslics_ssd/2022_NovSpawning/20221112_AMaggieTenuis'

    # hostnames = ['cslics02', 'cslics04'] # TODO automatically grab hostnames in root_dir
    # hostnames = os.listdir(root_dir) # we assume a folder structure as shown below


    if len(sys.argv) == 1:
        print('Missing required argument: [REMOTE HOSTNAME]')
        sys.exit(1)

    host = sys.argv[1]

    spawn_table(host)
    
