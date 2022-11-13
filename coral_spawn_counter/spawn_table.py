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
from pprint import *
import shutil
import PIL.Image as PIL_Image

from coral_spawn_counter.CoralImage import CoralImage

img_folder = 'images'
table_name = 'spawn_counts.csv'
img_detections = 'detections'
metadata_folder = 'metadata'

FORCE_REDO = False

# detection parameters for far focus cslics: cslics2:
det_param_far = {'blur': 5,
                'dp': 1.6,
                'minDist': 25,
                'param1': 75,
                'param2': 0.5,
                'maxRadius': 40,
                'minRadius': 25}

# detection parameters for near focus cslics: cslics04
det_param_close_cslics04 = {'blur': 9,
                'dp': 2.5,
                'minDist': 50,
                'param1': 50,
                'param2': 0.5,
                'maxRadius': 80,
                'minRadius': 50}

det_param_close_cslics03 = {'blur': 9,
                'dp': 2.5,
                'minDist': 50,
                'param1': 50,
                'param2': 0.5,
                'maxRadius': 80,
                'minRadius': 50}

# parameters for HOUGH_GRADIENT (not HOUGH_GRADIENT_ALT)
# det_param_close_cslics03 = {'blur': 5,
#                 'dp': 1.35,
#                 'minDist': 50,
#                 'param1': 75,
#                 'param2': 20,
#                 'maxRadius': 80,
#                 'minRadius': 50}

det_param_wide = {'blur': 3,
                'dp': 2.5,
                'minDist': 5,
                'param1': 50,
                'param2': 0.5,
                'maxRadius': 12,
                'minRadius': 5}  # no detection parameters for wide FOV yet

host_det_param = {"cslics01": det_param_far,
              "cslics02": det_param_far,
              "cslics03": det_param_close_cslics03,
              "cslics04": det_param_close_cslics04,
              "cslics06": det_param_wide,
              "cslics07": det_param_wide,
              "cslicsdt": det_param_wide
             }

wide_lens = ['cslics06', 'cslics07']

# for each host, grab all images, process them (count spawn), read metadata, save table
# for host in hostnames:
    
def spawn_table(host):
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
    phases = []

    # evaluate how many there are to do. If # is greater than 60, then don't do  because we'll get overlapping processes?

    for img_name in img_list:

        if not FORCE_REDO:
            if img_name in img_list0:
                print(f'Skipping {img_name}')
                continue

        print(f'Processing {img_name}')
        
        cimg = CoralImage(os.path.join(img_dir, img_name))

        # HACK because wide lens is not suited for circle detection at the moment
        if host in wide_lens:
            cimg.count = 0
            shutil.copy(os.path.join(img_dir, img_name), os.path.join(det_dir, img_name))
        else:
            # only open/save single image at a time to reduce memory requirements
            # (previously, each img was saved with Image)
            img = PIL_Image.open(os.path.join(img_dir, img_name))
            cimg.count_spawn(img, det_param=host_det_param[host])
            cimg.save_detection_img(img, save_dir=os.path.join(root_dir, host, img_detections))

        if 'phase' in cimg.metadata:
            phases.append(cimg.metadata['phase'])
        else:
            phases.append('n/a')
        
        print(f'{img_name}: {cimg.count}')
        imgs.append(cimg)

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
    root_dir = '/home/cslics/cslics_ws/src/rrap-downloader/cslics_data'
    # root_dir = '/home/cslics/Pictures/cslics_data_test'

    # hostnames = ['cslics02', 'cslics04'] # TODO automatically grab hostnames in root_dir
    # hostnames = os.listdir(root_dir) # we assume a folder structure as shown below


    if len(sys.argv) == 1:
        print('Missing required argument: [REMOTE HOSTNAME]')
        sys.exit(1)

    host = sys.argv[1]

    spawn_table(host)
    
