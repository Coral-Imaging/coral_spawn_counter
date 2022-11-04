#! /usr/bin/env python3

"""
create spawn count table given images in a folder
"""

import cv2 as cv
import os
import pandas as pd
from pprint import *

from circle_detector import CircleDetector

# directories
root_dir = '/home/cslics/cslics_ws/src/coral_spawn_counter/some_root_dir'
hostnames = ['cslics03', 'cslics04']
img_folder = 'images'
table_name = 'spawn_counts.csv'
img_detections = 'detections'
metadata_folder = 'metadata'


SpawnCounter = CircleDetector()

for host in hostnames:
    print(host)
    img_dir = os.path.join(root_dir, host, img_folder)
    img_list = os.listdir(img_dir)
    img_list.sort()
    pprint(img_list)

    det_dir = os.path.join(root_dir, host, img_detections)
    os.makedirs(det_dir, exist_ok=True)

    spawn_count = []
    cam_id = []
    cap_time = []

    for img_name in img_list:
        img = cv.imread(os.path.join(img_dir, img_name))

        count, circles = SpawnCounter.count_spawn(img)
        SpawnCounter.save_detections(img, img_name, det_dir)
        
        print(f'{img_name}: {count}')
        spawn_count.append(count)

        # TODO read the camera id, capture time, etc from png metadata
        # but for now, just grab from hostid, etc
        cam_id.append(int(host[6:8]))
        cap_time.append(img_name[0:22])


    # save as dataframe
    df = pd.DataFrame({"camera_id": cam_id,
                      "image_name": img_list,
                      "capture_time": cap_time,
                      "count": spawn_count})
    os.makedirs(os.path.join(root_dir, host, metadata_folder), exist_ok=True)
    df.to_csv(os.path.join(root_dir, host, metadata_folder, table_name))
    
print('done')