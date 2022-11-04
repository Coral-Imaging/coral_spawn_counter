#! /usr/bin/env python3

"""
create spawn count table given images in a folder
"""

import cv2 as cv
import os
import pandas as pd
from pprint import *

# directories
root_dir = '/home/cslics/cslics_ws/src/coral_spawn_counter/some_root_dir'
hostnames = ['cslics03', 'cslics04']
img_folder = 'images'
table_name = 'spawn_counts.csv'
img_detections = 'detections'
metadata_folder = 'metadata'

def find_circles(img, 
                 blur=5, 
                 method=cv.HOUGH_GRADIENT, 
                 dp=0.9, 
                 minDist=30,
                 param1=100,
                 param2=30,
                 maxRadius=200,
                 minRadius=20):
    # TODO rename parameters for more readability
    # convert image to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # blur image (Hough transforms work better on smooth images)
    img = cv.medianBlur(img, blur)
    # find circles using Hough Transform
    circles = cv.HoughCircles(image = img,
                              method=method,
                              dp=dp,
                              minDist=minDist,
                              param1=param1,
                              param2=param2,
                              maxRadius=maxRadius,
                              minRadius=minRadius)
    return circles


def draw_circles(img, circles, outer_circle_color=(255, 0, 0), thickness=8):
    """ draw circles onto image"""
    for circ, i in enumerate(circles[0,:], start=1):
        cv.circle(img, 
                  (int(i[0]), int(i[1])), 
                  radius=int(i[2]), 
                  color=outer_circle_color, 
                  thickness=thickness)
    return img


def count_spawn(img, img_name, save_dir):
    # proxy for ML detector, which will come later in the project
    # counts the spawn
    # saves a figure of the detections

    circles = find_circles(img)

    if circles is not None:
        _, count, _ = circles.shape
        img_c = draw_circles(img, circles)
    else:
        count = 0
        img_c = img

    img_name_circle = img_name[:-4] + '_circ.jpeg'
    cv.imwrite(os.path.join(save_dir, img_name_circle), img_c)



    return count





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

        count = count_spawn(img, img_name, det_dir)
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