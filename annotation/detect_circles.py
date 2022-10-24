#! /usr/bin/env python3

"""
code to find circles in images and 
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from pprint import *
import xml.etree.ElementTree as ET


def find_circles(img, 
                 blur=5, 
                 method=cv.HOUGH_GRADIENT, 
                 dp=0.9, 
                 minDist=80,
                 param1=110,
                 param2=39,
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


def convert_circle_to_box(x,y,r, image_width, image_height):
    # no less than zero
    # no more than image width, height
    xmin = max(0, x - r)
    xmax = min(image_width, x + r)
    ymin = max(0, y - r)
    ymax = min(image_height, y + r)
    # bbox format following Pascal VOC dataset:
    # [xmin, ymin, xmax, ymax]
    return [xmin, ymin, xmax, ymax]


def convert_box_to_dict(box, label='sphere'):
    add_box = {'label': label,
               'occluded': '0',
               'source': 'manual', 
               'xbr': str(box[2]),
               'xtl': str(box[0]), 
               'ybr': str(box[3]),
               'ytl': str(box[1]), 
               'z_order': '0'}
    return add_box


# get list of images from directory
data_dir = '/home/dorian/Code/cslics_ws/src/coral_spawn_counter/microsphere_dataset_unlabelled'
img_dir = os.path.join(data_dir, 'images')
img_list = os.listdir(img_dir)
pprint(img_list)

save_dir = '/home/dorian/Code/cslics_ws/src/coral_spawn_counter/images_circ'
os.makedirs(save_dir, exist_ok=True)

# cvat annotation file location:
output_cvat_file = 'annotations_mod.xml'
cvat_file = 'annotations.xml'
tree = ET.ElementTree(file=os.path.join(data_dir, cvat_file))
root = tree.getroot()

# for each image in the annotation file
for elem in root.iterfind('.//image'):
    
    # get image name from annotation file
    img_name = elem.attrib['name']
    # print(img_name)

    # read in image
    img = cv.imread(os.path.join(img_dir, img_name))

    img_height, img_width, _ = img.shape

    # find circles
    circles = find_circles(img, 
                           blur=5, 
                           method=cv.HOUGH_GRADIENT,
                           dp = 0.9,
                           minDist=30,
                           param1=80,
                           param2=30,
                           maxRadius=150)
    
    if circles is not None:
        _ , n_circ, _ = circles.shape
    else:
        n_circ = 0
    print(f'Image: {img_name}: circles detected = {n_circ}')

    # draw circles
    if n_circ > 0:
        img_c = draw_circles(img, circles)
    else:
        img_c = img
    # save image
    img_name_circle = img_name[:-4] + '_circ.jpeg'
    cv.imwrite(os.path.join(save_dir, img_name_circle), img_c)

    

    # get bounding box of the circles:
    # for i in range(n_circ):
    #     x = circles[0][i, 0]
    #     y = circles[0][i, 1]
    #     r = circles[0][i, 2]
    #     box = convert_circle_to_box(x,y,r,img_width, img_height)
    #     box_dict = convert_box_to_dict(box)
    #     box_elem = ET.SubElement(elem, 'box', box_dict)
    #     tree.write(output_cvat_file)


    # now, have to put boxes into cvat annotation format
    

# print(f'annotation file written: {output_cvat_file}')




import code
code.interact(local=dict(globals(), **locals()))
