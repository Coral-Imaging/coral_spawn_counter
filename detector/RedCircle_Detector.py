#! /usr/bin/env python3

"""
code pulled from detect_circle_annotations.py to be a detector
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import op


# get list of images from directory
data_dir = '/mnt/c/20221113_amtenuis_cslics04'
img_dir = os.path.join(data_dir, 'images_jpg')
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
print(img_list)

save_dir = os.path.join(data_dir, 'red_circles')
os.makedirs(save_dir, exist_ok=True)

def find_circles(img, 
                 blur=5, 
                 method=cv.HOUGH_GRADIENT, 
                 dp=0.9, 
                 minDist=30,
                 param1=100,
                 param2=30,
                 maxRadius=125,
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

for i, img_name in enumerate(img_list):
    if i > 3
        break
    
    print(f'{i}: img_name = {img_name}')  
    # read in image
    img = cv.imread(img_name)
    img_height, img_width, _ = img.shape
    # find circles
    circles = find_circles(img)
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

    # get bounding box of the circles
    pred = []
    for i in range(n_circ):
        x = circles[0][i, 0]
        y = circles[0][i, 1]
        r = circles[0][i, 2]
        box = convert_circle_to_box(x,y,r,img_width, img_height)
        print(box)
        pred.append(box)
    
