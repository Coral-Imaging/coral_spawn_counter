#!/usr/bin/env/python3

# annotation pipeline

# combine input from 
# Sift
# Edge
# Hue
# Saturation
# TODO also, Hough transform
# TODO also Laplacian

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from FilterEdge import EdgeFilter
from FilterSift import SiftFilter   
from FilterHue import HueFilter
from FilterSaturation import SaturationFilter


# the idea is that each Filter outputs a binary mask
# each image in the dataset is run through each filter
# at the end of each filter, a binary mask is output 
# that indicates putative regions of interest where an in-focus coral is likely to be
# by ANDing adding up these regions, the intersection of those are most likely to be 
# target corals
# we can then form bounding boxes over these regions (based on whichever measure has the tightest bbox)
# form these as annotations to upload to CVAT in the YOLO format

# TODO might tighten down the dilations as a result

img_pattern = '*.jpg'
img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/images'
img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))

save_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/output/sift'
os.makedirs(save_dir, exist_ok=True)

# init filters (NOTE: options/parameters)
sift = SiftFilter()
sat = SaturationFilter()

max_img = 10
for i, img_name in enumerate(img_list):
    print()
    print(f'{i}: {img_name}')
    if i >= max_img:
        print('reached max image limit')
        break
    img_bgr = cv.imread(img_name)
    
    # SIFT FILTER:
    kp = sift.get_best_sift_features(img_bgr)
    
    # draw
    img_ftr = sift.draw_keypoints(img_bgr, kp)
    sift.save_image(img_ftr, img_name, save_dir, '_sift.jpg')

    # draw mask of sift regions
    mask_sift = sift.create_sift_mask(img_bgr, kp)
    mask__sift_overlay = sift.display_mask_overlay(img_bgr, mask_sift)
    sift.save_image(mask__sift_overlay, img_name, save_dir, '_siftoverlay.jpg')
        
    # SATURATION FILTER:
    mask_sat = sat.create_saturation_mask(img_bgr)
    mask_sat_overlay = sat.display_mask_overlay(img_bgr, mask_sat)
    
    # save
    sat.save_image(mask_sat, img_name, save_dir, '_sat.jpg')
    sat.save_image(mask_sat_overlay, img_name, save_dir, '_satoverlay.jpg')

    # TODO
    # combine two masks
    # show respective overlays onto original image
    # output bboxes from each connected component/region