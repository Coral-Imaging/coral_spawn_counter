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
import yaml
import code 

from FilterEdge import FilterEdge
from FilterSift import FilterSift   
from FilterHue import FilterHue
from FilterSaturation import FilterSaturation


# the idea is that each Filter outputs a binary mask
# each image in the dataset is run through each filter
# at the end of each filter, a binary mask is output 
# that indicates putative regions of interest where an in-focus coral is likely to be
# by ANDing adding up these regions, the intersection of those are most likely to be 
# target corals
# we can then form bounding boxes over these regions (based on whichever measure has the tightest bbox)
# form these as annotations to upload to CVAT in the YOLO format

# TODO might tighten down the dilations as a result for tighter boxes

img_pattern = '*.jpg'
img_dir = '/Users/doriantsai/Code/cslics_ws/cslics_2023_subsurface_dataset/20231102_aant_tank3_cslics06/images'
img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))

save_dir = '/Users/doriantsai/Code/cslics_ws/cslics_2023_subsurface_dataset/20231102_aant_tank3_cslics06/output'
os.makedirs(save_dir, exist_ok=True)

# init filters
config_file = './cslics_annotation_01.yaml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
sift = FilterSift(config=config['sift'])
sat = FilterSaturation(config=config['saturation'])
edge = FilterEdge(config=config['edge'])
hue = FilterHue(config=config['hue'])

max_img = 1
for i, img_name in enumerate(img_list):
    print()
    print(f'{i}: {img_name}')
    if i >= max_img:
        print('reached max image limit')
        break
    img_bgr = cv.imread(img_name)
    
    # EDGE FILTER:
    mask_edge = edge.create_edge_mask(img_bgr)
    mask_edge_overlay = edge.display_mask_overlay(img_bgr, mask_edge)
    
    edge.save_image(mask_edge, img_name, save_dir, '_edge.jpg')
    edge.save_image(mask_edge_overlay, img_name, save_dir, '_edgeoverlay.jpg')
    
    # SIFT FILTER:
    kp = sift.get_best_sift_features(img_bgr)
    
    # draw
    img_ftr = sift.draw_keypoints(img_bgr, kp)
    sift.save_image(img_ftr, img_name, save_dir, '_sift.jpg')

    # draw mask of sift regions
    mask_sift = sift.create_sift_mask(img_bgr, kp)
    mask_sift_overlay = sift.display_mask_overlay(img_bgr, mask_sift)
    sift.save_image(mask_sift_overlay, img_name, save_dir, '_siftoverlay.jpg')
        
    # SATURATION FILTER:
    mask_sat = sat.create_saturation_mask(img_bgr)
    mask_sat_overlay = sat.display_mask_overlay(img_bgr, mask_sat)
 
    sat.save_image(mask_sat, img_name, save_dir, '_sat.jpg')
    sat.save_image(mask_sat_overlay, img_name, save_dir, '_satoverlay.jpg')

    
    # HUE FILTER:
    mask_hue = hue.create_hue_mask(img_bgr)
    mask_hue_overlay = hue.display_mask_overlay(img_bgr, mask_hue)
    
    hue.save_image(mask_hue, img_name, save_dir, '_hue.jpg')
    hue.save_image(mask_hue_overlay, img_name, save_dir, '_hueoverlay.jpg')

    # COMBINE MASKS
    mask_combined = mask_sift & mask_sat & mask_edge & mask_hue
    # show respective overlays onto original image
    mask_combined_overlay = hue.display_mask_overlay(img_bgr, mask_combined)
    
    hue.save_image(mask_combined, img_name, save_dir, '_combined.jpg')
    hue.save_image(mask_combined_overlay, img_name, save_dir, '_combinedoverlay.jpg')
    
    # import code
    # code.interact(local=dict(globals(), **locals()))

    
    # TODO output bboxes from each connected component/region in YOLO format
    