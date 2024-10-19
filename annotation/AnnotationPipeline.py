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
import re

from FilterEdge import FilterEdge
from FilterSift import FilterSift   
from FilterHue import FilterHue
from FilterSaturation import FilterSaturation


def save_image_predictions(predictions, img, imgname, imgsavedir, class_colors, quality=50, imgformat='.jpg'):
        """
        save predictions/detections (assuming predictions in yolo format) on image
        """
        # assuming input image is rgb, need to convert back to bgr:
        
        imgw, imgh = img.shape[1], img.shape[0]
        for p in predictions:
            cls = int(p[0])
            xcen, ycen, w, h = p[1], p[2], p[3], p[4]
            
            #extract back into cv lengths
            x1 = (xcen - w/2) *imgw
            x2 = (xcen + w/2) *imgw
            y1 = (ycen - h/2) *imgh
            y2 = (ycen + h/2 ) *imgh    
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), class_colors, 3)
            # cv.putText(img, f"{class_name}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[self.classes[cls]], 2)

        imgsavename = os.path.basename(imgname)
        # add day into save directory to prevent an untenable number of images in a single folder
        os.makedirs(os.path.join(imgsavedir), exist_ok=True)
        
        imgsave_path = os.path.join(imgsavedir, imgsavename.rsplit('.',1)[0] + '_annotated' + imgformat)
        
        # to save on memory, reduce quality of saved image
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        cv.imwrite(imgsave_path, img, encode_param)
        return True
    
def save_text_predictions(annotations, imgname, txtsavedir, txtformat='.txt'):
        """
        save annotations/predictions/detections into text file
        [class x1 y1 x2 y2]
        """
        txtsavename = os.path.basename(imgname).rsplit('.',1)[0]
        os.makedirs(os.path.join(txtsavedir), exist_ok=True)
        txtsavepath = os.path.join(txtsavedir, txtsavename + txtformat)

        # predictions [ pix pix pix pix conf class ]
        with open(txtsavepath, 'w') as f:
            for a in annotations:
                class_label = int(a[0])
                x1, y1, x2, y2 = a[1], a[2], a[3], a[4]
                f.write(f'{class_label:g} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n')
        return True

# the idea is that each Filter outputs a binary mask
# each image in the dataset is run through each filter
# at the end of each filter, a binary mask is output 
# that indicates putative regions of interest where an in-focus coral is likely to be
# by ANDing adding up these regions, the intersection of those are most likely to be 
# target corals
# we can then form bounding boxes over these regions (based on whichever measure has the tightest bbox)
# form these as annotations to upload to CVAT in the YOLO format

# TODO might tighten down the dilations as a result for tighter boxes

#####################
img_pattern = '*.jpg'
img_dir = '/Users/doriantsai/Code/cslics_ws/cslics_2023_subsurface_dataset/20231102_aant_tank3_cslics06/images'
img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))

# save output images
save_dir = '/Users/doriantsai/Code/cslics_ws/cslics_2023_subsurface_dataset/20231102_aant_tank3_cslics06/output'
os.makedirs(save_dir, exist_ok=True)

# save dataset export directory
save_export_dir = '/Users/doriantsai/Code/cslics_ws/cslics_2023_subsurface_dataset/20231102_aant_tank3_cslics06/export'
# save output annotations
txt_save_dir = os.path.join(save_export_dir, 'obj_train_data')
os.makedirs(txt_save_dir, exist_ok=True)

# save train.txt file
train_txt_name = os.path.join(save_export_dir, 'train.txt' )
# wipe it clean
with open(train_txt_name, 'w') as train_file:
    pass
        
# save obj.names file
obj_names_file = os.path.join(save_export_dir, 'obj.names')

# save obj.data file
obj_data_file = os.path.join(save_export_dir, 'obj.data')
with open(obj_data_file, 'w') as file:
    file.write('classes = 1\n')
    file.write('train = data/train.txt\n')
    file.write('names = data/obj.names\n')
    file.write('backup = backup/')

######################

# init filters
config_file = '../data/annotation_20231102_aant_tank3_cslics06.yaml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

class_name = config['class']['name']
class_label = config['class']['label']
# print(type(class_name))
print(f'class_name = {class_name}')
class_color = tuple(config['class']['color_bgr'])
# print(type(class_color))
print(f'class_color = {class_color}')

# write obj.names file:
with open(obj_names_file, 'w') as file:
    file.write(class_name)


sift = FilterSift(config=config['sift'])
sat = FilterSaturation(config=config['saturation'])
edge = FilterEdge(config=config['edge'])
hue = FilterHue(config=config['hue'])

max_img = 4
for i, img_name in enumerate(img_list):
    print()
    print(f'{i}: {img_name}')
    if i >= max_img:
        print('reached max image limit')
        break
    img_bgr = cv.imread(img_name)
    
    # EDGE FILTER:
    #mask_edge = edge.create_edge_mask(img_bgr)
    #mask_edge_overlay = edge.display_mask_overlay(img_bgr, mask_edge)
    
    #edge.save_image(mask_edge, img_name, save_dir, '_edge.jpg')
    #edge.save_image(mask_edge_overlay, img_name, save_dir, '_edgeoverlay.jpg')
    
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
    # mask_combined = mask_sift & mask_sat & mask_edge & mask_hue
    mask_combined = mask_sift & mask_sat & mask_hue
    # show respective overlays onto original image
    mask_combined_overlay = hue.display_mask_overlay(img_bgr, mask_combined)
    
    hue.save_image(mask_combined, img_name, save_dir, '_combined.jpg')
    hue.save_image(mask_combined_overlay, img_name, save_dir, '_combinedoverlay.jpg')
    
    # output bboxes from each connected component/region in YOLO format
    img_width, img_height = mask_combined.shape[1], mask_combined.shape[0]
    contours, _ = cv.findContours(mask_combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bb = []
    for c in contours:
        # get bbox:
        x,y,w,h = cv.boundingRect(c)
        xcen = (x + w/2.0)/img_width
        ycen = (y + h/2.0)/img_height
        
        # x1 = x / img_width
        # y1 = y / img_height
        # x2 = (x + w)/img_width
        # y2 = (y + h)/img_height
        # class x_center y_center width height
        bb.append([class_label, xcen, ycen, w/img_width, h/img_height])
        
    # export/save to text file
    save_text_predictions(bb, img_name, txt_save_dir)
    
    # draw exported annotations
    save_image_predictions(bb, img_bgr, img_name, save_dir, class_color)
        
    # append name to train.txt file
    write_line = os.path.join('data/obj_train_data/', os.path.basename(img_name))
    with open(train_txt_name, 'a') as train_file:
        train_file.write(write_line + '\n')
    

print('done')
# import code
# code.interact(local=dict(globals(), **locals()))
    