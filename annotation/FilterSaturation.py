#!/usr/bin/env python3

# inspired by saturation_shape_threshold.py
# given an image, find the blobs that come out wrt saturation in HSV space
# find the blobs/corals accordingly, output a binary map/mask of putative corals

import os
import glob
import cv2 as cv
import numpy as np
from FilterCommon import FilterCommon

DENOISE_TEMPLATE_WINDOW_SIZE = 7
DENOISE_SEARCH_WINDOW_SIZE = 21
DENOISE_STRENGTH = 3
# templateWindowSize	Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels
# searchWindowSize	Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels
# h	Parameter regulating filter strength. Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise

FILTER_MIN_AREA = 1000
FILTER_MAX_AREA = 20000
FILTER_MIN_CIRCULARITY = 0.5
FILTER_MAX_CIRCULARITY = 1.0

KERNEL_SIZE=11

class FilterSaturation(FilterCommon):
    
    def __init__(self,
                 template_window_size: int = DENOISE_TEMPLATE_WINDOW_SIZE,
                 search_window_size: int = DENOISE_SEARCH_WINDOW_SIZE,
                 denoise_strength: float = DENOISE_STRENGTH,
                 min_area: float = FILTER_MIN_AREA,
                 max_area: float = FILTER_MAX_AREA,
                 min_circ: float = FILTER_MIN_CIRCULARITY,
                 max_circ: float = FILTER_MAX_CIRCULARITY,
                 kernel_size: int = KERNEL_SIZE,
                 config: dict = None):
        
        if config:
            FilterCommon.__init__(self, 
                                config['denoise_template_window_size'], 
                                config['denoise_search_window_size'],
                                config['denoise_strength'],
                                config['min_area'],
                                config['max_area'],
                                config['min_circularity'],
                                config['max_circularity'],
                                config['kernel_size'],
                                config['process_denoise'],
                                config['process_thresh'],
                                config['process_morph'],
                                config['process_fill'],
                                config['process_filter'])
        else:
            FilterCommon.__init__(self, 
                                template_window_size, 
                                search_window_size,
                                denoise_strength,
                                min_area,
                                max_area,
                                min_circ,
                                max_circ,
                                kernel_size,
                                process_denoise=True,
                                process_thresh=True,
                                process_morph=True,
                                process_fill=True,
                                process_filter=True
                                )
   
  
    
    def create_saturation_mask(self, image_bgr):
        # process saturation image:
        # denoise
        # apply thresholding
        # use connected components to get blobs
        # filter blobs
        
        # Saturation:
        #     Definition: Measures the intensity or purity of the color. A high saturation means the color is vivid and intense, while low saturation means the color is more muted or washed out (closer to gray).
        #     Range: Usually ranges from 0% (gray, no color) to 100% (pure color).
        #     Usage: Saturation helps to control the vividness of colors in an image.
        image_hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
        image_s = image_hsv[:,:,1]
        
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(image_s)
        # plt.show()
        
        mask = self.process(image_s, SAVE_STEPS=False)
        return mask
    
    

if __name__ == "__main__":
    print('SaturationFilter.py')
    
    img_pattern = '*.jpg'
    img_dir = '/home/dorian/Data/cslilcs_2024_october_subsurface_dataset/100000009c23b5af/images'
    img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))
    
    # save_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231103_aten_tank4_cslics08/output/hue'
    save_dir = '/home/dorian/Data/cslilcs_2024_october_subsurface_dataset/100000009c23b5af/output/saturation'
    os.makedirs(save_dir, exist_ok=True)
    
    config = {}
    config['denoise_template_window_size'] = 14
    config['denoise_search_window_size'] = 31
    config['denoise_strength'] = 5
    config['min_area'] = 2000
    config['max_area'] = 5000
    config['min_circularity'] = 0.2
    config['max_circularity'] = 1.0
    config['kernel_size'] = 31
    config['process_denoise'] = True
    config['process_thresh'] = True
    config['process_morph'] = False
    config['process_fill'] = False
    config['process_filter'] = False
    # config['hue_min'] = 0
    # config['hue_max'] = 40
    # config['edge_dilation_kernel_size'] = 31
                 
                 
    sat = FilterSaturation(config=config)
    max_img = 5
    for i, img_name in enumerate(img_list):
        print()
        print(f'{i}: {img_name}')
        if i >= max_img:
            print('reached max image limit')
            break
        img_bgr = cv.imread(img_name)
        
        mask = sat.create_saturation_mask(img_bgr)
        
        mask_overlay = sat.display_mask_overlay(img_bgr, mask)
        
        # save
        sat.save_image(mask, img_name, save_dir, '_sat.jpg')
        sat.save_image(mask_overlay, img_name, save_dir, '_satoverlay.jpg')
        
    print('done')

# import code
# code.interact(local=dict(globals(), **locals()))