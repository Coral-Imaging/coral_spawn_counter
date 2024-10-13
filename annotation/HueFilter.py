#!/usr/bin/env python3


# inspired by color_threshold.py
# given an image, find the blobs that come out wrt hue in HSV space
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

KERNEL_SIZE=11
FILTER_MIN_AREA = 1000
FILTER_MAX_AREA = 20000
FILTER_MIN_CIRCULARITY = 0.5
FILTER_MAX_CIRCULARITY = 1.0

HUE_MIN = 0
HUE_MAX = 30

class HueFilter(FilterCommon):
    
    def __init__(self,
                 template_window_size: int = DENOISE_TEMPLATE_WINDOW_SIZE,
                 search_window_size: int = DENOISE_SEARCH_WINDOW_SIZE,
                 denoise_strength: float = DENOISE_STRENGTH,
                 min_area: float = FILTER_MIN_AREA,
                 max_area: float = FILTER_MAX_AREA,
                 min_circ: float = FILTER_MIN_CIRCULARITY,
                 max_circ: float = FILTER_MAX_CIRCULARITY,
                 kernel_size: int = KERNEL_SIZE,
                 hue_min: float = HUE_MIN,
                 hue_max: float = HUE_MAX):
        
        FilterCommon.__init__(self, 
                              template_window_size, 
                              search_window_size,
                              denoise_strength,
                              min_area,
                              max_area,
                              min_circ,
                              max_circ,
                              kernel_size)
        self.hue_min = hue_min
        self.hue_max = hue_max
   
    
    def create_hue_mask(self, image_bgr):
        # process hue image to create mask of where relevant blobs are
        image_hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
        image_h = image_hsv[:,:,0]
        
        image_h = cv.GaussianBlur(image_h, ksize=(self.kernel_size,self.kernel_size),sigmaX=0 )
        mask = self.process(image_h, 
                            thresh_min=self.hue_max, 
                            thresh_max=255, 
                            thresh_meth=cv.THRESH_BINARY_INV,
                            DENOISE=True,
                            THRESHOLD=True,
                            MORPH=False,
                            FILL_HOLES=False,
                            FILTER_CC=False)
        return mask
    
    

if __name__ == "__main__":
    print('HueFilter.py')
    
    img_pattern = '*.jpg'
    img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/images'
    img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))
    
    save_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/output/hue'
    os.makedirs(save_dir, exist_ok=True)
    
    hue = HueFilter()
    max_img = 4
    for i, img_name in enumerate(img_list):
        print()
        print(f'{i}: {img_name}')
        if i >= max_img:
            print('reached max image limit')
            break
        img_bgr = cv.imread(img_name)
        
        img_hue_mask = hue.create_hue_mask(img_bgr)
        
        # save
        basename = os.path.basename(img_name).rsplit('.', 1)[0]
        save_name = os.path.join(save_dir, basename + '_hue.jpg')
        cv.imwrite(save_name, img_hue_mask)
        
        # TODO save image as alpha on original image for easier comparison
        
    print('done')

# import code
# code.interact(local=dict(globals(), **locals()))