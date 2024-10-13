#!/usr/bin/env python3

# inspired by saturation_shape_threshold.py
# given an image, find the blobs that come out wrt saturation in HSV space
# find the blobs/corals accordingly, output a binary map/mask of putative corals

import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


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

class SaturationFilter:
    
    def __init__(self,
                 template_window_size: int = DENOISE_TEMPLATE_WINDOW_SIZE,
                 search_window_size: int = DENOISE_SEARCH_WINDOW_SIZE,
                 denoise_strength: float = DENOISE_STRENGTH,
                 min_area: float = FILTER_MIN_AREA,
                 max_area: float = FILTER_MAX_AREA,
                 min_circ: float = FILTER_MIN_CIRCULARITY,
                 max_circ: float = FILTER_MAX_CIRCULARITY,
                 kernel_size: int = KERNEL_SIZE):
        
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
        self.denoise_strength = denoise_strength
        self.min_area = min_area
        self.max_area = max_area
        self.min_circ = min_circ
        self.max_circ = max_circ
         
        # NOTE kernel size must be odd
        if kernel_size%2 == 0: # even
            print(f'kernel size received was {kernel_size}. Must be odd, adding 1 to make odd.')
            kernel_size+=1
        self.kernel_size = kernel_size
        
    
    
    # TODO should save HSV image?
    
    def filter_components(self, image_filter, num_labels, labels, stats):

        label_list = []
        circularity = []
        perimeter = []
        
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA] # 4th column of stats
            
            # create mask of current component
            mask = (labels==i).astype(np.uint8) * 255 # binary mask of label
            
            # calculate perimeter
            p = cv.arcLength(cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE,)[0][0], True)
            perimeter.append(p)
            
            # calculate circularity
            if p > 0:
                c = (4 * np.pi * area) / (p ** 2)
                if c > 1.0: # sometimes by numerical calculations or pixel artifacts, holes, etc
                    c = 1.0
            else:
                c = 0
            circularity.append(c)
            
            # filter by area and circularity
            if self.min_area <= area <= self.max_area and \
                self.min_circ <= c <= self.max_circ:
                    image_filter[labels==i] = 255 
                    label_list.append(i)
        
        return image_filter, label_list
    
    
    
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
        
        # denoise
        image_den = cv.fastNlMeansDenoising(image_s, 
                                            templateWindowSize=self.template_window_size,
                                            searchWindowSize=self.search_window_size,
                                            h=self.denoise_strength)
        
        # threshold using Otsu's method to automatically get threshold
        thresh_value, image_bin = cv.threshold(image_den, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # fill in any holes from original threshold
        contour, _ = cv.findContours(image_bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        for cont in contour:
            cv.drawContours(image_bin, [cont], 0, 255, -1)
        
        # group blobs into connected components for analysis
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(image_bin, 
                                                                              connectivity=8)
        
        
        image_filter, label_list = self.filter_components(np.zeros_like(image_bin), num_labels, labels, stats)
        
        # apply morphological operations
        # to make image filter components smoothed out, and a bit nicer
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        mask = cv.dilate(image_filter, kernel, iterations = 1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        
        return mask
    
    

if __name__ == "__main__":
    print('SaturationFilter.py')
    
    img_pattern = '*.jpg'
    img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/images'
    img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))
    
    save_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/output/saturation'
    os.makedirs(save_dir, exist_ok=True)
    
    sat = SaturationFilter()
    max_img = 10
    for i, img_name in enumerate(img_list):
        print()
        print(f'{i}: {img_name}')
        if i >= max_img:
            print('reached max image limit')
            break
        img_bgr = cv.imread(img_name)
        
        img_sat_mask = sat.create_saturation_mask(img_bgr)
        
        # save
        basename = os.path.basename(img_name).rsplit('.', 1)[0]
        save_name = os.path.join(save_dir, basename + '_sat.jpg')
        cv.imwrite(save_name, img_sat_mask)
        
        # TODO save image as alpha on original image for easier comparison
        # TODO should be a base class, since all filter methods will want this functionality
        
    print('done')

# import code
# code.interact(local=dict(globals(), **locals()))