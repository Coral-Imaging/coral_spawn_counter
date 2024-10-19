#!/usr/bin/env python3

# edge detection filter for annotating data pipeline

import os
import glob
import cv2 as cv
import numpy as np
from FilterCommon import FilterCommon

DENOISE_TEMPLATE_WINDOW_SIZE = 7
DENOISE_SEARCH_WINDOW_SIZE = 21
DENOISE_STRENGTH = 3
KERNEL_SIZE=11

KERNEL_GAUSSIAN_SMOOTH = 37

CANNY_LOWER_THRESHOLD = 8
CANNY_UPPER_THRESHOLD = 15

FILTER_MIN_AREA = 2000
FILTER_MAX_AREA = 40000
FILTER_MIN_CIRCULARITY = 0.3
FILTER_MAX_CIRCULARITY = 1.0
KERNEL_EDGE_DILATION_SIZE = 31

# NOTE use edge_detection.py to determine what canny edge thresholds to use, as they are quite sensitive

class FilterEdge(FilterCommon):
    
    def __init__(self,
                 template_window_size: int = DENOISE_TEMPLATE_WINDOW_SIZE,
                 search_window_size: int = DENOISE_SEARCH_WINDOW_SIZE,
                 denoise_strength: float = DENOISE_STRENGTH,
                 canny_lower_thresh: float = CANNY_LOWER_THRESHOLD,
                 canny_upper_thresh: float = CANNY_UPPER_THRESHOLD,
                 min_area: float = FILTER_MIN_AREA,
                 max_area: float = FILTER_MAX_AREA,
                 min_circ: float = FILTER_MIN_CIRCULARITY,
                 max_circ: float = FILTER_MAX_CIRCULARITY,
                 kernel_size: int = KERNEL_SIZE,
                 edge_dilation: int = KERNEL_EDGE_DILATION_SIZE,
                 gaussian_kernel: int = KERNEL_GAUSSIAN_SMOOTH,
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
                                config['kernel_size'])
            self.canny_lower_thresh = config['canny_lower_thresh']
            self.canny_upper_thresh = config['canny_upper_thresh']
            self.edge_dilation = config['edge_dilation']
            self.gaussian_kernel = config['gaussian_kernel']
        else:
            FilterCommon.__init__(self, 
                                template_window_size, 
                                search_window_size,
                                denoise_strength,
                                min_area,
                                max_area,
                                min_circ,
                                max_circ,
                                kernel_size)
            self.canny_lower_thresh = canny_lower_thresh
            self.canny_upper_thresh = canny_upper_thresh
            self.edge_dilation = edge_dilation
            self.gaussian_kernel = gaussian_kernel
        
    
    def create_edge_mask(self, image_bgr):
        """create_edge_mask

        Args:
            image_bgr (_type_): image, bgr format
        """
        
        # grayscale
        image_g = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        
        # denoise
        image_den = cv.fastNlMeansDenoising(image_g,
                                            templateWindowSize=self.template_window_size,
                                            searchWindowSize=self.search_window_size,
                                            h=self.denoise_strength)
        
        # blur/smooth
        image_g = cv.GaussianBlur(image_den, ksize=(self.gaussian_kernel, self.gaussian_kernel), sigmaX=0)
        
        # canny edge detection
        image_edges = cv.Canny(image_den, 
                               threshold1=self.canny_lower_thresh, 
                               threshold2=self.canny_upper_thresh)
        
        # process
        mask = self.process(image_edges,
                            DENOISE=False,
                            THRESHOLD=False,
                            MORPH=True,
                            FILL_HOLES=True,
                            FILTER_CC=True,
                            SAVE_STEPS=False)

        # expand the surviving regions
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.edge_dilation, self.edge_dilation))
        mask_dilated = cv.dilate(mask, kernel, iterations=1)
        
        return mask_dilated
    

if __name__ == "__main__"        :
    
    print('EdgeFilter.py')
    
    img_pattern = '*.jpg'
    # img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/images'
    img_dir = '/Users/doriantsai/Code/cslics_ws/cslics_2023_subsurface_dataset/20231102_aant_tank3_cslics06/images'
    img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))
    
    # save_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/output/edge'
    save_dir = '/Users/doriantsai/Code/cslics_ws/cslics_2023_subsurface_dataset/20231102_aant_tank3_cslics06/edge'
    os.makedirs(save_dir, exist_ok=True)
    
    edge = FilterEdge()
    max_img = 4
    
    for i, img_name in enumerate(img_list):
        print()
        print(f'{i}: {img_name}')
        
        if i >= max_img:
            print('reached max image limit')
            break
        
        img_bgr = cv.imread(img_name)
        
        mask = edge.create_edge_mask(img_bgr)
        
        mask_overlay = edge.display_mask_overlay(img_bgr, mask)
        
        edge.save_image(mask, img_name, save_dir, '_edge.jpg')
        edge.save_image(mask_overlay, img_name, save_dir, '_edgeoverlay.jpg')
        
    print('done')
    

# import code
# code.interact(local=dict(globals(), **locals()))