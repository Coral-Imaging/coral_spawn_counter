#!/usr/bin/env python3

import glob
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from FilterCommon import FilterCommon

DENOISE_TEMPLATE_WINDOW_SIZE = 11
DENOISE_SEARCH_WINDOW_SIZE = 31
DENOISE_STRENGTH = 5
KERNEL_SIZE=31

FILTER_MIN_AREA = 3000
FILTER_MAX_AREA = 40000
FILTER_MIN_CIRCULARITY = 0.3
FILTER_MAX_CIRCULARITY = 1.0

KERNEL_EDGE_DILATION_SIZE = 51
LAPLACIAN_KERNEL_SIZE = 5

class FilterLaplacian(FilterCommon):
    
    def __init__(self,
                 template_window_size: int = DENOISE_TEMPLATE_WINDOW_SIZE,
                 search_window_size: int = DENOISE_SEARCH_WINDOW_SIZE,
                 denoise_strength: float = DENOISE_STRENGTH,
                 min_area: float = FILTER_MIN_AREA,
                 max_area: float = FILTER_MAX_AREA,
                 min_circ: float = FILTER_MIN_CIRCULARITY,
                 max_circ: float = FILTER_MAX_CIRCULARITY,
                 kernel_size: int = KERNEL_SIZE,
                 laplacian_kernel_size: int = LAPLACIAN_KERNEL_SIZE,
                 edge_dilation_kernel_size: int = KERNEL_EDGE_DILATION_SIZE,
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
            self.laplacian_kernel_size = config['laplacian_kernel_size']
            self.edge_dilation_kernel_size = config['edge_dilation_kernel_size']
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
            self.laplacian_kernel_size = laplacian_kernel_size
            self.edge_dilation_kernel_size = edge_dilation_kernel_size
        
        
    def create_laplacian_mask(self, image_bgr):
        
        # convert to grayscale
        image_g = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        
        # serious denoise
        image_d = cv.fastNlMeansDenoising(image_g, 
                                        templateWindowSize=self.template_window_size,
                                        searchWindowSize=self.search_window_size,
                                        h=self.denoise_strength)
        
        # compute laplacian response and scale it to absolute
        lapl = cv.Laplacian(image_d, cv.CV_16S, ksize=self.laplacian_kernel_size)
        abs_lapl = cv.convertScaleAbs(lapl)
        
        
        # plt.figure()
        # plt.imshow(abs_lalpl)
        # plt.show()
        
        # import code
        # code.interact(local=dict(globals(), **locals()))
        mask = self.process(abs_lapl, 
                            thresh_min=25,
                            thresh_max=255,
                            thresh_meth=cv.THRESH_BINARY,
                            DENOISE=True,
                            THRESHOLD=True,
                            MORPH=True,
                            FILL_HOLES=True,
                            FILTER_CC=True,
                            SAVE_STEPS=True)
        
        # need a big dilation at the end to expand the region
        # expand the surviving regions
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.edge_dilation_kernel_size, self.edge_dilation_kernel_size))
        mask_dilated = cv.dilate(mask, kernel, iterations=1)
        return mask_dilated
    
    
if __name__ == "__main__":
    print('FilterLaplacian.py')
    
    img_pattern = '*.jpg'
    img_dir = '/Users/doriantsai/Code/cslics_ws/cslics_2023_subsurface_dataset/20231102_aant_tank3_cslics06/images'
    img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))
    
    save_dir = '/Users/doriantsai/Code/cslics_ws/cslics_2023_subsurface_dataset/20231102_aant_tank3_cslics06/laplacian'
    os.makedirs(save_dir, exist_ok=True)
    
    lap = FilterLaplacian()
    max_img = 10
    for i, img_name in enumerate(img_list):
        print()
        print(f'{i}: {img_name}')
        if i >= max_img:
            print('reached max image limit')
            break
        img_bgr = cv.imread(img_name)
        
        mask = lap.create_laplacian_mask(img_bgr)
        mask_overlay = lap.display_mask_overlay(img_bgr, mask)
        
        lap.save_image(mask, img_name, save_dir, '_lap.jpg')
        lap.save_image(mask_overlay, img_name, save_dir, '_lapoverlay.jpg')
        
    
    print('done')
        