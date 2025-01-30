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
LAPLACIAN_THRESHOLD = 25
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
                 laplacian_threshold: int = LAPLACIAN_THRESHOLD,
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
            self.laplacian_kernel_size = config['laplacian_kernel_size']
            self.laplacian_threshold = config['laplacian_threshold']
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
                                kernel_size,
                                process_denoise=True,
                                process_thresh=True,
                                process_morph=True,
                                process_fill=True,
                                process_filter=True
                                )
            self.laplacian_kernel_size = laplacian_kernel_size
            self.laplacian_threshold = laplacian_threshold
            self.edge_dilation_kernel_size = edge_dilation_kernel_size
        
        
    def create_laplacian_mask(self, image_bgr, image_name=None, save_dir=None):
        
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
        
        # uncomment to see absolute laplacian image/values
        # plt.figure()
        # plt.imshow(abs_lapl)
        # plt.show()
        # TODO create a histogram of this to help find?
        
        # import code
        # code.interact(local=dict(globals(), **locals()))
        mask = self.process(abs_lapl, 
                            thresh_min=self.laplacian_threshold,
                            thresh_max=255,
                            thresh_meth=cv.THRESH_BINARY,
                            SAVE_STEPS=False, # TODO fix if True, but image_name and save_dir not passed, crashes
                            image_name=image_name,
                            save_dir=save_dir)
        
        # need a big dilation at the end to expand the region
        # expand the surviving regions
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.edge_dilation_kernel_size, self.edge_dilation_kernel_size))
        mask_dilated = cv.dilate(mask, kernel, iterations=1)
        return mask_dilated
    
    
if __name__ == "__main__":
    print('FilterLaplacian.py')
    
    img_pattern = '*.jpg'
    # root_dir = 
    # img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231103_aten_tank4_cslics08/images'
    img_dir = '/home/dorian/Data/cslilcs_2024_october_subsurface_dataset/100000009c23b5af/images'
    img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)), reverse=True)
    
    # save_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231103_aten_tank4_cslics08/output/hue'
    save_dir = '/home/dorian/Data/cslilcs_2024_october_subsurface_dataset/100000009c23b5af/output/laplacian'
    os.makedirs(save_dir, exist_ok=True)
    
    config = {}
    config['denoise_template_window_size'] = 14
    config['denoise_search_window_size'] = 31
    config['denoise_strength'] = 7
    config['min_area'] = 400
    config['max_area'] = 10000
    config['min_circularity'] = 0.2
    config['max_circularity'] = 1.0
    config['kernel_size'] = 11
    config['process_denoise'] = True
    config['process_thresh'] = True
    config['process_morph'] = True
    config['process_fill'] = True
    config['process_filter'] = True
    config['laplacian_kernel_size'] = 5
    config['laplacian_threshold'] = 20
    config['edge_dilation_kernel_size'] = 40
    
    lap = FilterLaplacian(config=config)
    max_img = 10
    # iterate over several images
    for i, img_name in enumerate(img_list):
        print()
        print(f'{i}: {img_name}')
        if i >= max_img:
            print('reached max image limit')
            break
        img_bgr = cv.imread(img_name)
        
        basename = os.path.basename(img_name).rsplit('.', 1)[0] # temp hack to save progress
        mask = lap.create_laplacian_mask(img_bgr, image_name=basename, save_dir=save_dir)
        mask_overlay = lap.display_mask_overlay(img_bgr, mask)
        
        lap.save_image(mask, img_name, save_dir, '_lap.jpg')
        lap.save_image(mask_overlay, img_name, save_dir, '_lapoverlay.jpg')
    
    # repeat with different thresholds for one image
    # img_name = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231204_alor_tank3_cslics06/images/cslics06_20231206_202458_816291_img.jpg'
    # save_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231204_alor_tank3_cslics06/laplacian'
    # os.makedirs(save_dir, exist_ok=True)
    
    # n = 10
    # lap_thresh_array = np.linspace(10, n*10, n)
    # img_bgr = cv.imread(img_name)
    # print(lap_thresh_array)
    # for i, lap_thresh in enumerate(lap_thresh_array):
    #     print()
    #     print(f'{i}: lap_thresh = {lap_thresh}')
    #     lap = FilterLaplacian(laplacian_threshold=lap_thresh)
    #     mask = lap.create_laplacian_mask(img_bgr)
    #     mask_overlay = lap.display_mask_overlay(img_bgr, mask)
    #     lap.save_image(mask, img_name, save_dir, '_lap'+str(lap_thresh)+'.jpg')
    #     lap.save_image(mask_overlay, img_name, save_dir, '_lapoverlay'+str(lap_thresh)+'.jpg')
    
    
    print('done')
        