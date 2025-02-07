#!/usr/bin/env python3


# filter common functionality, like plotting and blob functions

import cv2 as cv
import numpy as np
import os


class FilterCommon:
    
    DENOISE_TEMPLATE_WINDOW_SIZE = 7
    DENOISE_SEARCH_WINDOW_SIZE = 21
    DENOISE_STRENGTH = 3
    FILTER_MIN_AREA = 1000
    FILTER_MAX_AREA = 20000
    FILTER_MIN_CIRCULARITY = 0.5
    FILTER_MAX_CIRCULARITY = 1.0
    KERNEL_SIZE=11
    
    PROCESS_DENOISE = True
    PROCESS_THRESH = True
    PROCESS_MORPH = True
    PROCESS_FILL = True
    PROCESS_FILTER = True
    
    def __init__(self,
                 template_window_size: int = DENOISE_TEMPLATE_WINDOW_SIZE,
                 search_window_size: int = DENOISE_SEARCH_WINDOW_SIZE,
                 denoise_strength: float = DENOISE_STRENGTH,
                 min_area: float = FILTER_MIN_AREA,
                 max_area: float = FILTER_MAX_AREA,
                 min_circ: float = FILTER_MIN_CIRCULARITY,
                 max_circ: float = FILTER_MAX_CIRCULARITY,
                 kernel_size: int = KERNEL_SIZE,
                 process_denoise: bool = PROCESS_DENOISE,
                 process_thresh: bool = PROCESS_THRESH,
                 process_morph: bool = PROCESS_MORPH,
                 process_fill: bool = PROCESS_FILL,
                 process_filter: bool = PROCESS_FILTER):
        
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
        
        self.process_denoise = process_denoise
        self.process_thresh = process_thresh
        self.process_morph = process_morph
        self.process_fill = process_fill
        self.process_filter = process_filter


    def filter_components(self, mask):
        
        image_filter = np.zeros_like(mask)
        # group blobs into connected components for analysis
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
        label_list = []
        circularity = []
        perimeter = []
        
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA] # 4th column of stats
            
            # create mask of current component
            mask_cc = (labels==i).astype(np.uint8) * 255 # binary mask of label
            
            # calculate perimeter
            p = cv.arcLength(cv.findContours(mask_cc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE,)[0][0], True)
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


    def fill_holes(self, mask):
        contour, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        for cont in contour:
            # draw in the holes, making them "255"
            cv.drawContours(mask, [cont], 0, 255, -1)
        return mask

    # TODO these should probably just be individual functions, and then an overall function that chains them together
    def process(self, 
                image, 
                thresh_min=0, 
                thresh_max=255, 
                thresh_meth=cv.THRESH_BINARY + cv.THRESH_OTSU,
                SAVE_STEPS=False,
                image_name='image',
                save_dir='.'):
        # 1) denoise
        # 2) threshold (Otsu's)
        # 3) morphological ops for nicer blobs (maybe should be after fill-in holes?)
        # 4) fill in holes
        # 5) connected components
        # 6) filter
        

        if self.process_denoise:
            # denoise
            image = cv.fastNlMeansDenoising(image, 
                                            templateWindowSize=self.template_window_size,
                                            searchWindowSize=self.search_window_size,
                                            h=self.denoise_strength)
            if SAVE_STEPS:
                self.save_image(image, image_name=image_name+'_01denoise.jpg', save_dir=save_dir)
        
        if self.process_thresh:
            # threshold using Otsu's method to automatically get threshold
            thresh_value, mask = cv.threshold(image, thresh_min, thresh_max, thresh_meth)
            if SAVE_STEPS:
                self.save_image(mask, image_name=image_name + '_02thresh.jpg', save_dir=save_dir)
        else:
            mask = image
            
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(image)   
        # plt.title('denoised image')
        # plt.show()
        # import code
        # code.interact(local=dict(globals(), **locals()))
        
        if self.process_morph:
            # apply morphological operations
            # to make image filter components smoothed out, and a bit nicer
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
            mask = cv.dilate(mask, kernel, iterations = 1)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            if SAVE_STEPS:
                self.save_image(mask, image_name=image_name + '_03morph.jpg',save_dir=save_dir)
        
        if self.process_fill:
            # fill in any holes from original threshold
            mask = self.fill_holes(mask)
            
            if SAVE_STEPS:
                self.save_image(mask, image_name=image_name + '_04fill.jpg',save_dir=save_dir)
        
        if self.process_filter:
            
            mask, _ = self.filter_components(mask)
            if SAVE_STEPS:
                self.save_image(mask, image_name=image_name + '_05filter.jpg',save_dir=save_dir)
        return mask
    
    
    def display_mask_overlay(self, image, mask, mask_alpha=0.25, mask_color=(0,125,240)):
        # display image with mask as an overlay
        # assume the mask is the same size as the input image TODO should check this
        # mask_alpha is the transparency factor
        
        # resize mask to match image size
        mask = cv.resize(mask, (image.shape[1], image.shape[0]))
        mask = mask.astype(bool)
        
        mask_bgr = np.zeros_like(image)
        mask_bgr[:,:,0] = mask * mask_color[0]
        mask_bgr[:,:,1] = mask * mask_color[1]
        mask_bgr[:,:,2] = mask * mask_color[2]

        mask_bgr = cv.cvtColor(mask_bgr, cv.COLOR_BGR2BGRA)
        
        if image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)

        blended = cv.addWeighted(mask_bgr, mask_alpha, image, 1-mask_alpha, 0)
        
        return blended
        
        
        
    def save_image(self, image, image_name, save_dir='./', str_pattern='.jpg', quality=50):
        # save image with image name, 
        # assume that image_name is absolute filename/path coming from glob
        # save the image in dir
        # according to str_pattern, which specifies the filetype
        # quality to save space
        
        basename = os.path.basename(image_name).rsplit('.', 1)[0]
        save_name = os.path.join(save_dir, basename + str_pattern)
        os.makedirs(save_dir, exist_ok=True)
        
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        cv.imwrite(save_name, image, encode_param)
        
        
        