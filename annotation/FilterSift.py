#!/usr/bin/env python3

# inspired from test_sift_features.py
# sift packaged as an object class with parameters, etc
# to be incorporated into annotation pipeline

import os
import glob
import cv2 as cv
# import matplotlib.pyplot as plt
import numpy as np
from FilterCommon import FilterCommon

CONTRAST_THRESHOLD=0.02
EDGE_THRESHOLD=200
SIGMA=1.2
MIN_SIZE=10
MAX_SIZE=200
DILATE=100

class SiftFeatures(FilterCommon):
    
    def __init__(self,
                 contrast_threshold: float = CONTRAST_THRESHOLD,
                 edge_threshold: float = EDGE_THRESHOLD,
                 sigma: float = SIGMA,
                 min_size: float = MIN_SIZE,
                 max_size: float = MAX_SIZE,
                 dilate: int = DILATE):
        
        FilterCommon.__init__(self)
        
        # sift feature parameters
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        
        # filter sift features
        self.min_size = min_size
        self.max_size = max_size
        
        # when converting sift features to binary mask of putative coral matches
        # how much to dilate around the given features
        # NOTE could consider doing a variable dilation wrt size, but some features have not matched according to size/visually
        # instead, we choose a constant value appropriate for the size of the corals in the image (max radius) in units of pixels
        self.dilate = dilate
        
        # contrastThreshold	The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
        
        # edgeThreshold	The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        
        # 	The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        
        self.sift = cv.SIFT_create(contrastThreshold=self.contrast_threshold, 
                                   edgeThreshold=self.edge_threshold, 
                                   sigma=self.sigma)
        
    
    def get_keypoints(self, image, mask=None):
        # assume image coming in is bgr
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # NOTE: mask can be a binary area of where to look for keypoints, and is optional in the image
        # TODO checks for valid image, numpy array
        keypoints = self.sift.detect(image_gray, mask)
        return keypoints
        
    
    def filter_size(self, keypoints):
        # print(f' length of kp before size filter: {len(keypoints)}')
        kp = []
        for k in keypoints:
            if k.size > self.min_size and k.size < self.max_size:
                kp.append(k)
        # print(f' length of kp after size filter: {len(keypoints)}')
        return kp
        
    
    def filter_response(self, keypoints):
        # take the keypoint with the highest response. We don't care about angle
        best_kp = {}
        for k in keypoints:
            if k.pt in best_kp: # xy tuple
                # check the existing keypoint is in the dictionary
                existing_kp = best_kp[k.pt]
                # if new kp is the same, but has a greater response, replace it
                if k.response > existing_kp.response:
                    best_kp[k.pt] = k
            else:
                best_kp[k.pt] = k
        return list(best_kp.values())
    
    
    def print_keypoints(self, kp):
        for i, k in enumerate(kp):

            print(f'{i}: pt = {k.pt}, size = {k.size}, resp = {k.response}, angle = {k.angle}')
            
    
    def draw_keypoints(self, image_bgr, kp):
        # image should be in bgr format
        
        color = (0, 0, 255) # red in bgr format
        img_ftr = cv.drawKeypoints(image_bgr, 
                                   kp,
                                   0, 
                                   color,
                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img_ftr
        
        
    def get_best_sift_features(self, image):
        kp = self.get_keypoints(image)
        kp = self.filter_size(kp)
        kp = self.filter_response(kp)
        return kp
    
    
    def create_sift_mask(self, image, kp):
        # in areas around SIFT features in kp, create binary mask that says areas around here should be considered as putative matches for corals
        # we do this, so that we can just AND binary masks from other steps in the pipeline that are mask-oriented 
        
        # create mask
        # for each kp, draw on mask with specified dilation radius
        # return mask
        
        mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
        for k in kp:
            center = (int(k.pt[0]), int(k.pt[1]))
            radius = self.dilate
            color = 255 # binary mask, white
            thickness = -1 # filled circle

            cv.circle(mask, center, radius, color, thickness)
        
        return mask
        
    
if __name__ == "__main__":
    print('SiftFeatures.py')
    
    img_pattern = '*.jpg'
    img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/images'
    img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))
    
    save_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/output/sift'
    os.makedirs(save_dir, exist_ok=True)
    sift = SiftFeatures()
    max_img = 10
    for i, img_name in enumerate(img_list):
        print()
        print(f'{i}: {img_name}')
        if i >= max_img:
            print('reached max image limit')
            break
        img_bgr = cv.imread(img_name)
        kp = sift.get_best_sift_features(img_bgr)
        
        # check:
        sift.print_keypoints(kp)
        
        # draw
        img_ftr = sift.draw_keypoints(img_bgr, kp)
        
        # save
        sift.save_image(img_ftr, img_name, save_dir, '_sift.jpg')
    
        # draw mask of sift regions
        mask_sift = sift.create_sift_mask(img_bgr, kp)
        mask_overlay = sift.display_mask_overlay(img_bgr, mask_sift)
        sift.save_image(mask_overlay, img_name, save_dir, '_siftoverlay.jpg')
        
        
    print('done')
        
        
        
    # import code
    # code.interact(local=dict(globals(), **locals()))
        
        
        