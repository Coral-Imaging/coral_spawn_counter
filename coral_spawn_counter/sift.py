# test 2D feature detectors on subsurface images 


import os
import glob as glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# image directory/file
img_dir = '/home/dorian/Data/cslics_subsurface_150_dataset/images'
save_dir = '/home/dorian/Data/cslics_subsurface_150_dataset/output'
os.makedirs(save_dir, exist_ok=True)

# define sift features
sift = cv.SIFT_create(contrastThreshold=0.04, edgeThreshold=100, sigma=1.2)
# https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html#ad337517bfdc068ae0ba0924ff1661131
# The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
# The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
# The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number. 
# The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
# The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).

# image files, sorted
img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
print(len(img_files))
# iterate through all images

MAX_IMG = 200
keypoints_list = []

for i, img_name in enumerate(img_files):
    base_img_name = os.path.basename(img_name)
    print(f'{i}/{len(img_files)}: {base_img_name}')
    
    if i >= MAX_IMG:
        print('hit max img')
        break
    else:
    
        # read in image, 
        im = cv.imread(img_name)

        # resize image for consistency:
        desired_width = 1280 # pixels
        aspect_ratio = desired_width / im.shape[1]
        height = int(im.shape[0] * aspect_ratio)
        im_r = cv.resize(im, (desired_width, height), interpolation=cv.INTER_AREA)
        
        # convert to grayscale
        im_gray = cv.cvtColor(im_r, cv.COLOR_BGR2GRAY)
        
        # SIFT 
        # https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html
        keypoints = sift.detect(im_gray, None)
        
        
        
        # TODO this should be a function
        # need to eliminate redundant/overlapping SIFT features due different orientation and size
        # elimination heuristics:
        # 1) Just take the keypoint with the highest response. We don't care about angle
        # TODO we also need to bound the SIFT features to within a min and max size
        # TODO eventually, would want to filter out by response
        best_keypoints = {}
        for k in keypoints:
            if k.pt in best_keypoints: # xy tuple
                # check the existing keypoint is in the dict
                existing_keypoint = best_keypoints[k.pt]
                # if new keypoint is the same pt, but a greater response, replace it
                if k.response > existing_keypoint.response:
                    best_keypoints[k.pt] = k
            else:
                best_keypoints[k.pt] = k

        select_keypoints = list(best_keypoints.values())
        
        # draw keypoints for reference
        im_sift = cv.drawKeypoints(im_r, select_keypoints,0, (0,0,255),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # save keypoints into list
        keypoints_list.append(select_keypoints)
        
        # save image
        fname = os.path.join(save_dir, base_img_name[:-4] + '_sift.jpg')
        cv.imwrite(fname, im_sift)
    
    
print('done')

import code
code.interact(local=dict(globals(), **locals()))

# kp_size = [kp.size for kp in keypoints_list[3]]
# kp_response = [kp.response for kp in keypoints_list[3]]
# keypoints_list[3]