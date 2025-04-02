#!/usr/bin/env python3

# apply surf features to cslics subsurface imagery
# to get putative detections for building annotations

import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt


img_pattern = '*.jpg'
img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231103_aten_tank4_cslics08/images'
img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))

i = 1

img_name = img_list[i]
img_bgr = cv.imread(img_name) # BGR format
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

img = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# surf features, initialise and detect
# surf = cv.xfeatures2d.SURF_create(400)
# kp, des = surf.detectAndCompute(img, None)
# print(len(kp))
img = cv.fastNlMeansDenoising(img, 
                            templateWindowSize=7,
                            searchWindowSize=21,
                            h=5)

# sift
# contrastThreshold	The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
# edgeThreshold	The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
# sigma	The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number. 
sift = cv.SIFT_create(contrastThreshold=0.01, edgeThreshold=200, sigma=1.0)
kp = sift.detect(img, None)

# need to eliminate redundant/overlapping SIFT features due different orientation and size
# elimination heuristics:

# 0) Remove keypoints with too small radii, in kp[i].size
# should sort the list of kp wrt size, then remove everything below and above values
min_size = 1 # pixels
max_size = 200 # pixels

kp_size = []
print(f' length of kp before size filter: {len(kp)}')
for i, k in enumerate(kp):
    if k.size > min_size and k.size < max_size:
        kp_size.append(k)
        
print(f' length of kp after size filter: {len(kp_size)}')

for i, k in enumerate(kp_size):
    print(f'{i}: size={k.size}, resp = {k.response}')

# 1) Just take the keypoint with the highest response. We don't care about angle
# TODO we also need to bound the SIFT features to within a min and max size
# TODO eventually, would want to filter out by response
best_keypoints = {}

print(f' length of kp before response filter: {len(kp_size)}')
for k in kp_size:
    if k.pt in best_keypoints: # xy tuple
        # check the existing keypoint is in the dict
        existing_keypoint = best_keypoints[k.pt]
        # if new keypoint is the same pt, but a greater response, replace it
        if k.response > existing_keypoint.response:
            best_keypoints[k.pt] = k
    else:
        best_keypoints[k.pt] = k

select_keypoints = list(best_keypoints.values())
        
print(f' length of kp before response filter: {len(select_keypoints)}')
img_ftr = cv.drawKeypoints(img_bgr, select_keypoints,0, (0,0,255),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', img_ftr)
cv.waitKey(0)
cv.destroyAllWindows()

for i, k in enumerate(select_keypoints):
    print(f'{i}: size={k.size}, resp = {k.response}')
    
import code
code.interact(local=dict(globals(), **locals()))