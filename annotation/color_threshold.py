#!/usr/bin/env python3

# sort through images and threshold by colour
# convert to HSV and focus on H


import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_pattern = '*.jpg'
img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/images'
img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))

i = 0

img_name = img_list[0]
img_bgr = cv.imread(img_name) # BGR format
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

# show image
plt.figure()
plt.imshow(img_rgb)
plt.title('original rgb image')

# show HSV as individual layers
plt.figure()
plt.imshow(img_hsv[:,:,0])
plt.title('H Hue')
# Hue:
#     Definition: Represents the color type and is measured as an angle on a color wheel. It indicates the dominant wavelength of color.
#     Range: Typically ranges from 0° to 360°:
#         0° (or 360°) is red,
#         120° is green,
#         240° is blue.
#     Usage: Hue allows for easy differentiation of colors (e.g., red, green, blue).
    
# apply smoothing
img_h = img_hsv[:,:,0]
k = 11 # want to choose smoothing kernel less than 10% of radius of corals? (50 px)
img_h = cv.GaussianBlur(img_h, ksize=(k,k),sigmaX=1 )

# threshold 
# reds will be around h < 30 and possibly h > 240 (unsure of the latter)
h_min = 0
h_max = 30

thresh_value, img_hb = cv.threshold(img_h, h_max, 255, cv.THRESH_BINARY_INV)

plt.figure()
plt.imshow(img_hb)
plt.title('binary from hue for coral')

# TODO lastly, just have to AND with the blobs obtained from saturation
    

plt.show()

# TODO lastly, we want to combine Hue, Saturation filtering of blobs