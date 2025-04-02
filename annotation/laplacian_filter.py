#!/usr/bin/env python3

import glob
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# generate putative annotations through principled mechanisms in computer vision

# ideas:

n_circ = 4
height, width = 200, 200
overall_width = width * n_circ
channels = 3 # RGB

# for i in range(n_circ):
#     print(f'{i}: {int(width/2 + width*i)}')
#     img = cv.circle(img, (int(width/2 + width*i), int(height/2)), radius=30, color=(255,0,0), thickness=-1)
    
# create 4 images of 4 circles, apply different smoothings to each, then combine into one wide image
kernel = [9, 15, 21, 27] # vary kernel size for blurring
img_list = []
for i in range(n_circ):
    # create empty image
    img = np.zeros((height, width, channels), dtype=np.uint8)

    # draw circle
    img = cv.circle(img, (int(width/2), int(height/2)), radius=30, color=(255,0,0), thickness=-1)
    
    # blur
    img = cv.GaussianBlur(img, (kernel[i], kernel[i]), sigmaX=1)
    
    # save for stacking horizontally
    img_list.append(img)

# stack images horizontally:
img_stack = img_list[0]
for img in img_list:
    img_stack = np.hstack((img_stack, img))
    

# do Laplacian filter for edges

# img_pattern = '*.jpg'
# img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/images'
# img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))

# create an image that has a circle and then an image pattern with gradually smoother sides


# read in image
# i = 0
# img = cv.imread(img_list[i])
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# reduce small local noise via Gaussian filter
# img = cv.GaussianBlur(img, (5,5), 0)

# convert to grayscale
img_gray = cv.cvtColor(img_stack, cv.COLOR_BGR2GRAY)

# apply laplacian
ddepth = cv.CV_16S
kernel_size = 5
lapl = cv.Laplacian(img_gray, ddepth, ksize=kernel_size)
abs_lapl = cv.convertScaleAbs(lapl)


# show image
plt.figure()
plt.imshow(img_stack)
plt.title('progressively blurred circles')

plt.figure()
plt.imshow(abs_lapl)
plt.title('absolute laplacian response')

# from the plots, the conclusion is you get peak response when you use kernels/filter 
# plt.show()


# now, applied to real images




