#!/usr/bin/env python3

# tool to find canny edge detection thresholds - via GUI

# detect edges in the image
# try to provide putative detections for corals (subsurface)
# sharp edges = in focus, and should be counted


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

# show image
plt.figure()
plt.imshow(img_rgb)
plt.title('original rgb image')

# smooth image
img_g = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
img_den = cv.fastNlMeansDenoising(img_g, 
                                  templateWindowSize=7,
                                  searchWindowSize=21,
                                  h=3)

plt.figure()
plt.subplot(121), plt.imshow(img_g)
plt.subplot(122), plt.imshow(img_den)

# plt.show()

################################
def callback(x):
    print(x)

TOOL = False
kernel_size = np.arange(1,200,2)
if TOOL:


    # img = cv.imread('your_image.png', 0) #read image as grayscale

    canny = cv.Canny(img_den, 85, 255) 
    img_blur = cv.GaussianBlur(img_den, ksize=(3,3),sigmaX=0)

    cv.namedWindow('image', cv.WINDOW_NORMAL) # make a window with name 'image'
    cv.createTrackbar('B', 'image', 0, 100, callback) #smoothing value
    cv.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
    cv.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

    

    while(1):
        numpy_horizontal_concat = np.concatenate((img_blur, canny), axis=1) # to display image side by side
        cv.imshow('image', numpy_horizontal_concat)
        k = cv.waitKey(1) & 0xFF
        if k == 27: #escape key
            break
        b = cv.getTrackbarPos('B', 'image')
        l = cv.getTrackbarPos('L', 'image')
        u = cv.getTrackbarPos('U', 'image')

        ksize = kernel_size[b]
        
        img_blur = cv.GaussianBlur(img_den, ksize=(ksize,ksize),sigmaX=0 )

        canny = cv.Canny(img_blur, l, u)

    cv.destroyAllWindows()
else:
    ksize = kernel_size[18]
    l = 6
    u = 7
    


# get final thresholds:


# want to choose smoothing kernel less than 10% of radius of corals? (50 px)
img_g = cv.GaussianBlur(img_den, ksize=(ksize,ksize),sigmaX=0)

# detect edges via Canny edge detection
# t1	first threshold for the hysteresis procedure. 
t1 =  l
print(f't1 = {t1}')
# t2	second threshold for the hysteresis procedure. 
t2 = u
print(f't2 = {t2}')
# aperture size, size fothe Sobel Operator
ap = 7
edges = cv.Canny(img_g, threshold1=t1, threshold2=t2)


# show edge image
plt.figure()
plt.imshow(edges, cmap='gray')
plt.title('edges')

# now, apply morphological operations
img_thresh = edges
km = 9
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(km, km))
img_thresh = cv.dilate(img_thresh, kernel, iterations = 1)
img_thresh = cv.morphologyEx(img_thresh, cv.MORPH_CLOSE, kernel)
img_thresh = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel)

plt.figure()
plt.imshow(img_thresh)
plt.title('img thresh morph')

# contour fill:
# img_fill = cv.bitwise_not(filtered_image)
img_fill = img_thresh
contour, hierarchy = cv.findContours(img_fill, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    cv.drawContours(img_fill, [cnt], 0, 255, -1)
# output = cv.bitwise_not(img_fill)

plt.figure()
plt.imshow(img_fill)
plt.title('image filled in from threshold')

# do connected components, and remove all blobs of too small an area/too big an area

numlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img_thresh, connectivity=8)
filtered_image = np.zeros_like(img_thresh)

min_area = 2000
max_area = 40000

min_circ = 0.3
max_circ = 1.0
label_list = []
circularity = []
perimeter = []
for i in range(1, numlabels):
    area = stats[i, cv.CC_STAT_AREA] # 4th column of stats
    
    # create mask for current component
    mask = (labels == i).astype(np.uint8) * 255 # binary mask of the label
    
    # calculate perimeter
    p = cv.arcLength(cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE,)[0][0], True)
    perimeter.append(p)
    # circularity
    if p > 0:
        c = (4 * np.pi * area) / (p ** 2)
        if c > 1.0: # sometimes by numerical calculations/pixel artifacts
            c = 1.0 # perfect circle
    else:
        c = 0
    circularity.append(c)
    
    if min_area <= area <= max_area and min_circ <= c <= max_circ:
        filtered_image[labels == i] = 255 # keep the component in the filtered image
        label_list.append(i)


for c in circularity:
    print(c)

plt.figure()
plt.imshow(filtered_image)
plt.title('filtered image for area and circularity')





# expand these surviving regions as areas that could be logically ANDed with other measures
kf = 101 # kernel size for expansion, half-that of an expected coral radius
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kf, kf))
img_edge_final = cv.dilate(filtered_image, kernel, iterations = 1)

plt.figure()
plt.imshow(img_edge_final)
plt.show()

import code
code.interact(local=dict(globals(), **locals()))

