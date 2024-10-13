#!/usr/bin/env python3

# sort through images and filter by colour thresholding
# possibly in the HSV space?

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
    
plt.figure()
plt.imshow(img_hsv[:,:,1])
plt.title('S Saturation')

# Saturation:
#     Definition: Measures the intensity or purity of the color. A high saturation means the color is vivid and intense, while low saturation means the color is more muted or washed out (closer to gray).
#     Range: Usually ranges from 0% (gray, no color) to 100% (pure color).
#     Usage: Saturation helps to control the vividness of colors in an image.
    
plt.figure()
plt.imshow(img_hsv[:,:,2])
plt.title('V Value')

# Value (also known as Brightness):
#     Definition: Represents the brightness of the color. It indicates how light or dark a color is.
#     Range: Ranges from 0% (black) to 100% (full brightness of the color).
#     Usage: Value is useful for adjusting the overall brightness of the image without altering the color itself.
    


# so for the current aant bright-back images, corals stand out wrt Saturation,
# so we focus on grabbing edges from the Saturation space

# apply smoothing
img_s = img_hsv[:,:,1]
k = 5 # want to choose smoothing kernel less than 10% of radius of corals? (50 px)
img_s = cv.GaussianBlur(img_s, ksize=(k,k),sigmaX=0 )


# show histogram of the image
plt.figure()
plt.hist(img_s.ravel(), 256)
# plt.yscale('log')
plt.title('image saturation histogram')


# apply thresholding
# blockSize	Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
# C	Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
# thresh = cv.adaptiveThreshold(img_s, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10)
thresh_value, img_sb = cv.threshold(img_s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)



plt.figure()
plt.imshow(img_sb)
plt.title('binary from saturation')


numlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img_sb, connectivity=8)

# show labels image
plt.figure()
plt.imshow(labels)
plt.title('labelled connected components')

# for visualisation, create a new image of the filtered components
filtered_image = np.zeros_like(img_sb)

min_area = 1000
max_area = 20000

min_circ = 0.5
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


plt.show()

import code
code.interact(local=dict(globals(), **locals()))
