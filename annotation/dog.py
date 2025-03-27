# script to test out effect of Difference of Gaussians on CSLICS images to reduce noise 
# there are currently certain frequencies in the CSLICS images which need to be removed for more reliable blob counting
# band-pass filters can help with this process

# a standard method of band-pass filtering is subtracting an image blurred with a Gaussian kernel from a less-blurred image 
# often found in SIFT/SURF detectors, etc

# https://scikit-image.org/docs/stable/auto_examples/filters/plot_dog.html

import os
import glob as glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import gravel
from skimage.filters import difference_of_gaussians, window
from scipy.fft import fftn, fftshift


# image directory/file
root_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics03'
img_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics03/images_subsurface'
save_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics03/test_output'



# just choose the first image
img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
img_name = img_files[0]
im = cv.imread(img_name)
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

# show original image:
plt.figure()
plt.imshow(im)
plt.title('original colour RGB image')



# resize image for consistency:
desired_width = 1280 # pixels
aspect_ratio = desired_width / im.shape[1]
height = int(im.shape[0] * aspect_ratio)
im_r = cv.resize(im, (desired_width, height), interpolation=cv.INTER_AREA)

# how to show the low frequencies via thresholding with the Laplacian operator:
kblur = 11
kmorph = 1
kfocus = 5
kfocusblur = 5

# cvt for mono:
im_mono = cv.cvtColor(im_r, cv.COLOR_RGB2GRAY)
plt.figure()
plt.imshow(im_mono)
plt.title('mono image')


# blurred image
im_blur = cv.GaussianBlur(im_mono, (kblur, kblur), 0)
# try applying non-local means noising to smooth out flat sections, but preserve edges
im_blur = cv.fastNlMeansDenoising(im_blur, None, 10, 7 ,21)

plt.figure()
plt.imshow(im_blur)
plt.title('blurred image')


focus_measure = cv.Laplacian(im_blur, cv.CV_16S, ksize=kfocus)
im_focus = cv.convertScaleAbs(focus_measure)

im_focus = cv.GaussianBlur(im_focus, (kfocusblur, kfocusblur), 0)


plt.figure()
plt.imshow(im_focus)
plt.title('focused image, which shows the low frequency edges that need removing')


# NOTE: https://stackoverflow.com/questions/14191967/opencv-efficient-difference-of-gaussian


im_focus1 = im_focus
kdiff = 12
im_focus2 = cv.GaussianBlur(im_focus, (kfocusblur+kdiff, kfocusblur+kdiff), 0)

dog = im_focus1 - im_focus2

plt.figure()
plt.imshow(dog)
plt.title('Diff of Gaussians')


wimage = im_focus * window('hann', im_focus.shape)
filtered_image = difference_of_gaussians(im_focus, 1, 4)

plt.figure()
plt.imshow(filtered_image)
plt.title('Diff of Gaussians - 1-12')




# image = gravel()
# wimage = image * window('hann', image.shape)
# filtered_image = difference_of_gaussians(image, 1, 12)
# filtered_wimage = filtered_image * window('hann', image.shape)
# im_f_mag = fftshift(np.abs(fftn(wimage)))
# fim_f_mag = fftshift(np.abs(fftn(filtered_wimage)))
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
# ax[0, 0].imshow(image, cmap='gray')
# ax[0, 0].set_title('Original Image')
# ax[0, 1].imshow(np.log(im_f_mag), cmap='magma')
# ax[0, 1].set_title('Original FFT Magnitude (log)')
# ax[1, 0].imshow(filtered_image, cmap='gray')
# ax[1, 0].set_title('Filtered Image')
# ax[1, 1].imshow(np.log(fim_f_mag), cmap='magma')
# ax[1, 1].set_title('Filtered FFT Magnitude (log)')
# plt.show()

# just find SURF features?

sift = cv.SIFT_create(contrastThreshold=0.04, edgeThreshold=10) # TODO can set edgeThreshold here, etc
# https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html
kp = sift.detect(im_mono, None)

im_sift = cv.drawKeypoints(im_mono, kp,0, (255,0,0),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

import code
code.interact(local=dict(globals(), **locals()))

plt.figure()
plt.imshow(im_sift)
plt.title('sift features shown (red)')

# surf = cv.SIFT(400)
# kp, des = surf.detectAndCompute(im_focus, None)
# len(kp)

# im_ftr = cv.drawKeypoints(im_focus, kp, (255,0,0), 4)
# plt.figure()
# plt.imshow(im_ftr)
# plt.title('surf features shown')





plt.show()
print('done dog.py')
# end code
