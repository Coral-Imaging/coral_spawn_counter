#! /usr/bin/env python3

""" 
blob class for blob annotation and elimination
from Machine Vision Toolbox
"""

import os
import cv2 as cv
import numpy as np
import glob
import random as rng
import matplotlib.pyplot as plt

import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image

img_dir = '/home/dorian/Data/cslics_2022_datasets/subsurface_data/20221113_amtenuis_cslics04/images_subset'
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
save_dir = 'output2'
os.makedirs(save_dir, exist_ok=True)

max_img = 10
for i, img_name in enumerate(img_list):
    if i >= max_img:
        print('hit max img')
        break
    
    print(f'{i}: img_name = {img_name}')    
    img_base_name = os.path.basename(img_name)[:-4]
    
    # read in image
    im = Image(img_name)
    
    # grayscale
    im_mono = im.mono()
    
    # step-by-step save the image
    im_mono.write(os.path.join(save_dir, img_base_name + '_00_orig.jpg'))
    
    # blur image
    # im_mono.smooth(sigma=1,hw=31) # this is really really slow!
    # HACK make a new object using GaussianBlur
    ksize = 61
    im_blur = Image(cv.GaussianBlur(im_mono.image, (ksize, ksize), 0))
    im_blur.write(os.path.join(save_dir, img_base_name + '_01_blur.jpg'))
    
    # edge detection
    canny = cv.Canny(im_blur.image, 3, 5, L2gradient=True)
    im_canny = Image(canny)
    # im_canny = Image(im_blur.canny(th0=5, th1=15))
    im_canny.write(os.path.join(save_dir, img_base_name + '_02_edge.jpg'))
    
    # TODO MVT GUI related to image edge thresholds?
    
    # morph!
    k = 21
    kernel = np.ones((k,k), np.uint8)
    im_morph = Image(im_canny.dilate(kernel))
    im_morph = im_morph.close(kernel)
    im_morph = im_morph.open(kernel)
    im_morph.write(os.path.join(save_dir, img_base_name + '_03_morph.jpg'))
    
    # call Blobs class
    print('call blobs')
    blobby = mvt.Blob(im_morph)
    
    # show blobs
    imblobs = blobby.drawBlobs(im_morph, None, None, None, contourthickness=-1)
    imblobs.write(os.path.join(save_dir, img_base_name + '_04_blob.jpg'))
    # imblobs.disp()
    
    # blobby.printBlobs()
    
    # TODO reject too-small and too weird blobs
    area_min = 10000
    area_max = 40000
    circ_min = 0.65
    circ_max = 1.0
    
    b0 = [b for b in blobby if ((b.area < area_max and b.area > area_min) and (b.circularity > circ_min and b.circularity < circ_max))]
    idx_blobs = np.arange(0, len(blobby))
    b0_area = [b.area for b in b0]
    b0_circ = [b.circularity for b in b0]
    b0_cent = [b.centroid for b in b0]
    # can do more...
    # icont = [i for i in idx_blobs if (blobby[i].area in b0_area and blobby[i].circularity in b0_circ)]
    # icont = [i for i, b in enumerate(blobby) if b in b0] # this should work, but doesn't - not yet implemented...
    icont = [i for i, b in enumerate(blobby) if (blobby[i].centroid in b0_cent and blobby[i].area in b0_area and blobby[i].circularity in b0_circ)] # this should work, but doesn't - not yet implemented...
    
    # b0.printBlobs()
    imblobs_area = blobby.drawBlobs(im_morph, None, icont, None, contourthickness=-1)
    imblobs_area.write(os.path.join(save_dir, img_base_name + '_05_blob_filter.jpg'))


# TODO for now, just count these blobs over time
# TODO write to xml file for uploading blob annotations    
import code
code.interact(local=dict(globals(), **locals()))