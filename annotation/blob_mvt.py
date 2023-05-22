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

max_img = 1
for i, img_name in enumerate(img_list):
    if i >= max_img:
        print('hit max img')
        break
    
    print(f'{i}: img_name = {img_name}')    
    img_base_name = os.path.basename(img_name)[:-4]
    save_orig_img_name = os.path.join(save_dir, img_base_name + '_00_orig.jpg')
    
    # read in image
    im = Image(img_name)
    
    # grayscale
    im_mono = im.mono()
    
    im_mono.write(save_orig_img_name)
    
    # call Blobs class
    print('call blobs')
    b = mvt.Blob(im_mono)
    
    # show blobs
    imblobs = b.drawBlobs(im_mono, None, None, None, contourthickness=-1)
    imblobs.disp()
    
    b.printBlobs()
    
    import code
    code.interact(local=dict(globals(), **locals()))