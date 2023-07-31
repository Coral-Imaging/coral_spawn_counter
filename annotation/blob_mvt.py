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
import seaborn as sns
import matplotlib as mpl
import seaborn.objects as so
import pickle
import pandas as pd
from datetime import datetime

import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image

from coral_spawn_counter.CoralImage import CoralImage


# subset of images
img_dir = '/home/dorian/Data/cslics_2022_datasets/subsurface_data/20221113_amtenuis_cslics04/images_subset'
save_dir = '/home/dorian/Data/cslics_2022_datasets/subsurface_data/20221113_amtenuis_cslics04/images_subset_detections'

# full image set
# img_dir = '/home/dorian/Data/cslics_2022_datasets/subsurface_data/20221113_amtenuis_cslics04/images_jpg'
# save_dir = '/home/dorian/Data/cslics_2022_datasets/subsurface_data/20221113_amtenuis_cslics04/subsurface_detections'

img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

save_plot_dir = os.path.join(save_dir, 'plots')
save_img_dir = os.path.join(save_dir, 'images')

subsurface_det_file = 'subsurface_det.pkl'
subsurface_det_path = os.path.join(save_dir, subsurface_det_file)
    
# save_dir = 'output2'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_plot_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

blobs_count = []
blobs_list = []
image_index = []
capture_time = []

LOAD = False
SAVE_PRELIM_IMAGES = False
MAX_IMG = 10

def convert_to_decimal_days(dates_list):
    if not dates_list:
        return []  # Return an empty list if the input list is empty

    time_zero = dates_list[0]  # Time zero is the first element date in the list
    decimal_days_list = []

    for date in dates_list:
        time_difference = date - time_zero
        decimal_days = time_difference.total_seconds() / (60 * 60 * 24)
        decimal_days_list.append(decimal_days)

    return decimal_days_list



if LOAD:
    # load pickle file for blobs_list and blobs_count
    
    with open(subsurface_det_path, 'rb') as f:
        save_data = pickle.load(f)
        
    blobs_list = save_data['blobs_list']
    blobs_count = save_data['blobs_count']
    image_index = save_data['image_index']
    capture_time = save_data['capture_time']
    
else:
    # do the thing
    
    for i, img_name in enumerate(img_list):
        if i >= MAX_IMG:
            print(f'{i}: hit max img - stop here')
            break
        
        print(f'{i}: img_name = {img_name}')    
        img_base_name = os.path.basename(img_name)[:-4]
        
        # create coral image:
        # TODO - loop the Hough transform method into here,
        # also has metadata -> capture times
        coral_image = CoralImage(img_name=img_name, txt_name = 'placeholder.txt')
        capture_time.append(coral_image.metadata['capture_time'])
        
        # read in image
        im = Image(img_name)
        
        # grayscale
        im_mono = im.mono()
        
        # step-by-step save the image
        if SAVE_PRELIM_IMAGES:
            im_mono.write(os.path.join(save_img_dir, img_base_name + '_00_orig.jpg'))
        
        # TODO try this, see if it improves detections?
        # try histogram normalization (same as equalization)
        # spreads out the cumulative distribution of pixels in a linear fashion
        # brings out the shadows
        # im_mono = im_mono.normhist()
        
        # if SAVE_PRELIM_IMAGES:
        #     im_mono.write(os.path.join(save_img_dir, img_base_name + '_00_normhist.jpg'))
        
        
        # blur image
        # im_mono.smooth(sigma=1,hw=31) # this is really really slow!
        # HACK make a new object using GaussianBlur
        # ksize = 61
        ksize = 61
        im_blur = Image(cv.GaussianBlur(im_mono.image, (ksize, ksize), 0))
        if SAVE_PRELIM_IMAGES:
            im_blur.write(os.path.join(save_img_dir, img_base_name + '_01_blur.jpg'))
        
        # edge detection
        canny = cv.Canny(im_blur.image, 3, 5, L2gradient=True)
        im_canny = Image(canny)
        # im_canny = Image(im_blur.canny(th0=5, th1=15))
        if SAVE_PRELIM_IMAGES:
            im_canny.write(os.path.join(save_img_dir, img_base_name + '_02_edge.jpg'))
        
        # TODO MVT GUI related to image edge thresholds?
        
        # morph!
        k = 11
        kernel = np.ones((k,k), np.uint8)
        im_morph = Image(im_canny.dilate(kernel))
        im_morph = im_morph.close(kernel)
        kernel = 11
        im_morph = im_morph.open(kernel)
        if SAVE_PRELIM_IMAGES:
            im_morph.write(os.path.join(save_img_dir, img_base_name + '_03_morph.jpg'))
        
        image_index.append(i)
        # call Blobs class
        print('call blobs')
        # try:
        blobby = mvt.Blob(im_morph)
        

        # show blobs
        imblobs = blobby.drawBlobs(im_morph, None, None, None, contourthickness=-1)
        if SAVE_PRELIM_IMAGES:
            imblobs.write(os.path.join(save_img_dir, img_base_name + '_04_blob.jpg'))
        # imblobs.disp()
        
        # blobby.printBlobs()
        
        # TODO reject too-small and too weird blobs
        area_min = 5000
        area_max = 40000
        circ_min = 0.5
        circ_max = 1.0
        
        b0 = [b for b in blobby if ((b.area < area_max and b.area > area_min) and (b.circularity > circ_min and b.circularity < circ_max))]
        idx_blobs = np.arange(0, len(blobby))
        b0_area = [b.area for b in b0]
        b0_circ = [b.circularity for b in b0]
        b0_cent = [b.centroid for b in b0]
        # can do more...
        # icont = [i for i in idx_blobs if (blobby[i].area in b0_area and blobby[i].circularity in b0_circ)]
        # icont = [i for i, b in enumerate(blobby) if b in b0] # this should work, but doesn't - not yet implemented...
        icont = [i for i, b in enumerate(blobby) if (blobby[i].centroid in b0_cent and 
                                                        blobby[i].area in b0_area and 
                                                        blobby[i].circularity in b0_circ)] # this should work, but doesn't - not yet implemented...
        
        # b0.printBlobs()
        imblobs_area = blobby.drawBlobs(im_morph, None, icont, None, contourthickness=-1)
        if SAVE_PRELIM_IMAGES:
            imblobs_area.write(os.path.join(save_img_dir, img_base_name + '_05_blob_filter.jpg'))

        
        image1 = im.image
        image2 = imblobs_area.image
        image2_mask = imblobs_area.image > 0
        # save side-by-side image/collage
        # concatenated_image = Image(cv.hconcat([image1, image2]))
        # concatenated_image.write(os.path.join(save_dir, img_base_name + '_concate_blobs.jpg'))
        
        # image overlay (probably easier to view)
        # opacity = 0.35
        # image_overlay = Image(cv.addWeighted(image1, 
        #                                     1-opacity, 
        #                                     image2,
        #                                     opacity,
        #                                     0))
        # image_overlay.write(os.path.join(save_dir, img_base_name + '_blobs_overlay.jpg'))
        
        # just plot the contours of the blobs based on imblobs_area and icont:
        image_contours = image1
        contour_colour = [0, 0, 255] # red in BGR
        contour_thickness = 10
        for i in icont:
            cv.drawContours(image_contours,
                            blobby._contours,
                            i,
                            contour_colour,
                            thickness=contour_thickness,
                            lineType=cv.LINE_8)
        image3 = Image(image_contours)
        image3.write(os.path.join(save_img_dir, img_base_name + '_blobs_contour.jpg'))

        # TODO for now, just count these blobs over time
        # output into csv file or into yolo txt file format?
        blobs_list.append(blobby)
        blobs_count.append(icont)
        
        
        # except:
        #     print(f'No blob detection for {img_name}')
            
        #     # append empty to keep indexes consistent
        #     blobs_list.append([])
        #     blobs_count.append([])
    # save results as pkl file just in case the latter crashes
    # name of the most recent spawn file:
    
    
    with open(subsurface_det_path, 'wb') as f:
        save_data ={'blobs_list': blobs_list,
                    'blobs_count': blobs_count,
                    'image_index': image_index,
                    'capture_time': capture_time}
        pickle.dump(save_data, f)
    




# capture time convert from strings to decimal days
# parse capture_time into datetime objects so we can sort them
capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time]
decimal_days = convert_to_decimal_days(capture_time_dt)


# convert blobs_count into actual count, not interior list of indices
count = [len(blobs_index) for blobs_index in blobs_count]
count = np.array(count)

# counts per image to density counts:
# need volume:
# calculated by hand to be approximately 0.1 mL 
# 2.23 cm x 1.675 cm x 0.267 cm
image_volume = 0.10 # mL

# density count:
# density_count = [c / image_volume for c in count]
density_count = count * image_volume

# overall tank count: 
tank_volume = 500 * 1000 # 500 L * 1000 mL/L
tank_count = density_count * tank_volume

# show averages to apply rolling means
plotdatadict = {
    'index': image_index,
    'image_count': count,
    'density_count': density_count,
    'tank_count': tank_count
}
df = pd.DataFrame(plotdatadict)

window_size = 20
image_count_mean = df['image_count'].rolling(window_size).mean()
image_count_std = df['image_count'].rolling(window_size).std()

density_count_mean = df['density_count'].rolling(window_size).mean()
density_count_std = df['density_count'].rolling(window_size).std()

tank_count_mean = df['tank_count'].rolling(window_size).mean()
tank_count_std = df['tank_count'].rolling(window_size).std()
n = 1 # how many std deviations to show

# TODO showcase counts over time?
sns.set_theme(style='whitegrid')


fig1, ax1 = plt.subplots()
plt.plot(image_index, image_count_mean, label='count')
plt.fill_between(image_index, 
                 image_count_mean - n*image_count_std,
                 image_count_mean + n*image_count_std,
                 alpha=0.2)
plt.xlabel('index')
plt.ylabel('image count')
plt.title('Subsurface Counts vs Image Index - Prelim')
plt.savefig(os.path.join(save_plot_dir, 'subsubfacecounts.png'))


fig2, ax2 = plt.subplots()
plt.plot(image_index, density_count_mean, label='density [count/mL]')
plt.fill_between(image_index, 
                 density_count_mean - n*density_count_std,
                 density_count_mean + n*density_count_std,
                 alpha=0.2)
plt.xlabel('index')
plt.ylabel('density')
plt.title('Subsurface Density count/mL vs Image Index - Prelim')
plt.savefig(os.path.join(save_plot_dir, 'subsubface_densitycounts.png'))


fig3, ax3 = plt.subplots()
plt.plot(image_index, tank_count_mean, label='tank count [count]')
plt.fill_between(image_index, 
                 tank_count_mean - n*tank_count_std,
                 tank_count_mean + n*tank_count_std,
                 alpha=0.2)
plt.xlabel('index')
plt.ylabel('tank count')
plt.title('Overall tank count vs Image Index - Prelim')
plt.savefig(os.path.join(save_plot_dir, 'subsubface_tankcounts.png'))



print('done blob_mvt.py')
# TODO need to ascertain the validity of the blobs - not all edges are ideal - chat with Andrew
# TODO save blob counts to txt file for reading/later usage?
# TODO write to xml file for uploading blob annotations    
import code
code.interact(local=dict(globals(), **locals()))