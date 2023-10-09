#! /usr/bin/env python3

"""
coral spawn counting using blobs with manual counts overlayed (copied from blob_mvt.py)
TODO: also calls yolov5 detector onto cslics surface embryogenesis
TODO: shows the two plots on same graph
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
import time

import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image

from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.read_manual_counts import read_manual_counts

class Surface_Detector:
    DEFAULT_IMG_DIR = "/mnt/c/20221113_amtenuis_cslics04/images_jpg"
    DEFAULT_SAVE_DIR = "/mnt/c/20221113_amtenuis_cslics04/combined_detections"
    DEFAULT_ROOT_DIR = "/mnt/c/20221113_amtenuis_cslics04"
    DEFAULT_DETECTION_FILE = 'subsurface_det.pkl'
    DEFAULT_OBJECT_NAMES_FILE = 'metadata/obj.names'
    DEFAULT_WINDOW_SIZE = 20
    MAX_IMG = 10000

    def __init__(self,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 root_dir: str = DEFAULT_ROOT_DIR,
                 detection_file: str = DEFAULT_DETECTION_FILE,
                 onject_names_file: str = DEFAULT_OBJECT_NAMES_FILE,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 max_img: int = MAX_IMG):
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.root_dir = root_dir
        self.detection_file = detection_file
        self.object_names_file = object_names_file
        self.window_size = window_size
        self.max_img = max_img

    def prep_img(self, img_name, SAVE_PRELIM_IMAGES):
        # create coral image:
        # TODO - loop the Hough transform method into here, also has metadata -> capture times
        coral_image = CoralImage(img_name=img_name, txt_name = 'placeholder.txt')
        capture_time.append(coral_image.metadata['capture_time'])
        
        # read in image
        im = Image(img_name)
        # grayscale
        im_mono = im.mono()
        # blur image
        # HACK make a new object using GaussianBlur
        ksize = 61
        im_blur = Image(cv.GaussianBlur(im_mono.image, (ksize, ksize), 0))
        # edge detection
        canny = cv.Canny(im_blur.image, 3, 5, L2gradient=True)
        im_canny = Image(canny)
        # TODO MVT GUI related to image edge thresholds?
        # morph
        k = 11
        kernel = np.ones((k,k), np.uint8)
        im_morph = Image(im_canny.dilate(kernel))
        im_morph = im_morph.close(kernel)
        kernel = 11
        im_morph = im_morph.open(kernel)

        # step-by-step save the image
        if SAVE_PRELIM_IMAGES:
            img_base_name = os.path.basename(img_name)[:-4]
            im_mono.write(os.path.join(save_img_dir, img_base_name + '_00_orig.jpg'))
            im_blur.write(os.path.join(save_img_dir, img_base_name + '_01_blur.jpg'))
            im_canny.write(os.path.join(save_img_dir, img_base_name + '_02_edge.jpg'))
            im_morph.write(os.path.join(save_img_dir, img_base_name + '_03_morph.jpg'))
        
        return im_morph

    def attempt_blobs(self, im_morph, SAVE_PRELIM_IMAGES, im):
        try:
            blobby = mvt.Blob(im_morph)
            print(f'{len(blobby)} blobs initially found')
            # show blobs
            imblobs = blobby.drawBlobs(im_morph, None, None, None, contourthickness=-1)

            #  reject too-small and too weird blobs
            area_min = 5000
            area_max = 40000
            circ_min = 0.5
            circ_max = 1.0
            b0 = [b for b in blobby if ((b.area < area_max and b.area > area_min) and (b.circularity > circ_min and b.circularity < circ_max))]
            idx_blobs = np.arange(0, len(blobby))
            b0_area = [b.area for b in b0]
            b0_circ = [b.circularity for b in b0]
            b0_cent = [b.centroid for b in b0]
            # can do more... TODO ??
            # icont = [i for i in idx_blobs if (blobby[i].area in b0_area and blobby[i].circularity in b0_circ)]
            # icont = [i for i, b in enumerate(blobby) if b in b0] # this should work, but doesn't - not yet implemented...
            icont = [i for i, b in enumerate(blobby) if (blobby[i].centroid in b0_cent and 
                                                            blobby[i].area in b0_area and 
                                                            blobby[i].circularity in b0_circ)] # this should work, but doesn't - not yet implemented...
            # b0.printBlobs()
            imblobs_area = blobby.drawBlobs(im_morph, None, icont, None, contourthickness=-1)

            self.save_side_blobs(im, img_base_name = os.path.basename(img_name)[:-4])
            image1 = im.image

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

            if SAVE_PRELIM_IMAGES:
                imblobs.write(os.path.join(save_img_dir, img_base_name + '_04_blob.jpg'))
                imblobs_area.write(os.path.join(save_img_dir, img_base_name + '_05_blob_filter.jpg'))

            # TODO for now, just count these blobs over time
            return blobby, icont
        except:
            print(f'No blob detection for {img_name}')
            # append empty to keep indexes consistent
            return [], []
    
    def save_side_blobs(self, im, img_base_name):
        # save side-by-side image/collage for blobs.
        # NOTE most of this code is commented out, as orginally given to me
        image1 = im.image
        image2 = imblobs_area.image
        image2_mask = imblobs_area.image > 0
        # save side-by-side image/collage
        # concatenated_image = Image(cv.hconcat([image1, image2]))
        # concatenated_image.write(os.path.join(self.save_dir, img_base_name + '_concate_blobs.jpg'))
            
        # image overlay (probably easier to view)
        # opacity = 0.35
        # image_overlay = Image(cv.addWeighted(image1, 
        #                                     1-opacity, 
        #                                     image2,
        #                                     opacity,
        #                                     0))
        # image_overlay.write(os.path.join(self.save_dir, img_base_name + '_blobs_overlay.jpg'))

    def run(self, SUBSURFACE_LOAD = False, SAVE_PRELIM_IMAGES = True):
        img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        subsurface_det_path = os.path.join(self.save_dir, self.detection_file)

        blobs_count = []
        blobs_list = []
        image_index = []
        capture_time = []

        start_time = time.time()

        if(SUBSURFACE_LOAD):
            #load pickel file
            with open(subsurface_det_path, 'rb') as f:
                save_data = pickle.load(f)

            blobs_list = save_data['blobs_list']
            blobs_count = save_data['blobs_count']
            image_index = save_data['image_index']
            capture_time = save_data['capture_time']
        else:
            for i, img_name in enumerate(img_list):
                if i >= MAX_IMG:
                    print(f'{i}: hit max img - stop here')
                    break
        
                print(f'{i}: img_name = {img_name}')  
                im_morph = prep_img(img_name, SAVE_PRELIM_IMAGES) 
                print('image prepared')
                image_index.append(i)

                blobby, icont = attempt_blobs(im_morph, SAVE_PRELIM_IMAGES, im = Image(img_name))
                blobs_list.append(blobby)
                blobs_count.append(icont)


            with open(subsurface_det_path, 'wb') as f:
                save_data ={'blobs_list': blobs_list,
                    'blobs_count': blobs_count,
                    'image_index': image_index,
                    'capture_time': capture_time}
                pickle.dump(save_data, f)

        end_time = time.time()
        duration = end_time - start_time
        print('run time: {} sec'.format(duration))
        print('run time: {} min'.format(duration / 60.0))
        print('run time: {} hrs'.format(duration / 3600.0))

        capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time]
        decimal_days = convert_to_decimal_days(capture_time_dt)

        return blobs_count, blobs_list, image_index, capture_time


def convert_to_decimal_days(dates_list, time_zero=None):
    if time_zero is None:
        time_zero = dates_list[0]  # Time zero is the first element date in the list
    else:
        time_zero = time_zero
        
    decimal_days_list = []

    for date in dates_list:
        time_difference = date - time_zero
        decimal_days = time_difference.total_seconds() / (60 * 60 * 24)
        decimal_days_list.append(decimal_days)

    return decimal_days_list
    

########################################################

# read manual counts file

dt, mc, tw = read_manual_counts(manual_counts_file)
zero_time = capture_time_dt[0]
manual_decimal_days = convert_to_decimal_days(dt, zero_time)

########################################################



# convert blobs_count into actual count, not interior list of indices
image_count = [len(blobs_index) for blobs_index in blobs_count]
image_count = np.array(image_count)

# counts per image to density counts:
# need volume:
# calculated by hand to be approximately 0.1 mL 
# 2.23 cm x 1.675 cm x 0.267 cm
image_volume = 0.10 # mL

# density count:
# density_count = [c / image_volume for c in count]
density_count = image_count * image_volume

# overall tank count: 
tank_volume = 500 * 1000 # 500 L * 1000 mL/L
tank_count = density_count * tank_volume

# show averages to apply rolling means
plotdatadict = {
    'index': image_index,
    'capture_time_days': decimal_days,
    'image_count': image_count,
    'density_count': density_count,
    'tank_count': tank_count
}
df = pd.DataFrame(plotdatadict)


image_count_mean = df['image_count'].rolling(window_size).mean()
image_count_std = df['image_count'].rolling(window_size).std()

density_count_mean = df['density_count'].rolling(window_size).mean()
density_count_std = df['density_count'].rolling(window_size).std()

tank_count_mean = df['tank_count'].rolling(window_size).mean()
tank_count_std = df['tank_count'].rolling(window_size).std()
n = 1 # how many std deviations to show
mpercent = 0.1 # range for manual counts

# TODO showcase counts over time?
sns.set_theme(style='whitegrid')


fig1, ax1 = plt.subplots()
plt.plot(decimal_days, image_count_mean, label='count')
plt.fill_between(decimal_days, 
                 image_count_mean - n*image_count_std,
                 image_count_mean + n*image_count_std,
                 alpha=0.2)
plt.xlabel('days since stocking')
plt.ylabel('image count')
plt.title('Subsurface Counts vs Image Index - Prelim')
plt.savefig(os.path.join(save_plot_dir, 'subsubfacecounts.png'))


fig2, ax2 = plt.subplots()
plt.plot(decimal_days, density_count_mean, label='density [count/mL]')
plt.fill_between(decimal_days, 
                 density_count_mean - n*density_count_std,
                 density_count_mean + n*density_count_std,
                 alpha=0.2)
plt.xlabel('days since stocking')
plt.ylabel('density')
plt.title('Subsurface Density count/mL vs Image Index - Prelim')
plt.savefig(os.path.join(save_plot_dir, 'subsubface_densitycounts.png'))




# wholistic tank count
fig3, ax3 = plt.subplots()

# surface counts
plt.plot(surface_decimal_days, counttank_total, label='surface count', color='orange')
plt.fill_between(surface_decimal_days,
                 counttank_total - n * nimage_to_tank_surface * count_total_std,
                 counttank_total + n * nimage_to_tank_surface * count_total_std,
                 alpha=0.2,
                 color='orange')

# subsurface counts
plt.plot(decimal_days, tank_count_mean, label='subsurface count')
plt.fill_between(decimal_days, 
                 tank_count_mean - n*tank_count_std,
                 tank_count_mean + n*tank_count_std,
                 alpha=0.2)

# manual counts
plt.plot(manual_decimal_days, mc, label='manual count', color='green', marker='o')
plt.fill_between(manual_decimal_days, 
                 mc - mpercent * mc,
                 mc + mpercent * mc,
                 alpha=0.1,
                 color='green')
plt.xlabel('days since stocking')
plt.ylabel('tank count')
plt.title('Overall tank count vs Time')
plt.legend()
plt.savefig(os.path.join(save_plot_dir, 'subsubface_tankcounts.png'))



print('done blob_mvt.py')
# TODO need to ascertain the validity of the blobs - not all edges are ideal - chat with Andrew
# TODO save blob counts to txt file for reading/later usage?
# TODO write to xml file for uploading blob annotations    
import code
code.interact(local=dict(globals(), **locals()))