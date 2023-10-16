#! /usr/bin/env python3

"""
coral spawn counting using blobs (copied from count_coral_spawn.py)
TODO: also calls yolov5 detector onto cslics surface embryogenesis
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
import torch

import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image

from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.read_manual_counts import read_manual_counts

class SubSurface_Detector:
    DEFAULT_IMG_DIR = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04/images_jpg"
    DEFAULT_SAVE_DIR = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04/combined_detections"
    DEFAULT_ROOT_DIR = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04"
    DEFAULT_DETECTION_FILE = 'subsurface_det.pkl'
    DEFAULT_OBJECT_NAMES_FILE = 'metadata/obj.names'
    DEFAULT_WINDOW_SIZE = 20
    MAX_IMG = 10000

    def __init__(self,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 root_dir: str = DEFAULT_ROOT_DIR,
                 detection_file: str = DEFAULT_DETECTION_FILE,
                 save_prelim_img: bool = False,
                 object_names_file: str = DEFAULT_OBJECT_NAMES_FILE,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 max_img: int = MAX_IMG):
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.root_dir = root_dir
        self.detection_file = detection_file
        self.object_names_file = object_names_file
        self.window_size = window_size
        self.max_img = max_img
        self.save_img_dir = os.path.join(save_dir, 'images')
        self.save_prelim = save_prelim_img
        self.capture_time_list = []
        self.classes = self.get_classes(root_dir)
        #help with detection
        self.count = 0
        self.blobby = []
        self.icont = []
        os.makedirs(self.save_img_dir, exist_ok=True)

    def get_classes(self, root_dir):
        """
        get the classes from a metadata/obj.names file
        classes = [class1, class2, class3 etc.]
        """
        #TODO: make a function of something else, used in both detectors
        with open(os.path.join(root_dir, 'metadata','obj.names'), 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def prep_img(self, img_name):
        # create coral image:
        coral_image = CoralImage(img_name=img_name, txt_name = 'placeholder.txt')
        self.capture_time_list.append(coral_image.metadata['capture_time'])
        
        # read in image
        im = Image(img_name)
        # grayscale
        im_mono = im.mono()
        # blur image
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
        if self.save_prelim:
            img_base_name = os.path.basename(img_name)[:-4]
            im_mono.write(os.path.join(self.save_img_dir, img_base_name + '_00_orig.jpg'))
            im_blur.write(os.path.join(self.save_img_dir, img_base_name + '_01_blur.jpg'))
            im_canny.write(os.path.join(self.save_img_dir, img_base_name + '_02_edge.jpg'))
            im_morph.write(os.path.join(self.save_img_dir, img_base_name + '_03_morph.jpg'))
        
        return im_morph

    def attempt_blobs(self, img_name, im_morph, im):
        try:
            img_base_name = os.path.basename(img_name)[:-4]
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
            image3.write(os.path.join(self.save_img_dir, img_base_name + '_blobs_contour.jpg'))

            if self.save_prelim:
                imblobs.write(os.path.join(self.save_img_dir, img_base_name + '_04_blob.jpg'))
                imblobs_area.write(os.path.join(self.save_img_dir, img_base_name + '_05_blob_filter.jpg'))
            print(f'{len(icont)} blobs passed threasholds')
            # TODO for now, just count these blobs over time
            return blobby, icont
        except:
            print(f'No blob detection for {img_name}')
            # append empty to keep indexes consistent
            return [], []

    def save_results_2_pkl(self, subsurface_det_path, blobs_list, blobs_count, image_index, capture_time):
        with open(subsurface_det_path, 'wb') as f:
            save_data ={'blobs_list': blobs_list,
                'blobs_count': blobs_count,
                'image_index': image_index,
                'capture_time': capture_time}
            pickle.dump(save_data, f)

    def blob_2_box(self, img_name, icont, blobby):
        img = cv.imread(img_name)
        contours = blobby._contours
        imgw, imgh = img.shape[1], img.shape[0]
        bb = []
        for i, c in enumerate(contours):
            if i in icont:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
                x1 = x/(imgw*2)
                y1 = y/(imgh*2)
                x2 = (x+w)/(imgw*2)
                y2 = (y+h)/(imgh*2)
                bb.append([x1, y1, x2, y2, 0.5, 3, 3])
        #debug visualise
        #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        #plt.show()
        return torch.tensor(bb)

    def save_text_predictions(self, predictions, imgname, txtsavedir):
        """
        save predictions/detections into text file
        [x1 y1 x2 y2 conf class_idx class_name]
        """
        txtsavename = os.path.basename(imgname)
        txtsavepath = os.path.join(txtsavedir, txtsavename[:-4] + '_det.txt')

        # predictions [ pix pix pix pix conf class ]
        with open(txtsavepath, 'w') as f:
            for p in predictions:
                x1, y1, x2, y2 = p[0:4].tolist()
                conf = p[4]
                class_idx = int(p[5])
                class_name = self.classes[class_idx]
                f.write(f'{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.4f} {class_idx:g} {class_name}\n')
        return True

    def detect(self, image):
        img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        img_name = img_list[self.count]
        blobby, icont = self.attempt_blobs(img_name, image, im = Image(img_name))
        predictions = self.blob_2_box(img_name, icont, blobby)
        self.blobby = blobby
        self.icont = icont
        self.count += 1
        return predictions
    
    def run(self, SUBSURFACE_LOAD = False):
        print("running blob subsurface detector")
        img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        txtsavedir = os.path.join(self.root_dir, 'blob', 'textfiles')
        os.makedirs(txtsavedir, exist_ok=True)

        blobs_count = []
        blobs_list = []
        image_index = []

        start_time = time.time()

        for i, img_name in enumerate(img_list):
            if i >= self.max_img:
                print(f'{i}: hit max img - stop here')
                #import code
                #code.interact(local=dict(globals(), **locals()))
                break
    
            print(f'{i}: img_name = {img_name}')  
            im_morph = self.prep_img(img_name) 
            print('image prepared')
            image_index.append(i)

            predictions = self.detect(im_morph)
            self.save_text_predictions(predictions, img_name, txtsavedir)
            blobs_list.append(self.blobby)
            blobs_count.append(self.icont)
        
        subsurface_det_path = os.path.join(self.save_dir, self.detection_file)
        self.save_results_2_pkl(subsurface_det_path, blobs_list, blobs_count, image_index, self.capture_time_list) 

        end_time = time.time()
        duration = end_time - start_time
        print('run time: {} sec'.format(duration))
        print('run time: {} min'.format(duration / 60.0))
        print('run time: {} hrs'.format(duration / 3600.0))

        capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in self.capture_time_list]
        decimal_days = convert_to_decimal_days(capture_time_dt)

        import code
        code.interact(local=dict(globals(), **locals()))



# TODO add to different file, used in this file and Plot_Detectors_Results
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
    
def main():
    img_dir = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04/images_jpg"
    root_dir = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04"
    MAX_IMG = 10000
    detection_file = 'subsurface_det_testing.pkl'
    object_names_file = 'metadata/obj.names'
    window_size = 20

    Coral_Detector = SubSurface_Detector(root_dir = root_dir, img_dir = img_dir, detection_file=detection_file, 
        object_names_file = object_names_file, window_size=window_size, max_img = MAX_IMG)
    Coral_Detector.run()
    # import code
    # code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    main()

