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
from detector.Detector_helper import get_classes, save_text_predictions, convert_to_decimal_days

class SubSurface_Detector:
    DEFAULT_IMG_DIR = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04/images_jpg"
    DEFAULT_SAVE_DIR = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04/combined_detections"
    DEFAULT_ROOT_DIR = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04"
    DEFAULT_SAVE_IMG_DIR = os.path.join(DEFAULT_SAVE_DIR, 'images')
    DEFAULT_DETECTION_FILE = 'subsurface_det.pkl'
    DEFAULT_OBJECT_NAMES_FILE = 'metadata/obj.names'
    DEFAULT_WINDOW_SIZE = 20
    MAX_IMG = 10000

    def __init__(self,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 root_dir: str = DEFAULT_ROOT_DIR,
                 save_img_dir: str = DEFAULT_SAVE_IMG_DIR,
                 detection_file: str = DEFAULT_DETECTION_FILE,
                 save_prelim_img: bool = False,
                 object_names_file: str = DEFAULT_OBJECT_NAMES_FILE,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 max_img: int = MAX_IMG):
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.root_dir = root_dir
        self.save_img_dir = save_img_dir
        self.detection_file = detection_file
        self.save_prelim = save_prelim_img
        self.object_names_file = object_names_file
        self.window_size = window_size
        self.max_img = max_img

        self.capture_time_list = []
        self.classes = get_classes(root_dir)
        #help with detection
        self.count = 0
        self.blobby = []
        self.icont = []
        os.makedirs(self.save_img_dir, exist_ok=True)


    def prep_img(self, img_name):
        """
        from an img_name, load the image into the correct format for dections (greyscaled, blured and morphed)
        """
        # create coral image:
        coral_image = CoralImage(img_name=img_name, txt_name = 'placeholder.txt')
        self.capture_time_list.append(coral_image.metadata['capture_time'])
        
        # read in image
        im = Image(img_name)
        #process image
        im_mono = im.mono()        # grayscale
        ksize = 61
        im_blur = Image(cv.GaussianBlur(im_mono.image, (ksize, ksize), 0))         # blur image
        canny = cv.Canny(im_blur.image, 3, 5, L2gradient=True) # edge detection
        im_canny = Image(canny)
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
        """
        given an img_name, a morphed image and the original image,
        try to find blobs in the image and threashold these blobs
        """
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
            b0_area = [b.area for b in b0]
            b0_circ = [b.circularity for b in b0]
            b0_cent = [b.centroid for b in b0]
            #get index of blobbs that passed thresholds
            icont = [i for i, b in enumerate(blobby) if (blobby[i].centroid in b0_cent and 
                                                            blobby[i].area in b0_area and 
                                                            blobby[i].circularity in b0_circ)] 
            imblobs_area = blobby.drawBlobs(im_morph, None, icont, None, contourthickness=-1)

            # plot the contours of the blobs based on imblobs_area and icont:
            image_contours = im.image
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
            return blobby, icont
        except:
            print(f'No blob detection for {img_name}')
            # append empty to keep indexes consistent
            return [], []


    def save_results_2_pkl(self, pkl_save_path, blobs_list, blobs_count, image_index, capture_time):
        """
        saving blobs list, blobs count and other data to a pickle file
        """
        with open(pkl_save_path, 'wb') as f:
            save_data ={'blobs_list': blobs_list,
                'blobs_count': blobs_count,
                'image_index': image_index,
                'capture_time': capture_time}
            pickle.dump(save_data, f)


    def blob_2_box(self, img_name, icont, blobby):
        """
        convert the blobs into bounding boxes of yolo format ([x1 y1 x2 y2 conf class_idx class_name])
        only saving the blobs that passed the threashold in attempt_blobs
        """
        img = cv.imread(img_name)
        if icont == []: #incase of no blobs
            return []
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
        # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # plt.show()
        return torch.tensor(bb)


    def detect(self, image):
        """
        return detections from a single prepared image, 
        attempts to find blobs and then converts the blobs into yolo format [x1 y1 x2 y2 conf class_idx class_name]
        """
        img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        img_name = img_list[self.count]
        blobby, icont = self.attempt_blobs(img_name, image, im = Image(img_name))
        predictions = self.blob_2_box(img_name, icont, blobby)
        self.blobby = blobby
        self.icont = icont
        self.count += 1
        return predictions
    
    
    def run(self):
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
            save_text_predictions(predictions, img_name, txtsavedir, self.classes)
            blobs_list.append(self.blobby)
            blobs_count.append(self.icont)
        
        pkl_save_path = os.path.join(self.save_dir, self.detection_file)
        self.save_results_2_pkl(pkl_save_path, blobs_list, blobs_count, image_index, self.capture_time_list) 

        end_time = time.time()
        duration = end_time - start_time
        print('run time: {} sec'.format(duration))
        print('run time: {} min'.format(duration / 60.0))
        print('run time: {} hrs'.format(duration / 3600.0))

        capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in self.capture_time_list]
        decimal_days = convert_to_decimal_days(capture_time_dt)

        return decimal_days

        import code
        code.interact(local=dict(globals(), **locals()))

    
def main():
    img_dir = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04/images_jpg"
    root_dir = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04"
    MAX_IMG = 3
    detection_file = 'subsurface_det_testing.pkl'
    object_names_file = 'metadata/obj.names'

    Coral_Detector = SubSurface_Detector(root_dir = root_dir, img_dir = img_dir, detection_file=detection_file, 
        object_names_file = object_names_file, max_img = MAX_IMG)
    Coral_Detector.run()
    # import code
    # code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    main()

