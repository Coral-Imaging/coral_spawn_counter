#! /usr/bin/env python3

"""
coral spawn counting using blobs (copied from count_coral_spawn.py)
TODO: also calls yolov8 detector onto cslics surface embryogenesis
"""
import os
import cv2 as cv
import numpy as np
import glob
# import random as rng
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib as mpl
# import seaborn.objects as so
import pickle
# import pandas as pd
# from datetime import datetime
import time
import torch

import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image as MvtImage

from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.Detector import Detector


class SubSurface_Detector(Detector):

    DEFAULT_META_DIR = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    DEFAULT_IMG_DIR = '/home/cslics04/20231018_cslics_detector_images_sample/subsurface'
    DEFAULT_SAVE_DIR = '/home/cslics04/images/subsurface'
    
    DEFAULT_OUTPUT_FILE = 'subsurface_det.pkl'
    DEFAULT_WINDOW_SIZE = 20
    MAX_IMG = 1000

    DEFAULT_AREA_MIN  = 5000
    DEFAULT_AREA_MAX = 40000
    DEFAULT_CIRC_MIN = 0.5
    DEFAULT_CIRC_MAX = 1.0

    def __init__(self,
                 meta_dir: str = DEFAULT_META_DIR,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 max_img: int = MAX_IMG,
                 save_prelim_img: bool = False,
                 img_size: int = 1280,
                 detection_file: str = DEFAULT_OUTPUT_FILE,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 area_min: int = DEFAULT_AREA_MIN,
                 area_max: int = DEFAULT_AREA_MAX,
                 circ_min: int = DEFAULT_CIRC_MIN,
                 circ_max: int = DEFAULT_CIRC_MAX):
        
        self.detection_file = detection_file
        self.window_size = window_size
        self.capture_time_list = []
        #help with detection
        self.count = 0
        self.blobby = []
        self.icont = []
        
        # parameters for circle/blob thresholding/filtering:
        self.area_min = area_min
        self.area_max = area_max
        self.circ_min = circ_min
        self.circ_max = circ_max

        # parameters for plotting subsurface detections
        self.contour_colour = [0, 0, 255] # red in BGR
        self.contour_thickness = 10
        
        Detector.__init__(self, 
                          meta_dir = meta_dir,
                          img_dir = img_dir,
                          save_dir = save_dir,
                          max_img = max_img,
                          save_prelim_img = save_prelim_img,
                          img_size=img_size)


    def prep_img_name(self, img_name):
        # create coral image:
        # TODO might need to change txt_name to actual img_name? or it's just a method of getting the capture_time metadata
        # coral_image = CoralImage(img_name=img_name, txt_name = 'placeholder.txt')
        # self.capture_time_list.append(coral_image.metadata['capture_time'])
        
        # read in image
        im = MvtImage(img_name)
        return self.prep_img(im, img_name)
        
    def prep_img(self, im, img_name=None):
        """
        from an img_name, load the image into the correct format for dections (greyscaled, blured and morphed)
        """
        im = MvtImage(im)
        # # create coral image:
        # # TODO might need to change txt_name to actual img_name? or it's just a method of getting the capture_time metadata
        # coral_image = CoralImage(img_name=img_name, txt_name = 'placeholder.txt')
        # self.capture_time_list.append(coral_image.metadata['capture_time'])
        
        # # read in image
        # im = Image(img_name)
        # TODO resize image, can then reduce k_blur - should be faster!
        
        #process image
        # TODO expose these edge detection parameters to the detector level
        im_mono = im.mono()        # grayscale
        k_blur = 61
        im_blur = MvtImage(cv.GaussianBlur(im_mono.image, (k_blur, k_blur), 0))         # blur image
        canny = cv.Canny(im_blur.image, 3, 5, L2gradient=True) # edge detection
        im_canny = MvtImage(canny)
        # morph
        k_morph = 11
        kernel = np.ones((k_morph,k_morph), np.uint8)
        im_morph = MvtImage(im_canny.dilate(kernel))
        im_morph = im_morph.close(kernel)
        # kernel = 11
        im_morph = im_morph.open(kernel)

        # step-by-step save the image
        if self.save_prelim_img:
            img_base_name = os.path.basename(img_name)[:-4]
            im_mono.write(os.path.join(self.save_dir, img_base_name + '_00_orig.jpg'))
            im_blur.write(os.path.join(self.save_dir, img_base_name + '_01_blur.jpg'))
            im_canny.write(os.path.join(self.save_dir, img_base_name + '_02_edge.jpg'))
            im_morph.write(os.path.join(self.save_dir, img_base_name + '_03_morph.jpg'))
        
        # be sure to return a numpy array, just like other Detectors
        return im_morph.image


    def attempt_blobs(self, image): # , original_image):
        """
        given an img_name, a morphed image and the original image,
        try to find blobs in the image and threashold these blobs
        """
        try:
            
            # img_base_name = os.path.basename(img_name)[:-4]
            image = MvtImage(image)
            blobby = mvt.Blob(image)
            print(f'{len(blobby)} blobs initially found')
            # show blobs
            if self.save_prelim_img:
                imblobs = blobby.drawBlobs(image, None, None, None, contourthickness=-1)

            # reject too-small and too weird (non-circular) blobs
            b0 = [b for b in blobby if ((b.area < self.area_max and b.area > self.area_min) and (b.circularity > self.circ_min and b.circularity < self.circ_max))]
            b0_area = [b.area for b in b0]
            b0_circ = [b.circularity for b in b0]
            b0_cent = [b.centroid for b in b0]
            # get index of blobbs that passed thresholds
            icont = [i for i, b in enumerate(blobby) if (blobby[i].centroid in b0_cent and 
                                                            blobby[i].area in b0_area and 
                                                            blobby[i].circularity in b0_circ)] 
            
            # if self.save_prelim_img:
            #     imblobs_area = blobby.drawBlobs(im_morph, None, icont, None, contourthickness=-1)

            #     # plot the contours of the blobs based on imblobs_area and icont:
            #     image_contours = original_image.image
                
            #     for i in icont:
            #         cv.drawContours(image_contours,
            #                         blobby._contours,
            #                         i,
            #                         self.contour_colour,
            #                         thickness=self.contour_thickness,
            #                         lineType=cv.LINE_8)
            #     image3 = MvtImage(image_contours)
            #     image3.write(os.path.join(self.save_dir, img_base_name + '_blobs_contour.jpg'))

            #     imblobs.write(os.path.join(self.save_dir, img_base_name + '_04_blob.jpg'))
            #     imblobs_area.write(os.path.join(self.save_dir, img_base_name + '_05_blob_filter.jpg'))
            print(f'{len(icont)} blobs passed thresholds')
            return blobby, icont
        except:
            print(f'No blob detection for image')
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


    def blob_2_box(self, img, icont, blobby):
        """
        convert the blobs into bounding boxes of yolo format ([x1 y1 x2 y2 conf class_idx class_name])
        only saving the blobs that passed the threashold in attempt_blobs
        """
        # img = cv.imread(img_name)
        if icont == []: #incase of no blobs
            return []
        contours = blobby._contours
        imgw, imgh = img.shape[1], img.shape[0]
        bb = []
        for i, c in enumerate(contours):
            if i in icont:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
                x1 = x/imgw
                y1 = y/imgh
                x2 = (x+w)/imgw
                y2 = (y+h)/imgh
                bb.append([x1, y1, x2, y2, 0.5, 3, 3])
        #debug visualise
        # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # plt.show()
        return torch.tensor(bb)


    def detect(self, image: np.ndarray):
        """
        return detections from a single prepared image, 
        attempts to find blobs and then converts the blobs into yolo format [x1 y1 x2 y2 conf class_idx class_name]
        """
        # img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        # img_name = img_list[self.count]
        # blobby, icont = self.attempt_blobs(img_name, image, im = MvtImage(img_name))
        blobby, icont = self.attempt_blobs(image)
        predictions = self.blob_2_box(image, icont, blobby)
        self.blobby = blobby
        self.icont = icont
        self.count += 1
        return predictions
    
    def run(self):
        print("running blob subsurface detector")
        img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
        imgsave_dir = os.path.join(self.save_dir, 'detections', 'detection_images')
        os.makedirs(imgsave_dir, exist_ok=True)
        txtsavedir = os.path.join(self.save_dir, 'detections', 'detection_textfiles')
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
            self.save_image_predictions(predictions, MvtImage(img_name).image, img_name, imgsave_dir)
            self.save_text_predictions(predictions, img_name, txtsavedir)
            
            blobs_list.append(self.blobby)
            blobs_count.append(self.icont)
        
        pkl_save_path = os.path.join(self.save_dir, self.detection_file)
        self.save_results_2_pkl(pkl_save_path, blobs_list, blobs_count, image_index, self.capture_time_list) 

        end_time = time.time()
        duration = end_time - start_time
        print('run time: {} sec'.format(duration))
        print('run time: {} min'.format(duration / 60.0))
        print('run time: {} hrs'.format(duration / 3600.0))

        print(f'time[s]/image = {duration / len(self.img_list)}')
        
        # capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in self.capture_time_list]
        # decimal_days = convert_to_decimal_days(capture_time_dt)

        # return decimal_days

    
def main():

    root_dir = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    img_dir = '/home/cslics04/20231018_cslics_detector_images_sample/subsurface'
    save_dir = '/home/cslics04/images/subsurface'
    MAX_IMG = 5
    detection_file = 'subsurface_det_testing.pkl'

    Coral_Detector = SubSurface_Detector(meta_dir = root_dir, 
                                         img_dir = img_dir, 
                                         save_dir=save_dir, 
                                         detection_file=detection_file,
                                         max_img = MAX_IMG)
    Coral_Detector.run()
    # import code
    # code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    main()

