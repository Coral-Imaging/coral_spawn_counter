#!/usr/bin/env python3

""" base class for detectors
has detector.detect()
has detector.save_image_predictions
has detector.save_to_text_predictions
set class colours, etc
"""

import os
import cv2 as cv
import numpy as np
import glob as glob
# from datetime import datetime


class Detector(object):
    
    DEFAULT_META_DIR = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    DEFAULT_IMG_DIR = '/home/cslics04/20231018_cslics_detector_images_sample/microspheres'
    DEFAULT_SAVE_DIR = '/home/cslics04/images/redcircles'
    
    DEFAULT_MAX_IMG = 2
    
    def __init__(self, 
                 meta_dir: str = DEFAULT_META_DIR,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 max_img: int = DEFAULT_MAX_IMG,
                 save_prelim_img: bool = False,
                 img_size: int = 1280):
        
        # folders and files
        self.meta_dir = meta_dir
        self.img_dir = img_dir
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')) +
                               glob.glob(os.path.join(self.img_dir, '*.png')) + 
                               glob.glob(os.path.join(self.img_dir, '*.jpeg')))
        
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.classes = self.get_classes(self.meta_dir)
        self.class_colours = self.set_class_colours(self.classes)
        
        
        # common parameters for detectors
        self.max_img = max_img
        self.save_prelim_img = save_prelim_img
        self.img_size = img_size
        
    def prep_img(self, img_name):
        """
        from an img_name, load the image into the correct format for dections (cv.imread)
        """
        img = cv.imread(img_name)
        return img
    
        
    def get_classes(self, meta_dir):
        """
        get the classes from a metadata/obj.names file
        classes = [class1, class2, class3 etc.]
        """
        #TODO: make a function of something else, used in both detectors
        with open(os.path.join(meta_dir, 'metadata','obj.names'), 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    
    
    def set_class_colours(self, classes):
        """
        set classes to specific colours using a dictionary
        """
        #TODO: make a function of something else, used in both detectors
        orange = [255, 128, 0] # four-eight cell stage
        blue = [0, 212, 255] # first cleavage
        purple = [170, 0, 255] # two-cell stage
        yellow = [255, 255, 0] # advanced
        brown = [144, 65, 2] # damaged
        green = [0, 255, 00] # egg
        class_colours = {classes[0]: orange,
                        classes[1]: blue,
                        classes[2]: purple,
                        classes[3]: yellow,
                        classes[4]: brown,
                        classes[5]: green}
        return class_colours


    def save_text_predictions(self, predictions, imgname, txtsavedir, classes):
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
                class_name = classes[class_idx]
                f.write(f'{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.4f} {class_idx:g} {class_name}\n')
        return True
    
    
    def save_image_predictions(self, predictions, imgname, imgsavedir, class_colours, classes):
        """
        save predictions/detections (assuming predictions in yolo format) on image
        """
        img = cv.imread(imgname)
        imgw, imgh = img.shape[1], img.shape[0]
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])
            #extract back into cv lengths
            x1 = x1*imgw
            x2 = x2*imgw
            y1 = y1*imgh
            y2 = y2*imgh        
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), class_colours[classes[cls]], 2)
            cv.putText(img, f"{classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_colours[classes[cls]], 2)

        imgsavename = os.path.basename(imgname)
        imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_det.jpg')
        cv.imwrite(imgsave_path, img)
        return True

        
    def convert_to_decimal_days(self, dates_list, time_zero=None):
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
