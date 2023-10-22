#! /usr/bin/env python3

"""
code pulled from detect_circle_annotations.py to be a detector
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
import torch
import time


from coral_spawn_counter.Detector_helper import get_classes, set_class_colours, save_text_predictions, save_image_predictions

class RedCircle_Detector():
   
    DEFAULT_ROOT_DIR = '/home/cslics04/20231018_cslics_detector_images_sample/' # where the metadata is
    DEFAULT_IMG_DIR = os.path.join(DEFAULT_ROOT_DIR, 'microspheres')
    DEFAULT_SAVE_DIR = '/home/cslics04/images/redcircles'

    DEFAULT_MAX_DECT = 2
    
    # TODO all the parameters for detection here as defaults
    
    def __init__(self,
                 root_dir: str = DEFAULT_ROOT_DIR,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 max_dect: int = DEFAULT_MAX_DECT,
                 save_prelim_img: bool = True):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')) +
                               glob.glob(os.path.join(self.img_dir, '*.png')) + 
                               glob.glob(os.path.join(self.img_dir, '*.jpeg')))
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.classes = get_classes(self.root_dir)
        self.class_colours = set_class_colours(self.classes)
        self.max_dect = max_dect
        self.count = 0
        
        self.save_prelim = save_prelim_img
        
        self.input_image_size = 1280 # pixels
        self.conf_def = 0.5
        self.class_idx_def = 5 # eggs
        self.class_name_def = 5 # eggs
        


    def find_circles(self,
                    img, 
                    blur=3, 
                    method=cv.HOUGH_GRADIENT, 
                    dp=0.9, 
                    minDist=5,
                    maxEdgeThresh=80,
                    minEdgeThresh=20,
                    maxRadius=40,
                    minRadius=10):
        """
        Find circles in a given image with specified parameters
        img - as numpy array
        blur - kernel size to apply median blur to image (smoother image = better circles)
        dp - inverse ratio  of the accumulator resolution, dp=1: the accumulator is same res as img, dp=2: accumulator is half as big width, height
        mindist - min dist between detected circle centres
        # NOTE names aren't exactly correct for the Canny's hysteresis threshold procedure
        maxEdgeThresh - higher threshold of 2 thresholds, passed to Canny edge detector
        minEdgeThresh - lower threshold of 2 thresholds passed to Canny edge detector
        minRadius - min radius of circle
        maxRadius - max radius of circle
        """
        # TODO rename parameters for more readability
        
        
        # convert image to grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # blur image (Hough transforms work better on smooth images)
        img = cv.medianBlur(img, blur)
        # find circles using Hough Transform
        circles = cv.HoughCircles(image = img,
                                method=method,
                                dp=dp,
                                minDist=minDist,
                                param1=maxEdgeThresh,
                                param2=minEdgeThresh,
                                maxRadius=maxRadius,
                                minRadius=minRadius)
        
        # TODO filter based on colour RED circles
        
        return circles


    def draw_circles(self, img, circles, outer_circle_color=(255, 0, 0), thickness=8):
        """ draw circles onto image"""
        for i, circ in enumerate(circles[0,:], start=1):
            cv.circle(img, 
                    (int(circ[0]), int(circ[1])), 
                    radius=int(circ[2]), 
                    color=outer_circle_color, 
                    thickness=thickness)
        return img


    def convert_circle_to_box(self, x,y,r, image_width, image_height):
        """ convert a circle to a box, maxing sure to stay within the image bounds"""
        xmin = max(0, x - r)
        xmax = min(image_width, x + r)
        ymin = max(0, y - r)
        ymax = min(image_height, y + r)
        return xmin, ymin, xmax, ymax


    def prep_img(self, img_name):
        """
        from an img_name, load the image into the correct format for dections (cv.imread)
        """
        img = cv.imread(img_name)
        return img


    def detect(self, image):
        """
        return detections from a single prepared image, input as a numpy array
        attempts to find circles and then converts the circles into yolo format [x1 y1 x2 y2 conf class_idx class_name]
        """
        img_height, img_width, _ = image.shape
        
        # resize image to small size/consistency (detection parameters vs image size) and speed!
        # img_h, img_w, n_channels = img.shape
        img_scale_factor: float = self.input_image_size / img_width
        print(f'img_scale_factor = {img_scale_factor}')
        image: np.ndarray = cv.resize(image, None, fx=img_scale_factor, fy=img_scale_factor)
        img_h_resized, img_w_resized, _ = image.shape
        
        circles = self.find_circles(image)
        if circles is not None:
            _ , n_circ, _ = circles.shape
        else:
            n_circ = 0
        print(f'Circles detected = {n_circ}')
        
        # save image with circles drawn ontop
        if self.save_prelim:
            if n_circ > 0:
                img_c = self.draw_circles(image, circles)
            else:
                img_c = image
            # save image
            img_name = self.img_list[self.count]
            img_name_circle = os.path.basename(img_name[:-4]) + '_circ.jpg'
            self.count += 1
            cv.imwrite(os.path.join(self.save_dir, img_name_circle), img_c)

        # get bounding box of the circles
        pred = []
        for i in range(n_circ):
            x = circles[0][i, 0]
            y = circles[0][i, 1]
            r = circles[0][i, 2]
            xmin, ymin, xmax, ymax = self.convert_circle_to_box(x,y,r,img_w_resized, img_h_resized)
            
            # size-up boxes to original image sizes
            x1 = xmin/img_w_resized
            y1 = ymin/img_h_resized
            x2 = xmax/img_w_resized
            y2 = ymax/img_h_resized
            # print(xmin, ymin, xmax, ymax)
            pred.append([x1, y1, x2, y2, self.conf_def, self.class_idx_def, self.class_name_def]) # class definitions hard-coded here
        
        # print('predictions as tensors')
        return torch.tensor(pred)


    def run(self):   
        imgsave_dir = os.path.join(self.save_dir, 'detections', 'detections_images')
        os.makedirs(imgsave_dir, exist_ok=True)
        txtsavedir = os.path.join(self.save_dir, 'detections', 'detection_textfiles')
        os.makedirs(txtsavedir, exist_ok=True)
        
        start_time = time.time()
        for i, img_name in enumerate(self.img_list):
            if i > self.max_dect:
                print('hit max detections')
                break
            
            print(f'{i}: img_name = {img_name}')  
            # read in image
            img = self.prep_img(img_name)
            # detect circles
            predictions = self.detect(img)

            save_image_predictions(predictions, img_name, imgsave_dir, self.class_colours, self.classes)
            save_text_predictions(predictions, img_name, txtsavedir, self.classes)
            
        end_time = time.time()
        duration = end_time - start_time
        print('run time: {} sec'.format(duration))
        print('run time: {} min'.format(duration / 60.0))
        print('run time: {} hrs'.format(duration / 3600.0))
        
        print(f'time[s]/image = {duration / len(self.img_list)}')
        

        


def main():
    root_dir = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    img_dir = '/home/cslics04/20231018_cslics_detector_images_sample/microspheres'
    
    max_dect = 2 # number of images
    Coral_Detector = RedCircle_Detector(root_dir=root_dir, img_dir = img_dir, max_dect=max_dect)
    Coral_Detector.run()
    # import code
    # code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    main()