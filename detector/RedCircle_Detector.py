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

from detector.Detector_helper import get_classes, set_class_colours, save_text_predictions, save_image_predictions

class RedCircle_Detector():
    DEFAULT_DATA_DIR = '/home/java/Java/data/cslics_microsphere_data'
    DEFAULT_CLASS_DIR = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04"
    DEFAULT_IMG_DIR = os.path.join(DEFAULT_DATA_DIR, 'images')
    DEFAULT_SAVE_DIR = os.path.join(DEFAULT_DATA_DIR, 'red_circles')
    DEFAULT_MAX_DECT = 1000
    
    def __init__(self,
                 data_dir: str = DEFAULT_DATA_DIR,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 show_circle: bool = True,
                 class_dir: str = DEFAULT_CLASS_DIR,
                 max_dect: int = DEFAULT_MAX_DECT):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.classes = get_classes(class_dir)
        self.class_colours = set_class_colours(self.classes)
        self.show_circles = show_circle
        self.max_dect = max_dect
        self.count = 0

    def find_circles(self,
                    img, 
                    blur=5, 
                    method=cv.HOUGH_GRADIENT, 
                    dp=0.9, 
                    minDist=30,
                    param1=100,
                    param2=30,
                    maxRadius=125,
                    minRadius=20):
        """
        Find circles in a given image with specified parameters
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
                                param1=param1,
                                param2=param2,
                                maxRadius=maxRadius,
                                minRadius=minRadius)
        return circles


    def draw_circles(self, img, circles, outer_circle_color=(255, 0, 0), thickness=8):
        """ draw circles onto image"""
        for circ, i in enumerate(circles[0,:], start=1):
            cv.circle(img, 
                    (int(i[0]), int(i[1])), 
                    radius=int(i[2]), 
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
        return detections from a single prepared image, 
        attempts to find circles and then converts the circles into yolo format [x1 y1 x2 y2 conf class_idx class_name]
        """
        img_height, img_width, _ = image.shape
        circles = self.find_circles(image)
        if circles is not None:
            _ , n_circ, _ = circles.shape
        else:
            n_circ = 0
        print(f'Circles detected = {n_circ}')
        if self.show_circles:
            if n_circ > 0:
                img_c = self.draw_circles(image, circles)
            else:
                img_c = image
            # save image
            img_name = self.img_list[self.count]
            img_name_circle = img_name[:-4] + '_circ.jpeg'
            self.count += 1
            cv.imwrite(os.path.join(self.save_dir, img_name_circle), img_c)

        # get bounding box of the circles
        pred = []
        for i in range(n_circ):
            x = circles[0][i, 0]
            y = circles[0][i, 1]
            r = circles[0][i, 2]
            xmin, ymin, xmax, ymax = self.convert_circle_to_box(x,y,r,img_width, img_height)
            x1 = xmin/(img_width*2)
            y1 = ymin/(img_height*2)
            x2 = xmax/(img_width*2)
            y2 = ymax/(img_height*2)
            print(xmin, ymin, xmax, ymax)
            pred.append([x1, y1, x2, y2, 0.5, 3, 3])
        return torch.tensor(pred)

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


    def run(self):   
        txtsavedir = os.path.join(self.data_dir, 'textfiles')
        os.makedirs(txtsavedir, exist_ok=True)
        for i, img_name in enumerate(self.img_list):
            if i > self.max_dect:
                break
            
            print(f'{i}: img_name = {img_name}')  
            # read in image
            img = self.prep_img(img_name)
            # detect circles
            predictions = self.detect(img)

            save_image_predictions(predictions, img, img_name, self.save_dir, self.class_colours, self.classes)
            save_text_predictions(predictions, img_name, txtsavedir, self.classes)
            import code
            code.interact(local=dict(globals(), **locals()))

def main():
    data_dir = '/home/java/Java/data/cslics_microsphere_data'
    max_dect = 3
    Coral_Detector = RedCircle_Detector(data_dir=data_dir, max_dect=max_dect)
    Coral_Detector.run()
    # import code
    # code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    main()