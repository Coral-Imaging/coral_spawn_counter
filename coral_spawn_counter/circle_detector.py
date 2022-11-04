#! /usr/bin/env python3

"""
circle detector using Hough transforms
"""

import cv2 as cv
import os

class CircleDetector:


    BLUR_DEFAULT = 5
    METHOD_DEFAULT = cv.HOUGH_GRADIENT
    DP_DEFAULT = 0.9
    MINDIST_DEFAULT = 30
    PARAM1_DEFAULT = 100
    PARAM2_DEFAULT = 20
    MAXRADIUS_DEFAULT = 200
    MINRADIUS_DEFAULT = 20

    def __init__(self,
                 blur = BLUR_DEFAULT,
                 method = METHOD_DEFAULT,
                 dp = DP_DEFAULT,
                 minDist = MINDIST_DEFAULT,
                 param1 = PARAM1_DEFAULT,
                 param2 = PARAM2_DEFAULT,
                 maxRadius = MAXRADIUS_DEFAULT,
                 minRadius = MINRADIUS_DEFAULT):

        self.blur = blur
        self.method = method
        self.dp = dp
        self.minDist = minDist
        self.param1 = param1
        self.param2 = param2
        self.maxRadius = maxRadius
        self.minRadius = minRadius

        self.circles = []
        self.count = []


    def find_circles(self, img):
        # TODO rename parameters for more readability
        # convert image to grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # blur image (Hough transforms work better on smooth images)
        img = cv.medianBlur(img, self.blur)
        # find circles using Hough Transform
        circles = cv.HoughCircles(image = img,
                                method=self.method,
                                dp=self.dp,
                                minDist=self.minDist,
                                param1=self.param1,
                                param2=self.param2,
                                maxRadius=self.maxRadius,
                                minRadius=self.minRadius)
                
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


    def count_spawn(self, img):
        # proxy for ML detector, which will come later in the project
        # counts the spawn
        # saves a figure of the detections

        circles = self.find_circles(img)

        if circles is not None:
            _, count, _ = circles.shape
        else:
            count = 0
        self.circles = circles
        self.count = count
        return count, circles


    def save_detections(self, img, img_name, save_dir):

        if self.circles is not None:
            img_c = self.draw_circles(img, self.circles)
        else:
            img_c = img
            
        img_name_circle = img_name[:-4] + '_circ.png'
        cv.imwrite(os.path.join(save_dir, img_name_circle), img_c)