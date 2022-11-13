#! /usr/bin/env python3

"""
Image class, that (TODO: will import from MVT)
but for now, just has metadata reader/extracter
properties for counts, detections
"""

import PIL.Image as PIL_Image
from PIL.PngImagePlugin import PngInfo
import json
import numpy as np
import os
from pprint import *

from coral_spawn_counter.circle_detector import CircleDetector

class CoralImage:

    def __init__(self, img_name, img=None):

        # for now, only accepting PIL images, png files
        self.img_name = img_name
        self.img_basename = os.path.basename(img_name)
        # self.img = PIL_Image.open(img_name)
        self.count = 0
        self.detections = []
        self.metadata = self.read_metadata()

        self.SpawnCounter = CircleDetector()


    def count_spawn(self, img, det_param=None):    
        count, circles = self.SpawnCounter.count_spawn(np.array(img), det_param)
        self.count = count
        self.detections = circles


    def save_detection_img(self, img, img_name=None, save_dir=None):
        if img_name is None:
            img_name = self.img_basename
        if save_dir is None:
            path = os.path.dirname(__file__)
            save_dir = os.path.join(path, 'detections')
        
        # print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.SpawnCounter.save_detections(img=np.array(img), img_name=img_name, save_dir=save_dir)


    # def set_img(self, img):
    #     self.img = img

    def set_img_name(self, img_name):
        self.img_name = img_name


    def set_count(self, count):
        self.count = count

    def set_detections(self, detections):
        self.detections = detections

    def set_metadata(self, metadata):
        self.metadata = metadata
        

    # def get_img(self):
    #     return self.img
    
    def get_img_name(self):
        return self.img_name

    def get_count(self):
        return self.count

    def get_detections(self):
        return self.detections

    def get_metadata(self):
        return self.metadata


    def read_metadata(self, img_name=None):
        # print('reading metadata')
        if img_name is None:
            img_name = self.img_name
        img_md = PIL_Image.open(img_name)
        # print(img_md.text)
        return img_md.text

    def print_metadata(self):
        # print(f'metadata: {self.metadata}')
        pprint(self.metadata)


    def numpy_array(self):
        return np.array(self.img)



if __name__ == "__main__":

    print('CoralImage.py')

    img_dir = '/home/cslics/cslics_ws/src/rrap-downloader/cslics_data/cslics03/images'
    img_list = os.listdir(img_dir)
    img_list.sort()
    img = CoralImage(os.path.join(img_dir, img_list[-1]))
    img.read_metadata(img_name = os.path.join(img_dir, img_list[4]))


    import code
    code.interact(local=dict(globals(), **locals()))