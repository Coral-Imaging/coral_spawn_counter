#! /usr/bin/env python3

"""
Image class, which has annotation properties
"""

import PIL.Image as PIL_Image
from PIL.PngImagePlugin import PngInfo
import os
from pprint import *

class Image:

    def __init__(self, 
                 img_name: str, 
                 filesize: int = None,
                 width: int = None,
                 height: int = None, 
                 camera: str = None):
        self.img_name = img_name
        self.img_dir = os.path.basename(img_name)
        self.filesize = filesize
        self.width = width
        self.height = height
        self.camera = camera
        self.regions = []
        self.metadata = self.read_metadata()

    def get_img_name(self):
        return self.img_name
    
    def get_metadata(self):
        return self.metadata
    
    def set_img_name(self, img_name):
        self.img_name = img_name
        
    def set_metadata(self, metadata):
        self.metadata = metadata
    
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
        
    def print(self):
        print('Filename: ' + self.img_name)
        print(f'Filesize: {self.filesize}')
        print(f'Image (width, height): ({self.width, self.height}) pix')
        print('Camera: ' + self.camera)
        print('Regions: ')
        for region in self.regions:
            region.print()


if __name__ == "__main__":
    
    print('Image.py')
    
    import code
    code.interact(local=dict(globals(), **locals()))