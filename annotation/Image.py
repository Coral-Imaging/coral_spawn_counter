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
                 camera: str = 'raspberry pi hq camera'):
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
            
        
        img_suffix = os.path.splitext(img_name)[1]
        img_md = PIL_Image.open(img_name)

        # print(img_md.text)
        png_suffix = ['.png', '.PNG']
        jpg_suffix = ['.jpg', '.JPG', '.jpeg', '.JPEG']
        if img_suffix in png_suffix:
            return img_md.text
        elif img_suffix in jpg_suffix: # HACK until we have proper metadata file structure associated with cslics
            base_image_name = os.path.basename(img_name)
            camera_index = base_image_name[7] # NOTE only works for single-digit cslics
            capture_time = base_image_name[9:31]
            metadata_dict = {'camera_index': camera_index,
                             'capture_time': capture_time}
            return metadata_dict
        else:
            return TypeError('Unknown image type (not jpg or png)')
    
    def print_metadata(self):
        # print(f'metadata: {self.metadata}')
        pprint(self.metadata)
        
    def print(self):
        print('Filename: ' + self.img_name)
        print(f'Filesize: {self.filesize}')
        print(f'Image (width, height): ({self.width, self.height}) pix')
        print(f'Camera: {self.camera}')
        print('Regions: ')
        for region in self.regions:
            region.print()


if __name__ == "__main__":
    
    print('Image.py')
    
    import glob
    # img_dir = '/home/dorian/Data/cslics_2022_datasets/202211_amtenuis_1000/images_png'
    # img_list = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    
    img_dir = '/home/dorian/Data/cslics_2022_datasets/202211_amtenuis_1000/images_jpg'
    img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    i = 0
    
    img_name = img_list[i]
    img = Image(img_name)
    
    img.print()
    
    
    import code
    code.interact(local=dict(globals(), **locals()))