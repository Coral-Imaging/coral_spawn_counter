#! /usr/bin/env python3

"""
Annotation class to represent all the different annotation types wrt shape, species, attributes
"""

from __future__ import annotations
import os
from PIL import Image as PILImage
import xml.etree.ElementTree as ET

from AnnotationRegion import AnnotationRegion
from Image import Image

class Annotations:
    
    # class labels
    ATTR_STAGE = 'name'
    STAGE_EGG = 'Egg'
    STAGE_FIRST_CLEAVAGE = 'First Cleavage'
    STAGE_TWO_CELL = 'Two-Cell Stage'
    STAGE_FOUR_EIGHT_CELL = 'Four-Eight-Cell Stage'
    STAGE_ADVANCED = 'Advanced Stage'
    STAGE_DAMAGED = 'Damaged'
    
    # annotation region shapes
    SHAPE_POLY = 'polygon'
    SHAPE_RECT = 'rect'
    
    def __init__(self, ann_file: str, img_dir: str):
        self.ann_file = ann_file # absolute filepath
        
        # load annotation data
        self.annotations_raw = self.read_cvat_annotations_raw()
        
        # image directory
        self.img_dir = img_dir
        self.img_list = list(sorted(os.listdir(self.img_dir)))
        
        # check annotations for consistency
        
        # convert to internal annotations format
        self.annotations = self.convert_annotations()
        
        
    def read_cvat_annotations_raw(self):
        """
        read raw annotations from cvat annotations file,
        return img metadata as a list for each image
        """
        
        # get root of xml tree
        root = ET.parse(self.ann_file).getroot()
        
        return root
    
    def convert_annotations(self):
        """
        convert raw annotations into internal annotation format (nested classes of annotation regions)
        """
        data = []
        
        # get image height/width from an image file in img_dir
        # assume all images are of the same size in the same img_dir
        im = PILImage.open(os.path.join(self.img_dir, self.img_list[0]))
        width, height = im.size
        
        for image in self.annotations_raw:
            if image.tag == 'image':
                img_name = image.attrib['name']
                id = image.attrib['id']
                width = image.attrib['width']
                height = image.attrib['height']
                
                # TODO make Image object 
                
                for anno in image:
                    if anno.tag == 'box':
                        label = anno.attrib['label']
                        xtl = anno.attrib['xtl']
                        ytl = anno.attrib['ytl']
                        ybr = anno.attrib['ybr']
                        xbr = anno.attrib['xbr']
                    
                # TODO make an Annotated Region object
        
            # TODO append to list of Image objects
        return False