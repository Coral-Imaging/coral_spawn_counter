#! /usr/bin/env python3

"""
Annotation class to represent all the different annotation types wrt shape, species, attributes
"""

from __future__ import annotations
import os
from PIL import Image as PILImage
import xml.etree.ElementTree as ET

from annotation.AnnotationRegion import AnnotationRegion
from annotation.Image import Image

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
        convert raw annotations into internal annotation format (nested list of classes of annotation regions)
        """
        
        # get image height/width from an image file in img_dir
        # assume all images are of the same size in the same img_dir
        im = PILImage.open(os.path.join(self.img_dir, self.img_list[0]))
        width, height = im.size
        
        AnnotatedImages = []
        for image in self.annotations_raw:
            AnnImage = None
            if image.tag == 'image':
                img_name = image.attrib['name']
                id = image.attrib['id']
                width = image.attrib['width']
                height = image.attrib['height']
                
                # make Image object 
                AnnImage = Image(os.path.join(self.img_dir, img_name), 
                                 width=width, 
                                 height=height, 
                                 camera=str(id))

                Regions = []
                for anno in image:
                    if anno.tag == 'box':
                        label = str(anno.attrib['label'])
                        xtl = int(float(anno.attrib['xtl']))
                        ytl = int(float(anno.attrib['ytl']))
                        ybr = int(float(anno.attrib['ybr']))
                        xbr = int(float(anno.attrib['xbr']))
                        x = [xtl, xbr, xbr, xtl]
                        y = [ytl, ytl, ybr, ybr]
                        Box = AnnotationRegion(class_name=label,
                                               x = x,
                                               y = y,
                                               shape_type='rect')
                        Regions.append(Box)

                AnnImage.regions = Regions

            if AnnImage is not None:
                AnnotatedImages.append(AnnImage)
        
        return AnnotatedImages


if __name__ == "__main__":

    print('Annotations.py')

    # in the 100 images
    data_dir = '/home/dorian/Data/acropora_maggie_tenuis_dataset_100_renamed/combined_100'
    img_dir = os.path.join(data_dir, 'images')
    ann_file = os.path.join(data_dir, 'metadata/annotations_updated.xml')

    ImageAnnotations = Annotations(ann_file=ann_file, img_dir=img_dir)

    import code
    code.interact(local=dict(globals(), **locals()))