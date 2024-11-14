#! /usr/bin/env/python3

"""
script to run a trained yolov8 segment model on unlabeled images, saving these results in cvat annotation form
NOTE: Basefile must have been downloaded from cvat, with the images already loaded into the job
"""

from ultralytics import YOLO
import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import zipfile
import sys

### File locations ###
base_file = "/home/java/Downloads/annotations.xml"
base_img_location = "/home/java/Java/data/cslics_desktop_data/cslics_desktop_2024_October_maeq/images_all"
output_filename = "/home/java/Downloads/cslics_2024_1pm_complete.xml"
# base_file = sys.args[1]
# base_img_location = sys.args[2]
# output_filename = sys.args[3]

### Parameters ###
weight_file = "/home/java/Java/ultralytics/runs/detect/cslics_desktop_Nov_2024/weights/best.pt"

classes = ["Four-Eight-Cell Stage", "First Cleavage", "Two-Cell Stage", "Advanced Stage", "Damaged", "Egg", "Larvae"]
labeled = [0,1,2,3,4,5,6,7,8,9,10, 15, 25, 30, 50, 55, 60, 65, 70,71,72,73,74,75,76,77,78,79,80, 89]


class Detect2Cvat:
    BASE_FILE = "/home/java/Java/Cgras/cgras_settler_counter/annotations.xml"
    OUTPUT_FILE = "/home/java/Downloads/complete.xml"
    DEFAULT_WEIGHT_FILE = "/home/java/Java/ultralytics/runs/segment/train4/weights/best.pt"
    
    def __init__(self, 
                 img_location: str, 
                 output_file: str = OUTPUT_FILE, 
                 weights_file: str = DEFAULT_WEIGHT_FILE,
                 base_file: str = BASE_FILE, 
                 output_as_mask: str = False):
        self.img_location = img_location
        self.base_file = base_file
        self.output_file = output_file
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = YOLO(weights_file).to(self.device)
        self.output_as_mask = output_as_mask


    def run(self):
        tree = ET.parse(self.base_file)
        root = tree.getroot() 
        new_tree = ElementTree(Element("annotations"))
        # add version element
        version_element = ET.Element('version')
        version_element.text = '1.1'
        new_tree.getroot().append(version_element)
        # add Meta elements, (copy over from source_file)
        meta_element = root.find('.//meta')
        if meta_element is not None:
            new_meta_elem = ET.SubElement(new_tree.getroot(), 'meta')
            # copy all subelements of meta
            for sub_element in meta_element:
                new_meta_elem.append(sub_element)
        

        for i, image_element in enumerate(root.findall('.//image')):
            print(i,'images being processed')
            image_id = image_element.get('id')
            image_name = image_element.get('name')
            image_width = int(image_element.get('width'))
            image_height = int(image_element.get('height'))

            # create new image element in new XML
            new_elem = SubElement(new_tree.getroot(), 'image')
            new_elem.set('id', image_id)
            new_elem.set('name', image_name)
            new_elem.set('width', str(image_width))
            new_elem.set('height', str(image_height))
            
            #copy images already labeled
            if i in labeled:
                boxes = image_element.findall('.//box')
                for box in boxes:
                    new_box = SubElement(new_elem, 'box')
                    for attr, value in box.attrib.items():
                        new_box.set(attr, value)
                print('labeled image,', len(boxes),'coppied')
            # boxes = image_element.findall('.//box')
            else:
                image_file = os.path.join(self.img_location, image_name)
                results = self.model.predict(source=image_file, iou=0.5, agnostic_nms=True, max_det=1000)
                boxes = results[0].boxes
                class_list = [b.cls.item() for b in results[0].boxes]

                if boxes==None:
                    print('No boxes found in image',image_name)
                    continue

                for j, b in enumerate(boxes):
                    xtl, ytl, xbr, ybr = [round(coord, 2) for coord in b.xyxy[0].tolist()]
                    class_name = classes[int(b.cls.item())]
                    new_box = SubElement(new_elem, 'box')
                    new_box.set('label', class_name)
                    new_box.set('source', 'manual')
                    new_box.set('occluded', '0')
                    new_box.set('xtl', str(xtl))
                    new_box.set('ytl', str(ytl))
                    new_box.set('xbr', str(xbr))
                    new_box.set('ybr', str(ybr))
                    new_box.set('z_order', '0')


                print(len(class_list),'boxes converted in image',image_name)

        new_tree.write(self.output_file, encoding='utf-8', xml_declaration=True)

        zip_filename = self.output_file.split('.')[0] + '.zip'
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.output_file, arcname='output_xml_file.xml')
        print('XML file zipped')


print("Detecting corals and saving to annotation format.")
Det = Detect2Cvat(base_img_location, output_filename, weight_file, base_file)
Det.run()
print("Done detecting corals")
