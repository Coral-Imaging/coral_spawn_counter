#! /usr/bin/env python3

# rename images by adding cslics_id to front of image names from cvat annotations format
import os
import shutil
import xml.etree.ElementTree as ET


# get cslics id
# get list of images in cslics images folder
# rename all images
# repeat for each cslics id within folder name

# dataset folder name
data_dir = '/home/dorian/Data/acropora_maggie_tenuis_dataset_100_renamed/combined_100'
print(data_dir)

# annotations file:
ann_file = 'annotations.xml'
ann_path = os.path.join(data_dir, 'metadata', ann_file)

# images folder:
img_dir = os.path.join(data_dir, 'images')

# find all image names with cslicsid at front:
img_names = sorted(os.listdir(img_dir))

# create a dictionary with old names: new names
img_dict = {}
for img_name in img_names:
    img_dict[img_name[9:]] = img_name

# find all image names in annotations file:
root = ET.parse(ann_path).getroot()
for i, child in enumerate(root):
    if child.tag == 'image':
        print(f"{i}: {child.attrib['name']}")
        
        # update name
        # TODO if name does not already have cslics## at start:
        name = child.attrib['name']
        if name[0:6] == 'cslics':
            print('image name already updated')
        else: 
            child.attrib['name'] = img_dict[name]
            print(f'updating annotation name from {name} to: {img_dict[name]}')

# write out modifications to existing annotations.xml file
out_tree = ET.ElementTree(root)
out_ann = os.path.join(data_dir, 'metadata/annotations_updated.xml')
out_tree.write(out_ann)
print('done')

import code
code.interact(local=dict(globals(), **locals()))
        
    
    
