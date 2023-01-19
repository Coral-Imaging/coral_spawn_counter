#! /usr/env/bin python3

# check annotations
# check overlap
# check all labels are relevant strings
# check all relevant types

# plot relative distributions of labels!

import xml.etree.ElementTree as ET
import os

# data directory:
data_dir = '/home/dorian/Data/acropora_maggie_tenuis_dataset_100_renamed/combined_100'
print(data_dir)

ann_file = os.path.join(data_dir, 'metadata', 'annotations_updated.xml')

root = ET.parse(ann_file).getroot()

for child in root:
    if child.tag == 'image':
        print(child.attrib['name'])
        for anno in child:
            print(anno.attrib['label'])

