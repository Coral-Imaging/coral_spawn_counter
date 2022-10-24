#! /usr/bin/env python3

""" export cvat annotation format """

# assume annotation file already exists (we've uploaded the data to cvat and exported a bare annotations file)

import os
import xml.etree.ElementTree as ET

data_dir = '/home/dorian/Code/cslics_ws/src/coral_spawn_counter/test_cslics_data'
cvat_file = 'annotations.xml'



# ann_file = os.path.join(data_dir, cvat_file)

# parse cvat annotations file
tree = ET.ElementTree(file=os.path.join(data_dir, cvat_file))
root = tree.getroot()

print('Root: ')
print(root)
# read data from xml file
# with open(os.path.join(data_dir, cvat_file)) as f:
#     data = f.read()

# pass data xml to xml parser of beautiful soup
# xml_data = BeautifulSoup(data, 'xml')

output_cvat_file = 'annotations_mod.xml'
add_box = {'label': 'sphere', 
           'occluded': '0',
           'source': 'manual', 
           'xbr': '2100',
           'xtl': '2000', 
           'ybr': '2100',
           'ytl': '2000', 
           'z_order': '0'}

for elem in root.iterfind('.//image'):
    print (elem.attrib)

    # import code
    # code.interact(local=dict(globals(), **locals()))


    # create new element
    box_elem = ET.SubElement(elem, 'box', add_box)
    # elem.append(box_elem)
    tree.write(output_cvat_file)



import code
code.interact(local=dict(globals(), **locals()))


