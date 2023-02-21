#! /usr/bin/env python3

# combine cvat 1.1 annotations

# list annotation files
# single output annotation file

import os
import xml.etree.ElementTree as ET
import glob


data_dir = '/home/dorian/Data/annotationfiles_combined'
ann_file = 'annotations.xml'
ann_files = glob.glob(data_dir + '/*/' + ann_file)

# ann1 = ann_files[0] # for now, just parse the first one
# print(ann1)
# for ann in ann_files:
# root1 = ET.parse(ann1).getroot()

# print('original root 1')
# for child in root1.iterfind('.//image'):
#     print(child.attrib['name'])
    
# root2 = ET.parse(ann_files[1]).getroot()

# for child in root2:
#     if child.tag == 'image':
#         root1.append(child)
        
# print('appended root2 to root1')
# for child in root1.iterfind('.//image'):
#     print(child.attrib['name'])

all_root = None
for ann in ann_files:
    root = ET.parse(ann).getroot()
    if all_root is None:
        all_root = root
    else:
        for child in root:
            if child.tag == 'image':
                all_root.append(child)

out_tree = ET.ElementTree(all_root)
out_ann = os.path.join(data_dir, 'annotations_combined.xml')
out_tree.write(out_ann)

# print:
# print(ET.tostring(data))

# for img in root.iterfind('.//image'):
#     print(img)
    
#     if all_root is None:
#         all_root = root
#         insertion_point = all_root.findall(".//image")
#     else:
#         insertion_point.extend(img)
            
# if all_root is not None:
#     print(ET.tostring(all_root))

        
    # tree = ET.ElementTree(ann)
    # root = tree.getroot()

    # if i == 0:
    #     all_root = root
    # if i > 0:
    #     all_root.extend(root)
        

# all_tree = ET.ElementTree(all_root)
# 


import code
code.interact(local=dict(globals(), **locals()))
        
    
    
