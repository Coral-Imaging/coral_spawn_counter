#! /usr/bin/env python3

"""
read detections from ML model, saved as individual .txt files
plot them into time history
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

from coral_spawn_counter.CoralImage import CoralImage

# read them in line-by-line for each image
# save each image as list of annotations - maybe we can reuse the Image/Annotations?
# we can use CoralImage to hold image name and detections
# detectiosn are a list of pred = [x1 y1 x2 y2 conf class]


# read in classes
# root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000'
# root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/20221114_amtenuis_cslics01'
root_dir = "/mnt/c/20221113_amtenuis_cslics04"
with open(os.path.join(root_dir, 'metadata','obj.names'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]


# TODO put this into specific file, similar to agklepie project
# define class-specific colours
orange = [255, 128, 0] # four-eight cell stage
blue = [0, 212, 255] # first cleavage
purple = [170, 0, 255] # two-cell stage
yellow = [255, 255, 0] # advanced
brown = [144, 65, 2] # damaged
green = [0, 255, 00] # egg
class_colours = {classes[0]: orange,
                 classes[1]: blue,
                 classes[2]: purple,
                 classes[3]: yellow,
                 classes[4]: brown,
                 classes[5]: green}

# where images are saved:
imgsave_dir = os.path.join(root_dir, 'detections', 'detections_images')

# where text detections are asaved:
txtsavedir = os.path.join(root_dir, 'detections', 'detections_textfiles')

# read in each .txt file
txt_list = sorted(os.listdir(txtsavedir))

# for each txt name, open up and read
print(f'importing in detections for {root_dir}')
results = []
for i, txt in enumerate(txt_list):
    print(f'importing detections {i+1}/{len(txt_list)}')
    with open(os.path.join(txtsavedir, txt), 'r') as f:
        detections = f.readlines() # [x1 y1 x2 y2 conf class_idx class_name] \n
    detections = [det.rsplit() for det in detections]
    
    # corresponding image name:
    # TODO find corresponding image name in imgsave_dir
    # HACK for now, just truncate the detections name
    img_name = txt[:-8] + '.jpg' # + '.png' # NOTE only png has the metadata, jpgs were unable to carry over metadata?\
    img_name = os.path.join(root_dir, 'images_jpg', img_name)

    CImage = CoralImage(img_name=img_name, # TODO absolute vs relative? # want to grab the metadata
                        txt_name=txt,
                        detections=detections)
    # create CoralImage object for each set of detections
    # ultimately, will have a list of CoralImages
    results.append(CImage)
    
# sort results based on metadata capture time
results.sort(key=lambda x: x.metadata['capture_time'])

# save all variables to a file using pickle
savefile = os.path.join(root_dir, 'detection_results.pkl')
with open(savefile, 'wb') as f:
    pickle.dump(results, f)

print('done')
        
import code
code.interact(local=dict(globals(), **locals()))