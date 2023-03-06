#! /usr/bin/env python3

"""
read detections from ML model, saved as individual .txt files
plot them into time history
"""

import os
import matplotlib.pyplot as plt

from coral_spawn_counter.CoralImage import CoralImage



# read them in line-by-line for each image
# save each image as list of annotations - maybe we can reuse the Image/Annotations?
# we can use CoralImage to hold image name and detections
# detectiosn are a list of pred = [x1 y1 x2 y2 conf class]





# read in classes
root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000'
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
imgsave_dir = os.path.join(root_dir, 'images', 'detections_images')


# where text detections are asaved:
txtsavedir = os.path.join(imgsave_dir, 'detections_textfiles')

# read in each .txt file
txt_list = sorted(os.listdir(txtsavedir))

# for each txt name, open up and read
print('importing in detections')
results = []
for txt in txt_list:
    with open(os.path.join(txtsavedir, txt), 'r') as f:
        # [x1 y1 x2 y2 conf class_idx class_name] \n
        detections = f.readlines()
    
    detections = [det.split() for det in detections]
    
    # corresponding image name:
    # TODO find corresponding image name in imgsave_dir
    # HACK for now, just truncate the detections name
    img_name = txt[:-8] + '.png' # NOTE only png has the metadata, jpgs were unable to carry over metadata?
    
    CImage = CoralImage(img_name=os.path.join(root_dir, 'images_png', img_name), # TODO absolute vs relative? # want to grab the metadata
                        txt_name=txt,
                        detections=detections)
    # create CoralImage object for each set of detections
    # ultimately, will have a list of CoralImages
    results.append(CImage)
    
# sort results based on metadata capture time
results.sort(key=lambda x: x.metadata['capture_time'])

# get counts as arrays:
print('getting counts from detections')
count_eggs = []
count_first = []
capture_time_str = []
for res in results:
    # create a list of strings of all the detections for the given image
    counted_classes = [det[6] for det in res.detections]
    
    # do list comprehensions on counted_classes  # TODO consider putting this as a function into CoralImage/detections
    count_eggs.append(counted_classes.count('Egg'))
    count_first.append(counted_classes.count('First')) # TODO fix class names to be all one continuous string (no spaces)
    capture_time_str.append(res.metadata['capture_time'])
    

# parse capture_time into datetime objects so we can sort them
from datetime import datetime
capture_times = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time_str]
# count_eggs = [res.det for res in results]


# plot those detections into a matplotlib graph
plt.plot(capture_times, count_eggs, label='Egg')
plt.plot(capture_times, count_first, label='First')

plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Cell Counts over Time')
plt.legend()

plt.show()



print('done')
        
import code
code.interact(local=dict(globals(), **locals()))