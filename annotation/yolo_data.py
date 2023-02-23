#! /usr/bin/env python3

"""
 check yolo dataset
 class distribution from annotations
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# given location of dataset:
# image annotations (.txt files for each image name)
# and image directory

# for each image in the directory
# find the annotation file
# count each of the classes
# make a histogram wrt class count

# =========================================================

# location of dataset root directory (which houses images, labels)
root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000'

# read in classes
with open(os.path.join(root_dir, 'metadata','obj.names'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# read in list of image files
img_list = sorted(os.listdir(os.path.join(root_dir, 'images','train')))

# read in all annotations
ann_dir = os.path.join(root_dir, 'labels', 'train')
ann_list = sorted(os.listdir(ann_dir))

# remove annotations that do not appear in the img_list:

# img names from the list of annotations
img_ann = [ann[:-4] + '.jpg' for ann in ann_list]

# find img names from the annotations not in img_list
img_miss = set(img_list) ^ set(img_ann)
print(img_miss)
if len(img_miss) > 0:
    print('Warning: img_miss non-zero, len(annotations) inconsistent with len(images)')
# img_miss = list(img_miss)

# remove, annotation, because these should match-up 1:1
# for img in img_miss:
#     img_txt = img[:-4] + '.txt'
#     print(f'removing: {img_txt}')
#     os.remove(os.path.join(ann_dir, img_txt))

# count number of occurences of each class in the dataset
class_counts = {cls: 0 for cls in classes}
for ann_path in ann_list:
    with open(os.path.join(ann_dir, ann_path), 'r') as f:
        for line in f.readlines():
            cls = classes[int(line.strip().split()[0])]
            class_counts[cls] += 1

# plot counts as a histogram/distribution:
fig, ax = plt.subplots()
ax.barh(np.arange(len(classes)), list(class_counts.values()), align='center')
ax.set_yticks(np.arange(len(classes)))
ax.set_yticklabels(classes)
ax.invert_yaxis() # labels read top-to-bottom?
ax.set_xlabel('Number of occurences')
ax.set_title('Distribution of object classes in coral spawn dataset')


# save figure
plt.savefig('class_distribution_train_20220222.png', bbox_inches='tight')

plt.show()

print('done')

# import code
# code.interact(local=dict(globals(), **locals()))

