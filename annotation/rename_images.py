#! /usr/bin/env python3

# rename images by adding cslics_id to front of image names

import os
import shutil

# TODO: should update camera trigger code to add cslics_id to image name

# get cslics id
# get list of images in cslics images folder
# rename all images
# repeat for each cslics id within folder name

# dataset folder name
# data_dir = '/home/dorian/Data/acropora_maggie_tenuis_dataset_100/20221114_AMaggieTenuis'
# data_dir = '/home/agkelpie/Data/RRAP_2022_NovSpawning/acropora_maggie_tenuis_dataset/20221114_AMaggieTenuis'
data_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Dec_Spawning/20221214_aloripedes_cslics07'
print(data_dir)

# get all cslics within given dataset
# cslics_ids = os.listdir(data_dir)
cslics_ids = ['cslics07']
print(cslics_ids)


for cs in cslics_ids:
    
    img_dir = 'images_jpg'
    img_path = os.path.join(data_dir, img_dir)
    
    img_names = os.listdir(img_path)
    
    for name in img_names:
        # assume .png at end of name, which we first remove
        # then append _cslics##.png
        new_name = cs + '_' + name
        shutil.move(os.path.join(img_path, name), 
                    os.path.join(img_path, new_name))
        
print('done')