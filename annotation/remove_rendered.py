# Dorian Tsai
# in preparation for compressing 2024 spawning data, which has rendered, clean jpg and json files in it
# need to downsize the folders, the renders/json files are not needed post-spawning
# script to delete all _rendered.jpg and .json files in folders/sub-folders

import os
import glob
from pathlib import Path
import sys

DELETE_RENDER = True
DELETE_JSON = False

# target_dir = '/media/dtsai/CSLICSOct24/cslics_october_2024'
target_dir = '/media/dtsai/CSLICSNov24/cslics_november_2024'

# absolute paths
img_list = sorted(Path(target_dir).rglob('*_rendered.jpg'))

if DELETE_RENDER:
    # deleting images
    print('Deleting images')
    for i, img_name in enumerate(img_list):

        print(f'img {i+1}/{len(img_list)}')
        try:
            os.remove(img_name)
            print(f'removed: {img_name}')
        except FileNotFoundError:
            print(f'not found: {img_name}')
        except OSError as e:
            print(f'error deleting file')

if DELETE_JSON:
    print('Deleting json files')
    json_list = sorted(Path(target_dir).rglob('*_boxes.json'))

    # deleting images
    print('Deleting jsons')
    for i, json_name in enumerate(json_list):

        print(f'img {i+1}/{len(json_list)}')
        try:
            os.remove(json_name)
            print(f'removed: {json_name}')
        except FileNotFoundError:
            print(f'not found: {json_name}')
        except OSError as e:
            print(f'error deleting file')

print('done')