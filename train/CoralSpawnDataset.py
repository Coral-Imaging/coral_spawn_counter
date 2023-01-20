#! /usr/bin/env python3

"""
CoralSpawnDataset object for training the coral spawn and larvae counter    
"""

# TODO CoralSpawnDataset 
# inherit from Annotations.py
# init
# get_item
# transforms
# len

import torch
from torch.utils.data import Dataset, DataLoader
from annotation import Annotations

class CoralSpawnDataset(Dataset):
    def __init__(self, ann_file: str, img_dir: str):

        self.ann_file = ann_file # absolute filepath
        self.img_dir = img_dir # absolute filepath to image directory
        Ann = Annotations(self.ann_file, self.img_dir)
        self.annotations = Ann.annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        return self.data_list[index]


