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

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from annotation.Annotations import Annotations
import torchvision.transforms as T
from PIL import Image as PILImage
from torchvision.transforms import functional as tvtransfunc


CLASS_DICT = {"Background": 0,
              "Egg": 1,
              "First Cleavage": 2,
              "Two-Cell Stage": 3,
              "Four-Eight Cell Stage": 4,
              "Advanced": 5,
              "Damaged": 6}
CLASS_COLORS = ['black', 'green', 'blue', 'purple', 'orange', 'yellow', 'brown']
# TODO specific colours for opencv drawing

class CoralSpawnDataset(Dataset):
    def __init__(self, ann_file: str, img_dir: str, transforms=None):

        self.ann_file = ann_file # absolute filepath
        self.img_dir = img_dir # absolute filepath to image directory
        Ann = Annotations(self.ann_file, self.img_dir)
        self.annotations = Ann.annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        given an index, return the corresponding image and sample from the dataset
        output as tensors
        """
        
        if torch.is_tensor(index):
            idx = idx.tolist()
        
        # get image, full file path
        img_name = self.annotations[index].img_name
        image = PILImage.open(img_name).convert("RGB")
        
        # number of annotations in the given image
        nobj = len(self.annotations[index].regions)
        
        if nobj > 0:
            poly = []
            labels = []
            # if there are annotations, get labels and annotations
            for region in self.annotations[index].regions:
                x, y = region.shape.exterior.coords.xy
                # bounding box
                xmin = min(x)
                xmax = max(x)
                ymin = min(y)
                ymax = max(y)
                poly.append([xmin, ymin, xmax, ymax])
                label = CLASS_DICT[region.class_name] # convert string name into class number
                labels.append(label)
                
            poly = torch.as_tensor(poly, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        else:
            # if no annotations (eg, a negative image)
            poly = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0), dtype=torch.int64)
            
        sample = {}
        sample['labels'] = labels
        sample['poly'] = poly
        
        # apply transforms to image and sample
        if self.transforms:
            image, sample = self.transforms(image, sample)
            
        return image, sample

# NOTE this is done because the built-in PyTorch Compose transforms function
# only accepts a single (image/tensor) input. To operate on both the image
#  as well as the sample/target, we need a custom Compose transform function
class Compose(object):
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    """ convert ndarray to sample in pytorch tensors"""
    
    def __call__(self, image, sample):
        image = tvtransfunc.to_tensor(image)
        image = torch.as_tensor(image, dtype=torch.float32)
        
        poly = sample['poly']
        if not torch.is_tensor(poly):
            poly = torch.from_numpy(poly)
            poly = torch.as_tensor(poly, dtype=torch.float32)
        sample['poly'] = poly

        return image, sample
    

class Rescale(object):
    """ rescale image to given size """
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, image, sample=None):
        
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image) # convert to PIL image
            
        h, w = image.size[:2]
        
        if isinstance(self.output_size, int):
            # assume square image
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        
        # apply resize to image
        img = T.Resize((new_w, new_h))(image) # only works on PIL images
        
        # apply resize to sample polygons
        if sample is not None:
            poly = sample['poly'] # [xmin ymin xmax ymax]
            delX = float(new_w) / float(w)
            delY = float(new_h) / float(h)
            if len(poly) > 0:
                poly[:, 0] = poly[:, 0] * delY
                poly[:, 1] = poly[:, 1] * delX
                poly[:, 2] = poly[:, 2] * delY
                poly[:, 3] = poly[:, 3] * delX
                sample['poly'] = np.float64(poly)
        
            return img, sample
        else:
            return img
        
    # TODO random horizontal flip
    # TODO random vertical flip
    # TODO random blur
    # TODO then test to make sure each transform works, then try running training pipeline
    
    
if __name__ == "__main__":
    
    print('CoralSpawnDataset.py')
    
    data_dir = '/home/dorian/Data/acropora_maggie_tenuis_dataset_100_renamed/combined_100'
    img_dir = os.path.join(data_dir, 'images')
    ann_file = os.path.join(data_dir, 'metadata/annotations_updated.xml')
    CSData = CoralSpawnDataset(ann_file, img_dir)
    
    print(f'Length of CSData: {len(CSData)}')
    
    idx = 10
    print(CSData[idx])
    
    # image:
    print('showing PIL image:')
    CSData[idx][0].show()
    
    print('labels: {}'.format(CSData[idx][1]['labels']))
    print('poly: {}'.format(CSData[idx][1]['poly']))
    
    import code
    code.interact(local=dict(globals(), **locals()))