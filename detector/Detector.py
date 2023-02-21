#! /usr/bin/env python3

"""
use the trained yolov5 model, and run it on a given folder/sequence of images
"""

import os
import torch
import glob
import numpy as np
from PIL import Image as PILImage

# can probably follow along detect.py on how to run model
# location of model
# load model via pytorch
# data folder location
# output folder/file setup - should probably be a .csv due to Excel compatibility


# model
weightsfile = '/home/dorian/Code/cslics_ws/src/coral_spawn_counter/detector/yolov5l6_epoch1000.pt'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = torch.load((weightsfile), map_location='cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weightsfile)
# model = (model.get('ema') or model['model']).to(device).float()
model = model.to(device)
# get names
# if hasattr(model, 'names') and isinstance(model.names, (list, tuple)):
#     model.names = dict(enumerate(model.names)) # convert to dict
# else:
#     print('TODO: get names from names file')
model.eval() # model into evaluation mode
    
# source images
sourceimages = '/home/dorian/Data/acropora_maggie_tenuis_dataset_100_renamed/202211_atenuis_100/images'
batch_size = 1
# imgslist = sorted(os.listdir(sourceimages).endswidth(".png")) # assume correct input, probably should use glob
imglist = glob.glob(os.path.join(sourceimages, '*.png'))

# parameters
img_size = 1280
model.conf = 0.25
model.iou = 0.45
model.agnostic = True
model.max_det = 1000

# for each image:
for imgname in imglist:
    # load image
    img = PILImage.open(imgname)

    # inference
    pred = model([img], size=img_size)
    
    # nms
    # pred = non_max_suppression(pred, 
    #                            conf_thresh,
    #                            iou_thresh,
    #                            classes,
    #                            agnostic_nms,
    #                            max_det=max_det)
    pred.print()
    pred.save()
    pred.pandas().xyxy[0] # predictions as pandas dataframe object
    
    

# output
save_img = '~/Data/acropora_maggie_tenuis_dataset_100_renamed/202211_atenuis_100/detections'
# save_txt = # TODO

import code
code.interact(local=dict(globals(), **locals()))
    