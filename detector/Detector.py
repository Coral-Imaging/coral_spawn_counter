#! /usr/bin/env python3

"""
use the trained yolov5 model, and run it on a given folder/sequence of images
"""

import os
import torch
import torchvision
import glob
import numpy as np
from PIL import Image as PILImage
import cv2 as cv


# can probably follow along detect.py on how to run model
# location of model
# load model via pytorch
# data folder location
# output folder/file setup - should probably be a .csv due to Excel compatibility

def nms(pred, conf_thresh, iou_thresh, classes, max_det):
    """ perform non-maxima suppression on predictions 
    pred = [x1 y1 x2 y2 conf class] tensor
    """

    # Checks
    assert 0 <= conf_thresh <= 1, f'Invalid Confidence threshold {conf_thresh}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thresh <= 1, f'Invalid IoU {iou_thresh}, valid values are between 0.0 and 1.0'
    # if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
    #     prediction = prediction[0]  # select only inference output

    # conf = object confidence * class_confidence
    pred = pred[pred[:, 4] > conf_thresh]
    boxes = pred[:, :4]
    scores = pred[:, 4]
    keep = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, iou_thresh)
    return pred[keep, :]

# model
weightsfile = '/home/agkelpie/Code/cslics_ws/src/yolov5/weights/yolov5l6_20220223.pt'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = torch.load((weightsfile), map_location='cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weightsfile, trust_repo=True) # TODO make sure this can be run offline?
# model = (model.get('ema') or model['model']).to(device).float()
model = model.to(device)
# get names
# if hasattr(model, 'names') and isinstance(model.names, (list, tuple)):
#     model.names = dict(enumerate(model.names)) # convert to dict
# else:
#     print('TODO: get names from names file')
model.eval() # model into evaluation mode
    
# source images
sourceimages = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/images/test'
batch_size = 1
# imgslist = sorted(os.listdir(sourceimages).endswidth(".png")) # assume correct input, probably should use glob
imglist = glob.glob(os.path.join(sourceimages, '*.jpg'))

# parameters
img_size = 1280
model.conf = 0.25
model.iou = 0.45
model.agnostic = True
model.max_det = 1000

# classes:
# read in classes
root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000'
with open(os.path.join(root_dir, 'metadata','obj.names'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# for each image:
for imgname in imglist:
    # load image
    img = cv.imread(imgname)

    # inference
    pred = model([img], size=img_size)
    
    pred.print()
    pred.save()
    pred.pandas().xyxy[0] # predictions as pandas dataframe object
    

    # nms - possibly import from yolov5? but for now, just use torch's built-in
    # pred = non_max_suppression(pred, 
    #                            conf_thresh,
    #                            iou_thresh,
    #                            classes,
    #                            agnostic_nms,
    #                            max_det=max_det)

    predictions = nms(pred.pred[0], model.conf, model.iou, classes, model.max_det)

    for p in predictions:
        x1, y1, x2, y2 = p[0:4].tolist()
        conf = p[4]
        cls = int(p[5])
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv.putText(img, f"{cls}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv.imshow("Coral spawn detections", img)
    cv.waitKey(0)
    
    import code
    code.interact(local=dict(globals(), **locals()))
    cv.destroyAllWindows()
    
    

# output
save_img = '/home/agkelpie/Code/cslics_ws/src/coral_spawn_counter/detector/runs'
# create/draw on image

# save_txt = # TODO


    