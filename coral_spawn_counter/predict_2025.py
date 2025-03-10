#!/usr/bin/env python3

# subsurface detector just neural network 
# run it on all the folders/sub-folders, save results as .txt file in target_dir

# TODO - need to adapt code to run this on the HPC! Much faster and parallelisable

import json
import os
import glob
from pathlib import Path
from ultralytics import YOLO
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import time
from Detector import Detector
from datetime import datetime

print('model inference')

weights_path = '/home/dtsai/Data/cslics_datasets/models/cslics_subsurface_20250205_640p_yolov8n.pt'

# NOTE: img_dir can be high-level directory with nested sub-folders of images
# img_dir = '/media/dtsai/CSLICSOct24/cslics_october_2024/20241023_spawning/10000000f620da42/2024-10-28'
# img_dir = '/media/dtsai/CSLICSNov24/cslics_november_2024/100000009c23b5af'
# img_dir = '/media/dtsai/CSLICSNov24/cslics_november_2024/100000000846a7ff'
img_dir = '/media/dtsai/CSLICSOct24/cslics_october_2024/10000000f620da42'
print(f'img_dir = {img_dir}')


# model_name = os.path.basename(weights_path)

# NOTE: the intention is to have model-specific save directories underneath the CSLICS UUID folder
# save_dir = '/media/dtsai/CSLICSOct24/cslics_october_2024/20241023_spawning/10000000f620da42'
# save_dir = '/media/dtsai/CSLICSOct24/cslics_october_2024/detections/10000000f620da42/2024-10-28'
# save_dir = os.path.join('/media/dtsai/CSLICSNov24/cslics_november_2024/detections/100000009c23b5af/',Path(weights_path).stem)
# save_dir = os.path.join('/media/dtsai/CSLICSNov24/cslics_november_2024/detections/100000000846a7ff/',Path(weights_path).stem)
save_dir = os.path.join('/media/dtsai/CSLICSOct24/cslics_october_2024/detections/10000000f620da42',Path(weights_path).stem)
print(f'save_dir = {save_dir}')
classes = ['coral']

red = [0, 0, 255]
orange = [0, 128, 225]
cyan = [225, 212, 0]
purple = [255, 0, 170]
yellow = [0, 255, 255]
brown = [2, 65, 144]
green = [0, 255, 0]
dark_purple = [112, 21, 97]

class_colours = {classes[0]: red}

# def save_image_predictions(self, predictions, img, imgname, imgsavedir, BGR=False, quality=50):


def save_image_predictions_bb(predictions, imgname, imgsavedir, class_colours, classes):
    """
    save predictions/detections (assuming predictions in yolo format) on image as bounding box
    """
    FONT_SIZE = 2
    FONT_THICK = 2
    BOX_THICK = 2
    quality = 25
    
    img = cv.imread(imgname) # BGR
    imgw, imgh = img.shape[1], img.shape[0]
    for p in predictions:
        x1, y1, x2, y2 = p[0:4].tolist()
        conf = p[4]
        cls = int(p[5])
        #extract back into cv lengths
        x1 = x1*imgw
        x2 = x2*imgw
        y1 = y1*imgh
        y2 = y2*imgh        
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), class_colours[classes[cls]], BOX_THICK) #cv2.rectangle(image, start_point, end_point, color, thickness)
        cv.putText(img, f"{classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, class_colours[classes[cls]], FONT_THICK) #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])  
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_det.jpg')
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    cv.imwrite(imgsave_path, img, encode_param)
    return True

def save_txt_predictions_bb(predictions, imgname, txtsavedir):
    """
    save predictions/detections [xn1, yn1, xn2, yn2, conf, cls]) as bounding box (x and y values normalised)
    """
    imgsavename = os.path.basename(imgname)
    txt_save_path = os.path.join(txtsavedir, imgsavename[:-4] + '_det.txt')
    with open(txt_save_path, "w") as file:
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])
            line = f"{x1} {y1} {x2} {y2} {cls} {conf}\n"
            file.write(line)


# TODO should convert this whole script into an object and have weights_path as a property
def save_json_predictions_bb(predictions, imgname, txtsavedir, weights_path, current_datetime):
    """
    save predictions as a bounding box (x, y normalised values)
    """
    imgsavename = os.path.basename(imgname)
    json_save_path = os.path.join(txtsavedir, imgsavename[:-4] + '_det.json')

    # convert pytorch tensor to dictionary
    predictions_dict = {"model_name": Path(weights_path).stem,
                        "date run": current_datetime,
                        "detections [xn1, yn1, xn2, yn2, conf, cls]": predictions.tolist()}

    with open(json_save_path, 'w') as f:
        json.dump(predictions_dict, f, indent=4)

# get date-time for saving into json files
current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# load model
print(f'load model: {weights_path}')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_path).to(device)


# load detector functions
# detector = Detector()
# get predictions


print(f'fetching image list in all subfolders from: {img_dir}')
img_list = sorted(Path(img_dir).rglob('*_clean.jpg'))
print(f'number of images: {len(img_list)}')

# output dir
imgsave_dir = os.path.join(save_dir, 'detections_images')
txtsave_dir = os.path.join(save_dir, 'detections_txt')

os.makedirs(imgsave_dir, exist_ok=True)
os.makedirs(txtsave_dir, exist_ok=True)

SAVE_IMG = False # care - operating on large number of images, can run out of space quickly
SAVE_TXT = True
MAX_IMG = 100000
IOU_THRESH = 0.3
MAX_DET = 1000
# SKIP_DET = True

start_time = time.time()
for i, img_name in enumerate(img_list):
    print(f'img {i+1}/{len(img_list)}')
    if i >= MAX_IMG:
        print(f'MAX IMG LIMIT REACHED - STOPPING INFERENCE')
        break
    
    # check if detection file already exists?
    # if SKIP_DET:
    #     continue
    # else:
    results = model.predict(source=img_name, iou=IOU_THRESH, agnostic_nms=True, max_det=MAX_DET)
    boxes = results[0].boxes 
    pred = []
    for b in boxes:
        if torch.cuda.is_available():
            xyxyn = b.xyxyn[0]
            pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
    predictions = torch.tensor(pred)

    if SAVE_IMG or SAVE_TXT:
        # walk through source directory
        rel_path = os.path.relpath(os.path.dirname(img_name), img_dir)

    if SAVE_IMG:
        os.makedirs(os.path.join(imgsave_dir, rel_path), exist_ok=True)
        save_image_predictions_bb(predictions, img_name, os.path.join(imgsave_dir, rel_path), class_colours, classes)

    if SAVE_TXT:
        os.makedirs(os.path.join(txtsave_dir, rel_path), exist_ok=True)
        save_txt_predictions_bb(predictions, img_name, os.path.join(txtsave_dir, rel_path))
        save_json_predictions_bb(predictions, img_name, os.path.join(txtsave_dir, rel_path), weights_path, current_datetime )

print('Done')

end_time = time.time()
duration = end_time - start_time
print('run time: {} sec'.format(duration))
print('run time: {} min'.format(duration / 60.0))
print('run time: {} hrs'.format(duration / 3600.0))

print(f'time[s]/image = {duration / len(img_list)}')


import code
code.interact(local=dict(globals(), **locals()))
