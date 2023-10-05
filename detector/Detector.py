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

class Detector:
    DEFAULT_WEIGHT_FILE = "/mnt/c/20221113_amtenuis_cslics04/metadata/yolov5l6_20220223.pt"
    DEFAULT_ROOT_DIR = "/mnt/c/20221113_amtenuis_cslics04"
    DEFAULT_IMAGE_SIZE = 1280
    DEFAULT_CONFIDENCE_THREASHOLD = 0.25
    DEFAULT_IOU = 0.45
    DEFAULT_MAX_DET = 1000

    def __init__(self, 
                weights_file: str = DEFAULT_WEIGHT_FILE,
                root_dir: str = DEFAULT_ROOT_DIR,
                image_size: int = DEFAULT_IMAGE_SIZE,
                conf_thresh: float = DEFAULT_CONFIDENCE_THREASHOLD,
                iou: float = DEFAULT_IOU,
                agnostic: bool = True,
                max_det: int = DEFAULT_MAX_DET):
        self.weights_file = weights_file
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = load_model(weights_file)
        self.model.conf = conf_thresh
        self.model.iou = iou
        self.model.agnostic = agnostic
        self.model.max_det = max_det

        self.root_dir = root_dir
        self.classes, self.class_colours = get_classes_n_colours(root_dir)
        self.image_size = image_size

    def load_model(self, weights_file: str)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_file, trust_repo=True) # TODO make sure this can be run offline?
        model = model.to(self.device)
        model.eval()  # model into evaluation mode
        return model

    def get_classes_n_colours(root_dir):
        with open(os.path.join(root_dir, 'metadata','obj.names'), 'r') as f:
            classes = [line.strip() for line in f.readlines()]
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
        return classes, class_colours

    def save_text_predictions(predictions, imgname, txtsavedir, classes):
        """
        save predictions/detections into text file
        [x1 y1 x2 y2 conf class_idx class_name]
        """
        txtsavename = os.path.basename(imgname)
        txtsavepath = os.path.join(txtsavedir, txtsavename[:-4] + '_det.txt')

        # predictions [ pix pix pix pix conf class ]
        with open(txtsavepath, 'w') as f:
            for p in predictions:
                x1, y1, x2, y2 = p[0:4].tolist()
                conf = p[4]
                class_idx = int(p[5])
                class_name = classes[class_idx]
                f.write(f'{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.4f} {class_idx:g} {class_name}\n')
        return True


    def save_image_predictions(predictions, img, imgname, imgsavedir, class_colours, classes):
        """
        save predictions/detections on image
        """

        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])        
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), class_colours[classes[cls]], 2)
            cv.putText(img, f"{classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_colours[classes[cls]], 2)

        imgsavename = os.path.basename(imgname)
        imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_det.jpg')
        cv.imwrite(imgsave_path, img)
        return True


    def nms(pred, conf_thresh, iou_thresh, classes, max_det):
        """ perform class-agnostic non-maxima suppression on predictions 
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
        # keep = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, iou_thresh)

        # sort scores into descending order
        _, indices = torch.sort(scores, descending=True)

        # class-agnostic nms
        # iteratively adds the highest scoring detection to the list of final detections, 
        # calculates the IoU of this detection with all other detections, and 
        # removes any detections with IoU above the threshold. 
        # This process continues until there are no detections left.
        keep = []
        while indices.numel() > 0:
            # get the highest scoring detection
            i = indices[0]
            
            # add the detection to the list of final detections
            keep.append(i.item())

            # calculate the IoU highest scoring detection within all other detections
            if indices.numel() == 1:
                break
            else:
                overlaps = torchvision.ops.box_iou(boxes[indices[1:]], boxes[i].unsqueeze(0)).squeeze()

            # keep only detections with IOU below certain threshold
            indices = indices[1:]
            indices = indices[overlaps <= iou_thresh]

        return pred[keep, :]

    def detect(self, image):
        pred = model([image], size=self.img_size)
        return pred
    
    def run():
        sourceimages = os.path.join(root_dir, 'images_jpg')
        print('running Detector.py on:')
        print(f'source images: {sourceimages}')
        
        imgsave_dir = os.path.join(root_dir, 'detections', 'detections_images')
        os.makedirs(imgsave_dir, exist_ok=True)

        # where to save text detections
        txtsavedir = os.path.join(root_dir, 'detections', 'detections_textfiles')
        os.makedirs(txtsavedir, exist_ok=True)

        for i, imgname in enumerate(imglist):

            print(f'predictions on {i+1}/{len(imglist)}')
            # if i >= 3: # for debugging purposes
            #     break

            # load image
            try:
                img_bgr = cv.imread(imgname) # BGR
                img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB) # RGB

                pred = detect(image_rgb)# inference

                predictions = nms(pred.pred[0], self.model.conf, self.model.iou, self.classes, self.model.max_det)
                
                # save predictions as an image
                save_image_predictions(predictions, img_bgr, imgname, imgsave_dir, self.class_colours, self.classes)
                
                # save predictions as a text file (TODO make a funtion)
                save_text_predictions(predictions, imgname, txtsavedir, classes)
            except:
                print('unable to read image or do model prediction --> skipping')
                print(f'skipped: imgname = {imgname}')

        print('done')


Coral_Detector = Detector()

# model
# weightsfile = '/home/dorian/Code/cslics_ws/yolov5_coralspawn/weights/yolov5l6_20220223.pt'
weightsfile = "/mnt/c/20221113_amtenuis_cslics04/metadata/yolov5l6_20220223.pt"
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
# sourceimages = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/images/test'
# sourceimages = '/home/agkelpie/Code/cslics_ws/src/datasets/20221113_amtenuis_cslics01/images_jpg'
# root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/20221114_amtenuis_cslics01'
# root_dir = '/home/dorian/Data/cslics_2022_datasets/20221214_CSLICS04_images'
root_dir = "/mnt/c/20221113_amtenuis_cslics04"
sourceimages = os.path.join(root_dir, 'images_jpg')
batch_size = 1
# imgslist = sorted(os.listdir(sourceimages).endswidth(".png")) # assume correct input, probably should use glob
imglist = glob.glob(os.path.join(sourceimages, '*.jpg'))

print('running Detector.py on:')
print(f'source images: {sourceimages}')

# parameters
img_size = 1280
model.conf = 0.25
model.iou = 0.45
model.agnostic = True
model.max_det = 1000

# classes:
# read in classes
# root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000'
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

# where to save image detections
imgsave_dir = os.path.join(root_dir, 'detections', 'detections_images')
os.makedirs(imgsave_dir, exist_ok=True)

# where to save text detections
txtsavedir = os.path.join(root_dir, 'detections', 'detections_textfiles')
os.makedirs(txtsavedir, exist_ok=True)

# for each image:
for i, imgname in enumerate(imglist):

    print(f'predictions on {i+1}/{len(imglist)}')
    # if i >= 3: # for debugging purposes
    #     break

    # load image
    try:
        img_bgr = cv.imread(imgname) # BGR
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB) # RGB

        # inference
        
        pred = model([img_rgb], size=img_size)
        # import code
        # code.interact(local=dict(globals(), **locals()))
        # pred.print()
        # pred.save()
        # pred.pandas().xyxy[0] # save predictions as pandas dataframe object

        predictions = nms(pred.pred[0], model.conf, model.iou, classes, model.max_det)
        
        # save predictions as an image
        save_image_predictions(predictions, img_bgr, imgname, imgsave_dir, class_colours, classes)
        
        # save predictions as a text file (TODO make a funtion)
        save_text_predictions(predictions, imgname, txtsavedir, classes)
    except:
        print('unable to read image or do model prediction --> skipping')
        print(f'skipped: imgname = {imgname}')
        import code
        code.interact(local=dict(globals(), **locals()))
                               
    
# TODO actually make this a class and then a "have .run" method to peform detections
# TODO write a separate script that reads in these text files and generates a plot in post

print('done')

# import code
# code.interact(local=dict(globals(), **locals()))
    