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

from coral_spawn_counter.CoralImage import CoralImage
import pickle
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from Detector_helper import get_classes, set_class_colours, save_text_predictions, save_image_predictions

class Surface_Detector:
    DEFAULT_ROOT_DIR = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04"
    DEFAULT_IMAGE_SIZE = 640
    DEFAULT_CONFIDENCE_THREASHOLD = 0.25
    DEFAULT_IOU = 0.45
    DEFAULT_MAX_DET = 1000
    DEFAULT_SOURCE_IMAGES = os.path.join(DEFAULT_ROOT_DIR, 'images_jpg')
    DEFAULT_YOLO8 = os.path.join("/home/java/Java/from_PC", "cslics_20230905_yolov8m_640p_amtenuis1000.pt")

    def __init__(self,
                weights_file: str = DEFAULT_YOLO8,
                root_dir: str = DEFAULT_ROOT_DIR,
                source_img_folder: str = DEFAULT_SOURCE_IMAGES,
                image_size: int = DEFAULT_IMAGE_SIZE,
                conf_thresh: float = DEFAULT_CONFIDENCE_THREASHOLD,
                iou: float = DEFAULT_IOU,
                max_det: int = DEFAULT_MAX_DET):
        self.weights_file = weights_file
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = YOLO(weights_file)
        self.conf = conf_thresh
        self.iou = iou
        self.max_det = max_det

        self.root_dir = root_dir
        self.classes = get_classes(self.root_dir)
        self.class_colours = set_class_colours(self.classes)
        self.img_size = image_size

        self.sourceimages = source_img_folder


    def nms(self, pred, conf_thresh, iou_thresh, classes, max_det):
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


    def convert_results_2_pkl(self, txtsavedir, save_file_name):
        """
        convert textfiles and images into CoralImage and save data in pkl file,
        assuming textfiles in [[x1 y1 x2 y2 conf class_idx class_name] yolo formt
        """
        # read in each .txt file
        txt_list = sorted(os.listdir(txtsavedir))

        results = []
        for i, txt in enumerate(txt_list):
            print(f'importing detections {i+1}/{len(txt_list)}')
            with open(os.path.join(txtsavedir, txt), 'r') as f:
                detections = f.readlines() # [x1 y1 x2 y2 conf class_idx class_name] \n
            detections = [det.rsplit() for det in detections]
            # corresponding image name:
            img_name = txt[:-8] + '.jpg' # + '.png'
            img_name = os.path.join(self.root_dir, 'images_jpg', img_name)
            CImage = CoralImage(img_name=img_name, # TODO absolute vs relative? # want to grab the metadata
                                txt_name=txt,
                                detections=detections)
            results.append(CImage)
            
        # sort results based on metadata capture time
        results.sort(key=lambda x: x.metadata['capture_time'])

        savefile = os.path.join(self.root_dir, save_file_name)
        with open(savefile, 'wb') as f:
            pickle.dump(results, f)


    def show_image_predictions(self, predictions, img):
        """
        show predictions/detections (assurming predeictions in yolo format) for an image
        """
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])        
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], 2)
            cv.putText(img, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[self.classes[cls]], 2)
        cv.show(img)


    def prep_img(self, img_name):
        """
        from an img name, load the image into the correct format for dections (rgb)
        """
        img_bgr = cv.imread(img_name) # BGR
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB) # RGB
        return img_rgb


    def detect(self, image):
        """
        return detections from a single rgb image, 
        passes prediction through nms and returns them in yolo format [x1 y1 x2 y2 conf class_idx class_name]
        """
        pred = self.model.predict(source=image,
                                      save=False,
                                      save_txt=False,
                                      save_conf=True,
                                      imgsz=self.img_size,
                                      conf=self.conf)
        boxes: Boxes = pred[0].boxes 
        pred = []
        for b in boxes:
            cls = int(b.cls)
            conf = float(b.conf)
            xyxyn = b.xyxyn.numpy()[0]
            x1n = xyxyn[0]
            y1n = xyxyn[1]
            x2n = xyxyn[2]
            y2n = xyxyn[3]  
            pred.append([x1n, y1n, x2n, y2n, conf, cls])
        pred = torch.tensor(pred)
        predictions = self.nms(pred, self.conf, self.iou, self.classes, self.max_det)
        return predictions
    

    def run(self):
        print('running Surface_Detector.py on:')
        print(f'source images: {self.sourceimages}')
        imglist = glob.glob(os.path.join(self.sourceimages, '*.jpg'))
        # where to save image and text detections
        imgsave_dir = os.path.join(self.root_dir, 'detections', 'detections_images')
        os.makedirs(imgsave_dir, exist_ok=True)
        txtsavedir = os.path.join(self.root_dir, 'detections', 'detections_textfiles')
        os.makedirs(txtsavedir, exist_ok=True)

        for i, imgname in enumerate(imglist):
            print(f'predictions on {i+1}/{len(imglist)}')
            if i >= self.max_det: # for debugging purposes
            #    import code
            #    code.interact(local=dict(globals(), **locals()))
                break

            # load image
            try:
                img_rgb = self.prep_img(imgname)
                predictions = self.detect(img_rgb)# inference
                # save predictions as an image
                save_image_predictions(predictions, imgname, imgsave_dir, self.class_colours, self.classes)
                # save predictions as a text file
                save_text_predictions(predictions, imgname, txtsavedir, self.classes)
            except:
                print('unable to read image or do model prediction --> skipping')
                print(f'skipped: imgname = {imgname}')
                #import code
                #code.interact(local=dict(globals(), **locals()))

        print('done detection')

        pkl_file = 'detection_results1.pkl'
        self.convert_results_2_pkl(txtsavedir, pkl_file)
        print(f'results stored in {pkl_file} file')



def main():
    # weightsfile = '/home/dorian/Code/cslics_ws/yolov5_coralspawn/weights/yolov5l6_20220223.pt'
    # weightsfile = "/mnt/c/20221113_amtenuis_cslics04/metadata/yolov5l6_20220223.pt"
    #weightsfile = "/mnt/c/20221113_amtenuis_cslics04/metadata/yolov5l6_20220223.pt"
    # root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/20221114_amtenuis_cslics01'
    # root_dir = '/home/dorian/Data/cslics_2022_datasets/20221214_CSLICS04_images'
    root_dir = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04"
    source_img_folder = os.path.join(root_dir, 'images_jpg')
    yolo_weights = "/home/java/Java/from_PC/cslics_20230905_yolov8m_640p_amtenuis1000.pt"

    Coral_Detector = Surface_Detector(weights_file=yolo_weights, root_dir = root_dir, source_img_folder=source_img_folder, max_det=10000)
    Coral_Detector.run()

if __name__ == "__main__":
    main()


# import code
# code.interact(local=dict(globals(), **locals()))
    