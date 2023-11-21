#! /usr/bin/env python3

"""
use the trained yolov5 model, and run it on a given folder/sequence of images
"""

import os
import torch
import torchvision
import glob
# import numpy as np
# from PIL import Image as PILImage
import cv2 as cv
import time
import pickle
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import sys
sys.path.insert(0, '')
from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.Detector import Detector

import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import zipfile

class Surface_Detector(Detector):

    DEFAULT_META_DIR = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    DEFAULT_IMG_DIR = '/home/cslics04/20231018_cslics_detector_images_sample/surface'
    DEFAULT_SAVE_DIR = '/home/cslics04/images/surface'
    
    DEFAULT_DETECTOR_IMAGE_SIZE = 640
    DEFAULT_CONFIDENCE_THREASHOLD = 0.50
    DEFAULT_IOU = 0.45
    DEFAULT_MAX_IMG = 1000 # per image
    
    DEFAULT_YOLO8 = os.path.join('/home/cslics04/cslics_ws/src/ultralytics_cslics/weights', 'cslics_20230905_yolov8n_640p_amtenuis1000.pt')
    DEFAULT_OUTPUT_FILE = 'surface_detections.pkl'

                 
    def __init__(self,
                
                meta_dir: str = DEFAULT_META_DIR,
                img_dir: str = DEFAULT_IMG_DIR,
                save_dir: str = DEFAULT_SAVE_DIR,
                max_img: int = DEFAULT_MAX_IMG,
                img_size: int = DEFAULT_DETECTOR_IMAGE_SIZE,
                weights_file: str = DEFAULT_YOLO8,
                conf_thresh: float = DEFAULT_CONFIDENCE_THREASHOLD,
                iou: float = DEFAULT_IOU,
                output_file: str = DEFAULT_OUTPUT_FILE):
        
        self.weights_file = weights_file
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        if not torch.cuda.is_available():
            print('Note: torch cuda (GPU) is not available, so performance will be slower')
            
        self.model = YOLO(weights_file)
        self.conf = conf_thresh
        self.iou = iou

        self.output_file = output_file
        
        Detector.__init__(self, 
                          meta_dir = meta_dir,
                          img_dir = img_dir,
                          save_dir = save_dir,
                          max_img = max_img,
                          img_size=img_size)


    def nms(self, pred, conf_thresh, iou_thresh):
        """ perform class-agnostic non-maxima suppression on predictions 
        pred = [x1 y1 x2 y2 conf class] tensor
        # TODO handle the pred empty case!
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
        assuming detections have already been done (i.e. txtsavedir is full of txtfiles)
        convert textfiles and images into CoralImage and save data in pkl file,
        assuming textfiles in [[x1 y1 x2 y2 conf class_idx class_name] yolo formt
        """
        # read in each .txt file
        # txt_list = sorted(os.listdir(txtsavedir))
        txt_list = sorted(glob.glob(os.path.join(txtsavedir, '*.txt')))
        
        if len(txt_list) > 0:
            results = []
            for i, txt in enumerate(txt_list):
                print(f'importing detections {i+1}/{len(txt_list)}')
                with open(os.path.join(txtsavedir, txt), 'r') as f:
                    detections = f.readlines() # [x1 y1 x2 y2 conf class_idx class_name] \n
                detections = [det.rsplit() for det in detections]
                # corresponding image name:
                img_name = os.path.basename(txt[:-8]) + '.jpg' # + '.png'
                img_name = os.path.join(self.img_dir, img_name)
                CImage = CoralImage(img_name=img_name, # TODO absolute vs relative? # want to grab the metadata
                                    txt_name=txt,
                                    detections=detections)
                results.append(CImage)
                
            # sort results based on metadata capture time
            results.sort(key=lambda x: x.metadata['capture_time'])

            savefile = os.path.join(self.save_dir, save_file_name)
            with open(savefile, 'wb') as f:
                pickle.dump(results, f)
            return True
        else:
            print(f'ERROR: No txt files in {txtsavedir}')
            return False


    def show_image_predictions(self, predictions, img, SHOW=False):
        """
        show predictions/detections (assurming predeictions in yolo format) for an image
        """
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])        
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], 2)
            cv.putText(img, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[self.classes[cls]], 2)
            
        if SHOW:
            cv.show(img)
        return True


    # def prep_img(self, img_bgr):
    #     img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB) # RGB
    #     return img_rgb


    def detect(self, image):
        """
        return detections from a single rgb image (numpy array)
        passes prediction through nms and returns them in yolo format [x1 y1 x2 y2 conf class_idx class_name]
        """
        pred = self.model.predict(source=image,
                                  save=False,
                                  save_txt=False,
                                  save_conf=True,
                                  verbose=False,
                                  imgsz=self.img_size,
                                  conf=self.conf)
        boxes: Boxes = pred[0].boxes 
        pred = []
        for b in boxes:
            
            # hope to keep all variables on cuda/GPU for speed
            if torch.cuda.is_available():
                xyxyn = b.xyxyn[0]
                pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
            else:
                cls = int(b.cls)
                conf = float(b.conf)
                xyxyn = b.xyxyn.cpu().numpy()[0]
                x1n = xyxyn[0]
                y1n = xyxyn[1]
                x2n = xyxyn[2]
                y2n = xyxyn[3]  
                pred.append([x1n, y1n, x2n, y2n, conf, cls])
        
        # after iterating over boxes, make sure pred is on GPU if available (and a single tensor)
        if torch.cuda.is_available():
            pred = torch.tensor(pred, device="cuda:0")
        else:
            pred = torch.tensor(pred)
            
        # TODO should handle empty case!
        if len(pred) > 0:
            predictions = self.nms(pred, self.conf, self.iou)
        else:
            predictions = [] # empty/0 case
        return predictions
    

    def run(self):
        print('running Surface_Detector.py on:')
        print(f'source images: {self.img_dir}')
        imglist = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
        # where to save image and text detections
        imgsave_dir = os.path.join(self.save_dir, 'detection_images')
        os.makedirs(imgsave_dir, exist_ok=True)
        txtsavedir = os.path.join(self.save_dir, 'detection_textfiles')
        os.makedirs(txtsavedir, exist_ok=True)

        start_time = time.time()
        
        for i, imgname in enumerate(imglist):
            print(f'predictions on {i+1}/{len(imglist)}')
            if i >= self.max_img: # for debugging purposes
                break

            # load image
            try:
                img_rgb = self.prep_img_name(imgname)
            except:
                print('unable to read image --> skipping')
                predictions = []
            
            try: 
                predictions = self.detect(img_rgb)# inference
            except:
                print('no model predictions --> skipping')
                predictions = []
                
            # save predictions as an image
            self.save_image_predictions(predictions, img_rgb, imgname, imgsave_dir)
            # save predictions as a text file
            self.save_text_predictions(predictions, imgname, txtsavedir)

        end_time = time.time()
        duration = end_time - start_time
        print('run time: {} sec'.format(duration))
        print('run time: {} min'.format(duration / 60.0))
        print('run time: {} hrs'.format(duration / 3600.0))
        
        print(f'time[s]/image = {duration / len(self.img_list)}')
        
        print('done detection')
        self.convert_results_2_pkl(txtsavedir, self.output_file)
        print(f'results stored in {self.output_file} file')


def main():
    # meta_dir = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    # img_dir = '/home/cslics04/20231018_cslics_detector_images_sample/surface'
    # weights = '/home/cslics04/cslics_ws/src/ultralytics_cslics/weights/cslics_20230905_yolov8n_640p_amtenuis1000.pt'

    # Coral_Detector = Surface_Detector(weights_file=weights, meta_dir = meta_dir, img_dir=img_dir, max_img=5)
    meta_dir = '/home/java/Java/cslics'
    img_dir = '/home/java/Java/data/202212_aloripedes_500/images'
    weights = '/home/java/Java/cslics/cslics_surface_detectors_models/cslics_20230905_yolov8m_640p_amtenuis1000.pt'
    save_dir = '/home/java/Java/data/202212_aloripedes_500'
    Coral_Detector = Surface_Detector(meta_dir=meta_dir, img_dir=img_dir, save_dir=save_dir, weights_file=weights)
    Coral_Detector.run()

if __name__ == "__main__":
    main()


# import code
# code.interact(local=dict(globals(), **locals()))
    
############ XML #####################
#add in under main()
    base_file = "/home/java/Downloads/archive/annotations.xml"
    img_location = img_dir
    output_filename = "/home/java/Downloads/surface.xml"
    output_file = output_filename
    classes = ["Four-Eight-Cell Stage", "First Cleavage", "Two-Cell Stage", "Advanced Stage", "Damaged", "Egg"]

    tree = ET.parse(base_file)
    root = tree.getroot() 
    new_tree = ElementTree(Element("annotations"))
    # add version element
    version_element = ET.Element('version')
    version_element.text = '1.1'
    new_tree.getroot().append(version_element)
    # add Meta elements, (copy over from source_file)
    meta_element = root.find('.//meta')
    if meta_element is not None:
        new_meta_elem = ET.SubElement(new_tree.getroot(), 'meta')
        # copy all subelements of meta
        for sub_element in meta_element:
            new_meta_elem.append(sub_element)
        
    for i, image_element in enumerate(root.findall('.//image')):
        print(i,'images being processed')
        image_id = image_element.get('id')
        image_name = image_element.get('name')
        image_width = int(image_element.get('width'))
        image_height = int(image_element.get('height'))

        # create new image element in new XML
        new_elem = SubElement(new_tree.getroot(), 'image')
        new_elem.set('id', image_id)
        new_elem.set('name', image_name)
        new_elem.set('width', str(image_width))
        new_elem.set('height', str(image_height))
        
        image_file = os.path.join(img_location, image_name)
        img_bgr = cv.imread(image_file)
        
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        predictions = Coral_Detector.detect(img_rgb)

        for j, p in enumerate(predictions):
            try: 
                xyxy = p[0:4].tolist()
                label = classes[int(p[5].item())]
                xtl = min(xyxy[0],xyxy[2])*image_width
                xbr = max(xyxy[0],xyxy[2])*image_width
                ytl = min(xyxy[1],xyxy[3])*image_height
                ybr = max(xyxy[1],xyxy[3])*image_height
                box_elem = SubElement(new_elem, 'box')
                box_elem.set('label', label)
                box_elem.set('source', 'semi-auto')
                box_elem.set('occluded', '0')
                box_elem.set('xtl', str(xtl))
                box_elem.set('ytl', str(ytl))
                box_elem.set('xbr', str(xbr))
                box_elem.set('ybr', str(ybr))
                box_elem.set('z_order', '0')
            except:
                print(f'prediction {j} encountered problem p = {p}')
                import code
                code.interact(local=dict(globals(), **locals()))
            #import code
            #code.interact(local=dict(globals(), **locals()))

    new_tree.write(output_file, encoding='utf-8', xml_declaration=True)

    zip_filename = output_file.split('.')[0] + '.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname='output_xml_file.xml')
    print('XML file zipped')

    import code
    code.interact(local=dict(globals(), **locals()))
