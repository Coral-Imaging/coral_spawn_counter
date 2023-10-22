#! /usr/bin/env python3

"""
Helper file with some functions that are used by all detectors
"""

import os
import cv2 as cv

def save_image_predictions(predictions, imgname, imgsavedir, class_colours, classes):
    """
    save predictions/detections (assuming predictions in yolo format) on image
    """
    img = cv.imread(imgname)
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
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), class_colours[classes[cls]], 2)
        cv.putText(img, f"{classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_colours[classes[cls]], 2)

    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, img)
    return True

def convert_to_decimal_days(dates_list, time_zero=None):
    if time_zero is None:
        time_zero = dates_list[0]  # Time zero is the first element date in the list
    else:
        time_zero = time_zero
        
    decimal_days_list = []

    for date in dates_list:
        time_difference = date - time_zero
        decimal_days = time_difference.total_seconds() / (60 * 60 * 24)
        decimal_days_list.append(decimal_days)

    return decimal_days_list

def get_classes(root_dir):
    """
    get the classes from a metadata/obj.names file
    classes = [class1, class2, class3 etc.]
    """
    #TODO: make a function of something else, used in both detectors
    with open(os.path.join(root_dir, 'metadata','obj.names'), 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
    

def set_class_colours(classes):
    """
    set classes to specific colours using a dictionary
    """
    #TODO: make a function of something else, used in both detectors
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
    return class_colours



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