from ultralytics import YOLO
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics.engine.results import Results
 
# Load a model
model = YOLO('/home/dorian/Data/cslics_desktop/models/20241006_cslics_desktop_larvae_yolov8n_1280p.pt')  # load a custom model

# Validate the model
# metrics = model.val(data='cslics_desktop_embryogenesis_2023.yaml',
#             imgsz=1280,)  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category


# inference on a given folder:

out_dir = '/home/dorian/Data/cslics_desktop/data/2023_combined_larvae/split/detections/test'
os.makedirs(out_dir, exist_ok=True)

img_dir = '/home/dorian/Data/cslics_desktop/data/2023_combined_larvae/split/images/test'
img_list = sorted(glob.glob(os.path.join(img_dir,'*.jpg')))
print(f'img_list length = {len(img_list)}')

max_det = 1000
conf_thresh=0.3
for i, img_name in enumerate(img_list):
    print(f'{i}/{len(img_list)}: {os.path.basename(img_name)}')
    results = model.predict(img_name, 
                    save=True, 
                    save_txt=True,
                    save_conf=True,
                    boxes=True,
                    conf=conf_thresh,
                    agnostic_nms=True,
                    max_det=max_det)
    # print(type(results))
 
    res: Results = results[0]
    res_plotted = res.plot(conf=conf_thresh,
                           font_size=20,
                           line_width=2,
                           labels=True,
                           boxes=True,
                           probs=True)
        # how to adjust the plotting characteristics
    res_rgb = cv.cvtColor(res_plotted, cv.COLOR_BGR2RGB)
    img_save_name = os.path.basename(img_name).rsplit('.')[0] + '_det.jpg'
    plt.imsave(os.path.join(out_dir, img_save_name), res_rgb)
    # plt.show()

    # TODO plot/annotate the total number of counts/class in the corner of the image
    

print("Done")
import code
code.interact(local=dict(globals(), **locals()))   

