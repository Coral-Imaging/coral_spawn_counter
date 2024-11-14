from ultralytics import YOLO
import torch

# load pretrained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO('yolov8n.pt')

# train the model
model.train(data='data/cslics_desktop.yaml', 
            epochs=500, 
            imgsz=1280,
            workers=10,
            cache=True,
            amp=False,
            batch=1,
            )

print('done')