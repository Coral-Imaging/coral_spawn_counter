from ultralytics import YOLO
import torch

# load pretrained model
model = YOLO('weights/yolov8n.pt')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train the model
model.train(data='../data/side_imaging.yml', 
            epochs=200, 
            imgsz=1280,
            workers=4,
            cache=True,
            amp=False,
            batch=4,
            ).to(device)


print('done')