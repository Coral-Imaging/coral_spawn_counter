from ultralytics import YOLO

# load pretrained model
model = YOLO('yolov8n.pt')

# train the model
model.train(data='data/cslics_desktop.yaml', 
            epochs=500, 
            imgsz=1280,
            workers=10,
            cache=True,
            amp=False,
            batch=1,
            project='cslics_desktop_2024'
            )

print('done')