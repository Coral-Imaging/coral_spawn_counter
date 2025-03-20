from ultralytics import YOLO
import torch

# load pretrained model
model = YOLO('/home/tsaid/code/coral_spawn_counter/weights/yolov8n.pt')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train the model
model.train(data='/home/tsaid/code/coral_spawn_counter/data_yml_files/cslics_2023_2024_subsurface_M12_hpc.yaml', 
            epochs=10, 
            imgsz=640,
            workers=4,
            cache=True,
            amp=False,
            batch=4,
            ).to(device)


print('done')