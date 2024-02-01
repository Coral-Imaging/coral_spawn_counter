from ultralytics import YOLO

# Load a model
model = YOLO('/home/java/Java/ultralytics/runs/detect/train - aten_1000/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data='data/cslics_surface.yaml',
            imgsz=640,)  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

print("Done")
# import code
# code.interact(local=dict(globals(), **locals()))