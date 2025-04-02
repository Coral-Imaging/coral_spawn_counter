from ultralytics import YOLO

def print_metrics(metrics):
    # Extract the required metrics
    box_p = metrics.box.p.mean()
    box_r = metrics.box.r.mean()
    box_map50 = metrics.box.map50
    box_map = metrics.box.map

    # Print the metrics in the desired format
    print(f"{'Class':<15}{'Box(P':<10}{'R':<10}{'mAP50':<10}{'mAP50-95)':<10}")
    print(f"{'all':<15}{box_p:<10.3f}{box_r:<10.3f}{box_map50:<10.3f}{box_map:<10.3f}")
    metrics.confusion_matrix.print() #print RAW confusion metrix




model_640 = YOLO("/home/java/Java/cslics/resolution_test_results/models/resolution_test_640/weights/best.pt")  # load model
model_320 = YOLO("/home/java/Java/cslics/resolution_test_results/models/resolution_test_320/weights/best.pt")  
model_160 = YOLO("/home/java/Java/cslics/resolution_test_results/models/resolution_test_160/weights/best.pt")
model_128 = YOLO("/home/java/Java/cslics/resolution_test_results/models/resolution_test_128/weights/best.pt")
model_96 = YOLO("/home/java/Java/cslics/resolution_test_results/models/resolution_test_96/weights/best.pt")
datafile = "/media/java/cslics_ssd/cslics_data/cslics_subsurface_data/cslics_2023_2024_subsurface_M12.yaml"
# Validate the model
metrics_640 = model_640.val(data=datafile)
metrics_320 = model_320.val(data=datafile)
metrics_160 = model_160.val(data=datafile)
metrics_128 = model_128.val(data=datafile)
metrics_96 = model_96.val(data=datafile)

print("Metrics for 640")
print_metrics(metrics_640)
print("Metrics for 320")
print_metrics(metrics_320)
print("Metrics for 160")
print_metrics(metrics_160)
print("Metrics for 128")
print_metrics(metrics_128)
print("Metrics for 96")
print_metrics(metrics_96)

