from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolov8n.pt")  # you can also use yolov8m.pt or yolov8n.pt

# Train on your dataset
model.train(
    data="split_datasethuman/data.yaml",   # path to data.yaml
    epochs=40,                  # number of training epochs
    imgsz=640,                  # image size
    batch=16,                   # adjust based on GPU
    name="human_activity_yolo", # run name
    workers=4
)
