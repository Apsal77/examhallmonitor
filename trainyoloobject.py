from ultralytics import YOLO

# Path to merged dataset data.yaml
data_yaml = "merged_dataset/data.yaml"

# Load YOLOv8 model (you can change yolov8s.pt → yolov8n.pt or yolov8m.pt)
model = YOLO("yolov8n.pt")  

# Train the model
esults = model.train(
    data=data_yaml,      # dataset config
    epochs=50,                    # number of epochs
    imgsz=416,           # image size
    batch=16,            # batch size (adjust depending on GPU/CPU)
    name="phone_chit_model",
    
      # folder name under runs/detect/
)

print("✅ Training complete! Best weights saved in: runs/detect/phone_chit_model/weights/best.pt")
 
 