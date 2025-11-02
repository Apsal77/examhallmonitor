from ultralytics import YOLO
import cv2

# Load your trained model (use best.pt or last.pt from training)
model = YOLO("runs/detect/phone_chit_model/weights/best.pt")  

# Open webcam (0 = default camera)
cap = cv2.VideoCapture("a.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, conf=0.5, imgsz=416)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




