from ultralytics import YOLO
import cv2

# Load your trained model (replace with your path)
model = YOLO("runs/detect/human_activity_yolo/weights/best.pt")

# Open video file (replace 'video.mp4' with your video path)
cap = cv2.VideoCapture("g.mp4")

# For saving output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Run YOLO prediction
    results = model.predict(frame, conf=0.5, verbose=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show video
    cv2.imshow("Activity Recognition", annotated_frame)

    # Save video
    out.write(annotated_frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
