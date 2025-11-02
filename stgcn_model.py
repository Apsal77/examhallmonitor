import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from ultralytics import YOLO  # pip install ultralytics

# ---------------- ST-GCN MODEL ----------------
class SimpleSTGCN(torch.nn.Module):
    def __init__(self, num_joints, in_channels, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_joints * in_channels, 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ---------------- PARAMETERS ----------------
model_path = r"C:\Users\apsal\OneDrive\Desktop\examhallai\humanclassifydataset\stgcn_model.pth"
num_joints = 33
in_channels = 3
num_classes = 4
seq_len = 30
class_names = ['standing', 'bending', 'turningaround', 'normal']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stgcn_model = SimpleSTGCN(num_joints, in_channels, num_classes).to(device)
stgcn_model.load_state_dict(torch.load(model_path, map_location=device))
stgcn_model.eval()

# ---------------- DETECTORS ----------------
yolo_model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Keypoint buffers
keypoint_buffers = {}

# ---------------- UTILITIES ----------------
def extract_keypoints(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """Remove duplicate overlapping boxes using IoU threshold"""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        return indices
    else:
        return []

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture("e.mp4")  # or 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection (tracking optional)
    results = yolo_model(frame)[0]  # detection only

    # Filter for 'person' class
    boxes = []
    scores = []
    for box in results.boxes:
        if int(box.cls) == 0:  # person
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h for NMS
            scores.append(float(box.conf))

    # Apply NMS to remove duplicate overlapping boxes
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.4)
    unique_boxes = [boxes[i] for i in keep_indices]

    # Process each unique detected person
    for i, (x, y, w, h) in enumerate(unique_boxes):
        x1, y1, x2, y2 = x, y, x + w, y + h
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(person_rgb)

        if pose_results.pose_landmarks:
            keypoints = extract_keypoints(pose_results.pose_landmarks.landmark)

            pid = i  # assign unique ID for now

            if pid not in keypoint_buffers:
                keypoint_buffers[pid] = deque(maxlen=seq_len)
            keypoint_buffers[pid].append(keypoints)

            # Predict if enough frames
            if len(keypoint_buffers[pid]) == seq_len:
                seq_input = np.stack(keypoint_buffers[pid])
                tensor_input = torch.tensor(seq_input).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = stgcn_model(tensor_input)
                    class_id = torch.argmax(output, dim=1).item()
                    label = class_names[class_id]
            else:
                label = "Collecting..."

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            # Draw pose keypoints
            annotated_crop = person_crop.copy()
            mp_drawing.draw_landmarks(
                annotated_crop,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            frame[y1:y2, x1:x2] = annotated_crop
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "No pose", (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Real-time Activity Classification", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
