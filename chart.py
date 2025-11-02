import os
import cv2
import mediapipe as mp
import json

# Paths - update to your actual dataset folder
input_folder = r'C:\Users\apsal\OneDrive\Desktop\examhallai\datasethumanrec'  # folder containing subfolders for standing, bending, etc.
output_folder = r'C:\Users\apsal\OneDrive\Desktop\examhallai\humangbjhgh'  # folder to save keypoints and annotated images

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Create output dirs mirroring input class subfolders
for cls_name in os.listdir(input_folder):
    cls_input_dir = os.path.join(input_folder, cls_name)
    if not os.path.isdir(cls_input_dir):
        continue
    os.makedirs(os.path.join(output_folder, cls_name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, cls_name, 'keypoints'), exist_ok=True)

    # Initialize Pose detector
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        for img_file in os.listdir(cls_input_dir):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(cls_input_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read {img_path}")
                continue

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks is None:
                print(f"No keypoints detected for {img_file}")
                continue

            # Draw landmarks on image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save annotated image
            annotated_img_path = os.path.join(output_folder, cls_name, 'images', img_file)
            cv2.imwrite(annotated_img_path, annotated_image)

            # Extract keypoints into list: [{x, y, z, visibility}, ...]
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

            # Save keypoints as JSON
            keypoint_path = os.path.join(output_folder, cls_name, 'keypoints', os.path.splitext(img_file)[0] + '.json')
            with open(keypoint_path, 'w') as f:
                json.dump(keypoints, f)

print("Pose keypoint extraction and saving completed.")
