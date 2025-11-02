from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2, os, json, time, threading, csv
import numpy as np
import pandas as pd
from ultralytics import YOLO
import pyttsx3
from datetime import datetime
import builtins
import atexit
import torch
from collections import deque
import mediapipe as mp
import torchvision.models as models
import torch.nn as nn
import queue
import mediapipe as mp
import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  


# MediaPipe Pose setup (reusable for all cameras if needed)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

# ST-GCN parameters
STGCN_MODEL_PATH = r"C:\Users\apsal\OneDrive\Desktop\examhallai\stgcn_model.pth"
NUM_JOINTS = 33
IN_CHANNELS = 3
NUM_CLASSES = 4
SEQ_LEN = 30
CLASS_NAMES = ['standing', 'bending', 'turningaround', 'normal']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stgcn_model = SimpleSTGCN(NUM_JOINTS, IN_CHANNELS, NUM_CLASSES).to(device)
stgcn_model.load_state_dict(torch.load(STGCN_MODEL_PATH, map_location=device))
stgcn_model.eval()

# Buffer to store sequences of keypoints per person
keypoint_buffers = {}



# Prevent hard exit while debugging
builtins.exit = lambda *args, **kwargs: None

# ========== CONFIG ==========
VIDEO_SOURCES = {
    "front": "a.mp4",
    "front2": "a.mp4",
    "front3": "a.mp4",   
    "left":  "b.mp4",
    "right": "b.mp4",
}


FRONT_ROWS, FRONT_COLS = 3, 4
SIDE_ROWS,  SIDE_COLS  = 3, 2
WARP_SIZE = 800
STUDENTS_CSV = "students.csv"
CALIB_FILE   = "calib.json"
LOGS_DIR     = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
YOLO_WEIGHTS = "yolov8n.pt"
CONF_THR = 0.35
IOU_THR  = 0.45

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ----- Load students (ALWAYS use real names for logging) -----
students = []
if os.path.exists(STUDENTS_CSV):
    students = pd.read_csv(STUDENTS_CSV, header=None)[0].astype(str).str.strip().tolist()
else:
    students = []

# Activity state (for dashboard + de-dup logs)
activity_log = {name: [] for name in students}
activity_lock = threading.Lock()
last_front_status = {}  # {student_name: "present"/"absent"/"moved_out"}

# Load calibration
calib = {"front": [], "left": [], "right": []}
if os.path.exists(CALIB_FILE):
    try:
        with open(CALIB_FILE, "r") as f:
            calib = json.load(f)
    except Exception:
        pass

# OpenCV caps
caps = {}

# Models
yolo = YOLO(YOLO_WEIGHTS)




import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template

app = Flask(__name__)

LOG_DIR = "logs"
STATIC_DIR = "static"
CHART_PATH = os.path.join(STATIC_DIR, "attendance_bar.png")
VIOLATION_CHART_PATH = os.path.join(STATIC_DIR, "violation_pie.png")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def read_log_files():
    """Read all CSV log files and combine them into a DataFrame."""
    csv_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError("Error: No log files found in logs folder.")

    df_list = []
    for file in csv_files:
        file_path = os.path.join(LOG_DIR, file)
        try:
            df = pd.read_csv(file_path)
            if not {"student", "activity"}.issubset(df.columns):
                raise ValueError("CSV must have 'student' and 'activity' columns")
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not df_list:
        raise ValueError("No valid CSV files found.")
    return pd.concat(df_list, ignore_index=True)


def plot_attendance_bar_chart(df):
    """Plot attendance summary bar chart."""
    summary = (
        df.groupby("student")["activity"]
        .apply(lambda x: "absent" if (x == "absent").any() else "present")
        .reset_index()
    )

    counts = summary["activity"].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["red", "green"])
    plt.title("Attendance Summary")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CHART_PATH)
    plt.close()

    absent_students = summary.loc[summary["activity"] == "absent", "student"].tolist()
    present_students = summary.loc[summary["activity"] == "present", "student"].tolist()

    return absent_students, present_students


# ✅ New function to extract violated and honest students
def analyze_violations(df):
    """Detect violated and honest students and generate pie chart."""
    if "warning_level" not in df.columns:
        return [], [], None

    # Mark violation if 3rd warning or leave hall message present
    violated_students = df[
        df["warning_level"].str.contains("3rd warning", case=False, na=False)
    ]["student"].unique().tolist()

    all_students = df["student"].unique().tolist()
    honest_students = [s for s in all_students if s not in violated_students]

    # Plot pie chart
    labels = ["Violated", "Honest"]
    sizes = [len(violated_students), len(honest_students)]
    colors = ["#ff4d4d", "#66cc66"]

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    plt.title("Violation Summary")
    plt.tight_layout()
    plt.savefig(VIOLATION_CHART_PATH)
    plt.close()

    return violated_students, honest_students, VIOLATION_CHART_PATH







def extract_keypoints(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

# ---------- Helpers ----------
def safe_filename(name: str) -> str:
    base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return base.lower() or "unknown"

def save_calib(calib_dict):
    with open(CALIB_FILE, "w") as f:
        json.dump(calib_dict, f)

def load_cap(name):
    if name in caps and caps[name].isOpened():
        return caps[name]
    src = VIDEO_SOURCES.get(name, 0)
    cap = cv2.VideoCapture(src)
    caps[name] = cap
    return cap

def release_caps():
    for c in list(caps.values()):
        try:
            c.release()
        except:
            pass


LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
ALERTS_LOG_FILE = os.path.join(LOGS_DIR, "alerts_log.csv")

alert_tracker = {}  # Tracks alert times and counts
students_blocked_after_third_warning = set()  # Block after 3rd warning

def ensure_alert_log_exists():
    if not os.path.exists(ALERTS_LOG_FILE):
        with open(ALERTS_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "student", "activity", "warning_level"])

def _log_to_individual_csv(student_name, msg, activity):
    fn = os.path.join(LOGS_DIR, f"{safe_filename(student_name)}.csv")
    header = not os.path.exists(fn)
    with open(fn, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["time", "student", "activity"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), student_name, msg])

def log_student_activity(student_name, activity):
    if not student_name:
        return

    ensure_alert_log_exists()

    if student_name in students_blocked_after_third_warning:
        _log_to_individual_csv(student_name, activity, activity)
        return

    now = datetime.now()
    key = (student_name, activity)
    phone_chit_activities = ["phone", "chit"]
    escalate_activities = ["moved out of seat", "cheating", "not cheating"]

    if key not in alert_tracker:
        alert_tracker[key] = {"first_alert_time": None, "last_alert_time": None, "count": 0}

    record = alert_tracker[key]
    first_time = record["first_alert_time"]
    last_time = record["last_alert_time"]
    count = record["count"]

    if first_time is None:
        record["first_alert_time"] = now
        record["last_alert_time"] = now
        record["count"] = 1
        warning_level = "First Alert"
        msg = activity
    else:
        elapsed_since_last = (now - last_time).total_seconds()

        if activity in phone_chit_activities:
            if count == 1:
                record["count"] = 3
                record["last_alert_time"] = now
                warning_level = "leave the hall immediately"
                msg = activity
                students_blocked_after_third_warning.add(student_name)
            else:
                return
        elif activity in escalate_activities:
            if elapsed_since_last >= 20:
                count += 1
                record["count"] = count
                record["last_alert_time"] = now
                if count == 2:
                    warning_level = "this is your 2nd warning"
                    msg = activity
                elif count >= 3:
                    warning_level = "this is your 3rd warning, leave the hall immediately"
                    msg = activity
                    students_blocked_after_third_warning.add(student_name)
                else:
                    warning_level = "this is your 3rd warning, leave the hall immediately"
                    msg = activity
            else:
                return
        else:
            warning_level = None
            msg = activity
            record["last_alert_time"] = now
            record["count"] += 1

    _log_to_individual_csv(student_name, msg, activity)

    if warning_level is not None and activity.lower() != "absent":
        alert_header = not os.path.exists(ALERTS_LOG_FILE)
        with open(ALERTS_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if alert_header:
                w.writerow(["time", "student", "activity", "warning_level"])
            w.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), student_name, activity, warning_level])



speech_queue = queue.Queue()

def speech_thread_func():
    engine = pyttsx3.init('sapi5')
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    while True:
        text = speech_queue.get()
        if text is None:  # signal to exit
            break
        engine.stop()    # stop any ongoing speech immediately
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_thread_func, daemon=True)
speech_thread.start()

def speak_text(text):
    speech_queue.put(text)



def tail_alerts_log_file(log_file_path, poll_interval=1.0):
    last_position = 0
    while True:
        if not os.path.exists(log_file_path):
            time.sleep(poll_interval)
            continue
        with open(log_file_path, "r", newline='', encoding="utf-8") as f:
            f.seek(last_position)
            new_lines = f.readlines()
            last_position = f.tell()

        if new_lines:
            for line in new_lines:
                if line.strip() and not line.lower().startswith("time,student,activity,warning_level"):
                    reader = csv.reader([line])
                    for row in reader:
                        if len(row) >= 4:
                            alert_time, student, activity, warning_level = row[:4]
                            msg = f"Alert! {student} {activity}, {warning_level}"
                            print("DEBUG: Speaking alert:", msg)  # Debug print
                            speak_text(msg)

        time.sleep(poll_interval)


# Start the background thread on app startup
def start_voice_alert_thread():
    t = threading.Thread(target=tail_alerts_log_file, args=(ALERTS_LOG_FILE,), daemon=True)
    t.start()




def compute_homography(quad, size=WARP_SIZE):
    src = np.array(quad, dtype=np.float32)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype=np.float32)
    H    = cv2.getPerspectiveTransform(src, dst)
    Hinv = cv2.getPerspectiveTransform(dst, src)
    return H, Hinv

def warp_points(H, pts):
    if len(pts) == 0:
        return np.empty((0,2), dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((len(pts),1), dtype=np.float32)])
    w = (H @ pts_h.T).T
    w = w[:, :2] / w[:, 2:3]
    return w

def backproject_grid_polys(Hinv, size, rows, cols):
    cell_w = size / cols
    cell_h = size / rows
    polys = []
    for i in range(rows):
        for j in range(cols):
            x0, y0 = j*cell_w, i*cell_h
            x1, y1 = (j+1)*cell_w, (i+1)*cell_h
            poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
            poly_back = warp_points(Hinv, poly)
            polys.append(poly_back)
    return polys

def which_cell(warp_pt, size, rows, cols):
    x,y = warp_pt
    if x<0 or y<0 or x>=size or y>=size:
        return None
    cell_w = size / cols
    cell_h = size / rows
    j = int(x // cell_w)
    i = int(y // cell_h)
    if 0<=i<rows and 0<=j<cols:
        return (i,j)
    return None

def draw_polyline(img, pts, color=(0,255,0), thickness=2, closed=False):
    pts_int = pts.astype(np.int32).reshape(-1,1,2)
    cv2.polylines(img, [pts_int], closed, color, thickness, lineType=cv2.LINE_AA)

def put_label(img, text, org, bg=True):
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    if bg:
        (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
        cv2.rectangle(img, (org[0]-4, org[1]-tht-6), (org[0]+tw+4, org[1]+4), (0,0,0), -1)
    cv2.putText(img, text, org, font, scale, (255,255,255), th, cv2.LINE_AA)

# ---------- Core per-camera processing ----------
# app.py  (only showing updated logic parts)

# ---------------- Core per-camera processing ----------------
prior_seat_names = {}
alerts = []

def process_frame_for_camera(name, frame, rows=None, cols=None, H=None):
    global last_front_status

    alerts = []
    # Person detection (head proxy)
    results = yolo.predict(source=frame, conf=CONF_THR, iou=IOU_THR, classes=[0], verbose=False)
    head_boxes, centers = [], []

    if len(results) and len(results[0].boxes) > 0:
        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()[:4])
            h = max(1, y2 - y1)

            # Smaller rectangle (~1/4 head area)
            small_head_size = int(h * 0.17)

            cx = (x1 + x2) // 2
            y1n, y2n = y1, y1 + small_head_size
            x1n, x2n = cx - small_head_size // 2, cx + small_head_size // 2

            head_boxes.append((x1n, y1n, x2n, y2n))
            centers.append((cx, y1 + small_head_size // 2))

            cv2.rectangle(frame, (x1n, y1n), (x2n, y2n), (0, 200, 255), 2)


    
    if name == "front" and H is not None and rows and cols:
        centers_arr = (
            np.array(centers, dtype=np.float32).reshape(-1, 2)
            if len(centers) > 0
            else np.empty((0, 2), dtype=np.float32)
        )
        warped_centers = (
            warp_points(H, centers_arr)
            if len(centers_arr) > 0
            else np.empty((0, 2), dtype=np.float32)
        )
    
        cell_w = WARP_SIZE / cols
        cell_h = WARP_SIZE / rows
        seat_centers = {
            (i, j): ((j + 0.5) * cell_w, (i + 0.5) * cell_h) for i in range(rows) for j in range(cols)
        }
    
        seat_to_detections = {(i, j): [] for i in range(rows) for j in range(cols)}
    
        # Map each warped center to corresponding seat cell
        for k, wc in enumerate(warped_centers):
            cell = which_cell(wc, WARP_SIZE, rows, cols)
            if cell is not None:
                seat_to_detections[cell].append(k)
    
        # Dictionary holding student names detected per seat this frame
        current_seat_names = {}
    
        # Iterate over all students/seats to check their detection status
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(students):
                    continue
                detections = seat_to_detections[(i, j)]
    
                if len(detections) == 0:
                    # No detection in seat: student moved out or absent
                    alerts.append(f"{students[idx]} moved out of seat")
                    log_student_activity(students[idx], "moved out of seat")
                    current_seat_names[(i, j)] = None
                else:
                    # If any detection(s) present, assign first detected name to this seat
                    detected_name = students[idx]  # Or use detected identity if available
                    current_seat_names[(i, j)] = detected_name
    
        # Compare to prior frame seat names (needs to be stored externally with persistence across frames)
        for seat, name in current_seat_names.items():
            prior_name = prior_seat_names.get(seat)  # prior_seat_names must be a stored dict outside this function
            if prior_name is not None and name != prior_name:
                alerts.append(f"{prior_name} moved out of seat ")
                log_student_activity(prior_name, "moved out of seat")
    
        # Update prior names for next frame (must be external/stateful)
        prior_seat_names.clear()
        prior_seat_names.update(current_seat_names)
    
        # Draw bounding boxes with student names on the frame
        # Draw bounding boxes with student names on the frame
        for (i, j), detection_indices in seat_to_detections.items():
            if detection_indices:
                idx = i * cols + j
                if idx < len(students):
                    student_name = students[idx]
                    for k in detection_indices:  # Loop over all detections in this cell
                        if k < len(head_boxes):
                            box = head_boxes[k]
                            # Draw rectangle (box)
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            # Put student name above each box (all the same name)
                            cv2.putText(
                                frame,
                                student_name,
                                (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

    
        # Additional detection for warped centers outside grid cells stays as is
    
    return frame, alerts, head_boxes, centers
    
    
    

# ---------- Stream generators ----------
def generate_setup_frames(cam_name):
    cap = load_cap(cam_name)
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break
        frame = cv2.resize(frame, (960, 720))
        q = calib.get(cam_name, [])
        if len(q) == 4:
            pts = np.array(q, dtype=np.float32)
            draw_polyline(frame, pts, color=(255,0,0), thickness=3, closed=True)
        ret, buff = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buff.tobytes() + b'\r\n')

def generate_monitor_frames(cam_name):
    cap = load_cap(cam_name)

    # calibration handling
    quad = calib.get("front", []) if cam_name in ["front2", "front3"] else calib.get(cam_name, [])

    if len(quad) == 4:
        H, Hinv = compute_homography(quad, WARP_SIZE)

        if cam_name in ["front", "front2", "front3"]:
            rows, cols = FRONT_ROWS, FRONT_COLS  # 3x4 full grid
            names_for_grid = students[:]

        elif cam_name in ["left"]:
            rows, cols = SIDE_ROWS, SIDE_COLS  # 3x2
            # take col1 & col2
            names_for_grid = [
                students[r * FRONT_COLS + c]
                for r in range(FRONT_ROWS)
                for c in [0, 1]
                if (r * FRONT_COLS + c) < len(students)
            ]

        elif cam_name in ["right"]:
            rows, cols = SIDE_ROWS, SIDE_COLS  # 3x2
            # take col3 & col4
            names_for_grid = [
                students[r * FRONT_COLS + c]
                for r in range(FRONT_ROWS)
                for c in [2, 3]
                if (r * FRONT_COLS + c) < len(students)
            ]

        else:
            rows = cols = None
            names_for_grid = []

        grid_polys_back = backproject_grid_polys(Hinv, WARP_SIZE, rows, cols)

    else:
        H = Hinv = None
        rows = cols = None
        names_for_grid = []

    # ---- main loop ----
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (960, 720))
        alerts = []

        if H is not None:
            # ---- FRONT CAMERA LOGIC ----
            if cam_name == "front":
                proc_frame, frame_alerts, boxes, centers = process_frame_for_camera(
                    cam_name, frame, rows, cols, H
                )
                alerts.extend(frame_alerts)

            





            if cam_name == "front2":
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame = frame.copy()
             
                # Step 1: Detect people using YOLO
                results_yolo = yolo.predict(source=frame, conf=CONF_THR, iou=IOU_THR, classes=[0], verbose=False)
                detected_people = []
                centers = []
             
                if len(results_yolo) and len(results_yolo[0].boxes) > 0:
                    for b in results_yolo[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()[:4])
                        detected_people.append((x1, y1, x2, y2))
             
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        centers.append((cx, cy))
             
                        # Draw bbox and center point
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
             
                # Step 2: Process each detected person for pose estimation
                for pid, bbox in enumerate(detected_people):
                    x1, y1, x2, y2 = bbox
                    person_crop = frame[y1:y2, x1:x2]
                    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    results_pose = pose.process(rgb_crop)
                    label = "normal"  # Default class
             
                    if results_pose.pose_landmarks:
                        keypoints = extract_keypoints(results_pose.pose_landmarks.landmark)
             
                        if pid not in keypoint_buffers:
                            keypoint_buffers[pid] = deque(maxlen=SEQ_LEN)
                        keypoint_buffers[pid].append(keypoints)
             
                        if len(keypoint_buffers[pid]) == SEQ_LEN:
                            seq_input = np.stack(keypoint_buffers[pid])
                            tensor_input = torch.tensor(seq_input).unsqueeze(0).to(device)
                            with torch.no_grad():
                                output = stgcn_model(tensor_input)
                                class_id = torch.argmax(output, dim=1).item()
                                class_name = CLASS_NAMES[class_id]
                                if class_name in ["standing", "bending", "turning around"]:
                                    label = class_name
                                # If anything else (including "normal"), label remains "normal"
                            # Only log activity if it's not normal
                            if label != "normal":
                                # Map pid or bbox to student name as in your grid logic, then:
                                # log_student_activity(student_name, label)
                                pass
             
                        # Draw pose landmarks inside the person's bounding box crop
                        mp_drawing.draw_landmarks(
                            person_crop,
                            results_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                        # Put the updated crop back into the annotated frame 
                        annotated_frame[y1:y2, x1:x2] = person_crop
             
                    # Draw label above bounding box on annotated_frame
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if label == "normal" else (0, 0, 255),
                        2
                    )
             
                frame = annotated_frame

                        
            elif cam_name == "front3":
         # presence/absence logic
                results = yolo.predict(
                    source=frame, conf=CONF_THR, iou=IOU_THR, classes=[0], verbose=False
                )
                centers = []
                if len(results) and len(results[0].boxes) > 0:
                    for b in results[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()[:4])
                        h = max(1, y2 - y1)
                        
                        cx = (x1 + x2) // 2
                        
                        # Stomach point: 55% down from head top
                        cy = y1 + int(h * 0.55)
                        
                        centers.append((cx, cy))
                        
                        # Draw filled rectangle as marker instead of circle
                        rect_width, rect_height = 20, 10  # Size of rectangle
                        top_left = (cx - rect_width // 2, cy - rect_height // 2)
                        bottom_right = (cx + rect_width // 2, cy + rect_height // 2)
                        cv2.rectangle(frame, top_left, bottom_right, (0, 200, 255), -1)
            
                # Warp points for seating grid mapping, if any detected points
                centers_arr = np.array(centers, dtype=np.float32).reshape(-1, 2) if len(centers) > 0 else np.empty((0, 2), dtype=np.float32)
                warped_centers = warp_points(H, centers_arr) if len(centers_arr) > 0 else np.empty((0, 2), dtype=np.float32)
            
                present_cells = set()
                for wc in warped_centers:
                    cell = which_cell(wc, WARP_SIZE, rows, cols)
                    if cell is not None:
                        present_cells.add(cell)
            
                # Draw seat grid + names
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        poly = grid_polys_back[idx]
                        idx += 1
                        draw_polyline(frame, poly, color=(0, 255, 0), thickness=1, closed=True)
                        tl = poly[0]
                        label_idx = i * cols + j
                        if label_idx < len(students):
                            student_name = students[label_idx]
                            status = "present" if (i, j) in present_cells else "absent"
                            put_label(frame, f"{student_name} [{status}]", (int(tl[0]) + 6, int(tl[1]) + 22))
            
                            if status == "absent":
                                log_student_activity(student_name, "absent")
            
            # ---- SIDE CAMERA LOGIC ----
            elif cam_name in ["left", "right"]:
             # Detect objects (phone=0, chit=1) -> adjust IDs as per your custom dataset
                results = YOLO("runs/detect/phone_chit_model/weights/best.pt").predict(
                    source=frame, conf=0.5, imgsz=416, verbose=False
                )
            
                detected_objects = []  # (bbox, label)
            
                if len(results) and len(results[0].boxes) > 0:
                    for b in results[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()[:4])
                        cls_id = int(b.cls.cpu().numpy())
                        label = "unknown"
                        if cls_id == 0:   # update class IDs based on your training
                            label = "phone"
                        elif cls_id == 1:
                            label = "chit"
                        else:
                            continue
            
                        detected_objects.append(((x1, y1, x2, y2), label))
                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
                # Check overlap of detections with seat cells
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        poly = grid_polys_back[idx]; idx += 1
                        draw_polyline(frame, poly, color=(0, 255, 0), thickness=1, closed=True)
                        tl = poly[0]
                        label_idx = i * cols + j
                        if label_idx < len(names_for_grid):
                            student_name = names_for_grid[label_idx]
                            put_label(frame, f"{student_name}", (int(tl[0]) + 6, int(tl[1]) + 22))
            
                            # Check if any object falls inside this grid
                            for (x1, y1, x2, y2), obj_label in detected_objects:
                                box_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                                
                                # Count how much of bbox is inside grid polygon
                                inter = cv2.intersectConvexConvex(poly.astype(np.float32), box_poly)
                                if inter[0] > 0:  # intersection area > 0
                                    box_area = (x2 - x1) * (y2 - y1)
                                    overlap_ratio = inter[0] / box_area
                                    if overlap_ratio > 0.5:  # majority inside seat
                                        log_student_activity(student_name, f"using {obj_label}")
            
            # --- grid overlay for front/front2 only ---
            if cam_name in ["front", "front2"]:
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        poly = grid_polys_back[idx]; idx += 1
                        draw_polyline(frame, poly, color=(0, 255, 0), thickness=1, closed=True)
                        tl = poly[0]
                        label_idx = i * cols + j
                        label_name = students[label_idx] if label_idx < len(students) else f"({i+1},{j+1})"
                        put_label(frame, f"({i+1},{j+1}) {label_name}", (int(tl[0]) + 6, int(tl[1]) + 22))

        else:
            # fallback if no calibration
            frame, frame_alerts, _, _ = process_frame_for_camera(cam_name, frame)
            alerts.extend(frame_alerts)

        # draw alerts only on main front cam
        if cam_name == "front":
            y = 40
            for a in alerts:
                cv2.putText(frame, str(a), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                y += 30

        ret, buff = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buff.tobytes() + b'\r\n')



# ---------- Flask routes ----------
@app.route('/')
def index():
    return redirect(url_for('setup_page'))

@app.route('/setup')
def setup_page():
    return render_template('setup.html')

@app.route('/stream_setup/<cam>')
def stream_setup(cam):
    return Response(generate_setup_frames(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_calib', methods=['POST'])
def save_calib_route():
    global calib
    data = request.json
    if data:
        calib = data
        save_calib(calib)
        return jsonify({"status":"ok"})
    return jsonify({"status":"error"}), 400

@app.route('/monitor')
def monitor_page():
    return render_template('monitor.html')

@app.route('/stream_monitor/<cam>')
def stream_monitor(cam):
    return Response(generate_monitor_frames(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_monitor/front2')
def stream_monitor_front2():
    return Response(generate_monitor_frames("front2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_monitor/front3')
def stream_monitor_front3():
    return Response(generate_monitor_frames("front3"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_monitor/left2')
def stream_monitor_left2():
    return Response(generate_monitor_frames("left2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_monitor/right2')
def stream_monitor_right2():
    return Response(generate_monitor_frames("right2"), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- Dashboard ----------------
@app.route("/dashboard")
def dashboard():
    df = read_log_files()
    absent_students, present_students = plot_attendance_bar_chart(df)
    violated_students, honest_students, _ = analyze_violations(df)

    return render_template(
        "dashboard.html",
        absent_students=absent_students,
        present_students=present_students,
        violated_students=violated_students,
        honest_students=honest_students,
    )





        

@app.route('/student/<student_name>')
def student_logs(student_name):
    with activity_lock:
        logs = activity_log.get(student_name, [])
    return render_template('student_logs.html', student=student_name, logs=logs)

@app.route('/delete_student/<student_name>', methods=['POST'])
def delete_student_log(student_name):
    with activity_lock:
        if student_name in activity_log:
            activity_log[student_name] = []
    # also clear the CSV
    fn = os.path.join(LOGS_DIR, f"{safe_filename(student_name)}.csv")
    if os.path.exists(fn):
        try:
            os.remove(fn)
        except:
            pass
    return redirect(url_for('dashboard_page'))

@app.route('/delete_all', methods=['POST'])
def delete_all_logs():
    with activity_lock:
        for student in students:
            activity_log[student] = []
    # clear CSVs
    for student in students:
        fn = os.path.join(LOGS_DIR, f"{safe_filename(student)}.csv")
        if os.path.exists(fn):
            try:
                os.remove(fn)
            except:
                pass
    return redirect(url_for('dashboard_page'))

@app.route('/student_log/<name>')
def student_log(name):
    fn = os.path.join(LOGS_DIR, f"{safe_filename(name)}.csv")
    rows = []
    if os.path.exists(fn):
        with open(fn, newline='', encoding='utf-8') as f:
            r = csv.reader(f)
            headers = next(r, None)
            for row in r:
                rows.append(row)
    return render_template('student_log.html', name=name, rows=rows)

# cleanup
@atexit.register
def _cleanup():
    release_caps()

if __name__ == "__main__":
    start_voice_alert_thread()
    
    try:
        app.run(debug=True, threaded=True, use_reloader=False)
    finally:
        release_caps()
