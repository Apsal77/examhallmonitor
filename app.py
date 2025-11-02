# app.py
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
import torch_directml

# -------- AMD GPU Setup --------

device = torch_directml.device()
print("Using device:", device)
print("PyTorch device:", device)
x = torch.randn(2,2).to(device)
print("Test tensor on GPU:", x)



buffers = {}       

# Prevent hard exit while debugging
builtins.exit = lambda *args, **kwargs: None

# ========== CONFIG ==========
VIDEO_SOURCES = {
    "front": r"C:\Users\apsal\OneDrive\Desktop\examhallai\a.mp4",
    "front2": r"C:\Users\apsal\OneDrive\Desktop\examhallai\a.mp4",
    "front3": r"C:\Users\apsal\OneDrive\Desktop\examhallai\a.mp4",
    "left": r"C:\Users\apsal\OneDrive\Desktop\examhallai\b.mp4",
    "right": r"C:\Users\apsal\OneDrive\Desktop\examhallai\b.mp4",
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
    # if no CSV found, keep list empty; we won't fallback to "Student1"
    students = []

# Activity state (for dashboard + de-dup logs)
activity_log = {name: [] for name in students}
activity_lock = threading.Lock()
last_front_status = {}  # {student_name: "present"/"absent"/"moved_out"}

# Load calibration
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
yolo_main = YOLO(YOLO_WEIGHTS)   # front/front3 detection
yolo_main.fuse()
yolo_main.to(device)

yolo_activity = YOLO("runs/detect/human_activity_yolo/weights/best.pt")
yolo_activity.fuse()
yolo_activity.to(device)

yolo_objects = YOLO("runs/detect/phone_chit_model/weights/best.pt")
yolo_objects.fuse()
yolo_objects.to(device)

# TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)




def speak_async(message: str):
    def run():
        try:
            tts_engine.say(message)
            tts_engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()

# ---------- Helpers ----------
def safe_filename(name: str) -> str:
    # keep original name for dashboard, only sanitize filename
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

def log_student_activity(student_name, activity):
    if not student_name:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"time": ts, "activity": activity}

    with activity_lock:
        if student_name not in activity_log:
            activity_log[student_name] = []
        # append to in-memory (trim)
        activity_log[student_name].append(entry)
        if len(activity_log[student_name]) > 500:
            activity_log[student_name] = activity_log[student_name][-500:]

    # write to per-student CSV using real names from students.csv
    fn = os.path.join(LOGS_DIR, f"{safe_filename(student_name)}.csv")
    header = not os.path.exists(fn)
    with open(fn, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["time", "student", "activity"])
        w.writerow([ts, student_name, activity])

    speak_async(f"Alert! {student_name} {activity}")

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
def process_frame_for_camera(name, frame, rows=None, cols=None, H=None):
    global last_front_status

    alerts = []
    # Person detection (head proxy)
    
    results = yolo_main.predict(
        source=frame,
        conf=CONF_THR,
        iou=IOU_THR,
        device=device,
        imgsz=416,
        verbose=False,
        half=True
    )

    head_boxes, centers = [], []

    if len(results) and len(results[0].boxes) > 0:
        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()[:4])
            h = max(1, y2 - y1)
            head_size = int(h * 0.35)
            cx = (x1 + x2) // 2
            cy = y1 + head_size // 2
            x1n, x2n = cx - head_size // 2, cx + head_size // 2
            y1n, y2n = y1, y1 + head_size
            head_boxes.append((x1n, y1n, x2n, y2n))
            centers.append((cx, cy))
            cv2.rectangle(frame, (x1n, y1n), (x2n, y2n), (0, 200, 255), 2)

    

    # ---- FRONT camera logic ----
    if name == "front" and H is not None and rows and cols:
        centers_arr = np.array(centers, dtype=np.float32).reshape(-1,2) if len(centers)>0 else np.empty((0,2),dtype=np.float32)
        warped_centers = warp_points(H, centers_arr) if len(centers_arr)>0 else np.empty((0,2),dtype=np.float32)

        # Precompute seat centers
        cell_w = WARP_SIZE / cols
        cell_h = WARP_SIZE / rows
        seat_centers = { (i,j): ((j+0.5)*cell_w, (i+0.5)*cell_h) for i in range(rows) for j in range(cols) }

        # Track which seat has a detection
        seat_to_detection = { (i,j): None for i in range(rows) for j in range(cols) }

        # Assign detections to seats if inside
        for k, wc in enumerate(warped_centers):
            cell = which_cell(wc, WARP_SIZE, rows, cols)
            if cell is not None:
                seat_to_detection[cell] = k  # mark detection inside this seat

        # Now check each seat
        

        # Check detections that are not in their correct seat
        for k, wc in enumerate(warped_centers):
            cell = which_cell(wc, WARP_SIZE, rows, cols)
            if cell is None:
                # completely outside any grid → moved out
                nearest_cell, nearest_d = None, None
                for c, sc in seat_centers.items():
                    d = (wc[0]-sc[0])**2 + (wc[1]-sc[1])**2
                    if nearest_d is None or d < nearest_d:
                        nearest_cell, nearest_d = c, d
                if nearest_cell is not None:
                    idx = nearest_cell[0]*cols + nearest_cell[1]
                    if idx < len(students):
                        student_name = students[idx]
                        alerts.append(f"{student_name} moved out of seat")
                        log_student_activity(student_name, "moved out of seat")
            else:
                # inside a seat — check if it matches their assigned seat
                idx = cell[0]*cols + cell[1]
                if idx < len(students):
                    student_name = students[idx]
                    # ensure this detection is truly for this student, otherwise it's moved
                    # If multiple detections overlap wrongly, treat as intrusion
                    if seat_to_detection[cell] != k:
                        alerts.append(f"{student_name} moved out of seat")
                        log_student_activity(student_name, "moved out of seat")

    return frame, alerts, head_boxes, centers



# ---------- Stream generators ----------
def generate_setup_frames(cam_name):
    cap = load_cap(cam_name)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{cam_name}] Frame read failed, resetting")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue
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

            

            elif cam_name == "front2":
            # Use the trained human activity YOLO model
                
            
                try:
                    results = yolo_activity.predict(frame, conf=0.35, iou=0.45, device=device, imgsz=416, verbose=False, half=True)
                except Exception as e:
                    print(f"[front2] YOLO prediction failed: {e}")
                    results = None
                
            
                annotated_frame = results[0].plot()  # draw bounding boxes & labels
            
                # If detections exist
                if len(results) and len(results[0].boxes) > 0:
                    for b in results[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()[:4])
                        cls_id = int(b.cls.cpu().numpy())
                        activity_label = results[0].names[cls_id]  # human activity class name
            
                        # Draw custom label (optional)
                        cv2.putText(
                            annotated_frame,
                            activity_label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
            
                        # Determine which student is detected based on bbox center
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
            
                        row_idx = min(FRONT_ROWS - 1, (cy * FRONT_ROWS) // frame.shape[0])
                        col_idx = min(FRONT_COLS - 1, (cx * FRONT_COLS) // frame.shape[1])
                        student_idx = row_idx * FRONT_COLS + col_idx
            
                        if student_idx >= len(students):
                            continue
            
                        student_name = students[student_idx]
            
                        # Log the activity
                        log_student_activity(student_name, activity_label)
            
                frame = annotated_frame  # update frame for streaming

             

                        
            
                                    
                        
            elif cam_name == "front3":
                # presence/absence logic
                try:
                    results = yolo_activity.predict(frame, conf=0.35, iou=0.45, device=device, imgsz=416, verbose=False, half=True)
                except Exception as e:
                    print(f"[front2] YOLO prediction failed: {e}")
                    results = None
                
                centers = []
                if len(results) and len(results[0].boxes) > 0:
                    for b in results[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()[:4])
                        h = max(1, y2 - y1)
                    
                        cx = (x1 + x2) // 2
                    
                        # stomach point: move downward from head top
                        cy = y1 + int(h * 0.55)   # ~55% of height from top ≈ stomach region
                    
                        centers.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 5, (0, 200, 255), -1)

                centers_arr = np.array(centers, dtype=np.float32).reshape(-1, 2) if len(centers) > 0 else np.empty((0, 2), dtype=np.float32)
                warped_centers = warp_points(H, centers_arr) if len(centers_arr) > 0 else np.empty((0, 2), dtype=np.float32)

                present_cells = set()
                for wc in warped_centers:
                    cell = which_cell(wc, WARP_SIZE, rows, cols)
                    if cell is not None:
                        present_cells.add(cell)

                # draw seat grid + names
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        poly = grid_polys_back[idx]; idx += 1
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
                try:
                    results = yolo_objects.predict(
                        source=frame,
                        conf=0.5,
                        imgsz=416,
                        device=device,  
                        verbose=False,
                        half=False
                    )
                except Exception as e:
                    print(f"[{cam_name}] YOLO object detection failed: {e}")
                    results = None
                                
                
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

@app.route('/stream_monitor/left')
def stream_monitor_left():
    return Response(generate_monitor_frames("left"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_monitor/right')
def stream_monitor_right():
    return Response(generate_monitor_frames("right"), mimetype='multipart/x-mixed-replace; boundary=frame')




# ---------------- Dashboard ----------------
@app.route('/dashboard')
def dashboard_page():
    with activity_lock:
        logs = {student: activity_log.get(student, []) for student in students}
    return render_template('dashboard.html', students=students, logs=logs)

@app.route('/student/<student_name>')
def student_logs(student_name):
    with activity_lock:
        logs = activity_log.get(student_name, [])
    return render_template('student_logs.html', student=student_name, logs=logs)



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
    try:
        app.run(debug=True, threaded=True, use_reloader=False)
    finally:
        release_caps()
