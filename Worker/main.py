import cv2
import os
import uuid
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import mysql.connector
import subprocess
import re
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import traceback
import sys

# try to import ray to report progress
try:
    import ray
except ImportError:
    ray = None

# ===========================================================
# BITRATE FUNCTIONS (unchanged)
# ===========================================================
def get_video_bitrate(video_path):
    try:
        cmd = ["ffmpeg", "-i", video_path]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        match = re.search(r"bitrate:\s*(\d+)\s*kb/s", result.stderr)
        if match:
            return int(match.group(1))
        else:
            print("⚠️ Could not detect bitrate for", video_path)
            return None
    except Exception as e:
        print("FFmpeg bitrate error:", e)
        return None

def match_bitrate(input_video, processed_video, final_video):
    bitrate = get_video_bitrate(input_video)
    if bitrate is None:
        print("⚠️ Skipping bitrate matching (bitrate not found)")
        try:
            os.rename(processed_video, final_video)
        except Exception as e:
            print("Error renaming processed video:", e)
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video,
        "-b:v", f"{bitrate}k",
        "-bufsize", f"{bitrate*2}k",
        final_video
    ]
    subprocess.run(cmd)
    print(f"🎯 Final video bitrate matched: {bitrate} kbps")
    try:
        os.remove(processed_video)
    except Exception as e:
        print("Could not remove temp processed video:", e)

# ===========================================================
# MYSQL SETTINGS AND FUNCTIONS (unchanged except insert_mysql added Driving param)
# ===========================================================
MYSQL_HOST = "192.168.0.101"
MYSQL_USER = "root"
MYSQL_PASS = ""
MYSQL_PORT = 3306
MYSQL_DB   = "Traffic"

def init_mysql():
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            port=MYSQL_PORT
        )
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS Traffic")
        cursor.execute("USE Traffic")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Information (
                UID VARCHAR(50),
                LicensePlate VARCHAR(100),
                Type VARCHAR(50),
                Speed VARCHAR(50),
                Date VARCHAR(50),
                Time VARCHAR(50),
                Driving VARCHAR(50)
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("MySQL Ready.")
    except Exception as e:
        print("MySQL Init Error:", e)

def insert_mysql(uid, lp, vtype, speed, date, time, driving):
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            port=MYSQL_PORT,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        query = """
            INSERT INTO Information 
            (UID, LicensePlate, Type, Speed, Date, Time, Driving)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (uid, lp, vtype, str(speed), date, time, driving))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("MySQL Insert Error:", e)

init_mysql()

# ===========================================================
# PATHS AND DIRECTORIES (unchanged)
# ===========================================================
BASE_DIR = "Z:\\"
INPUT_DIR = os.path.join(BASE_DIR, "Input_video")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output_video")
IMG_SAVE_DIR = "Z:\\img"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# ===========================================================
# LOAD MODELS AND SCALER
# ===========================================================
path_YOLO = "Z:\\ML_Models\\vehicle_detection.pt"
path_License = "Z:\\ML_Models\\LP-detection.pt"
LSTM_Driving = "Z:\\ML_Models\\drifting_detection_lstm_86_A6.h5"

vehicle_model = YOLO(path_YOLO)
plate_model = YOLO(path_License)
ocr = PaddleOCR(use_angle_cls=True, lang='en')
tracker = DeepSort(max_age=30)

try:
    lstm_model = load_model(LSTM_Driving, compile=False)
except Exception as e:
    print("Error loading LSTM model:", e)
    print(traceback.format_exc())
    lstm_model = None

scaler = StandardScaler()

# ===========================================================
# DataFrame Create store for input LSTM Model (columns unchanged)
# ===========================================================
column_names = ['Speed', 'Angle', 'Acceleration', 'Jerk', 'Vx', 'Vy']
df = pd.DataFrame(columns=column_names)

# Dictionaries to hold previous frame data for speed, acceleration, jerk calculation
prev_positions = {}
prev_speeds = {}
prev_accelerations = {}

# ===========================================================
# HELPERS (unchanged)
# ===========================================================
def generate_unique_id():
    return str(uuid.uuid4())[:8]

def estimate_speed(prev_centroid, curr_centroid, fps, scale=0.05):
    if prev_centroid is None:
        return 0
    dx = curr_centroid[0] - prev_centroid[0]
    dy = curr_centroid[1] - prev_centroid[1]
    distance_pixels = np.sqrt(dx ** 2 + dy ** 2)
    distance_meters = distance_pixels * scale
    speed_mps = distance_meters * fps
    return round(speed_mps * 3.6, 2)

def detect_plate_text(frame, vehicle_box):
    x1, y1, x2, y2 = map(int, vehicle_box)
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return "Unknown"
    vehicle_crop = frame[y1:y2, x1:x2]
    if vehicle_crop.size == 0:
        return "Unknown"
    try:
        plate_results = plate_model(vehicle_crop, conf=0.4, verbose=False)[0]
        for pbox in plate_results.boxes:
            px1, py1, px2, py2 = map(int, pbox.xyxy[0].cpu().numpy())
            px1 += x1; py1 += y1; px2 += x1; py2 += y1
            px1 = max(0, min(px1, w-1)); px2 = max(0, min(px2, w-1))
            py1 = max(0, min(py1, h-1)); py2 = max(0, min(py2, h-1))
            plate_crop = frame[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            ocr_result = ocr.ocr(plate_crop, cls=True)
            if ocr_result and len(ocr_result[0]) > 0:
                return ocr_result[0][0][1][0]
    except Exception:
        pass
    return "Unknown"

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)

saved_uids = set()
visibility_frames = {}

# ===========================================================
# PROCESS VIDEO FUNCTION WITH LSTM PREDICTION INTEGRATED
# Added debug prints for file existence and opening video.
# ===========================================================
def process_video(video_path, progress_actor=None):
    video_name = os.path.basename(video_path).replace(".mp4", "")
    print(f"Processing {video_name}...")
    print(f"Full video path: {video_path}")
    print(f"File exists: {os.path.exists(video_path)}")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video:", video_path)
            if progress_actor:
                try:
                    progress_actor.update.remote(video_name, -1)
                except:
                    pass
            return

        ret, test_frame = cap.read()
        if not ret:
            print("❌ Cannot read first frame:", video_path)
            if progress_actor:
                try:
                    progress_actor.update.remote(video_name, -1)
                except:
                    pass
            cap.release()
            return

        height, width = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        temp_output = os.path.join(OUTPUT_DIR, f"{video_name}_temp.mp4")
        final_output = os.path.join(OUTPUT_DIR, f"{video_name}_out.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        if not out.isOpened():
            print("❌ VideoWriter failed:", temp_output)
            if progress_actor:
                try:
                    progress_actor.update.remote(video_name, -1)
                except:
                    pass
            cap.release()
            return

        prev_positions = {}
        prev_speeds = {}
        prev_accelerations = {}

        unique_id_map = {}
        frame_idx = 0
        last_percent = -1

        captured = set()  # NEW SET to prevent duplicate captures

        if progress_actor:
            try:
                progress_actor.update.remote(video_name, 0)
            except Exception:
                pass

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            try:
                results = vehicle_model(frame, conf=0.4, verbose=False)[0]
            except Exception as e:
                print("Error running vehicle_model on frame:", e)
                print(traceback.format_exc())
                continue

            detections = []
            yolo_boxes = []

            for box in results.boxes:
                try:
                    cls = int(box.cls)
                    if cls in [2, 3, 5, 7]:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  
                        conf = float(box.conf)
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
                        yolo_boxes.append((x1, y1, x2, y2))
                except Exception:
                    continue

            try:
                tracks = tracker.update_tracks(detections, frame=frame)
            except Exception as e:
                print("Tracker update error:", e)
                print(traceback.format_exc())
                tracks = []

            H, W = frame.shape[:2]
            bottom_line = H - 5

            for track in tracks:
                try:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    if track_id not in unique_id_map:
                        unique_id_map[track_id] = generate_unique_id()
                    unique_id = unique_id_map[track_id]

                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                    speed = estimate_speed(prev_positions.get(track_id), centroid, fps)
                    angle = 0.0
                    if prev_positions.get(track_id) is not None:
                        dx = centroid[0] - prev_positions[track_id][0]
                        dy = centroid[1] - prev_positions[track_id][1]
                        angle = np.degrees(np.arctan2(dy, dx))

                    prev_speed = prev_speeds.get(track_id)
                    acceleration = 0
                    if prev_speed is not None:
                        acceleration = speed - prev_speed

                    prev_acceleration = prev_accelerations.get(track_id)
                    jerk = 0
                    if prev_acceleration is not None:
                        jerk = acceleration - prev_acceleration

                    if prev_positions.get(track_id) is not None:
                        dx_vx = (centroid[0] - prev_positions[track_id][0]) * 0.05 * fps * 3.6
                        dy_vy = (centroid[1] - prev_positions[track_id][1]) * 0.05 * fps * 3.6
                    else:
                        dx_vx, dy_vy = 0, 0

                    data_row = np.array([[speed, angle, acceleration, jerk, dx_vx, dy_vy]])

                    try:
                        if not hasattr(scaler, 'mean_'):
                            try:
                                scaler.partial_fit(data_row)
                            except Exception:
                                scaler.fit(data_row)
                    except Exception:
                        try:
                            scaler.fit(data_row)
                        except Exception:
                            pass

                    try:
                        data_scaled = scaler.transform(data_row)
                        data_scaled = data_scaled.reshape((1, 1, data_scaled.shape[1]))
                    except Exception:
                        data_scaled = data_row.reshape((1, 1, data_row.shape[1]))

                    driving_label = "Normal"
                    if lstm_model is not None:
                        try:
                            prediction = lstm_model.predict(data_scaled, verbose=0)
                            try:
                                val = float(prediction.reshape(-1)[0])
                                driving_label = "Drifting" if val > 0.7 else "Normal"
                            except Exception:
                                try:
                                    if prediction.shape[-1] >= 2:
                                        val2 = float(prediction.reshape(-1)[1])
                                        driving_label = "Drifting" if val2 > 0.7 else "Normal"
                                except Exception:
                                    driving_label = "Normal"
                        except Exception as e:
                            print("Error during LSTM prediction:", e)
                            print(traceback.format_exc())
                            driving_label = "Normal"

                    print(f"Frame {frame_idx} - Track {track_id} | Speed: {speed} km/h | Angle: {angle:.2f} | Accel: {acceleration:.2f} | Jerk: {jerk:.2f} | Vx: {dx_vx:.2f} | Vy: {dy_vy:.2f} | Driving: {driving_label}")

                    try:
                        df.loc[len(df)] = [speed, angle, acceleration, jerk, dx_vx, dy_vy]
                    except Exception:
                        pass

                    prev_positions[track_id] = centroid
                    prev_speeds[track_id] = speed
                    prev_accelerations[track_id] = acceleration

                    cls_id = track.det_class if hasattr(track, 'det_class') else 0
                    class_map = {2: 'Car', 3: 'Motorbike', 5: 'Bus', 7: 'Truck'}
                    vehicle_type = class_map.get(cls_id, 'Vehicle')

                    plate_text = detect_plate_text(frame, (x1, y1, x2, y2))

                    now = datetime.now()
                    try:
                        insert_mysql(unique_id, plate_text, vehicle_type, speed,
                            now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), driving_label)
                    except Exception:
                        print("Insert mysql failed for UID", unique_id)

                    # ======= NEW IMAGE CAPTURE LOGIC STARTS HERE =======
                    # Match YOLO box by IoU to get clean crop box:
                    best_iou_score = 0
                    best_yolo_box = None
                    track_box = (x1, y1, x2, y2)

                    for yb in yolo_boxes:
                        iou_score = iou(track_box, yb)
                        if iou_score > best_iou_score:
                            best_iou_score = iou_score
                            best_yolo_box = yb

                    if best_yolo_box is not None:
                        cx1, cy1, cx2, cy2 = map(int, best_yolo_box)

                        # Capture when DeepSORT box bottom touches or crosses bottom line
                        if y2 >= bottom_line:
                            if unique_id not in captured:
                                crop = frame[cy1:cy2, cx1:cx2]
                                if crop.size > 0:
                                    fname = os.path.join(IMG_SAVE_DIR, f"{unique_id}_{uuid.uuid4().hex}.jpg")
                                    cv2.imwrite(fname, crop)
                                    captured.add(unique_id)
                                    print("Captured:", fname)
                    # ======= IMAGE CAPTURE LOGIC ENDS HERE =======

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{unique_id} | {vehicle_type} | {speed} km/h | {plate_text}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                except Exception as e:
                    print("Per-track processing error:", e)
                    print(traceback.format_exc())
                    continue

            out.write(frame)

            if total_frames > 0:
                percent = int((frame_idx / total_frames) * 100)
                if percent != last_percent:
                    last_percent = percent
                    if progress_actor:
                        try:
                            progress_actor.update.remote(video_name, percent)
                        except Exception:
                            pass

        cap.release()
        out.release()

        if progress_actor:
            try:
                progress_actor.update.remote(video_name, 100)
            except Exception:
                pass

        print(f"🟡 Processing finished. Matching bitrate...")

        try:
            match_bitrate(video_path, temp_output, final_output)
        except Exception as e:
            print("match_bitrate error:", e)
            print(traceback.format_exc())

        print(f"✅ Final Saved video: {final_output}")
        print("Sample DataFrame rows:")
        print(df.tail(10))

    except Exception as e:
        print("❌ CRASH inside process_video:", e)
        print(traceback.format_exc())
        if progress_actor:
            try:
                progress_actor.update.remote(video_name, -1)
            except Exception:
                pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            out.release()
        except Exception:
            pass
        return

    # Run human.py and challan_gen.py after video processing
    print("Running human.py...")
    human_path = "C:\\Users\\ussam\\OneDrive\Desktop\\Ray_Project\\human.py"
    if not os.path.exists(human_path):
        print("ERROR: human.py not found at:", human_path)
    else:
        try:
            subprocess.run([sys.executable, human_path], check=True)
            print("human.py completed!")
        except Exception as e:
            print("Error running human.py:", e)
            print(traceback.format_exc())

    print("Running challan_gen.py...")
    challan_path = "C:\\Users\\ussam\\OneDrive\\Desktop\\Ray_Project\\challan_gen.py"
    if not os.path.exists(challan_path):
        print("ERROR: challan_gen.py not found at:", challan_path)
    else:
        try:
            subprocess.run([sys.executable, challan_path], check=True)
            print("challan_gen.py completed!")
        except Exception as e:
            print("Error running challan_gen.py:", e)
            print(traceback.format_exc())

# =============================
# Ray actor class for progress reporting
# =============================
if ray:
    @ray.remote
    class ProgressActor:
        def __init__(self):
            self.progress = {}

        def update(self, video_name, percent):
            self.progress[video_name] = percent
            print(f"Progress update: {video_name} = {percent}%")

else:
    ProgressActor = None