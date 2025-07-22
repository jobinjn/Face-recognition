
from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime
import torch
from detection import get_model, detect_faces
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load detection and spoofing models
model = get_model("yolov8m_200e.pt")

known_face_encodings = []
known_face_names = []
attendance_marked = set()

# Load YOLO anti-spoofing model globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
spoof_model = YOLO("newmodel.pt").to(device)
print(spoof_model.names)
classNames = spoof_model.names

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def is_real_face_yolo(face_img_np, threshold=0.3):
    # Resize + pad the face image
    padded_img, _, _ = letterbox(face_img_np, new_shape=(640, 640))
    rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

    # Run YOLO model
    results = spoof_model(rgb_img, verbose=False)[0]  # Assume spoof_model is a YOLOv8 object

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = classNames[cls_id].upper()

            print(f"[YOLO] Label: {label}, Confidence: {conf:.2f}")

            if conf > threshold:
                if label == "REAL":
                    # Run face recognition
                    face_encodings = face_recognition.face_encodings(face_img_np)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        if True in matches:
                            best_match_index = np.argmin(face_distances)
                            name = known_face_names[best_match_index]
                            print(f"[✅] Real face recognized: {name}")
                            return name
                        else:
                            print("[⚠️] Real face detected but not recognized.")
                            return "Unknown Real Face"
                    else:
                        print("[⚠️] Face encoding failed.")
                        return "Unknown Real Face"
                else:
                    print("[❌] Fake face detected.")
                    return "Fake"
            else:
                print(f"[INFO] Confidence {conf:.2f} below threshold {threshold}")
    else:
        print("[INFO] No face detected by YOLO or empty boxes.")

    return "Fake"


def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for file in os.listdir("known_faces"):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = face_recognition.load_image_file(os.path.join("known_faces", file))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(file)[0])
    print(f"[✅] Loaded known faces: {known_face_names}")

def mark_attendance(name):
    os.makedirs("attendance", exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join("attendance", f"{date_str}.csv")
    if name not in attendance_marked:
        attendance_marked.add(name)
        file_exists = os.path.exists(file_path)
        with open(file_path, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Time"])
            writer.writerow([name, datetime.now().strftime("%H:%M:%S")])
        print(f"[📌] Marked: {name}")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        try:
            _, boxes, scores, _ = detect_faces(model, [pil_img], box_format='xywh', th=0.5)
            boxes = boxes[0]
        except Exception as e:
            print(f"[❌] Detection error: {e}")
            boxes = []

        for (x_c, y_c, w, h) in boxes:
            x1 = int(x_c - w / 2)
            y1 = int(y_c - h / 2)
            x2 = int(x_c + w / 2)
            y2 = int(y_c + h / 2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face = frame[y1:y2, x1:x2]
            name = "Unknown"
            if face.size != 0:
                try:
                    recognized_name = is_real_face_yolo(frame)
                    if recognized_name not in ["Fake", "Unknown Real Face"]:
                        name = recognized_name
                        mark_attendance(name)
                    else:
                        print("[❌] Fake or unknown face. Skipping attendance.")
                except Exception as e:
                    print(f"[⚠️] Face processing error: {e}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance/today')
def get_today_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    return get_attendance(today)

@app.route('/attendance/<date>')
def get_attendance(date):
    file_path = os.path.join("attendance", f"{date}.csv")
    records = []
    if os.path.exists(file_path):
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                records.append({"name": row[0], "time": row[1]})
    return jsonify(records)

@app.route('/download/<date>')
def download_attendance(date):
    return send_from_directory('attendance', f"{date}.csv", as_attachment=True)

@app.route('/dates')
def available_dates():
    if not os.path.exists("attendance"):
        return jsonify([])
    files = [f[:-4] for f in os.listdir("attendance") if f.endswith(".csv")]
    return jsonify(sorted(files))

if __name__ == '__main__':
    load_known_faces()
    app.run(debug=True)
