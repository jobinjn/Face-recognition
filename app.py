# app.py
"""
AI Face Sentinel - Real-Time Face Attendance with Anti-Spoofing Verification
FastAPI High-Performance Web Application Backend
"""

import os
import cv2
import csv
import base64
import re
from datetime import datetime
from contextlib import asynccontextmanager
import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

import face_recognition
from ultralytics import YOLO

from detection import get_model, detect_faces

# Ensure required storage directories exist
os.makedirs("known_faces", exist_ok=True)
os.makedirs("attendance", exist_ok=True)

# Determine PyTorch hardware device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[SYSTEM] Running on device: {device.upper()}")

# Load machine learning models
detector_model = get_model("yolov8m_200e.pt")
spoof_model = YOLO("newmodel.pt").to(device)
classNames = spoof_model.names
print(f"[SPOOF] Anti-spoofing model classes: {classNames}")

# Global state
known_face_encodings = []
known_face_names = []
attendance_marked = set()
system_stats = {
    "fake_attempts": 0,
    "total_scans": 0
}
activity_logs = []


def log_event(message: str, status: str, confidence: float = 0.0):
    """Maintain rolling log of recent detection events."""
    global activity_logs
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "status": status,
        "confidence": confidence
    }
    if activity_logs and activity_logs[0]["message"] == message and activity_logs[0]["time"] == entry["time"]:
        return
    activity_logs.insert(0, entry)
    if len(activity_logs) > 50:
        activity_logs.pop()


def load_today_attendance():
    """Synchronize today's attendance records into memory to prevent duplicates."""
    global attendance_marked
    attendance_marked = set()
    today = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join("attendance", f"{today}.csv")
    if os.path.exists(file_path):
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row and len(row) > 0:
                    attendance_marked.add(row[0])
    print(f"[📌] Synchronized today's marked attendance ({len(attendance_marked)} records): {attendance_marked}")


def load_known_faces():
    """Load and compute 128D facial vectors for all images in known_faces/."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    for file in os.listdir("known_faces"):
        if file.lower().endswith(valid_exts):
            path = os.path.join("known_faces", file)
            name = os.path.splitext(file)[0]
            try:
                img = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"[✅] Registered face loaded: {name}")
                else:
                    print(f"[⚠️] Warning: No face encoding found in {file}")
            except Exception as e:
                print(f"[❌] Error loading face file {file}: {e}")
    print(f"[✅] Total active face encodings: {len(known_face_names)}")


def mark_attendance(name: str):
    """Record verified attendance to today's CSV file if not already recorded."""
    os.makedirs("attendance", exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join("attendance", f"{today}.csv")

    if name not in attendance_marked:
        attendance_marked.add(name)
        file_exists = os.path.exists(file_path)
        with open(file_path, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Time"])
            writer.writerow([name, datetime.now().strftime("%H:%M:%S")])
        print(f"[📌] Marked Attendance: {name} at {datetime.now().strftime('%H:%M:%S')}")


def evaluate_face(frame, box, threshold: float = 0.35):
    """
    Evaluates anti-spoofing on cropped face ROI and performs face recognition if real.

    Returns:
        status: 'REAL', 'FAKE', or 'UNKNOWN'
        display_name: Recognized name or warning string
        confidence: Confidence percentage (0-100)
    """
    h_img, w_img, _ = frame.shape
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return 'UNKNOWN', 'Face Too Small', 0.0

    face_crop = frame[y1:y2, x1:x2]
    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # 1. Anti-Spoofing Check
    spoof_results = spoof_model(rgb_crop, verbose=False)[0]

    is_real = False
    spoof_conf = 0.0

    if spoof_results.boxes is not None and len(spoof_results.boxes) > 0:
        boxes = spoof_results.boxes
        best_idx = int(torch.argmax(boxes.conf).item())
        conf = float(boxes.conf[best_idx].cpu().numpy())
        cls_id = int(boxes.cls[best_idx].cpu().numpy())
        label_str = classNames[cls_id].lower()

        if conf >= threshold:
            spoof_conf = conf
            is_real = (label_str == 'real')
        else:
            is_real = False
    else:
        is_real = True
        spoof_conf = 0.5

    if not is_real:
        system_stats["fake_attempts"] += 1
        conf_pct = round(spoof_conf * 100, 1)
        log_event("Spoofing Attempt Prevented!", "FAKE", conf_pct)
        return 'FAKE', 'Fake / Spoof Detected', conf_pct

    # 2. Real Face Verified -> Vector Comparison
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_location = [(y1, x2, y2, x1)]
    encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_location)

    if encodings and len(known_face_encodings) > 0:
        face_encoding = encodings[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.48)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if True in matches:
            best_match_index = int(np.argmin(face_distances))
            match_dist = face_distances[best_match_index]
            confidence = round((1.0 - match_dist) * 100, 1)
            name = known_face_names[best_match_index]
            mark_attendance(name)
            log_event(f"Attendance Verified: {name}", "REAL", confidence)
            return 'REAL', name, confidence
        else:
            conf_pct = round(spoof_conf * 100, 1)
            log_event("Unknown Real Face", "UNKNOWN", conf_pct)
            return 'UNKNOWN', 'Unknown Person', conf_pct
    else:
        if len(known_face_encodings) == 0:
            return 'UNKNOWN', 'No Faces Registered', 0.0
        conf_pct = round(spoof_conf * 100, 1)
        return 'UNKNOWN', 'Unknown Person', conf_pct


def gen_frames():
    """Webcam streaming frame generator for MJPEG StreamingResponse."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break

        system_stats["total_scans"] += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        try:
            _, boxes, scores, _ = detect_faces(detector_model, [pil_img], box_format='xyxy', th=0.5)
            face_boxes = boxes[0] if boxes else []
        except Exception as e:
            print(f"[❌] Face detection error: {e}")
            face_boxes = []

        for box in face_boxes:
            status_str, name, confidence = evaluate_face(frame, box)
            x1, y1, x2, y2 = [int(v) for v in box]

            if status_str == 'REAL':
                color = (0, 230, 118)
                label_text = f"REAL: {name} ({confidence}%)"
            elif status_str == 'FAKE':
                color = (0, 0, 255)
                label_text = f"SPOOF / FAKE ({confidence}%)"
            else:
                color = (0, 191, 255)
                label_text = f"UNKNOWN ({confidence}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w_txt, h_txt), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + w_txt + 10, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if status_str == 'REAL' else (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# Lifespan Context Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_today_attendance()
    load_known_faces()
    yield

# Initialize FastAPI App
tags_metadata = [
    {"name": "System", "description": "Core system status, dashboard HTML interface, and live video feeds"},
    {"name": "Attendance", "description": "Attendance tracking records, daily history queries, and CSV file downloads"},
    {"name": "Face Registry", "description": "Face registration, thumbnail serving, and face profile deletion"},
]

app = FastAPI(
    title="AI Face Sentinel Portal",
    description="Real-Time Face Recognition & Liveness Anti-Spoofing System powered by FastAPI and PyTorch",
    version="2.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan
)

# Mount Static Files and Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Routes & Endpoints

@app.get("/", response_class=HTMLResponse, tags=["System"], summary="Main Dashboard Interface")
async def index(request: Request):
    """Render main Glassmorphism dashboard user interface."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/video_feed", tags=["System"], summary="Webcam Video Stream")
async def video_feed():
    """Stream live webcam feed with YOLO face detection and liveness anti-spoofing overlays (MJPEG)."""
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stats", tags=["System"], summary="Live System Statistics")
async def get_stats():
    """Fetch live counters for total registered users, marked attendance today, spoof attempts blocked, and GPU status."""
    today = datetime.now().strftime("%Y-%m-%d")
    return JSONResponse(content={
        "total_registered": len(known_face_names),
        "marked_today": len(attendance_marked),
        "fake_prevented": system_stats["fake_attempts"],
        "device": device.upper(),
        "today_date": today,
        "cuda_available": torch.cuda.is_available()
    })


@app.get("/attendance/today", tags=["Attendance"], summary="Today's Attendance Logs")
@app.get("/api/attendance/today", tags=["Attendance"], summary="Today's Attendance Logs (API)")
async def get_today_attendance():
    """Retrieve attendance verification records for today's date."""
    today = datetime.now().strftime("%Y-%m-%d")
    return await get_attendance(today)


@app.get("/attendance/{date}", tags=["Attendance"], summary="Attendance Logs for Specified Date")
@app.get("/api/attendance/{date}", tags=["Attendance"], summary="Attendance Logs for Specified Date (API)")
async def get_attendance(date: str):
    """Retrieve attendance verification records for a specific date (YYYY-MM-DD)."""
    file_path = os.path.join("attendance", f"{date}.csv")
    records = []
    if os.path.exists(file_path):
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    records.append({"name": row[0], "time": row[1]})
    return JSONResponse(content=records)


@app.get("/download/{date}", tags=["Attendance"], summary="Download Attendance CSV File")
async def download_attendance(date: str):
    """Download raw CSV file for a given attendance date."""
    file_path = os.path.join("attendance", f"{date}.csv")
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=f"{date}.csv",
            media_type="text/csv"
        )
    return JSONResponse(status_code=404, content={"error": "File not found"})


@app.get("/dates", tags=["Attendance"], summary="List Available Attendance Dates")
@app.get("/api/dates", tags=["Attendance"], summary="List Available Attendance Dates (API)")
async def available_dates():
    """List all calendar dates for which attendance CSV files exist."""
    if not os.path.exists("attendance"):
        return JSONResponse(content=[])
    files = [f[:-4] for f in os.listdir("attendance") if f.endswith(".csv")]
    return JSONResponse(content=sorted(files, reverse=True))


@app.get("/api/faces", tags=["Face Registry"], summary="List Registered Face Profiles")
async def list_faces():
    """List all registered person profiles and thumbnail image endpoints."""
    faces = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    for file in os.listdir("known_faces"):
        if file.lower().endswith(valid_exts):
            name = os.path.splitext(file)[0]
            faces.append({
                "name": name,
                "filename": file,
                "thumbnail_url": f"/api/faces/thumbnail/{name}"
            })
    return JSONResponse(content=faces)


@app.get("/api/faces/thumbnail/{name}", tags=["Face Registry"], summary="Get Face Thumbnail Image")
async def get_face_thumbnail(name: str):
    """Serve photo thumbnail for a registered face profile."""
    valid_exts = ['.jpg', '.jpeg', '.png', '.webp']
    for ext in valid_exts:
        path = os.path.join("known_faces", f"{name}{ext}")
        if os.path.exists(path):
            return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "Thumbnail not found"})


@app.post("/api/register_face", tags=["Face Registry"], summary="Register New Face Profile")
async def register_face(
    name: str = Form(...),
    image: UploadFile = File(None),
    image_data: str = Form(None)
):
    """
    Register a new person in the system by processing an uploaded photo or camera snapshot.
    Validates face visibility, extracts 128D facial embeddings, and saves profile to disk.
    """
    try:
        if not name or not name.strip():
            return JSONResponse(status_code=400, content={"success": False, "message": "Name is required"})

        sanitized_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', name).replace(' ', '_').lower()
        save_path = os.path.join("known_faces", f"{sanitized_name}.jpg")

        if image is not None and image.filename != '':
            contents = await image.read()
            with open(save_path, "wb") as f:
                f.write(contents)
        elif image_data:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            with open(save_path, 'wb') as f:
                f.write(img_bytes)
        else:
            return JSONResponse(status_code=400, content={"success": False, "message": "No image file or camera snapshot provided"})

        img = face_recognition.load_image_file(save_path)
        encodings = face_recognition.face_encodings(img)
        if not encodings:
            if os.path.exists(save_path):
                os.remove(save_path)
            return JSONResponse(status_code=400, content={"success": False, "message": "No clear face detected in the image. Please upload a clear front-facing photo."})

        load_known_faces()
        log_event(f"Registered new face: {sanitized_name}", "REAL", 100.0)
        return JSONResponse(content={"success": True, "message": f"Successfully registered '{sanitized_name}'!", "name": sanitized_name})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Error registering face: {str(e)}"})


@app.delete("/api/faces/{name}", tags=["Face Registry"], summary="Delete Registered Face Profile")
async def delete_face(name: str):
    """Delete a registered face profile from disk and update active feature vectors."""
    valid_exts = ['.jpg', '.jpeg', '.png', '.webp']
    deleted = False
    for ext in valid_exts:
        path = os.path.join("known_faces", f"{name}{ext}")
        if os.path.exists(path):
            os.remove(path)
            deleted = True
            break

    if deleted:
        load_known_faces()
        log_event(f"Deleted face: {name}", "UNKNOWN", 0.0)
        return JSONResponse(content={"success": True, "message": f"Deleted face '{name}'"})
    return JSONResponse(status_code=404, content={"success": False, "message": "Face not found"})


@app.get("/api/logs", tags=["System"], summary="Live Detection Logs")
async def get_logs():
    """Retrieve rolling activity log of recent detection events."""
    return JSONResponse(content=activity_logs)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)

