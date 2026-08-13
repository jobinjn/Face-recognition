# 🛡️ AI Face Sentinel: Real-Time Face Attendance & Anti-Spoofing System

An enterprise-grade, real-time web portal for automated face attendance verification and liveness anti-spoofing detection. Built with **FastAPI**, **Uvicorn**, **PyTorch**, **YOLOv8**, **face_recognition**, and a sleek **Glassmorphic Dark UI Dashboard**.

---

## 🌟 Key Features

- 📸 **Real-Time Video Stream Recognition**: High-FPS webcam stream processing using YOLOv8 face detection coupled with dlib 128D deep facial embedding matching.
- 🛡️ **AI Liveness Anti-Spoofing Verification**: Integrates custom-trained YOLOv8 anti-spoofing model (`newmodel.pt`) to detect and block presentation attacks (photos, phone screens, printed masks).
- ⚡ **FastAPI & Uvicorn High-Performance Backend**: Async event processing, lifespan startup handlers, high-concurrency streaming response, and OpenAPI auto-documentation (`/docs` and `/redoc`).
- 📊 **Glassmorphic Dark UI Dashboard**: Interactive dark-slate dashboard with real-time statistics (Registered Users, Attendance Today, Spoof Attempts Blocked, Hardware Device Status).
- 📅 **Idempotent Attendance Tracking**: Daily auto-generated CSV logs stored under `attendance/YYYY-MM-DD.csv`. Prevents duplicate attendance logging upon server restarts.
- 👤 **Face Registry Manager**: Web-based interface to manage registered users. Supports uploading local photo files or taking direct live snapshots from webcam.
- ⚡ **CUDA GPU & CPU Support**: Automatic hardware detection for optimal acceleration via PyTorch.
- 🔌 **Comprehensive REST API**: Full JSON REST API suite with auto-generated OpenAPI / Swagger docs.

---

## 📂 Project Architecture & Directory Structure

```
d:\face_attendance_web\
├── app.py                      # FastAPI Application Entry Point & REST APIs (Uvicorn)
├── detection.py                # YOLO Face Detection & ROI Utility Module
├── requirements.txt            # Python Package Dependencies File
├── model.pt                    # Anti-Spoofing YOLO Model Weights (Alternative)
├── newmodel.pt                 # Anti-Spoofing YOLO Model Weights (Primary: {0: fake, 1: real})
├── yolov8l_100e.pt             # YOLOv8 Large Face Detector Weights
├── yolov8m_200e.pt             # YOLOv8 Medium Face Detector Weights (Primary)
├── attendance.csv              # Legacy Sample CSV
├── attendance/                 # Daily CSV Storage Directory
│   ├── 2025-07-17.csv
│   └── 2025-07-18.csv
├── known_faces/                # Registered User Face Images (.jpg / .png)
├── static/
│   ├── style.css               # Glassmorphic Theme CSS
│   └── script.js               # Dashboard & Web Interactivity Script
├── templates/
│   └── index.html              # Dashboard Web Interface Template
├── README.md                   # System Documentation Guide
└── PROJECT_DOCUMENTATION.md    # Technical Architecture & Pipeline Deep Dive
```

---

## ⚙️ Prerequisites & Installation

### 1. Python Environment Setup
Ensure **Python 3.9+** is installed on your system.

### 2. Install Required Dependencies
Run the following command to install required Python libraries:

```bash
pip install -r requirements.txt
```

Alternatively, install dependencies manually:
```bash
pip install fastapi uvicorn[standard] jinja2 python-multipart opencv-python numpy torch torchvision ultralytics face_recognition Pillow
```

> 💡 **GPU Acceleration Note**: To enable CUDA GPU support for faster inference, install PyTorch matching your CUDA runtime version from [PyTorch.org](https://pytorch.org/).

---

## 🚀 Running the Web Portal

1. Open your terminal in the workspace directory:
   ```bash
   cd d:\face_attendance_web
   ```

2. Start the FastAPI application with Uvicorn:
   ```bash
   python app.py
   ```
   *or using Uvicorn CLI:*
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 5000 --reload
   ```

3. Open your web browser and navigate to:
   - **Web Dashboard**: `http://localhost:5000`
   - **Interactive API Docs (Swagger UI)**: `http://localhost:5000/docs`
   - **ReDoc Documentation**: `http://localhost:5000/redoc`

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Main Glassmorphism Web Dashboard |
| `/video_feed` | `GET` | MJPEG Web Camera Video Stream with Bounding Overlays |
| `/api/stats` | `GET` | Live System Stats (Total Registered, Marked Today, Fake Attempts Blocked, Device Type) |
| `/api/attendance/today` | `GET` | Attendance records for today's date in JSON format |
| `/api/attendance/{date}` | `GET` | Attendance records for specified date (`YYYY-MM-DD`) |
| `/download/{date}` | `GET` | Download raw CSV file for specified date |
| `/api/dates` | `GET` | List of available attendance dates |
| `/api/faces` | `GET` | List of registered face profiles and thumbnail URLs |
| `/api/faces/thumbnail/{name}` | `GET` | Serves registered face thumbnail image |
| `/api/register_face` | `POST` | Register a new face via file upload (`image`) or webcam snapshot (`image_data`) |
| `/api/faces/{name}` | `DELETE` | Removes registered person from disk and memory |
| `/api/logs` | `GET` | Rolling activity log of recent detection events |

---

## 🛡️ Anti-Spoofing & Liveness Pipeline

```
                     ┌───────────────────────┐
                     │   Webcam Frame Input  │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  YOLO Face Detector   │ (yolov8m_200e.pt)
                     └───────────┬───────────┘
                                 │
                     [ Bounding Box Face Crop ]
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ YOLO Anti-Spoofing    │ (newmodel.pt)
                     └───────────┬───────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
             [ Class = FAKE ]          [ Class = REAL ]
                    │                         │
                    ▼                         ▼
         ┌─────────────────────┐   ┌─────────────────────┐
         │ Block & Log Spoof   │   │  face_recognition   │
         │ Alert in Dashboard  │   │  128D Vector Match  │
         └─────────────────────┘   └───────────┬─────────┘
                                               │
                                               ▼
                                   ┌─────────────────────┐
                                   │  Mark Attendance in │
                                   │   CSV + Live UI     │
                                   └─────────────────────┘
```

---

## 📄 License & Attribution
Distributed for commercial and educational use in automated face recognition & attendance management.
