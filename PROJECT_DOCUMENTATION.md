# 📘 Comprehensive Technical & Architectural Documentation

## Executive Overview

The **AI Face Sentinel System** provides real-time biometric face attendance verification with integrated presentation attack detection (anti-spoofing). It combines deep neural network vision models for object localization, binary spoof classification, and metric feature space face vector comparison, served by a high-performance **FastAPI** backend with Uvicorn server integration.

---

## 🔬 System Components & Machine Learning Models

### 1. Primary Face Detection Model (`yolov8m_200e.pt`)
- **Architecture**: Ultralytics YOLOv8 Medium (YOLOv8m) customized anchor-free detector.
- **Training Duration**: 200 Epochs.
- **Input Resolution**: Flexible height and width (letterboxed to \(640 \times 640\)).
- **Output**: Bounding boxes in \(xyxy\) format \([x_1, y_1, x_2, y_2]\), confidence score \(S_{det} \in [0, 1]\), and class ID 0 (`face`).

### 2. Anti-Spoofing & Liveness Model (`newmodel.pt`)
- **Architecture**: YOLOv8 fine-tuned binary classifier for presentation attack detection (PAD).
- **Class Labels**:
  - `0`: `fake` (Printouts, digital screens, photos, cutouts)
  - `1`: `real` (Genuine human face)
- **Execution Pipeline**: Evaluated directly on cropped facial Regions of Interest (ROI) to isolate high-frequency spectral artifacts, screen moiré patterns, and edge reflection anomalies.

### 3. Face Recognition & Vector Matching (`face_recognition` / `dlib`)
- **Embedding Dimensions**: 128-dimensional dense continuous vector space.
- **Distance Metric**: Euclidean norm in 128D space:
  \[
  D(f_1, f_2) = \| \mathbf{v}_1 - \mathbf{v}_2 \|_2 = \sqrt{\sum_{i=1}^{128} (v_{1,i} - v_{2,i})^2}
  \]
- **Matching Threshold**: \(\text{Tolerance} = 0.48\).
- **Confidence Metric Calculation**:
  \[
  \text{Confidence (\%)} = \max\left(0, \min\left(100, \left(1.0 - D(f_1, f_2)\right) \times 100\right)\right)
  \]

---

## 🛠️ Data Flow & Sequential Pipeline

1. **Frame Capture**: Webcam captures raw BGR image array via OpenCV `cv2.VideoCapture(0)`.
2. **Face Detection**: Frame is passed to YOLOv8 detector (`yolov8m_200e.pt`). If faces are detected, bounding box coordinates are extracted.
3. **Region of Interest (ROI) Extraction**: Crop bounding box ROI:
   \[
   \text{ROI} = \text{Frame}[y_1:y_2, x_1:x_2]
   \]
   If ROI dimensions are less than \(20 \times 20\) pixels, detection is discarded as invalid/small.
4. **Anti-Spoofing Verification**:
   - `ROI` is converted to RGB and fed to `spoof_model` (`newmodel.pt`).
   - If predicted class is `fake` or confidence is below threshold (\(0.35\)), the frame overlay is drawn in **Red** with label `SPOOF / FAKE DETECTED`. A spoof attempt counter is incremented, and log event is triggered.
5. **Biometric Face Encoding & Matching**:
   - If anti-spoofing passes (`real`), exact face location \((y_1, x_2, y_2, x_1)\) is passed to `face_recognition.face_encodings(rgb_frame, known_face_locations=[...])`.
   - Compute Euclidean distances against pre-loaded `known_face_encodings`.
   - If minimum distance \(D \le 0.48\), retrieve matching name and mark attendance. Frame overlay is drawn in **Neon Green** with label `REAL: <Name> (<Confidence>%)`.
   - If distance \(D > 0.48\), label as `Unknown Person`. Frame overlay is drawn in **Cyan**.
6. **CSV Persistence & Idempotency**:
   - `mark_attendance(name)` appends `[Name, Time]` to `attendance/YYYY-MM-DD.csv`.
   - `attendance_marked` in-memory set prevents duplicate writes for the same individual on the same calendar day.

---

## 🌐 Web Architecture & REST API Endpoints

The system is constructed with a high-performance **FastAPI** async backend serving JSON APIs, MJPEG video streams via `StreamingResponse`, file downloads via `FileResponse`, and HTML template rendering via `Jinja2Templates`.

### Key FastAPI Architectural Highlights
- **Lifespan Context Manager**: `@asynccontextmanager` initializes face embeddings and synchronizes today's CSV attendance on application startup cleanly.
- **Asynchronous Multipart Handling**: `UploadFile` and `Form(...)` support multipart form processing for face registration via file upload or base64 webcam data URL.
- **Automated Interactive Documentation**: Generates Swagger UI (`/docs`) and ReDoc (`/redoc`) specifications automatically based on type annotations and endpoint metadata tags (`System`, `Attendance`, `Face Registry`).

### Security & Sanitization
- File upload filenames are sanitized using regex: `re.sub(r'[^a-zA-Z0-9_\- ]', '', name)` to eliminate path traversal vulnerabilities.
- Registered faces are validated immediately upon upload using `face_recognition.face_encodings()`. If no valid face is present in the image, registration is rejected with an informative error message.

---

## 🚀 Performance Metrics & Hardware Acceleration

| Configuration | Detection FPS | Spoof Check Latency | Embedding Latency |
| :--- | :--- | :--- | :--- |
| **CUDA GPU (NVIDIA)** | ~45-60 FPS | ~8 ms | ~15 ms |
| **CPU (Multi-Core)** | ~18-25 FPS | ~35 ms | ~45 ms |

---

## 🛠️ Maintenance & Troubleshooting

1. **Camera Not Detected**:
   Ensure no other application (Zoom, Teams, Skype) is using webcam `0`.
2. **Duplicate Attendance**:
   `app.py` automatically synchronizes today's CSV file on startup. If records were edited manually, trigger server restart to sync.
3. **Adding Users via CLI / Directory**:
   Save any front-facing photo as `<name>.jpg` into `known_faces/` directory and restart the FastAPI server to auto-index encodings.
