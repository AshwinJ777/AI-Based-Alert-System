# AI-Based Intersection Collision Prediction System (V2I Alert System)

An end-to-end intelligent transportation safety pipeline that monitors traffic intersections, detects and tracks vehicles, predicts future trajectories, calculates Time-to-Collision (TTC) risk, and simulates Vehicle-to-Infrastructure (V2I) collision warning alerts in real time.

---

## Architecture Overview

The system consists of **6 modular pipeline stages**:

```
[Module 1: Camera Feed]
         │  (Frames + Timestamps)
         ▼
[Module 2: Vehicle Detection (YOLOv8)]
         │  (Bounding Boxes + Classes + Confidences)
         ▼
[Module 3: Multi-Object Tracking (ByteTrack)]
         │  (Persistent Track IDs + Historical Coordinates)
         ▼
[Module 4: Trajectory Prediction (Kalman Filter)]
         │  (Future Trajectories + Velocity Vectors)
         ▼
[Module 5: Risk Scoring (TTC Calculation)]
         │  (Collision Risks + Severity Levels)
         ▼
[Module 6: Alert Generation & V2I Simulation]
         │  (Deduplicated Alerts + JSONL Log)
         ▼
[Streamlit Web Dashboard]
```

---

## Prerequisites & Installation

1. **Python 3.10+** installed.
2. Clone or navigate to the project directory:
   ```powershell
   cd c:\Users\charu\Desktop\Project\AI-Based-Alert-System
   ```
3. Install all required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   *(Includes `ultralytics`, `opencv-python`, `supervision`, `filterpy`, `streamlit`, `pytest`, `pyyaml`, `numpy`, `matplotlib`)*

---

## How to Run Every Module Individually

### Module 1: Camera Feed (Input Layer)
Handles video ingestion from local video files (`.mp4`, `.avi`), webcams, or RTSP streams with sequential indexing, timestamping, and frame skipping.

- **Test on a local video file:**
  ```powershell
  python test_camera_feed.py --source data/raw_videos/videoplayback.mp4
  ```
- **Test on a live webcam (device 0):**
  ```powershell
  python test_camera_feed.py --source 0
  ```
- **With frame skipping and custom resolution:**
  ```powershell
  python test_camera_feed.py --source data/raw_videos/videoplayback.mp4 --skip 2 --resize 640x480
  ```
  *(Press **Q** to exit the playback window)*

---

### Module 2: Vehicle Detection (YOLOv8)
Detects vehicles (cars, motorcycles, buses, trucks, auto rickshaws, bicycles, pedestrians) using YOLOv8 with configurable confidence thresholds and class filtering.

- **Run detection on video:**
  ```powershell
  python detection/test_detector.py --source data/raw_videos/videoplayback.mp4
  ```
- **Run detection on webcam:**
  ```powershell
  python detection/test_detector.py --source 0
  ```
- **With custom frame skip:**
  ```powershell
  python detection/test_detector.py --source data/raw_videos/videoplayback.mp4 --skip 2
  ```
  *(Configured via [`detection/config.yaml`](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/detection/config.yaml))*

---

### Module 3: Multi-Object Tracking (ByteTrack)
Maintains persistent IDs and historical center coordinates (last 10–15 positions) across frames using ByteTrack.

- **Run tracking on video:**
  ```powershell
  python tracking/test_tracker.py --source data/raw_videos/videoplayback.mp4
  ```
- **Run tracking on webcam:**
  ```powershell
  python tracking/test_tracker.py --source 0
  ```
- **With frame skip:**
  ```powershell
  python tracking/test_tracker.py --source data/raw_videos/videoplayback.mp4 --skip 2
  ```
  *(Configured via [`tracking/config.yaml`](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/tracking/config.yaml))*

---

### Module 4: Trajectory Prediction (Kalman Filter)
Estimates vehicle velocity vectors and predicts future 2D positions over configurable horizons (e.g. 0.5s, 1.0s, 1.5s, 2.0s) using a constant-velocity Kalman Filter. Includes camera-to-world homography calibration support.

- **Run the Visual Demo (generates plot `prediction/trajectory_prediction_demo.png`):**
  ```powershell
  python prediction/visualize_demo.py
  ```
  *Simulates straight, turning, and accelerating trajectories and plots predicted Kalman positions alongside ground truth.*

- **Run unit tests:**
  ```powershell
  python -m pytest prediction/test_prediction.py -v
  ```

---

### Module 5: Risk Scoring (TTC - Time To Collision)
Evaluates all pairs of tracked vehicles within proximity. Calculates relative distance, relative velocity, and Time-to-Collision (TTC). Flags collision risks into **HIGH** (TTC < 1.0s) or **MEDIUM** (1.0s ≤ TTC < 2.0s) severity.

- **Run unit tests (covers head-on, diverging, parallel, proximity filtering, and edge cases):**
  ```powershell
  python -m pytest risk_scoring/test_ttc.py -v
  ```
  *(Configured via [`risk_scoring/config.yaml`](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/risk_scoring/config.yaml))*

---

### Module 6: Alert Generation & V2I Simulation
Generates standardized V2I collision warning alert messages with cooldown-based deduplication (preventing alert spam for the same pair of vehicles) and logs every alert to a JSONL audit file (`alerts/alert_log.jsonl`).

- **Run unit tests (deduplication, cooldown window, JSONL logging):**
  ```powershell
  python -m pytest alerts/test_alerts.py -v
  ```
  *(Configured via [`alerts/config.yaml`](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/alerts/config.yaml))*

---

## Running the End-to-End Pipeline

### 1. Simulated Demo (No Video Required)
Runs Modules 4 → 5 → 6 on synthetic multi-vehicle intersection trajectories to verify the prediction, risk scoring, and alert generation pipeline without needing a video file or GPU:

```powershell
python pipeline/run_demo.py
```

### 2. Full 6-Module Live Pipeline (Video File)
Chains all 6 modules end-to-end: **CameraFeed → YOLOv8 → ByteTrack → Kalman Predictor → Risk Scorer → Alert Generator**:

```powershell
python pipeline/run_demo.py --video data/raw_videos/videoplayback.mp4
```

- Outputs real-time progress, tracked vehicle counts, and formatted `[ALERT]` events to console.
- Automatically writes all generated alerts to [`alerts/alert_log.jsonl`](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/alerts/alert_log.jsonl).

---

## Running the Streamlit Web Dashboard

An interactive dashboard providing real-time video playback with bounding boxes, predicted future trajectory trails, active risk overlays, live statistics, and alert history.

Launch the dashboard:
```powershell
python -m streamlit run alerts/dashboard.py
```

> **Note:** Use `python -m streamlit run` rather than bare `streamlit run` to ensure Python executes the package directly without requiring manual Windows PATH adjustments.

### Dashboard Features:
- **Live Tracking Mode:** Select any video from `data/raw_videos/`, configure frame skip, and click **Start Live Tracking** to process video frames with visual overlays and real-time alert cards.
- **Alert History Mode:** Displays total alerts, high/medium severity breakdown, unique vehicle pairs flagged, and a searchable/filterable historical table loaded from `alerts/alert_log.jsonl`.
- **Clear Alert Log:** Clear the alert log file directly from the sidebar.

---

## Running All Unit Tests

Run the complete test suite across all modules (44 tests total):

```powershell
python -m pytest prediction/ risk_scoring/ alerts/ -v
```

---

## Repository Structure

```
AI-Based-Alert-System/
├── camera_feed.py             # Module 1: Camera Feed reader
├── test_camera_feed.py        # Module 1: Test & standalone runner
├── detection/                 # Module 2: Vehicle Detection
│   ├── detector.py            # YOLOv8 vehicle detector class
│   ├── test_detector.py       # Module 2 test runner
│   └── config.yaml            # Confidence thresholds, target classes
├── tracking/                  # Module 3: Multi-Object Tracking
│   ├── tracker.py             # ByteTrack vehicle tracker
│   ├── test_tracker.py        # Module 3 test runner
│   └── config.yaml            # Track thresholds, max age, buffer size
├── prediction/                # Module 4: Trajectory Prediction
│   ├── kalman_filter.py       # 2D Constant Velocity Kalman Filter
│   ├── calibration.py         # Pixel-to-world homography calibration
│   ├── visualize_demo.py      # Trajectory visualization script
│   ├── test_prediction.py     # Unit test suite
│   └── config.yaml            # Prediction horizons, noise matrices
├── risk_scoring/              # Module 5: Risk Scoring (TTC)
│   ├── ttc.py                 # Time-to-Collision calculation engine
│   ├── test_ttc.py            # Unit test suite
│   └── config.yaml            # TTC thresholds (high/med), proximity limit
├── alerts/                    # Module 6: Alert Generation & V2I Simulation
│   ├── alert_generator.py     # Alert deduplication & JSONL logger
│   ├── dashboard.py           # Streamlit Web Dashboard
│   ├── alert_log.jsonl        # Output alert log
│   ├── test_alerts.py         # Unit test suite
│   └── config.yaml            # Cooldown seconds, log path
├── pipeline/                  # End-to-End Orchestration
│   └── run_demo.py            # Unified runner (simulated or full video)
├── data/raw_videos/           # Video input folder (e.g. videoplayback.mp4)
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation & run guide
```
