# AI-Based Alert System — Walkthrough

This document outlines the current capabilities and modules of the Intersection Collision Prediction System.

## Architecture

The system consists of the following modules connected sequentially:

1.  **Camera Feed**: Handles video input.
2.  **Detection (YOLOv8)**: Detects vehicles in the frame.
3.  **Tracking (ByteTrack)**: Assigns persistent IDs to vehicles.
4.  **Scene Analysis (NEW)**: Infers traffic patterns and conflict zones.
5.  **Prediction (Kalman Filter)**: Projects future trajectories with uncertainty bounds.
6.  **Risk Scoring (TTC)**: Evaluates pairs of vehicles for collision risk.
7.  **Alert Generation**: Filters and logs risk events.
8.  **Dashboard**: Visualizes the pipeline live using Streamlit.

## Recent Enhancements (Part 2)

> [!NOTE]
> The system now relies entirely on data-driven observations to understand the road topology instead of using hardcoded/manual polygon lane definitions.

### 1. Scene Analysis & Traffic Movement Detection
We implemented a robust `scene_analysis` package that processes the historical trajectories emitted by the ByteTrack tracker:
*   **Movement Analyzer**: Computes a smoothed velocity vector and determines the cardinal direction (e.g., `EAST`, `SOUTHWEST`) of a vehicle, along with a confidence metric.
*   **Trajectory Clusterer**: Groups confident trajectories into dominant movement flows (e.g., identifying that there is a major stream of Eastbound and Southbound traffic).
*   **Conflict Zone Detector**: Evaluates crossing points between non-parallel movement flows to automatically learn and map out intersection "conflict zones" (represented as circles/ellipses).

### 2. Trajectory Uncertainty
The Kalman Filter (`TrajectoryPredictor`) was upgraded to extract the positional covariance matrix at each prediction horizon.
*   The `PredictedTrajectory` dictionary now includes an `uncertainties` list containing $2 \times 2$ covariance matrices.
*   This represents the growing uncertainty of the vehicle's position further into the future.

### 3. Smart Risk Scoring
The `RiskScorer` now acts more intelligently to avoid false positives and reduce computational overhead:
*   **Direction Prefilter**: Vehicles moving in the same or strictly opposite (parallel lanes) directions are filtered out before calculating TTC.
*   **Zone Prefilter**: If a projected collision point falls outside all discovered conflict zones, the risk event is ignored, preventing alerts for benign interactions occurring far from the actual intersection.

### 4. Advanced Visualization
The `collision_renderer.py` module and `dashboard.py` were heavily updated to visualize the new capabilities:
*   **Bounding Boxes**: Now display the discovered movement direction (e.g., `[EAST]`).
*   **Uncertainty Ellipses**: The Kalman covariance is rendered as translucent ellipses around future trajectory points. The ellipses grow larger at longer prediction horizons.
*   **Debug Mode**: A new toggle in the Streamlit UI enables a live-updating debug panel that shows active movement groups and overlays translucent circles over automatically discovered conflict zones.

## How to Run

### Command-Line Demo
Run the synthetic pipeline (no video required):
```bash
python pipeline/run_demo.py
```

### Full Pipeline (With Video)
```bash
python pipeline/run_demo.py --video path/to/video.mp4
```

### Live Streamlit Dashboard
Open a separate terminal and run:
```bash
python -m streamlit run alerts/dashboard.py
```
Check the "Debug Mode" toggle in the sidebar to see the Scene Analysis logic in action!
