- [x] Part 1 — Automatic Traffic Movement Detection
    - [x] Create `scene_analysis/config.yaml` with parameters for movement analysis and trajectory clustering.
    - [x] Implement `scene_analysis/movement_analyzer.py` to calculate smoothed movement vectors from historical trajectories and map them to cardinal directions (e.g. "EAST").
    - [x] Implement `scene_analysis/trajectory_clusterer.py` to group vehicles into movement patterns (e.g., "Eastbound traffic") based on the tracked directions.

- [x] Part 2 — Automatic Intersection/Road Zone Understanding
    - [x] Implement `scene_analysis/conflict_zone_detector.py` to identify crossing points between conflicting movement groups and establish automatic conflict zones (circles/ellipses).
    - [x] Update `prediction/config.yaml` and `risk_scoring/config.yaml` to include toggles and parameters for conflict zone integration.
    - [x] Modify `risk_scoring/ttc.py`'s `evaluate` method. Add pre-filtering logic:
        - If two vehicles are moving in the same or opposite directions without intersection, skip TTC calculation (direction filter).
        - If the projected collision point is outside of any active conflict zone, skip TTC calculation (zone filter).

- [x] Part 3 — Trajectory Confidence / Uncertainty
    - [x] Update `prediction/kalman_filter.py` -> `TrajectoryPredictor._predict_future`.
    - [x] Extract the $2 \times 2$ position covariance matrix (`kf.P[:2, :2]`) at each prediction horizon.
    - [x] Include `uncertainties`: `[np.ndarray(2,2)]` in the returned `PredictedTrajectory` dictionary alongside the mean points.

- [x] Part 4 — Visualizing the Enhancements
    - [x] Modify `alerts/collision_renderer.py` to extract `uncertainties` and use OpenCV to draw translucent uncertainty ellipses at predicted locations.
    - [x] Add directional labels (e.g., `[EAST]`) to vehicle bounding boxes in `collision_renderer.py`.
    - [x] Implement a "Debug Mode" toggle in `dashboard.py` to show translucent circles for active conflict zones and an overlay panel summarizing discovered movement groups.

- [x] Verification
    - [x] Create `scene_analysis/test_scene_analysis.py` to unit test direction logic, zone clustering, and uncertainty propagation.
    - [x] Run `pytest` on `risk_scoring`, `prediction`, and `scene_analysis` to ensure changes haven't broken the pipeline.
    - [x] Test the modified dashboard manually using `run_demo.py` if needed.
