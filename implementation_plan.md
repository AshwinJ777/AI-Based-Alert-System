# Collision Visualization & TTC Improvement — Implementation Plan

## Goal

Make the dashboard immediately interpretable: *"These two vehicles are predicted to collide HERE in approximately X seconds."*  
Reduce false-positive alerts by validating TTC against actual trajectory intersections.

## Constraints

- Do **NOT** rewrite Modules 1–4.
- Preserve all existing interfaces (TrackedObject, PredictedTrajectory, RiskEvent, Alert).
- Do not remove existing functionality.

---

## Proposed Changes

### Part 4 — Improve TTC Validity (Module 5: risk_scoring)

> [!IMPORTANT]
> This is the highest-impact change. The current TTC uses a simple closing-speed formula that fires whenever two nearby vehicles have any positive closing speed — regardless of whether their *trajectories actually intersect*. This produces hundreds of false positives (588 alerts from your video).

#### [MODIFY] [ttc.py](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/risk_scoring/ttc.py)

**Current approach (closing-speed only):**
```
closing_speed = -(rel_pos · rel_vel) / |rel_pos|
TTC = |rel_pos| / closing_speed
```

**New approach (trajectory-intersection validated):**

1. **Stage 1 — Proximity filter** (unchanged, cheap).
2. **Stage 2 — Trajectory intersection check** (NEW):
   - Take the `predictions` list from both vehicles (these are the Kalman-predicted future positions from Module 4 at horizons [1.0s, 2.0s, 3.0s]).
   - For each pair of future positions at the same time horizon, compute the distance.
   - Find the **Closest Point of Approach (CPA)**: the horizon where the two predicted trajectories are nearest.
   - If `CPA distance > collision_distance_threshold` → **skip** (trajectories don't intersect).
3. **Stage 3 — Time synchronization check** (NEW):
   - Verify both vehicles reach the CPA point at approximately the same time.
   - If `|time_A_at_CPA − time_B_at_CPA| > time_difference_tolerance` → **skip** (paths cross but at different times).
4. **Stage 4 — Refined TTC** (MODIFIED):
   - Use the CPA-based TTC rather than the raw closing-speed TTC.
   - Only flag if `TTC < ttc_threshold`.
5. **RiskEvent output enhanced** (backward-compatible additions):
   - Add `collision_point: (x, y)` — the predicted intersection point.
   - Add `cpa_distance: float` — distance at closest approach.
   - Add `vehicle_a_class: str`, `vehicle_b_class: str` — class labels forwarded for the alert card.
   - Add `vehicle_a_velocity: (vx, vy)`, `vehicle_b_velocity: (vx, vy)` — for visualization.
   - Existing fields `vehicle_pair`, `ttc`, `severity` remain unchanged.

New method: `_find_closest_approach(traj_a, traj_b)` → returns `(cpa_distance, collision_point, ttc_estimate)`.

The `evaluate()` method signature remains the same; it still accepts `List[PredictedTrajectory]` and returns `List[RiskEvent]`.

---

#### [MODIFY] [config.yaml](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/risk_scoring/config.yaml)

Add three new configurable parameters:

```yaml
# Trajectory intersection validation
collision_distance_threshold: 50.0     # pixels — max CPA distance to consider a collision
trajectory_intersection_tolerance: 60.0 # pixels — tolerance for path proximity at any horizon
time_difference_tolerance: 1.5          # seconds — max arrival-time difference at CPA
```

---

### Part 1 — Collision Visualization (new rendering module)

#### [NEW] [collision_renderer.py](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/alerts/collision_renderer.py)

A standalone rendering module that replaces the simple `draw_risk_overlay()` and `draw_prediction_trails()` calls in the dashboard pipeline. Provides:

1. **`render_collision_frame(frame, tracked_objects, predictions, risk_events, focus_mode=False, debug_mode=False)`** — main entry point.
   - Builds a lookup of involved vehicle IDs from risk events.
   - **Non-involved vehicles**: draws them with 40% opacity (dimmed bounding boxes, no labels, thin trail).
   - **Involved vehicles**: draws them with thick bounding boxes (thickness=4), large labels showing:
     - Track ID, class, velocity/direction arrow, TTC.
   - **Historical trajectories**: solid colored line with dots from the track history.
   - **Predicted trajectories**: dashed/dotted line in a contrasting color with direction arrowheads at each predicted point.
   - **Collision point**: large pulsing marker (⊕ symbol or cross-hair drawn with cv2) at `risk_event["collision_point"]`.
   - **Connecting line**: red/orange dashed line between the two vehicles with TTC label at midpoint.
   - If `focus_mode=True`: crop/zoom the frame to the bounding region of the two involved vehicles + padding.
   - If `debug_mode=True`: overlay a translucent info panel showing CPA distance, velocity vectors, trajectory coordinates.

2. **`dim_frame_region(frame, mask)`** — applies an alpha overlay to dim non-involved regions.

3. **`draw_direction_arrow(frame, pos, velocity, color, scale)`** — draws an arrowhead showing motion direction.

This module uses only OpenCV and NumPy (no new dependencies).

---

### Part 2 — Collision Focus Mode (Dashboard)

#### [MODIFY] [dashboard.py](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/alerts/dashboard.py)

**Sidebar addition** (inside the "Live Tracking" mode controls):

```python
focus_mode = st.checkbox("🎯 Focus on Collision", value=False,
    help="When active, zooms into the collision region and dims unrelated vehicles.")
debug_mode = st.checkbox("🐛 Debug Mode", value=False,
    help="Show trajectory details, CPA distance, and velocity vectors.")
```

**Pipeline loop modification** (lines ~576–600):
Replace the three separate drawing calls:
```python
annotated = tracker.draw_tracks(annotated, tracked_objects)
annotated = draw_prediction_trails(annotated, predictions)
annotated = draw_risk_overlay(annotated, risk_events, tracked_objects)
```

With a single call to the new renderer:
```python
from alerts.collision_renderer import render_collision_frame
annotated = render_collision_frame(
    annotated, tracked_objects, predictions, risk_events,
    focus_mode=focus_mode, debug_mode=debug_mode
)
```

The old `draw_risk_overlay()` and `draw_prediction_trails()` functions remain in the file for backward compatibility (they are still usable by `run_demo.py`), but the dashboard uses the new renderer.

---

### Part 3 — Redesigned Alert Card

#### [MODIFY] [dashboard.py](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/alerts/dashboard.py) — `render_alert_card()`

Update the card HTML to match the new design:

```
╔══════════════════════════════════════╗
║  ⚠ COLLISION PREDICTED       [HIGH] ║
╠══════════════════════════════════════╣
║  Vehicle A:  ID 511 — car           ║
║  Vehicle B:  ID 525 — car           ║
║                                      ║
║  TTC:  1.89 seconds                  ║
║                                      ║
║  Predicted collision point:          ║
║  (423, 287)                          ║
║                                      ║
║  Risk severity:  HIGH                ║
║  Location:  Intersection_A           ║
╚══════════════════════════════════════╝
```

This requires the alert dict to carry additional fields. We'll extend Module 6's alert output.

#### [MODIFY] [alert_generator.py](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/alerts/alert_generator.py) — `_create_alert()`

Forward the new RiskEvent fields into the alert dict:

```python
alert = {
    "alert_type": "forward_collision_warning",
    "timestamp": ...,
    "vehicles_involved": ...,
    "time_to_collision": ...,
    "severity": ...,
    "location": ...,
    # New fields (backward-compatible — may be absent in old logs)
    "collision_point": risk_event.get("collision_point"),
    "vehicle_a_class": risk_event.get("vehicle_a_class", "unknown"),
    "vehicle_b_class": risk_event.get("vehicle_b_class", "unknown"),
}
```

---

### Part 5 — Debug Information

Handled by `debug_mode=True` in the collision renderer (Part 1). When active, renders a translucent panel on the frame showing:

```
──── DEBUG ────────────────
Vehicle A [ID:511] vel=(42.3, -8.1)
Vehicle B [ID:525] vel=(-38.7, 12.4)
CPA distance: 23.4 px
Collision point: (423, 287)
Estimated TTC: 1.89s
───────────────────────────
```

No separate file needed — this is part of `collision_renderer.py`.

---

### Part 6 — Test Cases

#### [NEW] [test_ttc_trajectory.py](file:///c:/Users/charu/Desktop/Project/AI-Based-Alert-System/risk_scoring/test_ttc_trajectory.py)

New test file specifically for the trajectory-intersection-based TTC validation. Tests:

| # | Scenario | Expected Result |
|---|----------|----------------|
| 1 | Two vehicles whose predicted trajectories cross at the same time | ✅ Collision risk, `collision_point` populated |
| 2 | Two vehicles moving in parallel | ❌ No collision |
| 3 | Two vehicles moving apart | ❌ No collision |
| 4 | Two vehicles whose paths cross but at very different times | ❌ No collision (time_difference_tolerance exceeded) |
| 5 | Two vehicles approaching the same point at approximately the same time | ✅ Collision risk |
| 6 | Vehicles far apart (beyond proximity radius) | ❌ No collision (proximity filter) |

Each test builds synthetic `PredictedTrajectory` dicts with explicit `predictions` lists that model the scenario geometry.

The existing `test_ttc.py` is **NOT modified** — it continues to pass. The new tests validate the enhanced behavior.

---

## Summary of Files

| Action | File | Component |
|--------|------|-----------|
| MODIFY | `risk_scoring/ttc.py` | Trajectory-intersection TTC validation |
| MODIFY | `risk_scoring/config.yaml` | New thresholds |
| NEW    | `alerts/collision_renderer.py` | Collision-focused frame rendering |
| MODIFY | `alerts/dashboard.py` | Focus mode, debug toggles, redesigned alert card, new renderer integration |
| MODIFY | `alerts/alert_generator.py` | Forward collision_point & class info to alerts |
| NEW    | `risk_scoring/test_ttc_trajectory.py` | 6 trajectory-intersection test scenarios |

---

## Verification Plan

### Automated Tests

```powershell
# Existing tests must still pass (backward compatibility)
python -m pytest risk_scoring/test_ttc.py -v

# New trajectory-intersection tests
python -m pytest risk_scoring/test_ttc_trajectory.py -v

# All module tests together
python -m pytest prediction/ risk_scoring/ alerts/ -v
```

### Manual Verification

1. Run the full pipeline and compare alert count (expect significantly fewer alerts):
   ```powershell
   python pipeline/run_demo.py --video data/raw_videos/videoplayback.mp4
   ```
2. Launch the dashboard and verify:
   - Focus mode zooms into collision region
   - Debug mode shows CPA details
   - Alert cards display the new format with collision point
   ```powershell
   python -m streamlit run alerts/dashboard.py
   ```

---

## Open Questions

> [!IMPORTANT]
> **collision_distance_threshold default (50px):** This threshold determines how close two predicted trajectories must pass to count as a potential collision. 50px is a reasonable start for standard-resolution intersection videos, but may need tuning for your specific video's scale. Should I start with 50px and let you tune, or do you have a preferred value?

> [!NOTE]
> **Prediction horizons:** The Kalman filter currently predicts at [1.0s, 2.0s, 3.0s]. The trajectory intersection check interpolates between these points. For finer-grained CPA detection, we could add intermediate horizons (e.g., [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) — but this would be a Module 4 config change, not a code change. Want me to increase the density of prediction horizons?
