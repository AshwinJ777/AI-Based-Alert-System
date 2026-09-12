"""
Collision Renderer — Advanced visualization for the Intersection Collision Prediction System.

Draws:
- Vehicle bounding boxes with direction labels
- Historical trajectory trails
- Predicted future trajectories with direction arrows
- Uncertainty ellipses around predicted positions (from Kalman covariance)
- Collision warning lines and TTC labels
- Collision point markers
- Conflict zone overlays (debug mode)
- Debug info panel
"""

import cv2
import numpy as np
import math


# Direction abbreviation map
_DIR_ABBREV = {
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW",
    "UNKNOWN": "?",
}


def dim_frame_region(frame, alpha=0.3):
    """Apply a dark overlay to dim the entire frame.

    :param alpha: 0.0 = no dimming, 1.0 = fully black. Default 0.3.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_direction_arrow(frame, start_pos, velocity, color, scale=1.0):
    """Draw a direction arrow indicating the velocity vector."""
    vx, vy = velocity
    speed = np.hypot(vx, vy)
    if speed < 1.0:
        return frame

    length = min(max(speed * 0.5, 10.0), 50.0) * scale
    end_pos = (
        int(start_pos[0] + (vx / speed) * length),
        int(start_pos[1] + (vy / speed) * length)
    )
    cv2.arrowedLine(frame, start_pos, end_pos, color, 2, tipLength=0.3)
    return frame


def draw_uncertainty_ellipse(frame, center, covariance, color=(0, 255, 255),
                              n_sigma=2.0, alpha=0.3):
    """
    Draw an uncertainty ellipse from a 2×2 covariance matrix.

    :param center: (x, y) center of the ellipse.
    :param covariance: 2×2 numpy array (position covariance).
    :param color: BGR color tuple.
    :param n_sigma: Number of standard deviations for the ellipse size.
    :param alpha: Transparency (0=invisible, 1=opaque).
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return frame

    # Ensure positive eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Semi-axis lengths (n_sigma standard deviations)
    width = int(2 * n_sigma * math.sqrt(eigenvalues[1]))
    height = int(2 * n_sigma * math.sqrt(eigenvalues[0]))

    # Cap very large ellipses
    width = min(width, 300)
    height = min(height, 300)

    if width < 2 or height < 2:
        return frame

    # Rotation angle (degrees)
    angle = math.degrees(math.atan2(eigenvectors[1, 1], eigenvectors[0, 1]))

    # Draw on an overlay for transparency
    overlay = frame.copy()
    cv2.ellipse(overlay, (int(center[0]), int(center[1])),
                (width // 2, height // 2), angle, 0, 360, color, 2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


def draw_conflict_zones(frame, conflict_zones, alpha=0.15):
    """
    Draw discovered conflict zones as translucent circles.

    :param conflict_zones: List of ConflictZone objects (or dicts with center, radius).
    """
    overlay = frame.copy()
    for zone in conflict_zones:
        if hasattr(zone, 'center'):
            cx, cy = int(zone.center[0]), int(zone.center[1])
            r = int(zone.radius)
            conf = getattr(zone, 'confidence', 0.5)
        else:
            cx, cy = int(zone["center"][0]), int(zone["center"][1])
            r = int(zone["radius"])
            conf = zone.get("confidence", 0.5)

        # Color based on confidence: green=low, yellow=medium, red=high
        g = int(255 * (1 - conf))
        b_val = 0
        r_val = int(255 * conf)
        color = (b_val, g, r_val)

        cv2.circle(overlay, (cx, cy), r, color, -1)
        cv2.circle(overlay, (cx, cy), r, color, 2)

        # Label
        label = f"Zone {getattr(zone, 'zone_id', '?')}"
        cv2.putText(overlay, label, (cx - 20, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def _draw_debug_panel(frame, active_events, tracked_objects, pred_lookup,
                       conflict_zones=None, movement_groups=None):
    """Draw a translucent debug information panel in the top-right corner."""
    lines = ["--- DEBUG ---"]

    # Scene analysis info
    if movement_groups:
        lines.append(f"Movement groups: {len(movement_groups)}")
        for g in movement_groups[:4]:
            if hasattr(g, 'direction'):
                lines.append(f"  {g.direction}: {g.trajectory_count} tracks")
            else:
                lines.append(f"  {g.get('direction','?')}: {g.get('count',0)} tracks")

    if conflict_zones:
        n_active = len(conflict_zones)
        lines.append(f"Conflict zones: {n_active}")

    lines.append("---")

    for event in active_events[:3]:
        id_a, id_b = event["vehicle_pair"]
        vel_a = event.get("vehicle_a_velocity", (0, 0))
        vel_b = event.get("vehicle_b_velocity", (0, 0))
        ttc = event["ttc"]
        cpa = event.get("cpa_distance", 0)
        cp = event.get("collision_point")
        zone_id = event.get("conflict_zone_id")

        lines.append(f"Pair ({id_a},{id_b})")
        lines.append(f"  vel_A=({vel_a[0]:.1f},{vel_a[1]:.1f})")
        lines.append(f"  vel_B=({vel_b[0]:.1f},{vel_b[1]:.1f})")
        lines.append(f"  CPA: {cpa:.1f}px  TTC: {ttc:.2f}s")
        if cp:
            lines.append(f"  Coll pt: ({int(cp[0])},{int(cp[1])})")
        if zone_id is not None:
            lines.append(f"  Zone: {zone_id}")
        lines.append("---")

    # Panel dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    line_height = 16
    padding = 8

    max_w = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, w)

    panel_w = max_w + 2 * padding
    panel_h = len(lines) * line_height + 2 * padding

    h_frame, w_frame = frame.shape[:2]
    x0 = max(0, w_frame - panel_w - 10)
    y0 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        ty = y0 + padding + (i + 1) * line_height
        cv2.putText(frame, line, (x0 + padding, ty),
                    font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    return frame


def render_collision_frame(frame, tracked_objects, predictions, risk_events,
                           focus_mode=False, debug_mode=False,
                           conflict_zones=None, movement_groups=None):
    """
    Render the frame with collision risk visualization, uncertainty ellipses,
    and scene analysis overlays.

    :param frame: BGR video frame.
    :param tracked_objects: List of TrackedObject dicts (with movement_direction).
    :param predictions: List of PredictedTrajectory dicts (with uncertainties).
    :param risk_events: List of RiskEvent dicts.
    :param focus_mode: If True, crop & zoom to the collision region.
    :param debug_mode: If True, show debug info, conflict zones, and extra details.
    :param conflict_zones: Optional list of ConflictZone objects for debug overlay.
    :param movement_groups: Optional list of MovementGroup objects for debug panel.
    """
    annotated = frame.copy()

    # 1. Identify involved vehicles
    involved_ids = set()
    collision_points = []
    active_events = []

    for event in risk_events:
        id_a, id_b = event["vehicle_pair"]
        involved_ids.add(id_a)
        involved_ids.add(id_b)
        active_events.append(event)
        if event.get("collision_point") is not None:
            collision_points.append(event["collision_point"])

    # 2. Dim background when risk events are active
    if active_events:
        annotated = dim_frame_region(annotated, alpha=0.3)

    # 3. Draw conflict zones in debug mode
    if debug_mode and conflict_zones:
        annotated = draw_conflict_zones(annotated, conflict_zones, alpha=0.15)

    # Build prediction lookup
    pred_lookup = {p["track_id"]: p for p in predictions}

    # 4. Draw tracked objects
    for obj in tracked_objects:
        t_id = obj["track_id"]
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        cls_name = obj.get("class", "?")
        history = obj.get("history", [])
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        direction = obj.get("movement_direction", "UNKNOWN")
        dir_abbrev = _DIR_ABBREV.get(direction, "?")

        is_involved = t_id in involved_ids

        # Styling
        if is_involved:
            is_high = any(
                e["severity"] == "high" and t_id in e["vehicle_pair"]
                for e in active_events
            )
            color = (0, 0, 255) if is_high else (0, 165, 255)
            box_thickness = 4
        else:
            color = (150, 150, 150)
            box_thickness = 1

        # Bounding box
        if is_involved or not active_events:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
        else:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

        # Label with direction
        if is_involved or not active_events:
            label = f"ID:{t_id} {cls_name} [{dir_abbrev}]"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Historical trail
            if len(history) > 1:
                pts = [(int(hx), int(hy)) for hx, hy, _ in history]
                for j in range(1, len(pts)):
                    trail_thick = max(1, int(3 * j / len(pts)) + (1 if is_involved else 0))
                    cv2.line(annotated, pts[j - 1], pts[j], color, trail_thick)

            # Predicted trajectory + uncertainty ellipses
            if t_id in pred_lookup:
                pred = pred_lookup[t_id]
                pred_pts = pred.get("predictions", [])
                uncertainties = pred.get("uncertainties", [])

                if pred_pts:
                    prev = center
                    for i, pt in enumerate(pred_pts):
                        px, py = int(pt[0]), int(pt[1])
                        pcolor = (0, 255, 255) if is_involved else (200, 200, 0)
                        cv2.circle(annotated, (px, py), 4, pcolor, -1)
                        cv2.line(annotated, prev, (px, py), pcolor,
                                 2 if is_involved else 1, cv2.LINE_AA)
                        prev = (px, py)

                        # Draw uncertainty ellipse
                        if i < len(uncertainties):
                            cov = uncertainties[i]
                            ellipse_alpha = 0.4 if is_involved else 0.2
                            annotated = draw_uncertainty_ellipse(
                                annotated, (px, py), cov,
                                color=pcolor, n_sigma=2.0, alpha=ellipse_alpha
                            )

                    # Arrowhead at last predicted point
                    if is_involved and len(pred_pts) >= 2:
                        p1 = (int(pred_pts[-2][0]), int(pred_pts[-2][1]))
                        p2 = (int(pred_pts[-1][0]), int(pred_pts[-1][1]))
                        cv2.arrowedLine(annotated, p1, p2, (0, 255, 255), 2,
                                        tipLength=0.3, line_type=cv2.LINE_AA)

                # Velocity arrow
                if is_involved and "velocity" in pred:
                    annotated = draw_direction_arrow(
                        annotated, center, pred["velocity"],
                        (0, 255, 0), scale=1.5
                    )

    # 5. Draw collision events
    for event in active_events:
        id_a, id_b = event["vehicle_pair"]
        severity = event["severity"]
        ttc = event["ttc"]

        line_color = (0, 0, 255) if severity == "high" else (0, 165, 255)

        ca = cb = None
        for obj in tracked_objects:
            if obj["track_id"] == id_a:
                bx1, by1, bx2, by2 = obj["bbox"]
                ca = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
            if obj["track_id"] == id_b:
                bx1, by1, bx2, by2 = obj["bbox"]
                cb = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))

        if ca is not None and cb is not None:
            cv2.line(annotated, ca, cb, line_color, 2, cv2.LINE_AA)

            mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
            label = f"TTC:{ttc:.1f}s"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (mid[0] - 2, mid[1] - h - 6),
                          (mid[0] + w + 4, mid[1] + 4), line_color, -1)
            cv2.putText(annotated, label, (mid[0], mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Collision point marker
        cp = event.get("collision_point")
        if cp is not None:
            cpx, cpy = int(cp[0]), int(cp[1])
            cv2.drawMarker(annotated, (cpx, cpy), line_color,
                           markerType=cv2.MARKER_CROSS, markerSize=30,
                           thickness=3, line_type=cv2.LINE_AA)
            cv2.circle(annotated, (cpx, cpy), 15, line_color, 2)

            if debug_mode:
                debug_txt = f"CPA:{event.get('cpa_distance', 0):.1f}px"
                cv2.putText(annotated, debug_txt, (cpx + 20, cpy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                            cv2.LINE_AA)

    # 6. Debug panel
    if debug_mode:
        annotated = _draw_debug_panel(
            annotated, active_events, tracked_objects, pred_lookup,
            conflict_zones=conflict_zones, movement_groups=movement_groups
        )

    # 7. Focus mode
    if focus_mode and active_events:
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        has_box = False

        for obj in tracked_objects:
            if obj["track_id"] in involved_ids:
                has_box = True
                bx1, by1, bx2, by2 = obj["bbox"]
                min_x = min(min_x, bx1)
                min_y = min(min_y, by1)
                max_x = max(max_x, bx2)
                max_y = max(max_y, by2)

        for cp_pt in collision_points:
            min_x = min(min_x, cp_pt[0])
            min_y = min(min_y, cp_pt[1])
            max_x = max(max_x, cp_pt[0])
            max_y = max(max_y, cp_pt[1])
            has_box = True

        if has_box:
            pad = 150
            frame_h, frame_w = frame.shape[:2]
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            box_w = max_x - min_x + 2 * pad
            box_h = max_y - min_y + 2 * pad

            size = max(box_w, box_h, 400)

            x_start = max(0, int(cx - size / 2))
            y_start = max(0, int(cy - size / 2))
            x_end = min(frame_w, int(cx + size / 2))
            y_end = min(frame_h, int(cy + size / 2))

            crop = annotated[y_start:y_end, x_start:x_end]
            if crop.size > 0:
                annotated = cv2.resize(crop, (frame_w, frame_h))

    return annotated
