"""
Collision Renderer — Advanced visualization for the Intersection Collision Prediction System.

Replaces simple bounding box drawing with a collision-focused view that highlights
vehicles involved in a predicted collision, dims unrelated traffic, and plots
trajectory predictions with collision points.
"""

import cv2
import numpy as np


def dim_frame_region(frame, alpha=0.3):
    """Apply a dark overlay to dim the entire frame.
    
    :param alpha: 0.0 = no dimming, 1.0 = fully black.  Default 0.3 gives
                  a subtle "background" effect without hiding context.
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

    # Normalize and scale
    length = min(max(speed * 0.5, 10.0), 50.0) * scale
    end_pos = (
        int(start_pos[0] + (vx / speed) * length),
        int(start_pos[1] + (vy / speed) * length)
    )
    cv2.arrowedLine(frame, start_pos, end_pos, color, 2, tipLength=0.3)
    return frame


def _draw_debug_panel(frame, active_events, tracked_objects, pred_lookup):
    """Draw a translucent debug information panel in the top-right corner."""
    if not active_events:
        return frame

    lines = ["--- DEBUG ---"]
    for event in active_events[:3]:  # Show at most 3 events
        id_a, id_b = event["vehicle_pair"]
        vel_a = event.get("vehicle_a_velocity", (0, 0))
        vel_b = event.get("vehicle_b_velocity", (0, 0))
        ttc = event["ttc"]
        cpa = event.get("cpa_distance", 0)
        cp = event.get("collision_point")

        lines.append(f"Veh A [ID:{id_a}] vel=({vel_a[0]:.1f}, {vel_a[1]:.1f})")
        lines.append(f"Veh B [ID:{id_b}] vel=({vel_b[0]:.1f}, {vel_b[1]:.1f})")
        lines.append(f"CPA distance: {cpa:.1f} px")
        if cp:
            lines.append(f"Collision pt: ({int(cp[0])}, {int(cp[1])})")
        lines.append(f"Est. TTC: {ttc:.2f}s")
        lines.append("---")

    # Compute panel dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 18
    padding = 10

    max_w = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, w)

    panel_w = max_w + 2 * padding
    panel_h = len(lines) * line_height + 2 * padding

    # Position: top-right
    h_frame, w_frame = frame.shape[:2]
    x0 = max(0, w_frame - panel_w - 10)
    y0 = 10

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Draw text
    for i, line in enumerate(lines):
        ty = y0 + padding + (i + 1) * line_height
        cv2.putText(frame, line, (x0 + padding, ty),
                    font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    return frame


def render_collision_frame(frame, tracked_objects, predictions, risk_events,
                           focus_mode=False, debug_mode=False):
    """
    Render the frame with a focus on collision risks.

    When risk_events is non-empty:
      - Non-involved vehicles are subtly dimmed.
      - Involved vehicles get thick bounding boxes, labels, trails, and arrows.
      - Collision points and TTC lines are overlaid.

    :param focus_mode: If True, crop & zoom to the collision region.
    :param debug_mode: If True, show a translucent debug panel with CPA/velocity details.
    """
    annotated = frame.copy()

    # 1. Identify involved vehicles and collision points
    involved_ids = set()
    collision_points = []
    active_events = []

    for event in risk_events:
        id_a, id_b = event["vehicle_pair"]
        involved_ids.add(id_a)
        involved_ids.add(id_b)
        active_events.append(event)
        if "collision_point" in event and event["collision_point"] is not None:
            collision_points.append(event["collision_point"])

    # 2. Dim background only when there are active risk events
    if active_events:
        annotated = dim_frame_region(annotated, alpha=0.3)

    # Build a lookup for predictions
    pred_lookup = {p["track_id"]: p for p in predictions}

    # 3. Draw tracked objects
    for obj in tracked_objects:
        t_id = obj["track_id"]
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        cls_name = obj["class"]
        history = obj["history"]
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        is_involved = t_id in involved_ids

        # Determine styling based on involvement
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

        # Draw bounding box
        if is_involved or not active_events:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
        else:
            # Dimmed bounding box for non-involved vehicles
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

        # Draw label
        if is_involved or not active_events:
            label = f"ID:{t_id} {cls_name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Draw motion trail (history)
            if len(history) > 1:
                pts = [(int(hx), int(hy)) for hx, hy, _ in history]
                for j in range(1, len(pts)):
                    trail_thick = max(1, int(3 * j / len(pts)) + (1 if is_involved else 0))
                    cv2.line(annotated, pts[j - 1], pts[j], color, trail_thick)

            # Draw predicted trajectory
            if t_id in pred_lookup and (is_involved or not active_events):
                pred_pts = pred_lookup[t_id].get("predictions", [])
                if len(pred_pts) >= 1:
                    # Draw dot at each predicted point and connecting line
                    prev = center  # start from current center
                    for i, pt in enumerate(pred_pts):
                        px, py = int(pt[0]), int(pt[1])
                        pcolor = (0, 255, 255) if is_involved else (200, 200, 0)
                        cv2.circle(annotated, (px, py), 4, pcolor, -1)
                        cv2.line(annotated, prev, (px, py), pcolor,
                                 2 if is_involved else 1, cv2.LINE_AA)
                        prev = (px, py)

                    # Draw arrowhead at the last predicted point for involved vehicles
                    if is_involved and len(pred_pts) >= 2:
                        p1 = (int(pred_pts[-2][0]), int(pred_pts[-2][1]))
                        p2 = (int(pred_pts[-1][0]), int(pred_pts[-1][1]))
                        cv2.arrowedLine(annotated, p1, p2, (0, 255, 255), 2,
                                        tipLength=0.3, line_type=cv2.LINE_AA)

                # Draw velocity arrow at current center for involved vehicles
                if is_involved and "velocity" in pred_lookup[t_id]:
                    annotated = draw_direction_arrow(
                        annotated, center, pred_lookup[t_id]["velocity"],
                        (0, 255, 0), scale=1.5
                    )

    # 4. Draw collision points and risk lines
    for event in active_events:
        id_a, id_b = event["vehicle_pair"]
        severity = event["severity"]
        ttc = event["ttc"]

        # Color: red for high, orange for medium
        line_color = (0, 0, 255) if severity == "high" else (0, 165, 255)

        # Find vehicle centers
        ca = None
        cb = None
        for obj in tracked_objects:
            if obj["track_id"] == id_a:
                bx1, by1, bx2, by2 = obj["bbox"]
                ca = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
            if obj["track_id"] == id_b:
                bx1, by1, bx2, by2 = obj["bbox"]
                cb = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))

        if ca is not None and cb is not None:
            # Draw warning line between the two vehicles
            cv2.line(annotated, ca, cb, line_color, 2, cv2.LINE_AA)

            # Draw TTC label at midpoint
            mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
            label = f"TTC:{ttc:.1f}s"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (mid[0] - 2, mid[1] - h - 6),
                          (mid[0] + w + 4, mid[1] + 4), line_color, -1)
            cv2.putText(annotated, label, (mid[0], mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw collision point marker
        cp = event.get("collision_point")
        if cp is not None:
            cpx, cpy = int(cp[0]), int(cp[1])
            # Cross-hair + circle
            cv2.drawMarker(annotated, (cpx, cpy), line_color,
                           markerType=cv2.MARKER_CROSS, markerSize=30,
                           thickness=3, line_type=cv2.LINE_AA)
            cv2.circle(annotated, (cpx, cpy), 15, line_color, 2)

            # Inline CPA label (always shown — it's not debug-only)
            if debug_mode:
                debug_txt = f"CPA: {event.get('cpa_distance', 0):.1f}px"
                cv2.putText(annotated, debug_txt, (cpx + 20, cpy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                            cv2.LINE_AA)

    # 5. Debug panel overlay
    if debug_mode:
        annotated = _draw_debug_panel(annotated, active_events, tracked_objects, pred_lookup)

    # 6. Focus mode — crop and zoom to collision region
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

        for cp in collision_points:
            min_x = min(min_x, cp[0])
            min_y = min(min_y, cp[1])
            max_x = max(max_x, cp[0])
            max_y = max(max_y, cp[1])
            has_box = True

        if has_box:
            pad = 150
            frame_h, frame_w = frame.shape[:2]
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            box_w = max_x - min_x + 2 * pad
            box_h = max_y - min_y + 2 * pad

            size = max(box_w, box_h, 400)  # At least 400×400

            x_start = max(0, int(cx - size / 2))
            y_start = max(0, int(cy - size / 2))
            x_end = min(frame_w, int(cx + size / 2))
            y_end = min(frame_h, int(cy + size / 2))

            crop = annotated[y_start:y_end, x_start:x_end]
            if crop.size > 0:
                annotated = cv2.resize(crop, (frame_w, frame_h))

    return annotated
