"""
Movement Analyzer — Infers the dominant movement direction of each tracked vehicle.

Uses the historical trajectory points from ByteTrack (Module 3) to compute a
smoothed movement vector and map it to one of 8 cardinal/ordinal directions.

No manually drawn lane polygons are used. Direction is derived entirely from
observed vehicle motion.
"""

import math
import numpy as np
import yaml


# Cardinal/ordinal direction labels
DIRECTIONS = [
    "EAST",       # 0°    (-22.5 to 22.5)
    "NORTHEAST",  # 45°   (22.5  to 67.5)
    "NORTH",      # 90°   (67.5  to 112.5)
    "NORTHWEST",  # 135°  (112.5 to 157.5)
    "WEST",       # 180°  (157.5 to 202.5)
    "SOUTHWEST",  # 225°  (202.5 to 247.5)
    "SOUTH",      # 270°  (247.5 to 292.5)
    "SOUTHEAST",  # 315°  (292.5 to 337.5)
]


class MovementAnalyzer:
    """
    Determines the dominant movement direction for each tracked vehicle
    using trajectory history smoothing.
    """

    def __init__(self, config_path="scene_analysis/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.min_history = self.config.get("min_history_points", 5)
        self.smoothing_window = self.config.get("smoothing_window", 5)
        self.min_displacement = self.config.get("min_displacement", 15.0)
        self.min_confidence = self.config.get("min_movement_confidence", 0.5)

    def analyze(self, tracked_objects):
        """
        Augment each TrackedObject with movement_direction and movement_confidence.

        :param tracked_objects: List of TrackedObject dicts from Module 3.
        :return: The same list, with each dict augmented in-place.
        """
        for obj in tracked_objects:
            direction, confidence, angle = self._compute_direction(obj["history"])
            obj["movement_direction"] = direction
            obj["movement_confidence"] = round(confidence, 3)
            obj["movement_angle"] = round(angle, 2) if angle is not None else None
        return tracked_objects

    def _compute_direction(self, history):
        """
        Compute the smoothed movement direction from the trajectory history.

        :param history: List of (x, y, timestamp) tuples.
        :return: (direction_str, confidence_float, angle_degrees_or_None)
        """
        if len(history) < self.min_history:
            return "UNKNOWN", 0.0, None

        # Extract positions
        pts = np.array([(h[0], h[1]) for h in history], dtype=np.float64)

        # Apply moving-average smoothing
        smoothed = self._smooth(pts)

        if len(smoothed) < 2:
            return "UNKNOWN", 0.0, None

        # Compute the net displacement vector (first → last of smoothed)
        displacement = smoothed[-1] - smoothed[0]
        dist = np.linalg.norm(displacement)

        if dist < self.min_displacement:
            return "UNKNOWN", 0.0, None

        # Compute angle in degrees.
        # NOTE: In image coordinates, y increases downward.
        # atan2(-dy, dx) converts to standard compass where:
        #   0° = East, 90° = North (up in the real world = -y in image)
        angle_rad = math.atan2(-displacement[1], displacement[0])
        angle_deg = math.degrees(angle_rad) % 360

        # Map angle to direction
        direction = self._angle_to_direction(angle_deg)

        # Compute confidence based on trajectory straightness
        # (ratio of net displacement to total path length)
        segment_lengths = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
        total_path = np.sum(segment_lengths)

        if total_path < 1e-6:
            return "UNKNOWN", 0.0, None

        straightness = dist / total_path  # 1.0 = perfectly straight
        confidence = min(straightness, 1.0)

        if confidence < self.min_confidence:
            return "UNKNOWN", confidence, angle_deg

        return direction, confidence, angle_deg

    def _smooth(self, pts):
        """
        Apply a simple moving-average smoothing to a 2D trajectory.

        :param pts: np.ndarray of shape (N, 2).
        :return: Smoothed np.ndarray of shape (M, 2) where M <= N.
        """
        if len(pts) <= self.smoothing_window:
            return pts

        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        sx = np.convolve(pts[:, 0], kernel, mode="valid")
        sy = np.convolve(pts[:, 1], kernel, mode="valid")
        return np.column_stack((sx, sy))

    @staticmethod
    def _angle_to_direction(angle_deg):
        """
        Map an angle (0–360°, 0°=East, CCW) to one of 8 compass directions.
        """
        # Each sector is 45° wide, centered on the direction
        idx = int((angle_deg + 22.5) / 45.0) % 8
        return DIRECTIONS[idx]


def are_directions_conflicting(dir_a, dir_b):
    """
    Check if two movement directions are potentially conflicting.

    Conflicting means the two vehicles are approaching from different
    directions and could intersect. Same or opposite directions along
    the same axis (e.g., both EAST, or EAST vs WEST) are generally
    non-conflicting unless one is overtaking the other.

    :return: True if the directions indicate a potential crossing conflict.
    """
    if dir_a == "UNKNOWN" or dir_b == "UNKNOWN":
        # Can't determine conflict without direction info
        return True  # Conservative: don't filter out

    # Map direction to angle sector index
    idx_a = DIRECTIONS.index(dir_a)
    idx_b = DIRECTIONS.index(dir_b)

    # Angular difference in sectors (0-4, wrapping around)
    diff = abs(idx_a - idx_b)
    diff = min(diff, 8 - diff)

    # Same direction (0) or directly opposite (4) = generally non-conflicting
    # 1 sector apart (45°) = marginal
    # 2-3 sectors apart (90°-135°) = clearly conflicting
    if diff == 0:
        return False  # Same direction
    if diff == 4:
        return False  # Head-on along same axis (handled differently)

    return True  # Crossing trajectory
