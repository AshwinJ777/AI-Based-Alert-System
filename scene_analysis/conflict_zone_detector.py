"""
Conflict Zone Detector — Automatically discovers intersection/conflict regions.

Analyzes where trajectories from different movement groups cross or converge.
These crossing points are clustered into ConflictZone objects that represent
areas where collisions are most likely.

No manual polygon drawing is required — zones are learned from traffic.
"""

import numpy as np
import yaml
from collections import defaultdict


class ConflictZone:
    """Represents an automatically discovered traffic conflict region."""

    def __init__(self, zone_id, center, radius=50.0):
        self.zone_id = zone_id
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.confidence = 0.0
        self.observation_count = 0
        self.movement_groups = set()  # Set of direction labels (e.g., {"EAST", "SOUTH"})
        self._points = []  # Raw crossing points used to refine center

    def add_observation(self, crossing_point, group_a_dir, group_b_dir):
        """Record a new trajectory crossing observation."""
        self._points.append(np.array(crossing_point, dtype=np.float64))
        self.movement_groups.add(group_a_dir)
        self.movement_groups.add(group_b_dir)
        self.observation_count += 1

        # Recompute center as the mean of all observations
        if self._points:
            self.center = np.mean(self._points, axis=0)

        # Recompute radius as 1.5× the std-dev of points from center
        if len(self._points) >= 3:
            dists = np.linalg.norm(
                np.array(self._points) - self.center, axis=1
            )
            self.radius = max(float(np.std(dists) * 1.5), 30.0)

        # Confidence grows with observations, capped at 1.0
        self.confidence = min(1.0, self.observation_count / 10.0)

        # Keep buffer bounded
        if len(self._points) > 500:
            self._points = self._points[-250:]

    def contains_point(self, point, multiplier=1.0):
        """Check if a point is within the zone (with optional radius multiplier)."""
        dist = np.linalg.norm(np.array(point) - self.center)
        return dist <= self.radius * multiplier

    def to_dict(self):
        return {
            "zone_id": self.zone_id,
            "center": (round(float(self.center[0]), 1), round(float(self.center[1]), 1)),
            "radius": round(self.radius, 1),
            "confidence": round(self.confidence, 3),
            "observation_count": self.observation_count,
            "movement_groups": sorted(self.movement_groups),
        }


class ConflictZoneDetector:
    """
    Discovers conflict zones by finding where trajectories from different
    movement groups cross each other.

    The detector operates in two phases:
    1. Learning phase: collects crossing points until min_observations is reached.
    2. Active phase: maintains and refines conflict zones.
    """

    def __init__(self, config_path="scene_analysis/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.min_observations = self.config.get("min_observations", 50)
        self.crossing_angle_thresh = self.config.get("crossing_angle_threshold", 30.0)
        self.zone_merge_radius = self.config.get("zone_merge_radius", 80.0)
        self.zone_min_crossings = self.config.get("zone_min_crossings", 3)
        self.zone_confidence_decay = self.config.get("zone_confidence_decay", 0.995)
        self.zone_max_radius = self.config.get("zone_max_radius", 200.0)

        self.zones = []  # List of ConflictZone
        self._next_zone_id = 1
        self._total_observations = 0
        self._pending_crossings = []  # Buffered during learning phase

    @property
    def is_learning(self):
        """True if the detector is still in the learning phase."""
        return self._total_observations < self.min_observations

    def update(self, tracked_objects, trajectory_clusterer, timestamp):
        """
        Analyze tracked objects for crossing trajectories and update zones.

        :param tracked_objects: List of TrackedObject dicts (with movement_direction).
        :param trajectory_clusterer: TrajectoryClustering instance.
        :param timestamp: Current frame timestamp.
        """
        self._total_observations = trajectory_clusterer.total_trajectories

        # Find crossing points between vehicles of different movement groups
        crossings = self._find_crossings(tracked_objects)

        for crossing in crossings:
            point = crossing["point"]
            dir_a = crossing["dir_a"]
            dir_b = crossing["dir_b"]

            # Find the nearest existing zone
            nearest_zone = self._find_nearest_zone(point)

            if nearest_zone is not None:
                nearest_zone.add_observation(point, dir_a, dir_b)
                # Cap radius
                if nearest_zone.radius > self.zone_max_radius:
                    nearest_zone.radius = self.zone_max_radius
            else:
                # Create a new zone
                zone = ConflictZone(self._next_zone_id, point)
                self._next_zone_id += 1
                zone.add_observation(point, dir_a, dir_b)
                self.zones.append(zone)

        # Apply confidence decay
        for zone in self.zones:
            zone.confidence *= self.zone_confidence_decay

        # Remove zones with very low confidence
        self.zones = [z for z in self.zones if z.confidence > 0.01 or z.observation_count >= self.zone_min_crossings]

    def _find_crossings(self, tracked_objects):
        """
        Find pairs of vehicles with crossing trajectories and compute
        their approximate crossing point.

        :return: List of dicts with {"point": (x,y), "dir_a": str, "dir_b": str}
        """
        crossings = []
        n = len(tracked_objects)

        for i in range(n):
            for j in range(i + 1, n):
                obj_a = tracked_objects[i]
                obj_b = tracked_objects[j]

                dir_a = obj_a.get("movement_direction", "UNKNOWN")
                dir_b = obj_b.get("movement_direction", "UNKNOWN")

                if dir_a == "UNKNOWN" or dir_b == "UNKNOWN":
                    continue

                if dir_a == dir_b:
                    continue  # Same direction, no crossing

                conf_a = obj_a.get("movement_confidence", 0.0)
                conf_b = obj_b.get("movement_confidence", 0.0)

                if conf_a < 0.5 or conf_b < 0.5:
                    continue

                hist_a = obj_a.get("history", [])
                hist_b = obj_b.get("history", [])

                if len(hist_a) < 3 or len(hist_b) < 3:
                    continue

                # Compute approximate crossing point using line intersection
                crossing_pt = self._compute_crossing_point(hist_a, hist_b)
                if crossing_pt is not None:
                    crossings.append({
                        "point": crossing_pt,
                        "dir_a": dir_a,
                        "dir_b": dir_b,
                    })

        return crossings

    def _compute_crossing_point(self, hist_a, hist_b):
        """
        Compute the approximate crossing point of two trajectories.

        Uses the current position and velocity vector of each trajectory
        to find where their lines intersect.

        :return: (x, y) crossing point or None if lines are parallel.
        """
        # Use last few points to get position and velocity
        pos_a = np.array([hist_a[-1][0], hist_a[-1][1]], dtype=np.float64)
        pos_b = np.array([hist_b[-1][0], hist_b[-1][1]], dtype=np.float64)

        # Velocity from last two points
        vel_a = pos_a - np.array([hist_a[-2][0], hist_a[-2][1]], dtype=np.float64)
        vel_b = pos_b - np.array([hist_b[-2][0], hist_b[-2][1]], dtype=np.float64)

        # Normalize to avoid scale issues
        speed_a = np.linalg.norm(vel_a)
        speed_b = np.linalg.norm(vel_b)

        if speed_a < 1e-6 or speed_b < 1e-6:
            return None

        # Check if they are sufficiently proximate to consider
        dist = np.linalg.norm(pos_b - pos_a)
        if dist > 500:  # Too far apart
            return None

        # Solve: pos_a + vel_a * t = pos_b + vel_b * s
        A = np.column_stack((vel_a, -vel_b))
        b = pos_b - pos_a

        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return None  # Parallel

        try:
            t, s = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        # Crossing point should be roughly in the "future" direction
        # (allow some tolerance for already-passed points)
        if t < -5 or s < -5:
            return None

        crossing = pos_a + vel_a * t
        return (float(crossing[0]), float(crossing[1]))

    def _find_nearest_zone(self, point):
        """Find the nearest existing zone within merge radius, or None."""
        pt = np.array(point, dtype=np.float64)
        best_zone = None
        best_dist = float("inf")

        for zone in self.zones:
            dist = np.linalg.norm(pt - zone.center)
            if dist < self.zone_merge_radius and dist < best_dist:
                best_dist = dist
                best_zone = zone

        return best_zone

    def get_active_zones(self):
        """
        Return zones that are considered stable and active.

        :return: List of ConflictZone objects with sufficient observations.
        """
        return [
            z for z in self.zones
            if z.observation_count >= self.zone_min_crossings
        ]

    def get_zone_summaries(self):
        """Return a list of dicts describing each active zone."""
        return [z.to_dict() for z in self.get_active_zones()]

    def is_near_conflict_zone(self, point, radius_multiplier=1.5):
        """
        Check if a point is near any active conflict zone.

        :param point: (x, y) position.
        :param radius_multiplier: Multiply the zone radius by this factor.
        :return: The nearest ConflictZone or None.
        """
        pt = np.array(point, dtype=np.float64)
        for zone in self.get_active_zones():
            if zone.contains_point(pt, multiplier=radius_multiplier):
                return zone
        return None
