"""
Risk Scoring Module — Time-to-Collision (TTC) computation for vehicle pairs.

Evaluates every pair of tracked vehicles to determine if their predicted
paths will cross, and how soon. Uses a physics-based closing-speed formula
rather than machine learning.

This module is intentionally independent of Module 4's internals. It consumes
the PredictedTrajectory interface contract and produces RiskEvent dicts that
Module 6 (Alert Simulation) will consume.

Interface contracts (from architecture.md §7):

    Input — PredictedTrajectory:
        {
            "track_id": int,
            "predictions": [(x, y, future_timestamp), ...],
            "velocity": (vx, vy)
        }

    Output — RiskEvent:
        {
            "vehicle_pair": (int, int),   # sorted (smaller_id, larger_id)
            "ttc": float,                 # seconds
            "severity": "high" | "medium"
        }
"""

import os
from itertools import combinations

import numpy as np
import yaml


class RiskScorer:
    """
    Computes Time-to-Collision (TTC) between all vehicle pairs that are
    within proximity range and on a closing trajectory.
    """

    def __init__(self, config_path="risk_scoring/config.yaml"):
        """
        Initialize the risk scorer.

        :param config_path: Path to the risk scoring config YAML file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.ttc_threshold = self.config.get("ttc_threshold", 2.0)
        self.high_threshold = self.config.get("high_severity_threshold", 1.0)
        self.medium_threshold = self.config.get("medium_severity_threshold", 2.0)
        self.proximity_radius = self.config.get("proximity_radius", 300.0)
        self.min_closing_speed = self.config.get("min_closing_speed", 1.0)
        
        self.collision_dist_thresh = self.config.get("collision_distance_threshold", 50.0)
        self.traj_intersect_tol = self.config.get("trajectory_intersection_tolerance", 60.0)
        self.time_diff_tol = self.config.get("time_difference_tolerance", 1.5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, predicted_trajectories):
        """
        Evaluate all vehicle pairs for collision risk using trajectory intersection.
        """
        if len(predicted_trajectories) < 2:
            return []

        by_id = {pt["track_id"]: pt for pt in predicted_trajectories}
        risk_events = []

        for id_a, id_b in combinations(sorted(by_id.keys()), 2):
            traj_a = by_id[id_a]
            traj_b = by_id[id_b]

            if not self._within_proximity(traj_a, traj_b):
                continue

            result = self._compute_ttc(traj_a, traj_b)
            if result is None:
                continue

            ttc, cpa_dist, coll_point = result

            if ttc < self.ttc_threshold:
                severity = self._classify_severity(ttc)
                risk_events.append({
                    "vehicle_pair": (id_a, id_b),
                    "ttc": round(ttc, 3),
                    "severity": severity,
                    "collision_point": coll_point,
                    "cpa_distance": round(cpa_dist, 2),
                    "vehicle_a_class": traj_a.get("class", "unknown"),
                    "vehicle_b_class": traj_b.get("class", "unknown"),
                    "vehicle_a_velocity": traj_a.get("velocity", (0.0, 0.0)),
                    "vehicle_b_velocity": traj_b.get("velocity", (0.0, 0.0))
                })

        return risk_events

    # ------------------------------------------------------------------
    # Core TTC calculation
    # ------------------------------------------------------------------

    def _compute_ttc(self, traj_a, traj_b):
        """
        Compute Time-to-Collision between two vehicles using trajectory intersection.
        """
        pos_a = self._get_current_position(traj_a)
        pos_b = self._get_current_position(traj_b)
        vel_a = np.array(traj_a["velocity"], dtype=np.float64)
        vel_b = np.array(traj_b["velocity"], dtype=np.float64)
        
        # Check if vehicles are moving towards each other at all
        relative_position = pos_b - pos_a
        relative_velocity = vel_b - vel_a
        distance = np.linalg.norm(relative_position)
        
        if distance < 1e-6:
            # Already colliding
            return 0.0, 0.0, (float(pos_a[0]), float(pos_a[1]))
            
        closing_speed = -np.dot(relative_position, relative_velocity) / distance
        if closing_speed <= self.min_closing_speed:
            return None # Not closing in
            
        # Line intersection: pos_a + vel_a * t_a = pos_b + vel_b * t_b
        A = np.column_stack((vel_a, -vel_b))
        b = pos_b - pos_a
        
        try:
            det = np.linalg.det(A)
            if abs(det) < 1e-6:
                # Parallel lines - check perpendicular distance
                speed_a = np.linalg.norm(vel_a)
                if speed_a > 1e-6:
                    u = vel_a / speed_a
                    w = pos_b - pos_a
                    cpa_dist = np.linalg.norm(w - np.dot(w, u) * u)
                else:
                    cpa_dist = distance
                if cpa_dist > self.collision_dist_thresh:
                    return None
                # If they are parallel, closing in, and close enough, we can use simple closing speed TTC
                ttc = distance / closing_speed
                return float(ttc), float(cpa_dist), (float((pos_a[0]+pos_b[0])/2), float((pos_a[1]+pos_b[1])/2))
                
            t_a, t_b = np.linalg.solve(A, b)
            
            # Intersection point
            point_a = pos_a + vel_a * t_a
            point_b = pos_b + vel_b * t_b
            cpa_dist = np.linalg.norm(point_a - point_b)
            
            # Ensure time is in the future
            if t_a < 0 or t_b < 0:
                return None
                
            if cpa_dist > self.collision_dist_thresh:
                return None
                
            if abs(t_a - t_b) > self.time_diff_tol:
                return None
                
            ttc = (t_a + t_b) / 2.0
            
            # Additional check: trajectory intersection tolerance based on horizon
            # The prompt says: Do their predicted trajectories intersect or come within a configurable collision distance?
            # We already checked this with cpa_dist and time_diff_tol.
            return float(ttc), float(cpa_dist), (float(point_a[0]), float(point_a[1]))
            
        except np.linalg.LinAlgError:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _within_proximity(self, traj_a, traj_b):
        """
        Fast proximity check — are the two vehicles close enough to
        warrant a TTC calculation?

        Uses the current position (first prediction's origin or the
        velocity-implied position) for the distance check.

        :return: True if distance ≤ proximity_radius.
        """
        pos_a = self._get_current_position(traj_a)
        pos_b = self._get_current_position(traj_b)

        distance = np.linalg.norm(pos_b - pos_a)
        return distance <= self.proximity_radius

    def _get_current_position(self, trajectory):
        """
        Extract the current position from a PredictedTrajectory.

        Uses the first prediction minus the horizon offset to back-calculate
        the current position, or derives it from predictions and velocity.

        :param trajectory: PredictedTrajectory dict.
        :return: np.array([x, y])
        """
        predictions = trajectory["predictions"]
        vx, vy = trajectory["velocity"]

        if predictions:
            # The first prediction is at t + horizon_1. Back-calculate
            # current position by subtracting velocity × horizon.
            px, py, pt = predictions[0]

            # We need the horizon (time offset from "now"). If there are
            # at least 2 predictions, we can infer the base timestamp.
            # Otherwise, assume horizon = 1.0s (the default first horizon).
            if len(predictions) >= 2:
                # Horizon spacing tells us the base offset
                dt_between = predictions[1][2] - predictions[0][2]
                current_time = predictions[0][2] - dt_between
            else:
                current_time = pt - 1.0

            dt = pt - current_time
            current_x = px - vx * dt
            current_y = py - vy * dt
            return np.array([current_x, current_y], dtype=np.float64)

        # Fallback: no predictions available
        return np.array([0.0, 0.0], dtype=np.float64)

    def _classify_severity(self, ttc):
        """
        Classify collision severity based on TTC value.

        :param ttc: Time-to-Collision in seconds.
        :return: "high" if TTC < 1.0s, "medium" if 1.0 ≤ TTC < 2.0s.
        """
        if ttc < self.high_threshold:
            return "high"
        return "medium"
