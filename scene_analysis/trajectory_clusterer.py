"""
Trajectory Clusterer — Groups vehicles into movement groups based on direction.

Maintains a rolling buffer of completed/long-lived trajectories and clusters
them by direction angle. Each cluster represents a dominant traffic movement
(e.g., "East → West traffic").

No manual lane annotations are used — movement groups are discovered from
observed vehicle motion.
"""

import numpy as np
import yaml
from collections import defaultdict

from scene_analysis.movement_analyzer import DIRECTIONS


class MovementGroup:
    """Represents a discovered traffic movement pattern."""

    def __init__(self, direction, representative_angle):
        self.direction = direction
        self.representative_angle = representative_angle
        self.trajectory_count = 0
        self.trajectory_positions = []  # List of (start_pos, end_pos) tuples
        self.last_update_time = 0.0

    def add_trajectory(self, start_pos, end_pos, timestamp):
        self.trajectory_count += 1
        self.trajectory_positions.append((start_pos, end_pos))
        self.last_update_time = timestamp
        # Keep buffer bounded
        if len(self.trajectory_positions) > 200:
            self.trajectory_positions = self.trajectory_positions[-100:]

    def to_dict(self):
        return {
            "direction": self.direction,
            "angle": round(self.representative_angle, 1),
            "count": self.trajectory_count,
        }


class TrajectoryClustering:
    """
    Clusters tracked vehicle trajectories into movement groups.

    Each time a vehicle's trajectory is sufficiently long and confident,
    it is assigned to a movement group based on its direction angle.
    """

    def __init__(self, config_path="scene_analysis/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.min_trajectories = self.config.get("cluster_min_trajectories", 5)
        self.angular_tolerance = self.config.get("cluster_angular_tolerance", 45.0)

        # Movement groups keyed by direction label
        self.groups = {}

        # Track IDs already registered (to avoid double-counting)
        self._registered_ids = set()

        # Total trajectories observed
        self.total_trajectories = 0

    def update(self, tracked_objects, timestamp):
        """
        Process the current frame's tracked objects and register
        new trajectories into movement groups.

        :param tracked_objects: List of TrackedObject dicts (with movement_direction).
        :param timestamp: Current frame timestamp.
        """
        for obj in tracked_objects:
            tid = obj["track_id"]
            direction = obj.get("movement_direction", "UNKNOWN")
            confidence = obj.get("movement_confidence", 0.0)
            history = obj.get("history", [])

            if direction == "UNKNOWN" or confidence < 0.5:
                continue

            if tid in self._registered_ids:
                continue

            # Only register trajectories with sufficient history
            if len(history) < 10:
                continue

            start_pos = (history[0][0], history[0][1])
            end_pos = (history[-1][0], history[-1][1])

            # Add to or create the movement group
            if direction not in self.groups:
                angle = obj.get("movement_angle", 0.0) or 0.0
                self.groups[direction] = MovementGroup(direction, angle)

            self.groups[direction].add_trajectory(start_pos, end_pos, timestamp)
            self._registered_ids.add(tid)
            self.total_trajectories += 1

        # Prune registered IDs for tracks that are no longer active
        active_ids = {obj["track_id"] for obj in tracked_objects}
        # Only clean up when the set gets too large
        if len(self._registered_ids) > 1000:
            self._registered_ids &= active_ids

    def get_groups(self):
        """
        Return the list of movement groups that have enough observations.

        :return: List of MovementGroup objects.
        """
        return [
            g for g in self.groups.values()
            if g.trajectory_count >= self.min_trajectories
        ]

    def get_all_groups(self):
        """Return all movement groups regardless of count."""
        return list(self.groups.values())

    def get_group_summaries(self):
        """Return a list of dicts describing each qualified group."""
        return [g.to_dict() for g in self.get_groups()]

    def get_crossing_pairs(self):
        """
        Find pairs of movement groups that cross each other.

        Two groups "cross" if their representative angles differ by
        roughly 60°–120° (i.e., they are not parallel or anti-parallel).

        :return: List of (group_a, group_b) tuples.
        """
        qualified = self.get_groups()
        pairs = []

        for i in range(len(qualified)):
            for j in range(i + 1, len(qualified)):
                ga = qualified[i]
                gb = qualified[j]
                angle_diff = abs(ga.representative_angle - gb.representative_angle)
                angle_diff = min(angle_diff, 360 - angle_diff)

                # Crossing if angle difference is 30°-150°
                if 30 <= angle_diff <= 150:
                    pairs.append((ga, gb))

        return pairs
