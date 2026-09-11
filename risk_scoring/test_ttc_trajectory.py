"""
Test suite for Module 5 — Trajectory Intersection TTC Validation.
"""

import sys
import os

import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from risk_scoring.ttc import RiskScorer
from risk_scoring.test_ttc import make_prediction

class TestTrajectoryIntersection:
    def setup_method(self):
        self.scorer = RiskScorer(config_path="risk_scoring/config.yaml")

    def test_trajectories_cross_same_time(self):
        # A moving right along y=200, B moving up along x=200
        # A starts at (125, 200), vx=50. Reaches (200, 200) at t=1.5
        # B starts at (200, 125), vy=50. Reaches (200, 200) at t=1.5
        traj_a = make_prediction(1, x=125, y=200, vx=50, vy=0)
        traj_b = make_prediction(2, x=200, y=125, vx=0, vy=50)

        events = self.scorer.evaluate([traj_a, traj_b])

        assert len(events) == 1
        event = events[0]
        assert event["vehicle_pair"] == (1, 2)
        assert abs(event["ttc"] - 1.5) < 1e-3
        assert "collision_point" in event
        # The exact calculation might have a small floating point difference
        cx, cy = event["collision_point"]
        assert abs(cx - 200) < 1.0
        assert abs(cy - 200) < 1.0

    def test_parallel_vehicles(self):
        # A and B moving parallel with same velocity but 100px apart
        traj_a = make_prediction(1, x=100, y=200, vx=50, vy=0)
        traj_b = make_prediction(2, x=100, y=300, vx=50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0

    def test_diverging_vehicles(self):
        # A moving left, B moving right
        traj_a = make_prediction(1, x=200, y=200, vx=-50, vy=0)
        traj_b = make_prediction(2, x=210, y=200, vx=50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0

    def test_crossing_different_times(self):
        # Paths cross at (200, 200), but at different times.
        # A reaches (200,200) at t=1.5
        traj_a = make_prediction(1, x=125, y=200, vx=50, vy=0)
        # B reaches (200,200) at t=5
        traj_b = make_prediction(2, x=200, y=150, vx=0, vy=10)

        events = self.scorer.evaluate([traj_a, traj_b])
        # Time difference is 3.5s, which is > time_difference_tolerance (1.5s)
        assert len(events) == 0

    def test_approaching_same_point(self):
        # A moving diagonally, B moving right, meeting at (300,300)
        # A starts at (150, 150), vel=(100, 100) -> t=1.5
        traj_a = make_prediction(1, x=150, y=150, vx=100, vy=100)
        # B starts at (225, 300), vel=(50, 0) -> t=1.5
        traj_b = make_prediction(2, x=225, y=300, vx=50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 1
        assert "collision_point" in events[0]

    def test_vehicles_far_apart(self):
        # Paths would cross but initial distance > proximity radius
        traj_a = make_prediction(1, x=0, y=500, vx=50, vy=0)
        traj_b = make_prediction(2, x=500, y=0, vx=0, vy=50)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
