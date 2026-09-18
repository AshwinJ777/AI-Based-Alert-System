"""
Session Data Collector — passive observer that records per-frame pipeline
statistics during video processing.

Attaches to the existing pipeline loop and collects:
- Detection counts by vehicle class
- Tracked vehicle count per time interval
- Risk events with TTC values
- Processing timing

Saves all collected data to a JSON file that the AnalyticsEngine reads.
Does NOT modify any pipeline logic.
"""

import json
import os
import time
from collections import defaultdict


class SessionDataCollector:
    """
    Collects pipeline statistics during a single video processing session.
    Call record_frame() each frame, then save() at the end.
    """

    def __init__(self):
        self.video_file = ""
        self.start_time = None
        self.end_time = None

        # Per-frame accumulators
        self.total_frames = 0
        self.total_detections = 0
        self.vehicle_classes = defaultdict(int)
        self.unique_track_ids = set()

        # Time-series data (sampled every 0.5 seconds of video time)
        self.density_timeline = []        # [(video_time, vehicle_count), ...]
        self.risk_events_timeline = []    # [(video_time, num_risk_events), ...]
        self._last_density_time = -1.0

        # Risk / TTC
        self.all_ttc_values = []
        self.all_risk_events = []         # Full risk event dicts
        self.high_risk_count = 0
        self.medium_risk_count = 0

    def start(self, video_file):
        """Mark the start of processing."""
        self.video_file = video_file
        self.start_time = time.time()

    def record_frame(self, timestamp, detections, tracked_objects,
                     risk_events, predictions=None):
        """
        Record statistics for one processed frame.

        :param timestamp: Video timestamp in seconds.
        :param detections: List of detection dicts from Module 2.
        :param tracked_objects: List of TrackedObject dicts from Module 3.
        :param risk_events: List of RiskEvent dicts from Module 5.
        :param predictions: Optional list of PredictedTrajectory dicts.
        """
        self.total_frames += 1

        # --- Detection class counts ---
        for det in detections:
            cls = det.get("class", "unknown")
            self.vehicle_classes[cls] += 1
            self.total_detections += 1

        # --- Track IDs ---
        for obj in tracked_objects:
            self.unique_track_ids.add(obj["track_id"])

        # --- Density timeline (sample every 0.5s of video time) ---
        if timestamp - self._last_density_time >= 0.5:
            self.density_timeline.append([
                round(timestamp, 2),
                len(tracked_objects)
            ])
            self.risk_events_timeline.append([
                round(timestamp, 2),
                len(risk_events)
            ])
            self._last_density_time = timestamp

        # --- Risk events ---
        for event in risk_events:
            ttc = event.get("ttc", 0)
            self.all_ttc_values.append(ttc)

            severity = event.get("severity", "medium")
            if severity == "high":
                self.high_risk_count += 1
            else:
                self.medium_risk_count += 1

            # Store a serializable copy
            self.all_risk_events.append({
                "timestamp": round(timestamp, 3),
                "vehicle_pair": list(event.get("vehicle_pair", [])),
                "ttc": round(ttc, 3),
                "severity": severity,
                "vehicle_a_class": event.get("vehicle_a_class", "unknown"),
                "vehicle_b_class": event.get("vehicle_b_class", "unknown"),
                "collision_point": event.get("collision_point"),
            })

    def finish(self):
        """Mark the end of processing."""
        self.end_time = time.time()

    def get_processing_fps(self):
        """Calculate the average processing FPS."""
        if self.start_time and self.end_time and self.total_frames > 0:
            elapsed = self.end_time - self.start_time
            if elapsed > 0:
                return round(self.total_frames / elapsed, 1)
        return 0.0

    def to_dict(self):
        """Export all collected data as a serializable dict."""
        return {
            "video_file": self.video_file,
            "total_frames": self.total_frames,
            "processing_fps": self.get_processing_fps(),
            "total_detections": self.total_detections,
            "vehicle_classes": dict(self.vehicle_classes),
            "unique_tracks": len(self.unique_track_ids),
            "unique_track_ids": sorted(self.unique_track_ids),
            "density_timeline": self.density_timeline,
            "risk_events_timeline": self.risk_events_timeline,
            "all_ttc_values": [round(v, 3) for v in self.all_ttc_values],
            "total_risk_events": len(self.all_risk_events),
            "high_risk_count": self.high_risk_count,
            "medium_risk_count": self.medium_risk_count,
            "all_risk_events": self.all_risk_events,
        }

    def save(self, path="analytics/session_data.json"):
        """Save collected data to a JSON file."""
        self.finish()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
