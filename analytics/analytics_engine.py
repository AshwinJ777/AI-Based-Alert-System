"""
Analytics Engine — computes KPIs, distributions, and statistics from
collected session data and alert logs.

Consumes:
- analytics/session_data.json (from SessionDataCollector)
- alerts/alert_log.jsonl (from AlertGenerator)

Does NOT duplicate any collision detection logic. All computations are
aggregations over already-generated data.
"""

import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime


SESSION_DATA_PATH = "analytics/session_data.json"
ALERT_LOG_PATH = "alerts/alert_log.jsonl"
EXPORT_DIR = "analytics/exports"


class AnalyticsEngine:
    """
    Loads session data and alert logs, then provides computed analytics
    for the dashboard.
    """

    def __init__(self, session_path=SESSION_DATA_PATH,
                 alert_log_path=ALERT_LOG_PATH):
        self.session_data = {}
        self.alerts = []

        self._load_session(session_path)
        self._load_alerts(alert_log_path)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_session(self, path):
        """Load session_data.json."""
        if os.path.exists(path):
            with open(path, "r") as f:
                self.session_data = json.load(f)

    def _load_alerts(self, path):
        """Load alert_log.jsonl."""
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def has_data(self):
        """Check if session data is available."""
        return bool(self.session_data) or bool(self.alerts)

    # ------------------------------------------------------------------
    # 1. KPI Overview
    # ------------------------------------------------------------------

    def get_kpis(self):
        """Return top-level KPI metrics."""
        ttc_values = self.session_data.get("all_ttc_values", [])
        alert_ttc = [a.get("time_to_collision", 0) for a in self.alerts]

        return {
            "total_frames": self.session_data.get("total_frames", 0),
            "total_detections": self.session_data.get("total_detections", 0),
            "unique_tracks": self.session_data.get("unique_tracks", 0),
            "total_risk_events": self.session_data.get("total_risk_events", 0),
            "high_risk_events": self.session_data.get("high_risk_count", 0),
            "medium_risk_events": self.session_data.get("medium_risk_count", 0),
            "total_alerts": len(self.alerts),
            "avg_ttc": round(statistics.mean(ttc_values), 3) if ttc_values else 0,
            "min_ttc": round(min(ttc_values), 3) if ttc_values else 0,
            "processing_fps": self.session_data.get("processing_fps", 0),
        }

    # ------------------------------------------------------------------
    # 2. Vehicle Class Distribution
    # ------------------------------------------------------------------

    def get_class_distribution(self):
        """
        Return detection counts per vehicle class.

        :return: List of {"class": str, "count": int, "percentage": float}
        """
        classes = self.session_data.get("vehicle_classes", {})
        total = sum(classes.values()) or 1

        result = []
        for cls, count in sorted(classes.items(), key=lambda x: -x[1]):
            result.append({
                "class": cls,
                "count": count,
                "percentage": round(count / total * 100, 1),
            })
        return result

    # ------------------------------------------------------------------
    # 3. Risk Distribution
    # ------------------------------------------------------------------

    def get_risk_distribution(self):
        """
        Return risk event counts by severity.

        :return: List of {"severity": str, "count": int, "percentage": float,
                          "avg_ttc": float}
        """
        risk_events = self.session_data.get("all_risk_events", [])
        if not risk_events:
            return []

        by_severity = defaultdict(list)
        for event in risk_events:
            sev = event.get("severity", "medium")
            by_severity[sev].append(event.get("ttc", 0))

        total = len(risk_events) or 1
        result = []
        for sev in ["high", "medium"]:
            ttc_list = by_severity.get(sev, [])
            if ttc_list:
                result.append({
                    "severity": sev,
                    "count": len(ttc_list),
                    "percentage": round(len(ttc_list) / total * 100, 1),
                    "avg_ttc": round(statistics.mean(ttc_list), 3),
                })
        return result

    # ------------------------------------------------------------------
    # 4. TTC Analytics
    # ------------------------------------------------------------------

    def get_ttc_stats(self):
        """Return TTC statistics."""
        values = self.session_data.get("all_ttc_values", [])
        if not values:
            return {"avg": 0, "min": 0, "max": 0, "median": 0, "count": 0}

        return {
            "avg": round(statistics.mean(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "median": round(statistics.median(values), 3),
            "count": len(values),
        }

    def get_ttc_histogram_data(self, bins=10):
        """
        Return TTC values binned for a histogram.

        :return: List of {"bin_start": float, "bin_end": float, "count": int}
        """
        values = self.session_data.get("all_ttc_values", [])
        if not values:
            return []

        min_v = min(values)
        max_v = max(values)
        if min_v == max_v:
            return [{"bin_start": min_v, "bin_end": max_v, "count": len(values)}]

        bin_width = (max_v - min_v) / bins
        result = []
        for i in range(bins):
            start = min_v + i * bin_width
            end = start + bin_width
            count = sum(1 for v in values if start <= v < end)
            if i == bins - 1:
                count = sum(1 for v in values if start <= v <= end)
            result.append({
                "bin_start": round(start, 3),
                "bin_end": round(end, 3),
                "count": count,
            })
        return result

    def get_ttc_timeline(self):
        """
        Return TTC values over video time for a scatter/line chart.

        :return: List of {"time": float, "ttc": float, "severity": str}
        """
        risk_events = self.session_data.get("all_risk_events", [])
        return [
            {
                "time": e.get("timestamp", 0),
                "ttc": e.get("ttc", 0),
                "severity": e.get("severity", "medium"),
            }
            for e in risk_events
        ]

    # ------------------------------------------------------------------
    # 5. Alert Timeline
    # ------------------------------------------------------------------

    def get_alert_timeline(self, severity_filter=None):
        """
        Return alerts formatted for a timeline display.

        :param severity_filter: "high", "medium", or None for all.
        :return: List of alert dicts with formatted fields.
        """
        filtered = self.alerts
        if severity_filter:
            filtered = [a for a in filtered if a.get("severity") == severity_filter]

        result = []
        for a in filtered:
            vehicles = a.get("vehicles_involved", [])
            result.append({
                "timestamp": a.get("timestamp", 0),
                "time_str": self._format_video_time(a.get("timestamp", 0)),
                "vehicle_a": vehicles[0] if len(vehicles) > 0 else "?",
                "vehicle_b": vehicles[1] if len(vehicles) > 1 else "?",
                "vehicle_a_class": a.get("vehicle_a_class", "unknown"),
                "vehicle_b_class": a.get("vehicle_b_class", "unknown"),
                "ttc": a.get("time_to_collision", 0),
                "severity": a.get("severity", "medium"),
                "location": a.get("location", "Intersection_A"),
            })
        return result

    # ------------------------------------------------------------------
    # 6. Most Frequently Involved Vehicles
    # ------------------------------------------------------------------

    def get_most_involved_vehicles(self, top_n=10):
        """
        Calculate which track IDs appear most often in risk events.

        :return: List of {"vehicle_id": int, "risk_events": int,
                          "min_ttc": float, "avg_ttc": float}
        """
        risk_events = self.session_data.get("all_risk_events", [])
        vehicle_stats = defaultdict(lambda: {"count": 0, "ttc_values": []})

        for event in risk_events:
            pair = event.get("vehicle_pair", [])
            ttc = event.get("ttc", 0)
            for vid in pair:
                vehicle_stats[vid]["count"] += 1
                vehicle_stats[vid]["ttc_values"].append(ttc)

        result = []
        for vid, stats in vehicle_stats.items():
            ttc_vals = stats["ttc_values"]
            result.append({
                "vehicle_id": vid,
                "risk_events": stats["count"],
                "min_ttc": round(min(ttc_vals), 3) if ttc_vals else 0,
                "avg_ttc": round(statistics.mean(ttc_vals), 3) if ttc_vals else 0,
            })

        result.sort(key=lambda x: -x["risk_events"])
        return result[:top_n]

    # ------------------------------------------------------------------
    # 7. Traffic Density Timeline
    # ------------------------------------------------------------------

    def get_density_timeline(self):
        """
        Return vehicle count over time.

        :return: List of {"time": float, "vehicle_count": int}
        """
        timeline = self.session_data.get("density_timeline", [])
        return [
            {"time": entry[0], "vehicle_count": entry[1]}
            for entry in timeline
        ]

    # ------------------------------------------------------------------
    # 8. Density vs Risk
    # ------------------------------------------------------------------

    def get_density_vs_risk(self, bucket_seconds=5.0):
        """
        Correlate traffic density with risk event frequency in time buckets.

        :return: List of {"time": float, "avg_density": float, "risk_count": int}
        """
        density_tl = self.session_data.get("density_timeline", [])
        risk_tl = self.session_data.get("risk_events_timeline", [])

        if not density_tl:
            return []

        max_time = max(d[0] for d in density_tl) if density_tl else 0
        buckets = []
        t = 0
        while t <= max_time:
            bucket_end = t + bucket_seconds

            # Average density in this bucket
            densities = [d[1] for d in density_tl if t <= d[0] < bucket_end]
            avg_d = round(statistics.mean(densities), 1) if densities else 0

            # Risk count in this bucket
            risks = sum(r[1] for r in risk_tl if t <= r[0] < bucket_end)

            buckets.append({
                "time": round(t, 1),
                "avg_density": avg_d,
                "risk_count": risks,
            })
            t += bucket_seconds

        return buckets

    # ------------------------------------------------------------------
    # 9. Filtered Alerts
    # ------------------------------------------------------------------

    def get_filtered_alerts(self, severity=None, vehicle_class=None,
                            vehicle_id=None, time_range=None):
        """
        Return alerts matching the given filters.

        :param severity: "high" or "medium" or None.
        :param vehicle_class: e.g. "car" or None.
        :param vehicle_id: int track ID or None.
        :param time_range: (start_time, end_time) or None.
        :return: Filtered list of alert dicts.
        """
        result = list(self.alerts)

        if severity:
            result = [a for a in result if a.get("severity") == severity]

        if vehicle_class:
            result = [a for a in result
                      if a.get("vehicle_a_class") == vehicle_class
                      or a.get("vehicle_b_class") == vehicle_class]

        if vehicle_id is not None:
            result = [a for a in result
                      if vehicle_id in a.get("vehicles_involved", [])]

        if time_range:
            t_start, t_end = time_range
            result = [a for a in result
                      if t_start <= a.get("timestamp", 0) <= t_end]

        return result

    # ------------------------------------------------------------------
    # 10. Session Summary
    # ------------------------------------------------------------------

    def get_session_summary(self):
        """Return a complete session summary dict."""
        kpis = self.get_kpis()
        ttc_stats = self.get_ttc_stats()
        return {
            "video_file": self.session_data.get("video_file", "N/A"),
            "total_frames": kpis["total_frames"],
            "processing_fps": kpis["processing_fps"],
            "unique_vehicles": kpis["unique_tracks"],
            "total_risk_events": kpis["total_risk_events"],
            "high_risk": kpis["high_risk_events"],
            "medium_risk": kpis["medium_risk_events"],
            "total_alerts": kpis["total_alerts"],
            "min_ttc": ttc_stats["min"],
            "avg_ttc": ttc_stats["avg"],
            "median_ttc": ttc_stats["median"],
            "max_ttc": ttc_stats["max"],
        }

    # ------------------------------------------------------------------
    # 11. Export
    # ------------------------------------------------------------------

    def export_csv(self, path=None):
        """
        Export alert history as a CSV file.

        :return: Path to the exported file.
        """
        if path is None:
            os.makedirs(EXPORT_DIR, exist_ok=True)
            path = os.path.join(EXPORT_DIR, "alert_report.csv")

        headers = [
            "Timestamp", "Vehicle_A", "Vehicle_B",
            "Vehicle_A_Class", "Vehicle_B_Class",
            "TTC", "Severity", "Location"
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for a in self.alerts:
                vehicles = a.get("vehicles_involved", [])
                writer.writerow({
                    "Timestamp": self._format_video_time(a.get("timestamp", 0)),
                    "Vehicle_A": vehicles[0] if len(vehicles) > 0 else "",
                    "Vehicle_B": vehicles[1] if len(vehicles) > 1 else "",
                    "Vehicle_A_Class": a.get("vehicle_a_class", "unknown"),
                    "Vehicle_B_Class": a.get("vehicle_b_class", "unknown"),
                    "TTC": a.get("time_to_collision", 0),
                    "Severity": a.get("severity", ""),
                    "Location": a.get("location", ""),
                })

        return path

    def export_analytics_csv(self, path=None):
        """
        Export session analytics summary as a CSV file.

        :return: Path to the exported file.
        """
        if path is None:
            os.makedirs(EXPORT_DIR, exist_ok=True)
            path = os.path.join(EXPORT_DIR, "analytics_summary.csv")

        summary = self.get_session_summary()
        class_dist = self.get_class_distribution()
        risk_dist = self.get_risk_distribution()

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["=== SESSION SUMMARY ==="])
            for key, val in summary.items():
                writer.writerow([key, val])

            writer.writerow([])
            writer.writerow(["=== VEHICLE CLASS DISTRIBUTION ==="])
            writer.writerow(["Class", "Count", "Percentage"])
            for item in class_dist:
                writer.writerow([item["class"], item["count"], item["percentage"]])

            writer.writerow([])
            writer.writerow(["=== RISK DISTRIBUTION ==="])
            writer.writerow(["Severity", "Count", "Percentage", "Avg TTC"])
            for item in risk_dist:
                writer.writerow([item["severity"], item["count"],
                                 item["percentage"], item["avg_ttc"]])

        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_video_time(timestamp):
        """Format a video timestamp (seconds) as MM:SS."""
        if timestamp < 0:
            return "00:00"
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
