"""
Tests for the Analytics module.
"""

import json
import os
import pytest

from analytics.data_collector import SessionDataCollector
from analytics.analytics_engine import AnalyticsEngine


@pytest.fixture
def sample_session_data(tmp_path):
    """Create a temporary session_data.json for testing."""
    data = {
        "video_file": "test_video.mp4",
        "total_frames": 100,
        "processing_fps": 10.0,
        "total_detections": 50,
        "vehicle_classes": {"car": 40, "truck": 10},
        "unique_tracks": 15,
        "density_timeline": [[0.0, 2], [0.5, 4], [1.0, 5]],
        "risk_events_timeline": [[0.0, 0], [0.5, 1], [1.0, 2]],
        "all_ttc_values": [1.5, 1.2, 0.8, 1.9, 0.5],
        "total_risk_events": 5,
        "high_risk_count": 2,
        "medium_risk_count": 3,
        "all_risk_events": [
            {"timestamp": 0.5, "vehicle_pair": [1, 2], "ttc": 1.5, "severity": "medium", "vehicle_a_class": "car", "vehicle_b_class": "car"},
            {"timestamp": 0.6, "vehicle_pair": [3, 4], "ttc": 1.2, "severity": "medium", "vehicle_a_class": "truck", "vehicle_b_class": "car"},
            {"timestamp": 0.8, "vehicle_pair": [1, 5], "ttc": 0.8, "severity": "high", "vehicle_a_class": "car", "vehicle_b_class": "car"},
            {"timestamp": 1.0, "vehicle_pair": [2, 6], "ttc": 1.9, "severity": "medium", "vehicle_a_class": "car", "vehicle_b_class": "car"},
            {"timestamp": 1.1, "vehicle_pair": [1, 2], "ttc": 0.5, "severity": "high", "vehicle_a_class": "car", "vehicle_b_class": "car"}
        ]
    }
    path = tmp_path / "session_data.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


@pytest.fixture
def sample_alert_log(tmp_path):
    """Create a temporary alert_log.jsonl for testing."""
    alerts = [
        {"alert_type": "forward_collision_warning", "timestamp": 0.5, "vehicles_involved": [1, 2], "time_to_collision": 1.5, "severity": "medium", "location": "Intersection_A", "vehicle_a_class": "car", "vehicle_b_class": "car"},
        {"alert_type": "forward_collision_warning", "timestamp": 0.6, "vehicles_involved": [3, 4], "time_to_collision": 1.2, "severity": "medium", "location": "Intersection_A", "vehicle_a_class": "truck", "vehicle_b_class": "car"},
        {"alert_type": "forward_collision_warning", "timestamp": 0.8, "vehicles_involved": [1, 5], "time_to_collision": 0.8, "severity": "high", "location": "Intersection_A", "vehicle_a_class": "car", "vehicle_b_class": "car"},
        {"alert_type": "forward_collision_warning", "timestamp": 1.1, "vehicles_involved": [1, 2], "time_to_collision": 0.5, "severity": "high", "location": "Intersection_A", "vehicle_a_class": "car", "vehicle_b_class": "car"}
    ]
    path = tmp_path / "alert_log.jsonl"
    with open(path, "w") as f:
        for a in alerts:
            f.write(json.dumps(a) + "\n")
    return str(path)


@pytest.fixture
def engine(sample_session_data, sample_alert_log):
    return AnalyticsEngine(session_path=sample_session_data, alert_log_path=sample_alert_log)


def test_kpi_calculation(engine):
    kpis = engine.get_kpis()
    assert kpis["total_frames"] == 100
    assert kpis["total_detections"] == 50
    assert kpis["unique_tracks"] == 15
    assert kpis["total_risk_events"] == 5
    assert kpis["high_risk_events"] == 2
    assert kpis["medium_risk_events"] == 3
    assert kpis["total_alerts"] == 4
    assert kpis["avg_ttc"] == 1.18  # (1.5 + 1.2 + 0.8 + 1.9 + 0.5) / 5
    assert kpis["min_ttc"] == 0.5
    assert kpis["processing_fps"] == 10.0


def test_class_distribution(engine):
    dist = engine.get_class_distribution()
    assert len(dist) == 2
    
    # Ordered by count descending
    assert dist[0]["class"] == "car"
    assert dist[0]["count"] == 40
    assert dist[0]["percentage"] == 80.0
    
    assert dist[1]["class"] == "truck"
    assert dist[1]["count"] == 10
    assert dist[1]["percentage"] == 20.0


def test_risk_distribution(engine):
    dist = engine.get_risk_distribution()
    assert len(dist) == 2
    
    high = next((d for d in dist if d["severity"] == "high"), None)
    assert high is not None
    assert high["count"] == 2
    assert high["percentage"] == 40.0
    assert high["avg_ttc"] == 0.65  # (0.8 + 0.5) / 2
    
    medium = next((d for d in dist if d["severity"] == "medium"), None)
    assert medium is not None
    assert medium["count"] == 3
    assert medium["percentage"] == 60.0
    assert medium["avg_ttc"] == round((1.5 + 1.2 + 1.9) / 3, 3)


def test_ttc_stats(engine):
    stats = engine.get_ttc_stats()
    assert stats["avg"] == 1.18
    assert stats["min"] == 0.5
    assert stats["max"] == 1.9
    assert stats["median"] == 1.2
    assert stats["count"] == 5


def test_most_involved_vehicles(engine):
    vehicles = engine.get_most_involved_vehicles()
    assert len(vehicles) > 0
    
    # ID 1 is in 3 events (1.5, 0.8, 0.5)
    id_1 = next((v for v in vehicles if v["vehicle_id"] == 1), None)
    assert id_1 is not None
    assert id_1["risk_events"] == 3
    assert id_1["min_ttc"] == 0.5
    assert id_1["avg_ttc"] == round((1.5 + 0.8 + 0.5) / 3, 3)
    
    # ID 2 is in 3 events (1.5, 1.9, 0.5)
    id_2 = next((v for v in vehicles if v["vehicle_id"] == 2), None)
    assert id_2 is not None
    assert id_2["risk_events"] == 3
    assert id_2["min_ttc"] == 0.5
    assert id_2["avg_ttc"] == round((1.5 + 1.9 + 0.5) / 3, 3)


def test_density_timeline(engine):
    timeline = engine.get_density_timeline()
    assert len(timeline) == 3
    assert timeline[0] == {"time": 0.0, "vehicle_count": 2}
    assert timeline[1] == {"time": 0.5, "vehicle_count": 4}
    assert timeline[2] == {"time": 1.0, "vehicle_count": 5}


def test_filtered_alerts(engine):
    # Filter by severity
    high_alerts = engine.get_filtered_alerts(severity="high")
    assert len(high_alerts) == 2
    assert all(a["severity"] == "high" for a in high_alerts)
    
    # Filter by class
    truck_alerts = engine.get_filtered_alerts(vehicle_class="truck")
    assert len(truck_alerts) == 1
    
    # Filter by ID
    id_1_alerts = engine.get_filtered_alerts(vehicle_id=1)
    assert len(id_1_alerts) == 3


def test_csv_export(engine, tmp_path):
    export_path = str(tmp_path / "test_report.csv")
    engine.export_csv(export_path)
    
    assert os.path.exists(export_path)
    with open(export_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 5  # Header + 4 alerts


def test_data_collector():
    collector = SessionDataCollector()
    collector.start("test.mp4")
    
    # Frame 1
    detections_1 = [{"class": "car"}, {"class": "car"}]
    tracked_1 = [{"track_id": 1}, {"track_id": 2}]
    risk_events_1 = [{"ttc": 1.5, "severity": "medium", "vehicle_pair": [1, 2]}]
    collector.record_frame(0.1, detections_1, tracked_1, risk_events_1)
    
    # Frame 2
    detections_2 = [{"class": "car"}, {"class": "truck"}]
    tracked_2 = [{"track_id": 2}, {"track_id": 3}]
    risk_events_2 = [{"ttc": 0.5, "severity": "high", "vehicle_pair": [2, 3]}]
    collector.record_frame(0.6, detections_2, tracked_2, risk_events_2)
    
    collector.finish()
    
    data = collector.to_dict()
    assert data["total_frames"] == 2
    assert data["total_detections"] == 4
    assert data["vehicle_classes"] == {"car": 3, "truck": 1}
    assert data["unique_tracks"] == 3
    assert data["unique_track_ids"] == [1, 2, 3]
    assert data["high_risk_count"] == 1
    assert data["medium_risk_count"] == 1
    assert len(data["all_ttc_values"]) == 2
    assert data["all_ttc_values"] == [1.5, 0.5]
