import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.anomaly_engine import detect_anomalies


def test_detects_obvious_outlier():
    df = pd.DataFrame({
        "revenue": [100, 110, 95, 105, 90, 115, 10000, 108, 98, 102, 101, 99],
    })
    result = detect_anomalies(df)
    flagged_indices = [a["row_index"] for a in result["anomalies"]]
    assert 6 in flagged_indices  # the 10000 outlier
    assert result["total_flagged"] >= 1


def test_no_anomalies_in_uniform_data():
    # Isolation Forest's contamination param always flags a small floor of points
    # even in uniform data, so we assert "very few flagged", not "zero".
    df = pd.DataFrame({"value": [50, 51, 49, 50, 52, 48, 50, 51, 49, 50] * 5})
    result = detect_anomalies(df, contamination=0.01)
    assert result["total_flagged"] <= 2


def test_handles_no_numeric_columns():
    df = pd.DataFrame({"name": ["a", "b", "c"]})
    result = detect_anomalies(df)
    assert result["anomalies"] == []
    assert result["method"] == "none"
