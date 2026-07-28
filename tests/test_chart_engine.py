import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.chart_engine import build_chart


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "region": ["North", "South", "North", "East"],
        "revenue": [100, 200, 150, 90],
    })


def test_bar_chart_aggregates_by_group(sample_df):
    result = build_chart(sample_df, "bar", x="region", y="revenue")
    assert result["chart_type"] == "bar"
    assert "chart_json" in result


def test_invalid_chart_type_raises(sample_df):
    with pytest.raises(ValueError):
        build_chart(sample_df, "pyramid", x="region", y="revenue")


def test_missing_column_raises(sample_df):
    with pytest.raises(ValueError):
        build_chart(sample_df, "bar", x="nonexistent", y="revenue")
