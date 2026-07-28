import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.data_manager import Session


def _write_csv(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def test_loads_valid_csv():
    path = _write_csv("a,b\n1,2\n3,4\n")
    session = Session("test-session")
    info = session.add_csv(path, "test.csv")
    assert info.n_rows == 2
    assert info.n_cols == 2


def test_rejects_empty_csv():
    path = _write_csv("a,b\n")
    session = Session("test-session")
    with pytest.raises(ValueError):
        session.add_csv(path, "empty.csv")


def test_flags_missing_values_as_warning():
    path = _write_csv("a,b\n1,\n3,4\n")
    session = Session("test-session")
    info = session.add_csv(path, "nulls.csv")
    assert any("Missing values" in w for w in info.warnings)


def test_sql_execution_on_loaded_table():
    path = _write_csv("region,revenue\nNorth,100\nSouth,200\n")
    session = Session("test-session")
    info = session.add_csv(path, "sales.csv")
    df = session.run_sql(f'SELECT SUM(revenue) as total FROM "{info.table_name}"')
    assert df["total"].iloc[0] == 300
