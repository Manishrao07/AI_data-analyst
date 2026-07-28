import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.agent.tools import run_sql_safely
from app.core.data_manager import Session


@pytest.fixture
def session_with_table(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("region,revenue\nNorth,100\nSouth,200\n")
    session = Session("test")
    session.add_csv(str(csv_path), "data.csv")
    return session


def test_allows_select(session_with_table):
    df = run_sql_safely(session_with_table, 'SELECT * FROM "data"')
    assert len(df) == 2


@pytest.mark.parametrize("bad_sql", [
    'DROP TABLE "data"',
    'DELETE FROM "data"',
    'UPDATE "data" SET revenue = 0',
    'INSERT INTO "data" VALUES (1, 2)',
])
def test_blocks_destructive_queries(session_with_table, bad_sql):
    with pytest.raises(ValueError):
        run_sql_safely(session_with_table, bad_sql)
