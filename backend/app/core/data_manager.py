"""
Handles CSV ingestion, validation, multi-file support, and gives every
session an in-memory DuckDB connection so the agent can run real SQL
against uploaded data (not just pandas string-munging).
"""
import os
import uuid
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetInfo:
    table_name: str
    filename: str
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    warnings: list[str] = field(default_factory=list)


class Session:
    """One session = one user's uploaded dataset(s) + their own DuckDB connection."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.con = duckdb.connect(database=":memory:")

        # Force single-threaded execution -- multi-threaded DuckDB result
        # materialization (fetchdf via pyarrow/mimalloc) segfaults on some
        # Apple Silicon setups. This trades a little query speed for stability.
        self.con.execute("PRAGMA threads=1;")

        self.datasets: dict[str, DatasetInfo] = {}
        self.history: list[dict] = []  # conversation memory

    def add_csv(self, file_path: str, filename: str) -> DatasetInfo:
        warnings = []

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Could not parse '{filename}' as CSV: {e}")

        if df.empty:
            raise ValueError(f"'{filename}' has no rows.")

        # basic data-quality checks (bonus feature)
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            warnings.append(f"Missing values detected in columns: {null_cols}")

        dup_count = df.duplicated().sum()
        if dup_count > 0:
            warnings.append(f"{dup_count} duplicate rows found")

        # try to parse obvious date columns for time-series questions
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

        table_name = self._safe_table_name(filename)
        self.con.register(table_name, df)

        # persist a real table (register() is a view tied to the df object's lifetime)
        self.con.execute(
            f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM "{table_name}"'
        )

        info = DatasetInfo(
            table_name=table_name,
            filename=filename,
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=list(df.columns),
            dtypes={c: str(t) for c, t in df.dtypes.items()},
            warnings=warnings,
        )

        self.datasets[table_name] = info
        logger.info(
            f"[{self.session_id}] Registered table '{table_name}' ({len(df)} rows)"
        )

        return info

    def get_dataframe(self, table_name: str) -> pd.DataFrame:
        return self._safe_fetch(f'SELECT * FROM "{table_name}"')

    def run_sql(self, query: str) -> pd.DataFrame:
        return self._safe_fetch(query)

    def _safe_fetch(self, query: str) -> pd.DataFrame:
        """
        Builds the DataFrame manually from fetchall() + column descriptions,
        bypassing DuckDB's default fetchdf() which routes through PyArrow's
        mimalloc allocator -- that path segfaults on some Apple Silicon setups
        under multi-threaded execution (uvicorn's threadpool). This is slightly
        slower for very large results but eliminates the crash entirely.
        """
        result = self.con.execute(query)
        rows = result.fetchall()
        columns = [desc[0] for desc in result.description]
        return pd.DataFrame(rows, columns=columns)

    def schema_summary(self) -> str:
        """
        Human-readable schema description fed to the LLM as grounding context.

        IMPORTANT: for text/categorical columns we also include a handful of
        actual distinct values (e.g. product = ['Notebook Set', 'Office
        Chair', ...]). Without this, the LLM has no way to know the exact
        string values stored in the data and will guess plausible-looking
        but wrong filters (e.g. WHERE product = 'notebook' instead of the
        real value 'Notebook Set'), silently returning NaN/empty results.
        """
        parts = []

        for t, info in self.datasets.items():
            df = self.get_dataframe(t)
            col_descriptions = []

            for c in info.columns:
                dtype = info.dtypes[c]
                col_desc = f"{c} ({dtype})"

                if (
                    df[c].dtype == object
                    or str(dtype).startswith("object")
                    or str(dtype).startswith("string")
                ):
                    uniques = df[c].dropna().unique()

                    if 0 < len(uniques) <= 30:
                        sample = list(uniques)
                    else:
                        sample = list(uniques[:10])

                    col_desc += f" -- sample values: {sample}"

                col_descriptions.append(col_desc)

            cols = ", ".join(col_descriptions)
            parts.append(f"Table '{t}' ({info.n_rows} rows): {cols}")

        return "\n".join(parts)

    @staticmethod
    def _safe_table_name(filename: str) -> str:
        base = os.path.splitext(filename)[0]
        safe = "".join(
            c if c.isalnum() else "_" for c in base
        ).strip("_").lower()

        return safe or f"table_{uuid.uuid4().hex[:6]}"


class SessionStore:
    """In-memory registry of active sessions. Swap for Redis in production."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        sid = uuid.uuid4().hex[:16]
        session = Session(sid)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_error(self, session_id: str) -> Session:
        session = self.get(session_id)

        if session is None:
            raise ValueError(
                "Session not found. Upload a CSV first to start a session."
            )

        return session


session_store = SessionStore()