"""
Detects anomalies in numeric columns using two complementary methods:
1. IQR (fast, explainable, good for single-column outliers)
2. Isolation Forest (catches multivariate anomalies IQR would miss)

Every flagged row comes with a plain-language reason -- the assignment
explicitly asks for "explain why they were flagged", not just a list of ids.

KEY FIX (v2): IQR was originally computed globally across the whole
column. On this dataset that's wrong -- e.g. unit_price mixes Notebook
Sets (~4.5), Desk Lamps (~20), and Office Chairs (~150) together, so
every normal Office Chair order gets flagged as an outlier just because
it's expensive compared to a notebook. The fix is to compute IQR bounds
*per category* (or per product, if you want tighter baselines) so each
row is compared against its own peer group, not the whole dataset.

We also exclude clearly derived/correlated columns (e.g. revenue =
quantity * unit_price) from the IQR check -- flagging revenue on top of
quantity and unit_price just triple-counts the same underlying event and
inflates severity without adding new information.

KEY FIX (v3): meaningless id-like columns (e.g. order_id) were being fed
into the Isolation Forest as a numeric feature via select_dtypes(number),
which distorted which row got flagged (a mid-range id could look "unusual"
next to genuinely extreme quantity/price values). We now exclude any
column whose name contains "id" the same way we exclude derived columns.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.logger import get_logger

logger = get_logger(__name__)

# Columns that are mathematically derived from other numeric columns.
# Flagging these independently double/triple-counts the same anomaly.
DEFAULT_EXCLUDE_DERIVED = {"revenue", "total", "amount", "total_price", "line_total"}

# Substring match: any numeric column whose name contains this is treated as
# an identifier (order_id, customer_id, row_id, ...) and never used as a
# model feature -- ids carry no statistical meaning and only add noise.
ID_LIKE_SUBSTRING = "id"


def detect_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    group_by: str | None = "category",
    exclude_derived: set[str] | None = None,
    contamination: float = 0.05,
    iqr_multiplier: float = 1.5,
    min_severity: float = 0.3,
) -> dict:
    exclude_derived = {c.lower() for c in (exclude_derived or DEFAULT_EXCLUDE_DERIVED)}

    numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c for c in numeric_cols
        if c in df.columns
        and c.lower() not in exclude_derived
        and ID_LIKE_SUBSTRING not in c.lower()
    ]

    if not numeric_cols:
        return {"anomalies": [], "method": "none", "message": "No numeric columns available for anomaly detection."}

    use_grouping = bool(group_by) and group_by in df.columns
    groups = df.groupby(group_by) if use_grouping else [(None, df)]

    iqr_flags = pd.Series(False, index=df.index)
    reasons: dict[int, list[str]] = {}
    bounds_by_row: dict[int, dict[str, tuple[float, float]]] = {}

    for group_val, gdf in groups:
        for col in numeric_cols:
            series = gdf[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
            mask = (gdf[col] < lower) | (gdf[col] > upper)
            for idx in gdf.index[mask]:
                group_note = f" within '{group_val}'" if group_val is not None else ""
                reasons.setdefault(idx, []).append(
                    f"'{col}' = {df.loc[idx, col]:.2f} is outside the normal range "
                    f"[{lower:.2f}, {upper:.2f}]{group_note}"
                )
                bounds_by_row.setdefault(idx, {})[col] = (lower, upper)
            iqr_flags |= mask.reindex(df.index, fill_value=False)

    # --- Isolation Forest (multivariate, dataset-wide) ---
    iso_flags = pd.Series(False, index=df.index)
    try:
        clean = df[numeric_cols].dropna()
        if len(clean) >= 10:
            model = IsolationForest(contamination=contamination, random_state=42, n_jobs=1)
            preds = model.fit_predict(clean)
            iso_series = pd.Series(preds == -1, index=clean.index)
            iso_flags.loc[iso_series.index] = iso_series
            for idx in clean.index[iso_series]:
                reasons.setdefault(idx, []).append(
                    "Flagged by Isolation Forest as an unusual combination across multiple columns"
                )
    except Exception as e:
        logger.warning(f"Isolation Forest skipped: {e}")

    combined = iqr_flags | iso_flags

    # Severity: max normalized deviation from the (per-group) IQR bounds
    # that actually flagged this row, plus a bonus for Isolation Forest hits.
    severity: dict[int, float] = {}
    for idx in df.index[combined]:
        score = 0.0
        for col, (lower, upper) in bounds_by_row.get(idx, {}).items():
            val = df.loc[idx, col]
            if pd.isna(val):
                continue
            if val > upper:
                score = max(score, (val - upper) / (upper - lower + 1e-9))
            elif val < lower:
                score = max(score, (lower - val) / (upper - lower + 1e-9))
        if iso_flags.get(idx, False):
            score += 0.5  # slight boost for multivariate flags
        severity[idx] = score

    # Only keep rows whose deviation clears the minimum severity floor --
    # this drops rows that barely poke past a bound while keeping genuinely
    # extreme ones.
    final_flagged = [idx for idx in df.index[combined] if severity.get(idx, 0) >= min_severity]

    results = []
    for idx in sorted(final_flagged, key=lambda i: severity.get(i, 0), reverse=True):
        row = df.loc[idx].to_dict()
        results.append({
            "row_index": int(idx),
            "row_data": {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()},
            "reasons": reasons.get(idx, ["Flagged by anomaly model"]),
            "severity": round(severity.get(idx, 0), 2),
        })

    return {
        "anomalies": results,
        "method": f"IQR (grouped by '{group_by}')" if use_grouping else "IQR (global)",
        "total_flagged": len(results),
        "total_rows": len(df),
        "columns_checked": numeric_cols,
    }