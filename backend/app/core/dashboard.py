"""
Computes summary metrics for the Dashboard bonus feature -- a quick
at-a-glance view of the dataset (total revenue, top region, top product,
order count, date range, anomaly count) rather than making the user ask
each of these as separate chat questions.

Runs directly against the session's DuckDB connection so numbers are exact,
not LLM-estimated.
"""
from app.core.anomaly_engine import detect_anomalies
from app.core.data_manager import Session
from app.core.logger import get_logger

logger = get_logger(__name__)


def build_dashboard_summary(session: Session, table_name: str) -> dict:
    df = session.get_dataframe(table_name)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower()]

    summary: dict = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }

    revenue_col = next((c for c in numeric_cols if c.lower() in {"revenue", "total", "amount", "sales"}), None)
    if revenue_col:
        summary["total_revenue"] = round(float(df[revenue_col].sum()), 2)
        summary["avg_order_value"] = round(float(df[revenue_col].mean()), 2)

    # "breakdowns" -- per-dimension top value + a small ranked list for charting.
    # Key name matches what the frontend dashboard tab expects.
    breakdowns = {}
    for col in text_cols:
        n_unique = df[col].nunique()
        if 1 < n_unique <= 20 and revenue_col:
            grouped = df.groupby(col)[revenue_col].sum().sort_values(ascending=False)
            top_n = grouped.head(6)
            breakdowns[col] = {
                "top_value": grouped.index[0],
                "top_value_total": round(float(grouped.iloc[0]), 2),
                "all_values": [
                    {"label": str(label), "value": round(float(val), 2)}
                    for label, val in top_n.items()
                ],
            }
    summary["breakdowns"] = breakdowns

    if date_cols:
        try:
            dcol = date_cols[0]
            summary["date_range"] = {
                "start": str(df[dcol].min()),
                "end": str(df[dcol].max()),
            }
        except Exception as e:
            logger.warning(f"Dashboard date range calc failed: {e}")

    try:
        anomaly_result = detect_anomalies(df)
        summary["anomaly_count"] = anomaly_result["total_flagged"]
    except Exception as e:
        logger.warning(f"Dashboard anomaly calc failed: {e}")
        summary["anomaly_count"] = None

    info = session.datasets.get(table_name)
    summary["data_quality_warnings"] = info.warnings if info else []

    return summary