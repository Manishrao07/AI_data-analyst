"""
Individual tools the LangGraph agent nodes invoke. Kept as plain functions
(not classes) so they're easy to unit test in isolation -- see tests/.
"""
import json
import re

import pandas as pd

from app.core.anomaly_engine import detect_anomalies
from app.core.chart_engine import build_chart
from app.core.data_manager import Session
from app.core.llm_client import chat
from app.core.logger import get_logger

logger = get_logger(__name__)


def generate_sql(question: str, schema_summary: str, history: list[dict]) -> str:
    """Ask the LLM to write a DuckDB-flavoured SQL query grounded in the real schema."""
    system = (
        "You are a senior data analyst. Write a single valid DuckDB SQL query that answers "
        "the user's question, using ONLY the tables/columns listed in the schema below. "
        "Never invent column names. Always SELECT the actual metric/aggregate value(s) needed "
        "to answer the question (e.g. include SUM(revenue) AS total_revenue), not just the "
        "grouping/label column -- the result must contain enough detail to state a concrete answer. "
        "Return ONLY the SQL query, no explanation, no markdown fences."
        f"\n\nSchema:\n{schema_summary}"
    )
    messages = [{"role": "system", "content": system}]
    for turn in history[-4:]:
        messages.append(turn)
    messages.append({"role": "user", "content": question})

    raw = chat(messages, temperature=0.1)
    sql = re.sub(r"^```sql\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return sql


def run_sql_safely(session: Session, sql: str) -> pd.DataFrame:
    """Blocks destructive statements -- this agent is read-only by design."""
    forbidden = ["drop ", "delete ", "update ", "insert ", "alter ", "truncate ", "create ", "attach "]
    lowered = sql.lower()
    if any(kw in lowered for kw in forbidden):
        raise ValueError("Only read-only SELECT queries are permitted.")
    if not lowered.strip().startswith("select"):
        raise ValueError("Generated query must be a SELECT statement.")
    return session.run_sql(sql)


def infer_chart_spec(question: str, schema_summary: str) -> dict:
    """Ask the LLM to pick chart type + columns as structured JSON."""
    system = (
        "You choose chart parameters for a plotting library based on a user's question and "
        "the dataset schema. Respond with ONLY valid JSON, no prose, no markdown fences, matching:\n"
        '{"chart_type": "bar|line|pie|scatter|histogram|box", "x": "<column>", '
        '"y": "<column or null>", "color": "<column or null>", "agg": "sum|mean|count", "title": "<short title>"}'
        f"\n\nSchema:\n{schema_summary}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    raw = chat(messages, temperature=0.1)
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Chart spec JSON parse failed, raw={raw}")
        raise ValueError("Could not determine chart parameters from the question.")


def generate_insight_summary(
    question: str,
    data_preview: str,
    history: list[dict],
    is_sql_result: bool = False,
    sql_query: str = "",
) -> str:
    if is_sql_result:
        limit_note = ""
        if "limit" in sql_query.lower():
            limit_note = (
                " Note: this query used LIMIT to return only the top result(s) -- "
                "the full dataset likely has more categories/rows than shown here, "
                "so do not claim this is the only category/value that exists in the dataset."
            )
        context_note = (
            "The data below is the RESULT of a SQL query that was already run to answer the "
            "user's question -- it is the answer, not a limited sample of the dataset. "
            "Do not say the data is 'limited' or that other categories/values are missing "
            "unless this note explicitly says so." + limit_note
        )
    else:
        context_note = (
            "The data below is a preview/sample from the full dataset (not the complete dataset)."
        )
    system = (
        "You are a business data analyst. Answer the user's question using ONLY the data preview "
        f"provided. {context_note} Be concrete: cite actual numbers and names from the data. "
        "Keep it to 3-6 sentences unless the user asked for more detail. Explain the reasoning "
        "briefly, don't just state a number."
        f"\n\nData:\n{data_preview}"
    )
    messages = [{"role": "system", "content": system}]
    for turn in history[-4:]:
        messages.append(turn)
    messages.append({"role": "user", "content": question})
    return chat(messages, temperature=0.3)


def summarize_anomalies(question: str, anomaly_result: dict) -> str:
    if anomaly_result["total_flagged"] == 0:
        return "No significant anomalies were detected in the dataset using IQR and Isolation Forest analysis."

    top = anomaly_result["anomalies"][:8]
    lines = []
    for a in top:
        lines.append(f"Row {a['row_index']}: {a['row_data']} -- {'; '.join(a['reasons'])}")
    preview = "\n".join(lines)

    system = (
        "You are a data analyst explaining detected anomalies to a business user in plain language. "
        "Summarize the pattern across the flagged rows below, and explain plausible business reasons "
        "for the outliers (e.g. bulk order, data entry error, seasonal spike). Be concise."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nFlagged rows:\n{preview}"},
    ]
    return chat(messages, temperature=0.3)


def classify_intent(question: str, schema_summary: str) -> tuple[str, str]:
    """Router: decide which tool(s) the question needs."""
    system = (
        "Classify the user's data-analysis question into exactly one category:\n"
        "- 'sql': needs a precise query/aggregation/lookup answer (e.g. top N, totals, filters)\n"
        "- 'chart': user wants a chart/graph/plot/visualization, OR asks to 'show' a trend, "
        "pattern, or breakdown over time/category (e.g. 'show monthly sales trends', "
        "'sales by region') -- these need a visual, not just prose, even without the word 'chart'\n"
        "- 'anomaly': user wants outliers/anomalies/unusual data detected\n"
        "- 'insight': open-ended business insight, explanation, or summary question that does NOT "
        "ask to 'show' a trend/breakdown (e.g. 'why did sales drop', 'summarize this dataset')\n"
        "- 'general': greeting or question unrelated to the dataset\n\n"
        "Respond with ONLY JSON: {\"intent\": \"<category>\", \"reasoning\": \"<one short sentence>\"}"
        f"\n\nSchema:\n{schema_summary}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    raw = chat(messages, temperature=0.0)
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed.get("intent", "insight"), parsed.get("reasoning", "")
    except json.JSONDecodeError:
        return "insight", "Fell back to general insight due to router parse failure."