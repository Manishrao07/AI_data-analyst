"""
Shared state object passed between LangGraph nodes.
Using a TypedDict keeps the graph state serializable and easy to log.
"""
from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    history: list[dict]          # prior turns, for conversational context
    schema_summary: str

    intent: str                  # router's decision: sql | chart | anomaly | insight | general
    intent_reasoning: str

    sql_query: Optional[str]
    sql_result_preview: Optional[str]

    chart_spec: Optional[dict]
    chart_result: Optional[dict]

    anomaly_result: Optional[dict]

    final_answer: str
    trace: list[dict]            # step-by-step log shown to the user as "reasoning"
