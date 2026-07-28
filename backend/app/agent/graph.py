"""
LangGraph orchestration for the Data Analyst agent.

Flow:
    router -> (sql_node | chart_node | anomaly_node | insight_node | general_node) -> END

Each node appends to `trace`, so the final response can show the user
*why* the agent did what it did (the assignment explicitly asks for this).
"""
import pandas as pd
from langgraph.graph import END, StateGraph

from app.agent import tools
from app.agent.state import AgentState
from app.core.data_manager import Session
from app.core.logger import get_logger

logger = get_logger(__name__)


def _router_node(state: AgentState) -> AgentState:
    intent, reasoning = tools.classify_intent(state["question"], state["schema_summary"])
    state["intent"] = intent
    state["intent_reasoning"] = reasoning
    state.setdefault("trace", []).append({"step": "router", "detail": f"Classified as '{intent}': {reasoning}"})
    return state


def _make_sql_node(session: Session):
    def _sql_node(state: AgentState) -> AgentState:
        sql = tools.generate_sql(state["question"], state["schema_summary"], state.get("history", []))
        state["sql_query"] = sql
        state["trace"].append({"step": "sql_generation", "detail": f"Generated query: {sql}"})
        try:
            df = tools.run_sql_safely(session, sql)
            preview = df.head(20).to_string(index=False)
            state["sql_result_preview"] = preview
            state["trace"].append({"step": "sql_execution", "detail": f"Returned {len(df)} rows"})
            answer = tools.generate_insight_summary(
                state["question"],
                preview,
                state.get("history", []),
                is_sql_result=True,
                sql_query=sql,
            )
            state["final_answer"] = answer
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            state["final_answer"] = f"I generated this SQL but it failed to run: `{sql}`\nError: {e}"
        return state

    return _sql_node


def _make_chart_node(session: Session):
    def _chart_node(state: AgentState) -> AgentState:
        spec = tools.infer_chart_spec(state["question"], state["schema_summary"])
        state["chart_spec"] = spec
        state["trace"].append({"step": "chart_spec", "detail": f"Chart parameters: {spec}"})

        # pull the underlying data across all registered tables (agent uses the first/most relevant table)
        table = next(iter(session.datasets.keys()))
        df = session.get_dataframe(table)
        try:
            result = tools.build_chart(
                df,
                chart_type=spec["chart_type"],
                x=spec["x"],
                y=spec.get("y"),
                color=spec.get("color"),
                title=spec.get("title"),
                agg=spec.get("agg", "sum"),
            )
            state["chart_result"] = result
            state["trace"].append({"step": "chart_render", "detail": f"Rendered {spec['chart_type']} chart"})
            state["final_answer"] = f"Here's the {spec['chart_type']} chart for: {spec.get('title', state['question'])}"
        except Exception as e:
            logger.error(f"Chart build failed: {e}")
            state["final_answer"] = f"Couldn't build that chart: {e}"
        return state

    return _chart_node


def _make_anomaly_node(session: Session):
    def _anomaly_node(state: AgentState) -> AgentState:
        table = next(iter(session.datasets.keys()))
        df = session.get_dataframe(table)
        result = detect_anomalies_wrapper(df)
        state["anomaly_result"] = result
        state["trace"].append({
            "step": "anomaly_detection",
            "detail": f"Flagged {result['total_flagged']}/{result['total_rows']} rows using {result['method']}",
        })
        state["final_answer"] = tools.summarize_anomalies(state["question"], result)
        return state

    return _anomaly_node


def detect_anomalies_wrapper(df):
    from app.core.anomaly_engine import detect_anomalies
    return detect_anomalies(df)


def _make_insight_node(session: Session):
    def _insight_node(state: AgentState) -> AgentState:
        table = next(iter(session.datasets.keys()))
        df = session.get_dataframe(table)
        preview = df.describe(include="all").to_string()[:3000]

        # If there's a date-like column, add a FULL monthly aggregation so
        # insight questions that touch on trends aren't limited to a 10-row
        # sample (this is what previously made "monthly trend" answers only
        # see January). This is a safety net in case the router still sends
        # a trend-style question here instead of to the chart node.
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            try:
                dcol = date_cols[0]
                tmp = df.copy()
                tmp[dcol] = pd.to_datetime(tmp[dcol])
                numeric_cols = tmp.select_dtypes("number").columns.tolist()
                if numeric_cols:
                    monthly = tmp.groupby(tmp[dcol].dt.to_period("M"))[numeric_cols].sum()
                    preview += "\n\nMonthly aggregation (full dataset, all rows):\n" + monthly.to_string()
            except Exception as e:
                logger.warning(f"Monthly aggregation for insight context failed: {e}")

        preview += "\n\nSample rows:\n" + df.head(10).to_string(index=False)
        state["trace"].append({
            "step": "insight_context",
            "detail": "Built statistical summary + monthly aggregation (if date column present) + sample rows",
        })
        state["final_answer"] = tools.generate_insight_summary(state["question"], preview, state.get("history", []))
        return state

    return _insight_node


def _general_node(state: AgentState) -> AgentState:
    state["trace"].append({"step": "general", "detail": "Handled as a general conversational message"})
    state["final_answer"] = (
        "I'm your AI data analyst — ask me things like 'which region had the highest revenue' "
        "or 'show monthly sales trends' and I'll query, chart, or analyze your uploaded data."
    )
    return state


def _route_decision(state: AgentState) -> str:
    return state["intent"]


def build_agent_graph(session: Session):
    """Builds a fresh compiled graph bound to this session's DuckDB connection."""
    graph = StateGraph(AgentState)

    graph.add_node("router", _router_node)
    graph.add_node("sql", _make_sql_node(session))
    graph.add_node("chart", _make_chart_node(session))
    graph.add_node("anomaly", _make_anomaly_node(session))
    graph.add_node("insight", _make_insight_node(session))
    graph.add_node("general", _general_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {"sql": "sql", "chart": "chart", "anomaly": "anomaly", "insight": "insight", "general": "general"},
    )
    for node in ["sql", "chart", "anomaly", "insight", "general"]:
        graph.add_edge(node, END)

    return graph.compile()


def run_agent(session: Session, question: str) -> AgentState:
    app = build_agent_graph(session)
    initial_state: AgentState = {
        "session_id": session.session_id,
        "question": question,
        "history": session.history,
        "schema_summary": session.schema_summary(),
        "trace": [],
    }
    result = app.invoke(initial_state)

    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": result.get("final_answer", "")})
    return result