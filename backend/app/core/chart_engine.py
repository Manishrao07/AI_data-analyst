"""
Builds Plotly charts from a dataframe + a simple chart spec.
Returns chart.to_json() so the frontend (or any client) can render it
without re-running the aggregation logic.
"""
import pandas as pd
import plotly.express as px

from app.core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_TYPES = {"bar", "line", "pie", "scatter", "histogram", "box"}


def build_chart(df: pd.DataFrame, chart_type: str, x: str, y: str | None = None,
                 color: str | None = None, title: str | None = None, agg: str = "sum") -> dict:
    chart_type = chart_type.lower()
    if chart_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported chart type '{chart_type}'. Choose from {SUPPORTED_TYPES}")

    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found in dataset.")
    if y and y not in df.columns:
        raise ValueError(f"Column '{y}' not found in dataset.")

    plot_df = df.copy()

    # aggregate when x is categorical and y is numeric (typical BI question: "revenue by region")
    if y and chart_type in {"bar", "line", "pie"} and plot_df[x].dtype == object:
        group_cols = [x] + ([color] if color and color != x else [])
        plot_df = plot_df.groupby(group_cols, as_index=False)[y].agg(agg)

    title = title or f"{y or x} by {x}"

    if chart_type == "bar":
        fig = px.bar(plot_df, x=x, y=y, color=color, title=title)
    elif chart_type == "line":
        fig = px.line(plot_df.sort_values(x), x=x, y=y, color=color, title=title, markers=True)
    elif chart_type == "pie":
        fig = px.pie(plot_df, names=x, values=y, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(plot_df, x=x, y=y, color=color, title=title)
    elif chart_type == "histogram":
        fig = px.histogram(plot_df, x=x, color=color, title=title)
    elif chart_type == "box":
        fig = px.box(plot_df, x=x, y=y, color=color, title=title)

    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40))
    return {"chart_json": fig.to_json(), "chart_type": chart_type, "title": title}
