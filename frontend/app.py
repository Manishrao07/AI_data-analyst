"""
Streamlit frontend for the AI Data Analyst.
Two tabs: Chat (with inline similar-past-question detection) and Dashboard
(polished metric cards + charts). The old separate "Search History" tab was
dropped in favor of proactive similar-question surfacing inside the chat.
"""
import os

import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

# ---------- light custom styling for metric cards ----------
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 18px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem;
        opacity: 0.7;
    }
    .breakdown-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .similar-q-banner {
        background: rgba(99, 102, 241, 0.12);
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "datasets" not in st.session_state:
    st.session_state.datasets = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dashboard" not in st.session_state:
    st.session_state.dashboard = None

# ---------- sidebar: upload ----------
with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.caption("Upload CSVs, then ask questions in plain English.")

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Start Session", use_container_width=True):
        with st.spinner("Validating and loading your data..."):
            files_payload = [("files", (f.name, f.getvalue(), "text/csv")) for f in uploaded_files]
            try:
                resp = requests.post(f"{BACKEND_URL}/api/upload", files=files_payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.datasets = data["datasets"]
                st.session_state.messages = []
                st.session_state.dashboard = None
                st.success(data["message"])
            except requests.exceptions.RequestException as e:
                st.error(f"Upload failed: {e}")

    if st.session_state.datasets:
        st.divider()
        st.subheader("Loaded datasets")
        for d in st.session_state.datasets:
            with st.expander(f"📄 {d['filename']} ({d['n_rows']} rows)"):
                st.write(f"**Columns:** {', '.join(d['columns'])}")
                if d["warnings"]:
                    for w in d["warnings"]:
                        st.warning(w)

    if st.session_state.messages:
        st.divider()
        if st.button("📄 Export session as PDF", use_container_width=True):
            with st.spinner("Building report..."):
                resp = requests.post(
                    f"{BACKEND_URL}/api/export/pdf",
                    json={"session_id": st.session_state.session_id},
                    timeout=30,
                )
                if resp.ok:
                    st.download_button(
                        "⬇️ Download report",
                        data=resp.content,
                        file_name="analysis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.error("Failed to build report.")

    st.divider()
    st.subheader("Try asking")
    for q in [
        "Which region generated the highest revenue?",
        "Show monthly sales trends",
        "Which products are underperforming?",
        "What are the top 5 customers?",
        "Detect anomalies in the dataset",
        "Generate SQL for total revenue by category",
    ]:
        st.code(q, language=None)

# ---------- main ----------
if not st.session_state.session_id:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    st.stop()

tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard"])

# ============================================================
# CHAT TAB
# ============================================================
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("similar_past_question"):
                spq = msg["similar_past_question"]
                pct = round(spq["similarity"] * 100)
                st.markdown(
                    f"""<div class="similar-q-banner">
                    📌 <b>Similar to a question you asked earlier</b> ({pct}% match):<br/>
                    <i>"{spq['question']}"</i>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            if msg.get("chart_json"):
                fig = pio.from_json(msg["chart_json"])
                st.plotly_chart(fig, use_container_width=True)
            if msg.get("sql_query"):
                st.code(msg["sql_query"], language="sql")
            if msg.get("trace"):
                with st.expander("🧠 Agent reasoning"):
                    for step in msg["trace"]:
                        st.markdown(f"**{step['step']}**: {step['detail']}")
            if msg.get("anomalies") and msg["anomalies"]["total_flagged"] > 0:
                with st.expander(f"⚠️ {msg['anomalies']['total_flagged']} anomalies flagged"):
                    st.dataframe(
                        [{"row": a["row_index"], "severity": a.get("severity"), **a["row_data"],
                          "reason": "; ".join(a["reasons"])}
                         for a in msg["anomalies"]["anomalies"][:20]]
                    )

    question = st.chat_input("Ask a question about your data...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/ask",
                        json={"session_id": st.session_state.session_id, "question": question},
                        timeout=90,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            if result.get("similar_past_question"):
                spq = result["similar_past_question"]
                pct = round(spq["similarity"] * 100)
                st.markdown(
                    f"""<div class="similar-q-banner">
                    📌 <b>Similar to a question you asked earlier</b> ({pct}% match):<br/>
                    <i>"{spq['question']}"</i>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(result["answer"])

            chart_json = None
            if result.get("chart"):
                chart_json = result["chart"]["chart_json"]
                fig = pio.from_json(chart_json)
                st.plotly_chart(fig, use_container_width=True)

            if result.get("sql_query"):
                st.code(result["sql_query"], language="sql")

            with st.expander("🧠 Agent reasoning"):
                for step in result["trace"]:
                    st.markdown(f"**{step['step']}**: {step['detail']}")

            if result.get("anomalies") and result["anomalies"]["total_flagged"] > 0:
                with st.expander(f"⚠️ {result['anomalies']['total_flagged']} anomalies flagged"):
                    st.dataframe(
                        [{"row": a["row_index"], "severity": a.get("severity"), **a["row_data"],
                          "reason": "; ".join(a["reasons"])}
                         for a in result["anomalies"]["anomalies"][:20]]
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "chart_json": chart_json,
                "sql_query": result.get("sql_query"),
                "trace": result.get("trace"),
                "anomalies": result.get("anomalies"),
                "similar_past_question": result.get("similar_past_question"),
            })

# ============================================================
# DASHBOARD TAB
# ============================================================
with tab_dashboard:
    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.dashboard = None

    if st.session_state.dashboard is None:
        with st.spinner("Building dashboard..."):
            try:
                resp = requests.get(
                    f"{BACKEND_URL}/api/dashboard/{st.session_state.session_id}", timeout=30
                )
                resp.raise_for_status()
                st.session_state.dashboard = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not load dashboard: {e}")
                st.stop()

    dash = st.session_state.dashboard

    # --- top metric row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"₹{dash.get('total_revenue', 0):,.2f}")
    m2.metric("Avg Order Value", f"₹{dash.get('avg_order_value', 0):,.2f}")
    m3.metric("Total Rows", f"{dash.get('total_rows', 0):,}")
    m4.metric("Anomalies Flagged", dash.get("anomaly_count", 0),
               delta=None if dash.get("anomaly_count", 0) == 0 else "review advised",
               delta_color="inverse")

    st.markdown("### 📈 Breakdown highlights")

    breakdowns = dash.get("breakdowns", {})
    if breakdowns:
        # render each breakdown as a small horizontal bar chart instead of raw dict text
        cols = st.columns(2)
        for i, (dim, info) in enumerate(breakdowns.items()):
            with cols[i % 2]:
                top_value = info.get("top_value")
                top_total = info.get("top_value_total")
                all_values = info.get("all_values")  # optional: list of {label, value}

                st.markdown(
                    f"""<div class="breakdown-card">
                    <b>Top {dim.replace('_', ' ').title()}</b><br/>
                    <span style="font-size:1.4rem;">{top_value}</span>
                    &nbsp;·&nbsp; ₹{top_total:,.2f}
                    </div>""",
                    unsafe_allow_html=True,
                )

                if all_values:
                    labels = [v["label"] for v in all_values][:6]
                    values = [v["value"] for v in all_values][:6]
                    fig = go.Figure(go.Bar(
                        x=values, y=labels, orientation="h",
                        marker_color="#6366f1",
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        height=220,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title=None, yaxis_title=None,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"dash_{dim}")
    else:
        st.info("No breakdown data available yet.")

    date_range = dash.get("date_range")
    if date_range:
        st.caption(f"📅 Data spans **{date_range.get('start', '?')[:10]}** to **{date_range.get('end', '?')[:10]}**")