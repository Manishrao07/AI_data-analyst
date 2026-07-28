"""
Minimal evaluation framework for the agent's router (bonus deliverable).

Measures intent-classification accuracy against a labeled set of
representative questions -- the kind of thing you'd expand into a
regression suite before shipping changes to the router prompt.

Run with: GROQ_API_KEY=xxx python eval/run_eval.py
(requires a real Groq key since it calls the LLM)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.agent.tools import classify_intent

LABELED_QUESTIONS = [
    ("Which region generated the highest revenue?", "sql"),
    ("Show monthly sales trends", "chart"),
    ("Plot revenue by category as a pie chart", "chart"),
    ("What are the top 5 customers?", "sql"),
    ("Detect anomalies in the dataset", "anomaly"),
    ("Are there any unusual orders in this data?", "anomaly"),
    ("Which products are underperforming?", "insight"),
    ("Give me a summary of overall business performance", "insight"),
    ("Generate SQL for total revenue by category", "sql"),
    ("hi there", "general"),
]


def run_eval():
    correct = 0
    rows = []
    for question, expected in LABELED_QUESTIONS:
        predicted, reasoning = classify_intent(question, schema_summary="Table 'sales_data': region, product, category, revenue, quantity")
        is_correct = predicted == expected
        correct += is_correct
        rows.append((question, expected, predicted, is_correct))

    accuracy = correct / len(LABELED_QUESTIONS)
    print(f"{'Question':<55} {'Expected':<10} {'Predicted':<10} {'OK'}")
    print("-" * 90)
    for q, exp, pred, ok in rows:
        print(f"{q:<55} {exp:<10} {pred:<10} {'✅' if ok else '❌'}")
    print("-" * 90)
    print(f"Router accuracy: {accuracy:.0%} ({correct}/{len(LABELED_QUESTIONS)})")


if __name__ == "__main__":
    run_eval()
