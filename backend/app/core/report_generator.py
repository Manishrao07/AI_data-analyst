"""
Exports a session's conversation history (questions + answers) as a PDF report.
Bonus feature: lets a user share analysis results without giving them app access.
"""
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.core.data_manager import Session


def build_pdf_report(session: Session) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = [Paragraph("AI Data Analyst -- Session Report", styles["Title"]), Spacer(1, 12)]

    story.append(Paragraph("<b>Datasets analyzed:</b>", styles["Heading2"]))
    for info in session.datasets.values():
        story.append(Paragraph(f"{info.filename} -- {info.n_rows} rows, {info.n_cols} columns", styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>Conversation</b>", styles["Heading2"]))
    for turn in session.history:
        role = "Q" if turn["role"] == "user" else "A"
        style = styles["Heading4"] if role == "Q" else styles["Normal"]
        prefix = "Question: " if role == "Q" else "Answer: "
        story.append(Paragraph(prefix + turn["content"].replace("\n", "<br/>"), style))
        story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()
