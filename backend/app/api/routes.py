import os
import time
import uuid
from dataclasses import asdict

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.agent.graph import run_agent
from app.core import cache
from app.core.config import settings
from app.core.data_manager import session_store
from app.core.dashboard import build_dashboard_summary
from app.core.history_search import search_history
from app.core.logger import get_logger, new_trace_id
from app.core.report_generator import build_pdf_report
from app.models.schemas import ExportRequest, QuestionRequest, QuestionResponse, UploadResponse

router = APIRouter()
logger = get_logger(__name__)

# Minimum similarity (0-1) before we bother showing the "you asked this before" banner.
# Below this, near-every question would loosely match something and it'd become noise.
SIMILARITY_THRESHOLD = 0.65


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(files: list[UploadFile] = File(...)):
    trace_id = new_trace_id()
    logger.info(f"[{trace_id}] Upload request: {[f.filename for f in files]}")

    session = session_store.create()
    dataset_infos = []

    for f in files:
        if not f.filename.lower().endswith(".csv"):
            raise HTTPException(400, f"'{f.filename}' is not a CSV file.")

        contents = await f.read()
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > settings.MAX_UPLOAD_MB:
            raise HTTPException(400, f"'{f.filename}' exceeds max size of {settings.MAX_UPLOAD_MB}MB.")

        tmp_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{f.filename}")
        with open(tmp_path, "wb") as out:
            out.write(contents)

        try:
            info = session.add_csv(tmp_path, f.filename)
        except ValueError as e:
            raise HTTPException(400, str(e))
        dataset_infos.append(asdict(info))

    return UploadResponse(
        session_id=session.session_id,
        datasets=dataset_infos,
        message=f"Uploaded {len(dataset_infos)} file(s). Session ready.",
    )


def _find_similar_past_question(session, question: str) -> dict | None:
    """
    Look at this session's history for a prior *user* turn that's semantically
    similar to the incoming question, and return it with the answer that
    followed it (the next "assistant" turn right after it in session.history).

    NOTE: search_history() returns a list[HistoryMatch] (a dataclass), not
    dicts -- attribute access only (.content / .score / .turn_index), never
    ["content"] / ["score"]. Indexing it like a dict was the bug that crashed
    the app once a later question became similar enough to an earlier one.
    """
    if not session.history:
        return None

    matches = search_history(session.history, question, top_k=1)
    if not matches:
        return None

    top = matches[0]
    if top.role != "user" or top.score < SIMILARITY_THRESHOLD:
        return None

    # The answer is the assistant turn immediately after the matched user turn.
    answer = ""
    next_idx = top.turn_index + 1
    if next_idx < len(session.history) and session.history[next_idx]["role"] == "assistant":
        answer = session.history[next_idx]["content"]

    return {
        "question": top.content,
        "answer": answer,
        "similarity": round(top.score, 3),
    }


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(payload: QuestionRequest):
    trace_id = new_trace_id()
    start = time.time()

    session = session_store.get(payload.session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Upload a CSV first.")

    cached = cache.get(payload.session_id, payload.question)
    if cached:
        logger.info(f"[{trace_id}] Cache hit for question")
        return cached

    # --- proactive similar-question detection (runs BEFORE the agent, against
    # only the questions asked so far -- so we don't match the question against itself) ---
    similar_past_question = _find_similar_past_question(session, payload.question)

    try:
        result = run_agent(session, payload.question)
    except RuntimeError as e:
        # e.g. missing GROQ_API_KEY
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.error(f"[{trace_id}] Agent failed: {e}")
        raise HTTPException(500, f"Agent failed to process the question: {e}")

    response = QuestionResponse(
        answer=result.get("final_answer", ""),
        intent=result.get("intent", "general"),
        trace=result.get("trace", []),
        sql_query=result.get("sql_query"),
        chart=result.get("chart_result"),
        anomalies=result.get("anomaly_result"),
        similar_past_question=similar_past_question,
    )
    cache.set(payload.session_id, payload.question, response)

    elapsed = (time.time() - start) * 1000
    logger.info(f"[{trace_id}] Answered in {elapsed:.0f}ms, intent={response.intent}")
    return response


@router.get("/session/{session_id}/datasets")
async def list_datasets(session_id: str):
    session = session_store.get_or_error(session_id)
    return {"datasets": [asdict(d) for d in session.datasets.values()]}


@router.post("/export/pdf")
async def export_pdf(payload: ExportRequest):
    session = session_store.get_or_error(payload.session_id)
    pdf_bytes = build_pdf_report(session)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=analysis_report.pdf"},
    )


@router.get("/dashboard/{session_id}")
async def get_dashboard(session_id: str, table_name: str | None = None):
    session = session_store.get_or_error(session_id)
    if table_name is None:
        table_name = next(iter(session.datasets.keys()))
    try:
        summary = build_dashboard_summary(session, table_name)
    except Exception as e:
        raise HTTPException(500, f"Failed to build dashboard: {e}")
    return summary


@router.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}