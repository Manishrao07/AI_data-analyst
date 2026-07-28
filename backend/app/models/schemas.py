from pydantic import BaseModel


class UploadResponse(BaseModel):
    session_id: str
    datasets: list[dict]
    message: str


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class SimilarPastQuestion(BaseModel):
    question: str
    answer: str
    similarity: float


class QuestionResponse(BaseModel):
    answer: str
    intent: str
    trace: list[dict]
    sql_query: str | None = None
    chart: dict | None = None
    anomalies: dict | None = None
    similar_past_question: SimilarPastQuestion | None = None


class ExportRequest(BaseModel):
    session_id: str