"""
Semantic search over the session's own conversation history (bonus feature:
"Semantic search"). Lets the agent -- or a UI search box -- find prior
questions/answers that are meaningfully similar to a new query, not just
exact keyword matches.

Kept lightweight: TF-IDF + cosine similarity via scikit-learn (already a
project dependency for anomaly detection), instead of a heavy embedding
model -- overkill for one session's chat history.
"""
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HistoryMatch:
    turn_index: int
    role: str
    content: str
    score: float


def search_history(history: list[dict], query: str, top_k: int = 3, min_score: float = 0.1) -> list[HistoryMatch]:
    if not history:
        return []

    texts = [turn.get("content", "") for turn in history]
    if not any(texts):
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts + [query])
    except ValueError:
        return []

    query_vec = matrix[-1]
    history_vecs = matrix[:-1]
    scores = cosine_similarity(query_vec, history_vecs).flatten()

    ranked = sorted(
        (
            HistoryMatch(turn_index=i, role=history[i].get("role", ""), content=texts[i], score=float(scores[i]))
            for i in range(len(texts))
        ),
        key=lambda m: m.score,
        reverse=True,
    )
    return [m for m in ranked if m.score >= min_score][:top_k]
