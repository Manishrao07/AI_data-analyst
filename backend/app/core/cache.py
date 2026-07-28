"""
Lightweight TTL cache to avoid re-calling the LLM / re-running heavy
aggregations for repeated questions within the same session.
Swap for Redis if this needs to scale across multiple app instances.
"""
import hashlib
import time

from app.core.config import settings

_store: dict[str, tuple[float, object]] = {}
TTL_SECONDS = 300


def _key(session_id: str, payload: str) -> str:
    return hashlib.sha256(f"{session_id}:{payload}".encode()).hexdigest()


def get(session_id: str, payload: str):
    if not settings.ENABLE_CACHE:
        return None
    k = _key(session_id, payload)
    hit = _store.get(k)
    if hit and time.time() - hit[0] < TTL_SECONDS:
        return hit[1]
    return None


def set(session_id: str, payload: str, value: object):
    if not settings.ENABLE_CACHE:
        return
    _store[_key(session_id, payload)] = (time.time(), value)
