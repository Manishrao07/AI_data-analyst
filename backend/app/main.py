"""
Entrypoint for the AI Data Analyst backend.
Run with: uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="Upload CSVs and interact with your data using natural language.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    logger.info(f"{settings.APP_NAME} backend starting up...")
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set -- LLM calls will fail until you configure .env")


@app.get("/")
async def root():
    return {"message": f"{settings.APP_NAME} API is running. See /docs for the API reference."}
