from fastapi import FastAPI
from app.config.settings import get_settings
from app.api.router import router

settings = get_settings()

app = FastAPI(
    title="Cost-Efficient RAG Application",
    description="RAG Pipeline built with FastAPI, Qdrant Local, and Ollama Gemma 2B",
    version="0.1.0"
)

app.include_router(router)
