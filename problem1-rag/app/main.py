from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config.settings import get_settings
from app.api.router import router

settings = get_settings()

app = FastAPI(
    title="Cost-Efficient RAG Application",
    description="RAG Pipeline built with FastAPI, Qdrant Local, and Ollama Gemma 2B",
    version="0.1.0"
)

# CORS middleware for local web frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Mount static frontend files if frontend directory exists
frontend_dir = Path(__file__).resolve().parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        """Serve frontend chat HTML interface."""
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"status": "ok", "message": "RAG API service running"}
