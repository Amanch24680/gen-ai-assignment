from fastapi import APIRouter
from app.schemas.query import RAGQueryRequest, RAGQueryResponse

router = APIRouter(prefix="/api/v1", tags=["RAG Application"])


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "cost-efficient-rag"}


@router.post("/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest):
    """Placeholder endpoint for RAG query processing."""
    raise NotImplementedError("RAG Query endpoint logic will be implemented in subsequent phases.")
