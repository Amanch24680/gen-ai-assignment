import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_rag_service
from app.generation.exceptions import (
    EmptyGenerationError,
    GenerationError,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.rag.service import RAGService
from app.retrieval.exceptions import (
    EmptyQueryError,
    InvalidRetrievalConfigError,
    RetrievalError,
)
from app.schemas.query import RAGQueryRequest, RAGQueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["RAG Application"])


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "cost-efficient-rag"}


@router.post("/query", response_model=RAGQueryResponse)
def query_rag(
    request: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGQueryResponse:
    """
    HTTP POST endpoint for processing grounded RAG queries.
    Delegates retrieval and generation to RAGService pipeline.
    """
    try:
        response = rag_service.query(
            query=request.query,
            top_k=request.top_k,
            relevance_threshold=request.relevance_threshold,
            metadata_filter=request.metadata_filter,
        )
        return response
    except (EmptyQueryError, InvalidRetrievalConfigError, ValueError) as exc:
        logger.warning(f"Bad request in RAG query endpoint: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except (OllamaConnectionError, OllamaTimeoutError) as exc:
        logger.error(f"LLM service unavailable: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM Generation Service unavailable: {exc}",
        )
    except (OllamaResponseError, EmptyGenerationError, RetrievalError, GenerationError) as exc:
        logger.error(f"RAG application execution error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG processing error: {exc}",
        )
    except Exception as exc:
        logger.exception("Unhandled unexpected error in RAG query endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal RAG application error.",
        )
