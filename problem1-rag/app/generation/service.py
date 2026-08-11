import logging
import time
from typing import List, Optional
import httpx

from app.config.settings import get_settings
from app.generation.base import BaseGenerator
from app.generation.exceptions import (
    EmptyGenerationError,
    GenerationError,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.schemas.document import DocumentChunk
from app.schemas.query import Citation, RAGQueryResponse

logger = logging.getLogger(__name__)

NO_CONTEXT_RESPONSE_TEXT = (
    "No relevant information was found in the provided document context to answer this query."
)


def format_context_prompt(chunks: List[DocumentChunk]) -> str:
    """
    Format a list of DocumentChunk objects into a deterministic, readable context block.
    Preserves the order of chunks as received.
    """
    context_blocks = []
    for idx, chunk in enumerate(chunks, 1):
        filename = chunk.metadata.get("filename", "Unknown")
        page_num = chunk.metadata.get("page_number")
        page_str = str(page_num) if page_num is not None else "N/A"

        block = (
            f"[Source {idx}]\n"
            f"Filename: {filename}\n"
            f"Page: {page_str}\n"
            f"Content:\n{chunk.text.strip()}\n"
        )
        context_blocks.append(block)

    return "\n".join(context_blocks)


def build_grounded_prompt(query: str, context_text: str) -> str:
    """
    Construct a strict grounding prompt instructing the LLM to answer
    using ONLY the provided context.
    """
    prompt = (
        "You are a helpful assistant. Answer the query based ONLY on the context below.\n"
        "Do not use external knowledge or invent facts. If the context does not contain the answer, "
        "state that the information is not available in the provided context.\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        f"Query: {query}\n\n"
        "Answer:"
    )
    return prompt


class OllamaGenerator(BaseGenerator):
    """
    Local LLM generation service utilizing Ollama HTTP API (e.g. gemma:2b).
    Generates grounded answers based strictly on retrieved document context.
    """

    def __init__(
        self,
        ollama_base_url: Optional[str] = None,
        generator_model: Optional[str] = None,
        timeout: float = 30.0,
        http_client: Optional[httpx.Client] = None,
    ):
        settings = get_settings()
        self.ollama_base_url = (
            ollama_base_url.rstrip("/") if ollama_base_url is not None else settings.ollama_base_url.rstrip("/")
        )
        self.generator_model = (
            generator_model if generator_model is not None else settings.generator_model
        )
        self.timeout = timeout
        self.client = http_client

    def _get_client(self) -> httpx.Client:
        if self.client is not None:
            return self.client
        return httpx.Client(timeout=self.timeout)

    def generate(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
    ) -> RAGQueryResponse:
        """
        Generate a grounded answer based on query and context_chunks.
        
        If context_chunks is empty:
        - Does NOT invoke Ollama API.
        - Immediately returns no-context response.
        """
        start_time = time.perf_counter()

        if not query or not query.strip():
            raise ValueError("User query string cannot be empty or whitespace-only.")

        # 1. Empty context handling - DO NOT call Ollama
        if not context_chunks:
            logger.info(f"No context chunks provided for query '{query[:40]}...'. Skipping Ollama call.")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RAGQueryResponse(
                query=query,
                answer=NO_CONTEXT_RESPONSE_TEXT,
                citations=[],
                retrieved_chunk_count=0,
                has_relevant_context=False,
                latency_ms=round(elapsed_ms, 2),
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )

        # 2. Build citations
        citations: List[Citation] = []
        for chunk in context_chunks:
            snippet = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
            score_val = chunk.score if chunk.score is not None else float(chunk.metadata.get("score", 0.0))
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    snippet=snippet,
                    score=score_val,
                )
            )

        # 3. Build deterministic prompt
        context_text = format_context_prompt(context_chunks)
        prompt = build_grounded_prompt(query, context_text)

        payload = {
            "model": self.generator_model,
            "prompt": prompt,
            "stream": False,
        }

        endpoint = f"{self.ollama_base_url}/api/generate"
        logger.info(f"Sending generation request to Ollama ({self.generator_model}) at {endpoint}")

        # 4. Execute HTTP POST request to Ollama
        client = self._get_client()
        should_close = self.client is None

        try:
            response = client.post(endpoint, json=payload)
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama service at {self.ollama_base_url}. Error: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama generation request timed out after {self.timeout} seconds."
            ) from exc
        except Exception as exc:
            raise GenerationError(f"Unexpected HTTP client error during Ollama generation: {exc}") from exc
        finally:
            if should_close:
                client.close()

        # 5. Handle HTTP status code & payload parsing
        if response.status_code != 200:
            raise OllamaResponseError(
                f"Ollama API returned HTTP error status {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise OllamaResponseError(f"Failed to parse JSON response from Ollama API: {exc}") from exc

        if not isinstance(data, dict) or "response" not in data:
            raise OllamaResponseError("Malformed response payload from Ollama API: missing 'response' field.")

        answer_text = str(data.get("response", "")).strip()

        if not answer_text:
            raise EmptyGenerationError("Ollama API returned an empty or whitespace-only answer string.")

        # Extract token usage counts if returned by Ollama API
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RAGQueryResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            retrieved_chunk_count=len(context_chunks),
            has_relevant_context=True,
            latency_ms=round(elapsed_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
