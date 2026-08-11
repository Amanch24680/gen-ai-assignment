# Problem 1 — Cost-Efficient RAG Application

A lightweight, production-grade, cost-efficient Retrieval-Augmented Generation (RAG) system built with FastAPI, Qdrant Vector Database, SentenceTransformers embeddings, and local Ollama (`gemma:2b`) text generation.

---

## 1. Executive Summary & Key Architecture

This repository provides an end-to-end, zero-cloud-lock-in RAG solution designed to run locally on commodity hardware without external API billing dependencies.

```
                  +-----------------------------------+
                  |         Document Corpus           |
                  |  (PDF, HTML, Markdown Files)      |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |         DocumentLoader            |
                  |      (PyMuPDF, BeautifulSoup)     |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |           TextChunker             |
                  |  (500 chars / 50 overlap / MD5)   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | SentenceTransformerEmbedding     |
                  |      (all-MiniLM-L6-v2)           |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |        Qdrant Vector DB           |
                  |   (Local Disk / Cosine Index)     |
                  +-----------------------------------+
                                    |
                                    v
+----------------+  Query   +---------------+  Context   +-------------------+
|  User / Client | -------> | VectorRetriever| ---------> |  Ollama Generator |
| (FastAPI / UI) | <------- | (Top-K / Cos) |            |    (gemma:2b)     |
+----------------+ Response +---------------+            +-------------------+
```

### Key Highlights:
- **Zero API Incurred Cost**: Runs 100% locally using open-source models (`all-MiniLM-L6-v2` for 384d dense embeddings, `gemma:2b` via Ollama for generation).
- **Deterministic Identifiers**: Generates content-hash `doc_id` and `chunk_id` for idempotent re-ingestion and deduplication.
- **Strict Observability & Safety**: Evaluates retrieval accuracy, answer fidelity, citation validity, and unanswerable query safety without external LLM judges.

---

## 2. Setup & Installation

### Prerequisites
- Python 3.11+
- Local [Ollama](https://ollama.com/) service running with the `gemma:2b` model pulled:
  ```bash
  ollama pull gemma:2b
  ollama serve
  ```

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   cd problem1-rag
   ```
2. Activate your virtual environment and install requirements:
   ```bash
   ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

---

## 3. Configuration & Environment Variables

All parameters are configured via `app/core/config.py` using Pydantic Settings and can be overridden via environment variables:

| Variable Name | Default Value | Description |
|---|---|---|
| `CHUNK_SIZE` | `500` | Target character size for text chunking |
| `CHUNK_OVERLAP` | `50` | Overlapping character count between adjacent chunks |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace SentenceTransformer model name |
| `QDRANT_DB_PATH` | `data/qdrant` | Disk path for persistent Qdrant storage |
| `QDRANT_COLLECTION_NAME` | `rag_chunks` | Qdrant collection name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for local Ollama HTTP API |
| `OLLAMA_MODEL_NAME` | `gemma:2b` | LLM model name served by Ollama |
| `DEFAULT_TOP_K` | `5` | Default top-K retrieved chunks per query |
| `DEFAULT_RELEVANCE_THRESHOLD` | `0.0` | Minimum cosine similarity relevance filter |

---

## 4. Ingestion Instructions

The pipeline supports ingestion of **PDF**, **HTML**, and **Markdown** documents.

### Single File Ingestion
```bash
python -m app.ingestion.cli --file path/to/document.pdf
```

### Directory Ingestion
```bash
python -m app.ingestion.cli --dir path/to/documents_folder/
```

*Note: Ingestion is fully idempotent. Re-ingesting identical files produces identical `chunk_id` keys in Qdrant without creating duplicates.*

---

## 5. API Endpoint Documentation

### Running the Server
```bash
python -m app.main
```
FastAPI server runs by default at `http://localhost:8000`.

### Health Check Endpoint
- **GET** `/health`
- **Response**: `200 OK` `{"status": "ok"}`

### RAG Query Endpoint
- **POST** `/api/v1/query`
- **Request Body**:
  ```json
  {
    "query": "What candidate low-cost vector stores are suggested for Problem 1?",
    "top_k": 5,
    "relevance_threshold": 0.5
  }
  ```
- **Response Body**:
  ```json
  {
    "query": "What candidate low-cost vector stores are suggested for Problem 1?",
    "answer": "The suggested low-cost vector stores include Qdrant, ChromaDB, and PGvector.",
    "citations": [
      {
        "doc_id": "10e0b8a88698680cfc3084cffc67ff6e",
        "chunk_id": "60e906b285514b553b944db54bd9790e",
        "source": "Gen AI_assignment.pdf",
        "page_number": 1,
        "score": 0.8173,
        "snippet": "Problem 1 requires building a RAG application with low-cost vector databases such as Qdrant or Chroma..."
      }
    ],
    "retrieved_chunk_count": 5,
    "has_relevant_context": true,
    "latency_ms": 24890.10,
    "metrics": {
      "retrieval_latency_ms": 34.24,
      "generation_latency_ms": 24855.86
    }
  }
  ```

---

## 6. Evaluation Methodology

The evaluation harness measures retrieval quality and answer generation deterministically against a fixed **20-case ground-truth dataset** (`evaluation/dataset.json`).

### Evaluation Metrics Defined:
- **Retrieval Metrics**:
  - **Recall@K**: Proportion of ground-truth relevant chunks present in top-K retrieved results.
  - **Precision@K**: Proportion of top-K retrieved chunks that belong to ground-truth.
  - **Hit Rate@K**: Binary indicator (1.0 if $\ge 1$ relevant chunk is in top-K, else 0.0).
  - **nDCG@K**: Normalized Discounted Cumulative Gain at rank K.
  - **MRR (Mean Reciprocal Rank)**: Mean reciprocal rank ($1 / \text{rank}$) of the first relevant chunk.
  - **Retrieval Latency Percentiles**: $p50$ (median) and $p95$ retrieval latency measured across queries.
- **Answer Metrics**:
  - **Token F1**: Token-level precision/recall overlap between LLM generated answer and ground-truth answer.
  - **Context Support**: Lexical overlap fraction of generated answer tokens present in retrieved snippets.
  - **Citation Coverage**: Ratio of valid citations pointing to retrieved chunks.
  - **Unanswerable Safe Handling Rate**: Proportion of out-of-corpus queries where the system explicitly refrains from hallucination.

### Running Evaluation CLI
```bash
# Run retrieval evaluation across all 20 cases
python -m evaluation.cli --mode retrieval

# Run end-to-end RAG answer evaluation
python -m evaluation.cli --mode rag

# Save detailed JSON output
python -m evaluation.cli --mode rag --output evaluation/results.json
```

---

## 7. Actual Measured Project Results

> **DATA CATEGORY 1: ACTUAL EMPIRICALLY MEASURED RESULTS**
>
> All metrics in this section were recorded from live execution of the pipeline against `evaluation/dataset.json` and saved in `evaluation/results.json`.

### Retrieval Layer Performance (20 Dataset Cases)
| Metric | Value |
|---|---|
| **Recall@1** | `0.3889` |
| **Recall@3** | `0.7778` |
| **Recall@5** | `0.8889` |
| **Precision@1** | `0.4444` |
| **Precision@3** | `0.3148` |
| **Precision@5** | `0.2333` |
| **Hit Rate@1** | `0.4444` |
| **Hit Rate@3** | `0.8889` |
| **Hit Rate@5** | `0.9444` |
| **nDCG@1** | `0.4444` |
| **nDCG@3** | `0.6627` |
| **nDCG@5** | `0.7199` |
| **MRR** | `0.6806` |
| **p50 Retrieval Latency** | `34.24 ms` |
| **p95 Retrieval Latency** | `71.53 ms` |
| **Unanswerable Empty Retrieval Rate** | `0.0000` |

### End-to-End Answer Generation Performance
| Metric | Value |
|---|---|
| **Mean Answer Token F1** | `0.8143` |
| **Mean Lexical Context Support** | `0.9428` |
| **Mean Citation Coverage** | `1.0000` |
| **Unanswerable Safe Handling Rate** | `1.0000` (100% safe refusal) |
| **Average End-to-End Latency** | `25412.30 ms` (~25.4s) |
| **Median End-to-End Latency** | `24890.10 ms` (~24.9s) |

---

## 8. Cost Analysis & Infrastructure Comparison

> **DATA CATEGORIES 2 & 3: ASSUMPTION-BASED COST ESTIMATES & CURRENT VENDOR LIST PRICING**
>
> The cost comparisons below model hypothetical scale projections using official vendor list pricing as of 2026.

### Explicit Modeling Assumptions (Category 2):
1. **Vector Dimensionality**: 384d (`all-MiniLM-L6-v2`, float32 = 1.5 KB raw vector size + 0.5 KB payload = ~2.0 KB storage per vector).
2. **Monthly Query Volume**: 100,000 queries per month.
3. **Model Exclusions**: Embedding inference (SentenceTransformers local CPU) and LLM token generation (Ollama local CPU) incur **$0.00** API cost and are excluded from vector DB storage comparison.

### Published Vendor List Pricing Sources (Category 3):
- **Local / Self-Hosted**: AWS EC2 On-Demand Linux pricing for `t3.small` ($0.0208/hr ≈ $15/mo), `t3.medium` ($0.0416/hr ≈ $30/mo), and `c6i.xlarge` ($0.17/hr ≈ $120–$140/mo).
- **Pinecone Serverless**: $0.33 per GB/month storage + $0.000002 per Read Unit ($2.00 per million RUs).
- **Qdrant Cloud**: Managed cluster pricing starting at $9/mo (1GB RAM), $25/mo (4GB RAM), $210/mo (32GB RAM).
- **Weaviate Cloud (WCS)**: Free Sandbox, Serverless/Standard clusters starting at ~$15–$35/mo.

### Monthly Vector Database Cost Comparison Matrix (USD)

| Scale | Local / Self-Hosted Qdrant | Pinecone (Serverless) | Qdrant Cloud | Weaviate Cloud (WCS) |
|---|---|---|---|---|
| **100K Vectors** (~200 MB) | **$0.00** (Local Dev) / **$15.00** (AWS t3.small EC2) | **~$1.50** ($0.33/GB storage + read units) | **~$9.00** (1GB Managed Cluster) | **~$0.00** (SandBox) / **$15.00** |
| **1M Vectors** (~2.0 GB) | **~$30.00** (AWS t3.medium / t3.large EC2) | **~$18.00** ($0.33/GB + Read/Write Units) | **~$25.00** (4GB RAM Cluster) | **~$35.00** (Standard Cluster) |
| **10M Vectors** (~20.0 GB) | **~$140.00** (AWS c6i.xlarge EC2, 16GB RAM) | **~$180.00** (Storage + High QPS Read Units) | **~$210.00** (32GB RAM Cluster) | **~$240.00** (Dedicated Instance) |

---

## 9. Architectural Trade-offs & Decisions

1. **Local Vector Database (Qdrant) vs. Managed Cloud (Pinecone / Qdrant Cloud)**:
   - *Trade-off*: Local Qdrant provides zero recurring API billing, zero network egress latency, and complete data privacy. However, managed cloud provides automated multi-region replication, horizontal sharding, and zero infrastructure maintenance.
2. **Small Dense Embeddings (`all-MiniLM-L6-v2`) vs. Commercial API (`text-embedding-3-small`)**:
   - *Trade-off*: 384d local embeddings consume minimal RAM (~2KB/vector vs ~6KB/vector for 1536d) and achieve **94.4% Hit Rate@5** at sub-35ms retrieval speeds with 0 external API cost.
3. **Local LLM (`gemma:2b` on CPU) vs. Cloud LLM (`gpt-4o-mini`)**:
   - *Trade-off*: Local generation guarantees strict privacy and zero per-token cost, but incurs high local CPU latency (~24.9s/query). Cloud API generation provides sub-second responses but introduces token billing and data privacy trade-offs.

---

## 10. When to Switch to Managed Infrastructure

Switching from self-hosted local Qdrant to managed cloud infrastructure (Qdrant Cloud / Pinecone) is recommended when reaching the following boundary conditions:

1. **Dataset Scale > 10M Vectors (>20 GB Index)**: RAM footprint exceeds single-node commodity server capacity.
2. **High Availability & SLA Requirements**: Multi-AZ failover, zero-downtime rolling upgrades, and guaranteed 99.99% uptime SLAs.
3. **Burst QPS Capacity**: Concurrent query throughput exceeds single-node CPU/memory bandwidth limits, requiring automated horizontal read-replica scaling.

---

## 11. Retriever vs. Generator Analysis: Identifying the Weaker Link

Based on empirical evaluation evidence:
- **Retrieval Layer Performance**: **STRONG**
  - **Hit Rate@5**: `94.44%`
  - **Recall@5**: `88.89%`
  - **p50 Latency**: `34.24 ms`
  - The retriever consistently fetches the correct grounding context in sub-50ms speeds.
- **Generation Layer Performance**: **WEAKER LINK**
  - **Execution Bottleneck**: CPU text generation averages **25.4 seconds per query** (~99.8% of total end-to-end latency).
  - **Model Size Constraints**: At 2 billion parameters, `gemma:2b` exhibits limited complex reasoning capacity for dense multi-document synthesis compared to larger models.

*Conclusion*: The **Generator** is the primary bottleneck in system latency and complex reasoning, while the **Retriever** provides high accuracy and sub-50ms performance.

---

## 12. Automated Test Suite

Run the full pytest suite:
```bash
cd problem1-rag
..\.venv\Scripts\python.exe -m pytest tests -v
```

### Verification Result:
```
================== 125 passed, 1 warning in 46.63s ==================
```
