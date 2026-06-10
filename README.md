# BravoBOT

BravoBOT is a production-style, multi-user Retrieval-Augmented Generation (RAG) system for question answering over PDF and TXT documents. It demonstrates the core work expected in an AI engineering project: document ingestion, structure-aware chunking, hybrid retrieval, query transformation, reranking, grounded generation, evaluation, streaming, persistence, and containerization.

This is a portfolio prototype, not a production-ready hosted service. See [Production Readiness](#production-readiness) for the remaining gaps.

## Highlights

- Structure-aware chunking using numbered headings and subheadings
- Dense retrieval with Chroma and HuggingFace embeddings
- Sparse retrieval with BM25
- HyDE query transformation for dense retrieval
- Reciprocal Rank Fusion (RRF) across dense and sparse results
- Cross-encoder reranking
- Gemini generation with Ollama fallback
- Streaming NDJSON responses and source citations
- RAGAS answer-level evaluation
- NDCG@5 retrieval-level evaluation
- Background ingestion jobs persisted in MySQL
- Development-level user isolation across documents, jobs, files, BM25, and Chroma
- Dockerized FastAPI backend

## Evaluation

Evaluation was performed against the paper *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers*.

### Retrieval

| Metric | Score | Evaluated queries |
| --- | ---: | ---: |
| NDCG@5 | 0.7507 | 5 |

NDCG uses manually assigned graded relevance labels for retrieved chunk IDs. Ten additional questions currently have no relevance labels and are excluded from the average.

### Answer Quality

| RAGAS metric | Score |
| --- | ---: |
| Faithfulness | 1.0000 |
| Answer relevancy | 0.7774 |
| Context precision | 1.0000 |
| Context recall | 1.0000 |

These RAGAS scores were produced from one evaluated question using a local Ollama judge. They validate the evaluation pipeline but are not statistically representative. A larger, independently reviewed test set is required before making production-quality claims.

Evaluation artifacts are stored in:

```text
data/eval/questions.json
data/eval/ndcg_results.json
data/eval/ragas_inputs.json
data/eval/ragas_results.json
```

## Architecture

```text
Browser UI
   |
   | X-User-ID + PDF/TXT upload
   v
FastAPI API
   |
   +--> MySQL upload job
   |
   +--> Per-user raw file storage
           |
           v
     PDF/TXT extraction
           |
           v
     Structural chunking
     heading -> subsection -> bounded chunks
           |
           +--> user_chunks.json
           |
           +--> HuggingFace embeddings -> Chroma

User question
   |
   +--> Original query ----------------------> BM25
   |
   +--> HyDE hypothetical document ----------> Dense search
                                                  |
                         BM25 + Dense results ----+
                                                  v
                                     Reciprocal Rank Fusion
                                                  |
                                                  v
                                      CrossEncoder reranker
                                                  |
                                                  v
                                      Grounded prompt + context
                                                  |
                                      Gemini / Ollama fallback
                                                  |
                                                  v
                                    Streamed answer + citations
```

## Retrieval Pipeline

1. The original user query is tokenized and searched with BM25.
2. HyDE generates a hypothetical answer-like document from the query.
3. The HyDE document is used for dense similarity search in Chroma.
4. Both retrieval paths are filtered by `user_id`.
5. Reciprocal Rank Fusion combines the two rankings.
6. Duplicate candidates are removed.
7. A cross-encoder reranks the fused candidates against the original query.
8. The highest-ranked chunks are formatted as grounded LLM context.
9. The answer is streamed with page-level source citations.

## Ingestion Pipeline

1. The API validates the user ID, file extension, and non-empty content.
2. A MySQL-backed background job is created.
3. Files are stored under a hashed per-user directory.
4. Duplicate documents are detected using a file hash within that user.
5. PDF or text content is loaded and divided by numbered heading hierarchy.
6. Oversized sections are split on paragraph boundaries.
7. Heading paths are included as chunk context and metadata.
8. Chunks are persisted to JSON and embedded into Chroma.
9. Replacing sources removes only the current user's indexed content.

## User Isolation

Every upload, job, chunk, and retrieval request is scoped by `user_id`.

- Raw uploads use separate hashed user directories.
- Document hashes are checked per user.
- BM25 builds and caches a separate index per user.
- Chroma search applies a `user_id` metadata filter.
- Upload jobs can only be read by their owning user.
- Replace operations preserve other users' documents.

The current browser sends identity through `X-User-ID`. This is appropriate for demonstrating tenant-aware RAG behavior, but it is not authentication because clients can spoof the header. A production system must derive the user ID from a validated session or JWT.

## Tech Stack

| Area | Technology |
| --- | --- |
| API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| PDF parsing | PyMuPDF |
| Dense retrieval | Chroma, HuggingFace embeddings |
| Sparse retrieval | rank-bm25 |
| Query transformation | HyDE |
| Fusion | Reciprocal Rank Fusion |
| Reranking | SentenceTransformers CrossEncoder |
| Generation | Gemini, Ollama fallback |
| Evaluation | RAGAS, NDCG@5 |
| Job persistence | MySQL, SQLAlchemy, PyMySQL |
| Packaging | Docker |

## Project Structure

```text
app/
  config.py              Runtime paths and model configuration
  database.py            SQLAlchemy database setup
  document_ingestion.py  Upload, chunk persistence, and Chroma writes
  document_registry.py   Per-user document registry and deduplication
  hybrid_retriever.py    Dense/BM25 fusion
  job_store.py           MySQL upload-job persistence
  jobs.py                Background ingestion lifecycle
  main.py                FastAPI endpoints
  rag_pipeline.py        HyDE, prompting, generation, and orchestration
  rerank.py              Cross-encoder reranking
  retriever.py           Chroma and per-user BM25 retrieval
  schemas.py             API and ingestion models

scripts/
  ingest.py                  Structural chunking and document conversion
  evaluate_ndcg.py           Retrieval evaluation
  generate_ragas_inputs.py   RAGAS dataset generation
  evaluate_ragas.py          Answer-level evaluation

ui/
  index.html
  styles.css
  app.js
```

## Local Setup

### 1. Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Example:

```env
GEMINI_API_KEY=your_gemini_api_key
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash

OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=mistral:latest

DATABASE_URL=mysql+pymysql://root:password@localhost:3306/rag_app
```

Do not commit `.env`.

### 3. Create the database

```sql
CREATE DATABASE IF NOT EXISTS rag_app;
USE rag_app;

CREATE TABLE IF NOT EXISTS upload_jobs (
    job_id VARCHAR(80) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    status VARCHAR(30) NOT NULL,
    result_json JSON NULL,
    error TEXT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);
```

### 4. Start the backend

```bash
python -m uvicorn app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the frontend

```bash
python -m http.server 5500 --directory ui
```

Open:

```text
http://127.0.0.1:5500
```

### 6. Optional Ollama fallback

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

## API

Tenant-scoped endpoints require:

```http
X-User-ID: user-a
```

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `GET` | `/sources/status` | Current user's indexed sources |
| `POST` | `/upload` | Upload and index PDF/TXT files |
| `GET` | `/jobs/{job_id}` | Poll an owned ingestion job |
| `POST` | `/chat` | Non-streaming chat or RAG answer |
| `POST` | `/chat/stream` | NDJSON streaming answer |

Example RAG request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-a" \
  -d '{
    "query": "What are the paper main recommendations?",
    "candidate_k": 8,
    "final_k": 5,
    "mode": "rag"
  }'
```

## Docker

The current Dockerfile packages the backend. MySQL and Ollama/Gemini remain external services.

Build:

```bash
docker build -t bravobot-backend .
```

When MySQL and Ollama run on macOS, container URLs must use `host.docker.internal` rather than `localhost`.

Example `.env.docker`:

```env
GEMINI_API_KEY=your_gemini_api_key
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
OLLAMA_URL=http://host.docker.internal:11434/api/generate
OLLAMA_MODEL=mistral:latest
DATABASE_URL=mysql+pymysql://root:password@host.docker.internal:3306/rag_app
```

Run:

```bash
docker run --rm \
  --name bravobot-backend \
  -p 8001:8000 \
  --env-file .env.docker \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/vectorstore:/app/vectorstore" \
  bravobot-backend
```

The API is then available at:

```text
http://127.0.0.1:8001/docs
```

The frontend can still be served locally and configured to call port `8001`.

## Performance

Observed warm-cache timings from local development:

```text
hybrid_search: ~0.23s
reranker:      ~0.07s
Gemini:        ~2.0s
total:         ~2.4s
```

Latency varies by model provider, network conditions, HyDE generation, hardware, and cache state. Local Ollama generation and RAGAS judging can take substantially longer.

Implemented optimizations include:

- BM25 cache per user
- Chroma client cache per active index
- Singleton cross-encoder model
- Streaming generation
- Incremental Chroma updates when appending sources

## Production Readiness

BravoBOT demonstrates production-style RAG design, but the current implementation is not production-ready.

Implemented:

- Tenant-aware ingestion and retrieval
- Persistent job state
- Background indexing
- Hybrid retrieval and reranking
- Evaluation pipelines
- Source-grounded streaming responses
- Containerized backend

Required before production:

- Real authentication and authorization instead of trusting `X-User-ID`
- Object storage for uploaded files
- Database-backed document/chunk metadata instead of shared JSON
- A production vector database or carefully managed persistent Chroma deployment
- Redis/Celery/RQ or another durable worker queue
- Concurrency and transactional safeguards for index updates
- File-size limits, rate limiting, malware scanning, and OCR
- Structured logging, tracing, metrics, alerting, and error reporting
- Automated unit, integration, isolation, load, and security tests
- CI/CD, secret management, backups, and deployment configuration
- A larger human-reviewed evaluation dataset

## Screenshots

### Indexed Source Library

![BravoBOT indexed source library](docs/images/bravobot-indexed-sources.png)

### Main Chat Workspace

![BravoBOT main chat workspace](docs/images/bravobot-main-workspace.png)

### Citation Preview Cards

![BravoBOT answer with citation previews](docs/images/bravobot-citation-previews.png)

## Resume Bullet

> Built a containerized, multi-user RAG system with structural chunking, HyDE-enhanced BM25/dense retrieval, Reciprocal Rank Fusion, cross-encoder reranking, streaming grounded answers, and RAGAS/NDCG evaluation.

## License

Add a license before publishing the repository. MIT is a common choice for a portfolio project.
