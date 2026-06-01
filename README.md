# BravoBOT

BravoBOT is a NotebookLM-style document assistant for private document question answering. Users can upload PDF or TXT sources, index them into a local retrieval system, ask questions over the uploaded documents, inspect citations, and use normal chat for casual assistant interactions.

The project combines a FastAPI backend, a custom HTML/CSS/JavaScript frontend, hybrid retrieval, cross-encoder reranking, Gemini streaming generation, and Ollama fallback.

## Features

- Upload PDF/TXT documents from the browser
- Background document indexing with job status polling
- MySQL-backed upload job tracking
- PyMuPDF-based PDF text extraction
- Chunking with document IDs and file hashes
- Duplicate document detection by file hash
- Chroma vector search with HuggingFace embeddings
- BM25 sparse retrieval
- Hybrid retrieval with reciprocal rank fusion
- Cross-encoder reranking
- Gemini API answer generation
- Ollama fallback when Gemini is unavailable or quota-limited
- Streaming answer UI
- Source citations with page numbers and text previews
- Normal chat fallback for greetings and assistant identity questions
- Caching for BM25, Chroma/vectorstore, and reranker model

## Screenshots

### Indexed Source Library

![BravoBOT indexed source library](docs/images/bravobot-indexed-sources.png)

### Main Chat Workspace

![BravoBOT main chat workspace](docs/images/bravobot-main-workspace.png)

### Citation Preview Cards

![BravoBOT answer with citation previews](docs/images/bravobot-citation-previews.png)

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Document parsing | PyMuPDF via LangChain loader |
| Dense retrieval | Chroma, HuggingFace embeddings |
| Sparse retrieval | rank-bm25 |
| Fusion | Reciprocal Rank Fusion |
| Reranking | SentenceTransformers CrossEncoder |
| LLM | Gemini API, Ollama fallback |
| Job persistence | MySQL, SQLAlchemy, PyMySQL |
| Streaming | FastAPI StreamingResponse, NDJSON |

## Architecture

```text
Browser UI
  |
  | upload PDF/TXT
  v
FastAPI /upload
  |
  | creates job_id
  v
MySQL upload_jobs table
  |
  | background indexing
  v
PyMuPDF -> chunks -> Chroma vectorstore + user_chunks.json
  |
  | user asks question
  v
FastAPI /chat/stream
  |
  | hybrid retrieval
  v
Dense search + BM25 search
  |
  | reciprocal rank fusion
  v
CrossEncoder reranker
  |
  | grounded prompt
  v
Gemini streaming response
  |
  | fallback on 429/503
  v
Ollama local model
```

## How The RAG Pipeline Works

1. A user uploads one or more PDF/TXT files.
2. FastAPI creates an upload job and returns a `job_id`.
3. The frontend polls `/jobs/{job_id}` until indexing completes.
4. The backend saves uploaded files locally.
5. PyMuPDF extracts document text.
6. Text is split into chunks.
7. Chunks are saved to `data/processed/user_chunks.json`.
8. New chunks are embedded and added to Chroma.
9. On chat, dense Chroma retrieval and BM25 retrieval run together.
10. Results are fused with Reciprocal Rank Fusion.
11. A cross-encoder reranks the fused candidates.
12. The final context is sent to Gemini.
13. The answer streams back to the UI with citations.

## Performance Notes

The app was optimized from slow local generation to a much faster streaming flow.

Observed local timings after cache warm-up:

```text
hybrid_search: ~0.23s
reranker:      ~0.07s
Gemini:        ~2.0s
total:         ~2.4s
```

Implemented optimizations:

- BM25 index cache keyed by `user_chunks.json` modified time
- Chroma/vectorstore cache keyed by active persist directory
- CrossEncoder reranker singleton cache
- Gemini streaming responses
- Ollama fallback for Gemini quota or availability errors
- Incremental Chroma append when `replace=false`

## Project Structure

```text
app/
  config.py              # Runtime paths and model/provider config
  database.py            # SQLAlchemy MySQL engine/session setup
  document_ingestion.py  # Upload ingestion, chunk persistence, Chroma writes
  document_registry.py   # Local document registry by file hash
  hybrid_retriever.py    # Dense + BM25 fusion
  job_store.py           # MySQL job CRUD
  jobs.py                # Upload job lifecycle functions
  main.py                # FastAPI app and API endpoints
  rag_pipeline.py        # Prompting, generation, streaming, RAG orchestration
  rerank.py              # CrossEncoder reranker
  retriever.py           # Chroma and BM25 retrieval
  schemas.py             # Pydantic API schemas and internal dataclasses
  utils.py               # Hash/id utilities

scripts/
  ingest.py              # Document loading/chunk creation helpers
  evaluate.py            # Evaluation script
  test_*.py              # Local development test scripts

ui/
  index.html             # Frontend layout
  styles.css             # Custom visual design
  app.js                 # Upload, polling, streaming chat
  Assets/                # UI image assets
```

## Setup

### 1. Clone The Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create A Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Environment File

Copy the example file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=mistral:latest
DATABASE_URL=mysql+pymysql://root:your_mysql_password@localhost:3306/rag_app
```

Do not commit `.env`.

### 5. Create MySQL Database

Open MySQL Workbench or a MySQL shell and run:

```sql
CREATE DATABASE IF NOT EXISTS rag_app;

USE rag_app;

CREATE TABLE IF NOT EXISTS upload_jobs (
    job_id VARCHAR(80) PRIMARY KEY,
    status VARCHAR(30) NOT NULL,
    result_json JSON NULL,
    error TEXT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);
```

### 6. Optional: Run Ollama Fallback

Install Ollama and pull a local fallback model:

```bash
ollama pull mistral
```

Run Ollama in the background before using fallback generation.

## Running Locally

Start the FastAPI backend:

```bash
python3 -m uvicorn app.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Start the frontend:

```bash
python3 -m http.server 5500 --directory ui
```

Open:

```text
http://127.0.0.1:5500
```

## API Endpoints

### Health Check

```http
GET /
```

### Source Status

```http
GET /sources/status
```

Returns whether indexed sources are available.

### Upload Documents

```http
POST /upload
```

Form data:

- `files`: one or more PDF/TXT files
- `replace`: `true` to replace current sources, `false` to append

Response:

```json
{
  "status": "accepted",
  "job_id": "job_...",
  "message": "Upload accepted. Indexing started."
}
```

### Job Status

```http
GET /jobs/{job_id}
```

Response:

```json
{
  "job_id": "job_...",
  "status": "completed",
  "result": {
    "files": ["paper.pdf"],
    "new_chunks": 42,
    "total_chunks": 42,
    "replace": true
  },
  "error": null,
  "created_at": "...",
  "updated_at": "..."
}
```

### Chat

```http
POST /chat
```

JSON body:

```json
{
  "query": "What is this document about?",
  "candidate_k": 8,
  "final_k": 5,
  "mode": "rag"
}
```

### Streaming Chat

```http
POST /chat/stream
```

Streams newline-delimited JSON:

```json
{"type":"token","text":"The"}
{"type":"token","text":" document"}
{"type":"sources","sources":[...]}
```

## Chat Modes

BravoBOT supports two modes:

### RAG Mode

```json
{
  "mode": "rag"
}
```

Uses uploaded documents and returns citations.

### Normal Chat Mode

```json
{
  "mode": "chat"
}
```

Skips document retrieval and answers like a normal assistant.

The backend also routes simple greetings and identity questions to normal chat automatically:

```text
hey
hello
what's your name
who are you
```

## GitHub Safety Checklist

Before pushing:

- Do not commit `.env`
- Do not commit `.venv`
- Do not commit uploaded PDFs
- Do not commit generated Chroma databases
- Do not commit `data/processed`
- Do not commit API keys or database passwords
- Replace any copyrighted visual assets before public/commercial use

This repository includes `.gitignore` rules for the common local artifacts.

## Known Limitations

- Scanned/image-only PDFs need OCR; PyMuPDF can only extract embedded text.
- FastAPI `BackgroundTasks` are acceptable for local/demo use but not ideal for production workers.
- Local Chroma persistence is fine for a demo, but production should use a managed vector database or persistent volume.
- MySQL stores job state, but document metadata is still partly stored in local JSON.
- Gemini free-tier quota can be exhausted; the app falls back to Ollama when configured.
- Multi-user isolation is not implemented yet.
- Authentication and authorization are not implemented.
- Uploaded file size limits and rate limits should be added before production deployment.

## Production Roadmap

Recommended upgrades:

1. Add user/workspace IDs.
2. Move uploaded files to S3, GCS, or Azure Blob Storage.
3. Replace FastAPI `BackgroundTasks` with Redis + RQ/Celery workers.
4. Move document registry into MySQL/Postgres.
5. Use Qdrant, Pinecone, Weaviate, Milvus, or pgvector for production vector storage.
6. Add OCR for scanned PDFs.
7. Add authentication.
8. Add request/file size limits.
9. Add Docker and deployment configuration.
10. Add structured logs, metrics, and error tracking.

## Resume Bullet

```text
Built BravoBOT, a NotebookLM-style document QA app with FastAPI, Chroma, BM25+dense hybrid retrieval, cross-encoder reranking, Gemini streaming responses, Ollama fallback, and MySQL-backed background ingestion jobs.
```

Performance-focused bullet:

```text
Optimized RAG latency from ~22.7s to ~2.4s after cache warm-up by adding BM25/vectorstore/reranker caching and replacing local generation with Gemini streaming plus Ollama fallback.
```

## License

Add a license before publishing publicly. MIT is a common choice for portfolio projects.
