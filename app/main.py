import json
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse

from app.document_ingestion import get_upload_status, has_indexed_uploads, ingest_uploaded_files
from app.jobs import create_jobs, get_job, run_ingestion_job
from app.rag_pipeline import answer_query, normal_chat, stream_answer_query, stream_normal_chat
from app.schemas import APIChatRequest, APIChatResponse, APISourceCitation


app = FastAPI(
    title="BravoBOT API",
    description="Document chat and normal chat API",
    version="0.1.0",
    openapi_version="3.0.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    upload_schema = (
        openapi_schema
        .get("components", {})
        .get("schemas", {})
        .get("Body_upload_documents_upload_post", {})
    )
    files_schema = upload_schema.get("properties", {}).get("files", {})
    file_item_schema = files_schema.get("items", {})

    if file_item_schema.get("contentMediaType") == "application/octet-stream":
        file_item_schema.pop("contentMediaType", None)
        file_item_schema["format"] = "binary"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Advanced Local RAG API is running"
    }


@app.get("/sources/status")
def sources_status():
    return get_upload_status()


@app.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(...)],
    replace: Annotated[bool, Form()] = True,
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one PDF or TXT file."
        )

    uploaded_files = []

    for file in files:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} is empty."
            )

        uploaded_files.append((file.filename, content))

    job_id = create_jobs()

    background_tasks.add_task(
        run_ingestion_job,
        job_id,
        ingest_uploaded_files,
        uploaded_files,
        replace
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Upload accepted. Indexing started."
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    return job

def extract_api_sources(sources):
    api_sources = []

    for source in sources:
        doc = source.get("doc")
        chunk = source.get("chunk")

        if doc is not None:
            metadata = doc.metadata
        else:
            metadata = chunk

        if doc is not None:
            preview_text = doc.page_content
        else:
            preview_text = chunk.get("text", "")

        preview_text = " ".join(preview_text.split())
        readable_chars = sum(char.isalnum() for char in preview_text)
        readability_ratio = readable_chars / max(len(preview_text), 1)

        if readability_ratio < 0.25:
            preview_text = "No readable preview available for this source."

        api_sources.append(
            APISourceCitation(
                source_file=metadata.get("source_file"),
                page_number=metadata.get("page_number"),
                preview_text=preview_text[:360],
                chunk_id=metadata.get("chunk_id"),
                rerank_score=source.get("reranked_scores"),
                rrf_score=source.get("rrf_score"),
                dense_rank=source.get("dense_rank"),
                bm25_rank=source.get("bm25_rank"),
            )
        )

    return api_sources


def should_use_normal_chat(query: str) -> bool:
    casual_inputs = {
        "hi",
        "hey",
        "hello",
        "yo",
        "sup",
        "thanks",
        "thank you",
    }

    normalized = query.strip().lower()
    identity_phrases = (
        "your name",
        "ur name",
        "who are you",
        "what are you called",
        "what is your name",
        "whats your name",
        "what's your name",
    )

    return normalized in casual_inputs or any(
        phrase in normalized
        for phrase in identity_phrases
    )


def clean_chat_error(error: Exception) -> str:
    message = str(error)

    if "503" in message or "UNAVAILABLE" in message or "high demand" in message:
        return "BravoBOT is busy right now. Please retry in a moment."

    if "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
        return "BravoBOT has hit the Gemini request limit. Please retry in a minute."

    return message


@app.post("/chat", response_model=APIChatResponse)
def chat(request: APIChatRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    if request.mode == "chat" or should_use_normal_chat(request.query):
        response = normal_chat(request.query)

        return APIChatResponse(
            answer=response["answer"],
            sources=[]
        )

    if not has_indexed_uploads():
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF or TXT source before asking questions."
        )

    response = answer_query(
        query=request.query,
        candidate_k=request.candidate_k,
        final_k=request.final_k
    )

    return APIChatResponse(
        answer=response["answer"],
        sources=extract_api_sources(response["sources"])
    )


@app.post("/chat/stream")
def chat_stream(request: APIChatRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    if request.mode == "chat" or should_use_normal_chat(request.query):
        def normal_event_stream():
            try:
                for event in stream_normal_chat(request.query):
                    yield json.dumps(event) + "\n"

            except Exception as exc:
                yield json.dumps({
                    "type": "error",
                    "message": clean_chat_error(exc),
                }) + "\n"

        return StreamingResponse(
            normal_event_stream(),
            media_type="application/x-ndjson",
        )

    if not has_indexed_uploads():
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF or TXT source before asking questions."
        )

    def event_stream():
        try:
            for event in stream_answer_query(
                query=request.query,
                candidate_k=request.candidate_k,
                final_k=request.final_k,
            ):
                if event["type"] == "sources":
                    event["sources"] = [
                        source.model_dump()
                        for source in extract_api_sources(event["sources"])
                    ]

                yield json.dumps(event) + "\n"

        except Exception as exc:
            yield json.dumps({
                "type": "error",
                "message": clean_chat_error(exc),
            }) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )
