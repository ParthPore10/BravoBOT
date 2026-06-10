from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import uuid
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import ACTIVE_PERSIST_PATH, CHUNKS_PATH, EMBEDDINGS_MODEL, PERSIST_DIR, REGISTRY_PATH
from scripts.ingest import create_document_chunks
from app.document_registry import (
    add_document_record,
    find_document_by_hash,
    remove_documents_for_user,
    update_document_status,
)
from app.utils import compute_file_hash,generate_document_id

RAW_UPLOAD_DIR = Path("data/raw/uploads")
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def clean_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace(" ", "_")
    return name or "uploaded_document"


def get_user_upload_dir(user_id: str) -> Path:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id is required")

    user_directory = hashlib.sha256(
        normalized_user_id.encode("utf-8")
    ).hexdigest()
    return RAW_UPLOAD_DIR / user_directory


def reset_storage() -> None:
    chunks_path = Path(CHUNKS_PATH)

    if chunks_path.exists():
        chunks_path.unlink()

    active_path = Path(ACTIVE_PERSIST_PATH)
    if active_path.exists():
        active_path.unlink()
    
    registry_path = Path(REGISTRY_PATH)
    if registry_path.exists():
        registry_path.unlink()


    RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)


def write_active_vectorstore_dir(persist_dir: Path) -> None:
    active_path = Path(ACTIVE_PERSIST_PATH)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(str(persist_dir), encoding="utf-8")


def get_active_vectorstore_dir() -> Path:
    active_path = Path(ACTIVE_PERSIST_PATH)

    if not active_path.exists():
        return Path(PERSIST_DIR)

    return Path(active_path.read_text(encoding="utf-8").strip())


def has_indexed_uploads(user_id:str) -> bool:
    chunks = [
        chunk
        for chunk in load_existing_chunks()
        if chunk.get("user_id") == user_id
    ]

    persist_dir = get_active_vectorstore_dir()

    return bool(chunks) and (persist_dir / "chroma.sqlite3").exists()


def get_upload_status(user_id: str) -> dict:
    chunks = [
        chunk
        for chunk in load_existing_chunks()
        if chunk.get("user_id") == user_id
    ]
    source_files = sorted({chunk.get("source_file", "unknown") for chunk in chunks})

    return {
        "has_sources": has_indexed_uploads(user_id),
        "total_chunks": len(chunks),
        "files": source_files,
    }


def save_upload(filename: str, content: bytes, user_id: str) -> Path:
    safe_name = clean_filename(filename)
    suffix = Path(safe_name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Upload one of: {allowed}")

    upload_dir = get_user_upload_dir(user_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_name
    file_path.write_bytes(content)
    return file_path


def load_existing_chunks() -> list[dict]:
    chunks_path = Path(CHUNKS_PATH)

    if not chunks_path.exists():
        return []

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    for chunk in chunks:
        chunk.setdefault("user_id", "legacy-user")

    return chunks


def save_chunk_dicts(chunks: list[dict]) -> None:
    chunks_path = Path(CHUNKS_PATH)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    with chunks_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)


def build_vectorstore_from_chunks(chunks: list[dict], persist_dir: Path) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "user_id": chunk["user_id"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "document_type": chunk["document_type"],
                "created_at": chunk["created_at"],
                "document_id" :chunk.get("document_id"),
                "file_hash" :chunk.get("file_hash"),
                "context":(chunk.get("metadata") or {}).get("context",""),
                "heading_path":(chunk.get("metadata")  or  {}).get("heading_path",""),
                "section_title":(chunk.get ("metadata") or {}).get("section_title","")
            },
        )
        for chunk in chunks
    ]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="langchain",
    )


def append_chunks_to_vectorstore(chunks: list[dict], persist_dir: Path) -> None:
    if not chunks:
        return

    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "user_id": chunk["user_id"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "document_type": chunk["document_type"],
                "created_at": chunk["created_at"],
                "document_id": chunk.get("document_id"),
                "file_hash": chunk.get("file_hash"),
                "context":(chunk.get("metadata")  or {}).get("context",""),
                "heading_path":(chunk.get("metadata")  or  {}).get("heading_path",""),
                "section_title":(chunk.get("metadata")  or {}).get("section_title","")
            },
        )
        for chunk in chunks
    ]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="langchain",
    )

    vectorstore.add_documents(documents)


def ingest_uploaded_files(uploaded_files: list[tuple[str, bytes]],
                          user_id: str,
                           replace: bool = True) -> dict:
    all_existing_chunks = load_existing_chunks()

    if replace:
        existing_chunks = [
            chunk
            for chunk in all_existing_chunks
            if chunk.get("user_id") != user_id
        ]
        remove_documents_for_user(user_id)
    else:
        existing_chunks = all_existing_chunks
    new_chunks = []
    saved_files = []
    skipped_files =[]
    for filename, content in uploaded_files:
        file_path = save_upload(filename, content, user_id)
        saved_files.append(file_path.name)
        
        file_hash = compute_file_hash(file_path)
        existing_document = find_document_by_hash(
            file_hash=file_hash,
            user_id=user_id,
        )
        
        if existing_document is not None and existing_document.get("status") == "completed":
            skipped_files.append(file_path.name)
            continue

        document_id = generate_document_id()

        document_record = {
            "document_id":document_id,
            "user_id":user_id,
            "source_file":str(file_path),
            "file_name": file_path.name,
            "document_type":Path(file_path).suffix.lower().lstrip("."),
            "file_hash":file_hash,
            "created_at" : datetime.now().isoformat(),
            "status": "pending",
            "num_chunks":0,
            "num_pages":None,
        }
        add_document_record(document_record)
        try:
            document_chunks = create_document_chunks(
                file_path=file_path,
                chunk_size=800,
                chunk_overlap=150,
                user_id=user_id,
                document_id=document_id,
                file_hash=file_hash
            )

            if not document_chunks:
                raise ValueError(
                    f"{file_path.name} did not contain extractable text. "
                    "Upload a text-based PDF/TXT file."
                )
        except Exception:
            update_document_status(
                document_id=document_id,
                status="failed",
                num_chunks=0
            )
            raise 

        new_chunks.extend(asdict(chunk) for chunk in document_chunks)

        update_document_status(
            document_id=document_id,
            status="completed",
            num_chunks=len(document_chunks),
        )
    all_chunks = existing_chunks + new_chunks

    if not all_chunks:
        return {
        "files": saved_files,
        "skipped_files": skipped_files,
        "new_chunks": 0,
        "total_chunks": 0,
        "replace": replace,
    }
    if not new_chunks:
        return {
            "files": saved_files,
            "skipped_files": skipped_files,
            "new_chunks": 0,
            "total_chunks": len(all_chunks),
            "replace": replace,
        }

    save_chunk_dicts(all_chunks)

    active_persist_dir = get_active_vectorstore_dir()
    active_index_exists = (active_persist_dir / "chroma.sqlite3").exists()

    if replace or not active_index_exists:
        persist_dir = Path(PERSIST_DIR) / f"index_{uuid.uuid4().hex}"
        build_vectorstore_from_chunks(all_chunks, persist_dir=persist_dir)
        write_active_vectorstore_dir(persist_dir)
    else:
        append_chunks_to_vectorstore(new_chunks, persist_dir=active_persist_dir)

    return {
        "files": saved_files,
        "skipped_files":skipped_files,
        "new_chunks": len(new_chunks),
        "total_chunks": len(all_chunks),
        "replace": replace,
    }
