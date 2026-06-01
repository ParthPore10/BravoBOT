from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH = Path("data/processed/user_chunks.json")
PERSIST_DIR = Path("vectorstore/UserUploads")
ACTIVE_PERSIST_PATH = Path("data/processed/active_vectorstore.txt")
REGISTRY_PATH = Path("data/processed/document_registry.json")


EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")

RRF_K = 60
