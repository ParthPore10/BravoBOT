import json
import re
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from app.config import CHUNKS_PATH, EMBEDDINGS_MODEL
from app.document_ingestion import get_active_vectorstore_dir

BM25_CACHE = {
    "mtime": None,
    "chunks" : None,
    "bm25" : None
}

VECTORSTORE_CACHE = {
    'persist_dir' : None,
    "vectorstore" :None
}

def load_vectorstore():
    persist_dir = get_active_vectorstore_dir()

    if not (persist_dir / "chroma.sqlite3").exists():
        raise FileNotFoundError("The given directory path doesnt exist. Enter a valid path")
    
    if VECTORSTORE_CACHE['persist_dir'] == persist_dir:
        return VECTORSTORE_CACHE['vectorstore']

    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDINGS_MODEL
    )

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )

    VECTORSTORE_CACHE['persist_dir'] =persist_dir
    VECTORSTORE_CACHE["vectorstore"] = vectorstore
    return vectorstore

def dense_search(query :str, top_k :int=5):
    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k
    )
    
    dense_rank=[]
    for rank, (doc, score) in enumerate(results, start=1):
        dense_rank.append(
            {
                "chunk_id" : doc.metadata.get("chunk_id"),
                "doc":doc,
                "dense_score" : score,
                "dense_rank" :rank
            }
        )
    return dense_rank

def load_chunks():
    json_path = Path(CHUNKS_PATH)

    if not json_path.exists():
        raise FileNotFoundError("Invalid path. Enter a valid path")
    
    with open(json_path,"r",encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

def tokenize(text :str):
    text =text.lower()
    tokens = re.findall(r"\b\w+\b",text)

    return tokens

def build_bm25(chunks):
    tokenize_corpus = []

    for chunk in chunks:
        text = chunk['text']
        tokens = tokenize(text)
        tokenize_corpus.append(tokens)
    bm25 = BM25Okapi(tokenize_corpus)
    return bm25


def get_bm25_index():
    json_path = Path(CHUNKS_PATH)

    if not json_path.exists():
        raise FileNotFoundError("Invalid path. Enter a valid path")

    current_mtime = json_path.stat().st_mtime

    if BM25_CACHE["mtime"] == current_mtime:
        return BM25_CACHE["chunks"], BM25_CACHE["bm25"]

    chunks = load_chunks()
    if not chunks:
        raise ValueError("NO Chunks available for bm25 search")
    
    bm25 = build_bm25(chunks)

    BM25_CACHE["mtime"] = current_mtime
    BM25_CACHE["chunks"] = chunks
    BM25_CACHE["bm25"] = bm25

    return chunks, bm25

def search_bm25(query :str, top_k : int=3):
    chunks,bm25 = get_bm25_index()

    tokenize_query = tokenize(query)
    scores =bm25.get_scores(tokenize_query)

    ranked_index = sorted(
        range(len(scores)),
        key= lambda x: scores[x],
        reverse=True
    )

    bm25_results = []

    for rank,index in enumerate(ranked_index[:top_k],start=1):
        chunk = chunks[index]
        bm25_results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "chunk": chunk,
                "bm25_score": scores[index],
                "bm25_rank": rank
            }
        )
    return bm25_results