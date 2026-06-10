from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Optional,Any

@dataclass
class DocumentRecord:
    document_id :str
    user_id :str
    source_file :str
    file_name :str
    document_type :str
    file_hash :str
    created_at :str
    status :str="pending"
    num_chunks :int=0
    num_pages :Optional[str]=None

@dataclass
class DocumentChunk:
    chunk_id :str
    user_id :str
    source_file : str
    page_number : int
    chunk_index :int
    text : str
    document_type : str
    created_at : str

    document_id : Optional[str] = None
    file_hash :Optional[str] = None
    metadata :dict[str,Any] = None

@dataclass
class ChatRequest:
    query : str
    top_k : int

@dataclass
class SourceCitation:
    source_file :str
    page_number : int
    chunk_id : str
    score :float

@dataclass
class ChatResponse:
    answer : str
    sources : list[SourceCitation]

class APISourceCitation(BaseModel):
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    preview_text: Optional[str] = None
    chunk_id: Optional[str] = None
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None

class APIChatResponse(BaseModel):
    answer: str
    sources: List[APISourceCitation]

class APIChatRequest(BaseModel):
    query: str
    candidate_k: int = 8
    final_k: int = 5
    mode: str = "rag"
