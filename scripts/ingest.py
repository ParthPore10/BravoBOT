from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import json

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.schemas import DocumentChunk
from app.utils import generate_chunk_id

def load_documents(file_path : str|Path)->list:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError("The given Path Doesnt Exists, enter a valid path")
    
    if file_path.suffix.lower() == ".pdf":
        loader = PyMuPDFLoader(str(file_path))
    elif file_path.suffix.lower() == ".txt":
        loader = TextLoader(str(file_path),encoding="utf-8")
    else:
        raise ValueError("Currently accepting only .pdf and .txt files")

    document = loader.load()
    return document

def split_text_into_chunks(file_path:str|Path,chunk_size : int, chunk_overlap :int)->list:
    documents = load_documents(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n","\n"," "]
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_chunk_id(source_file : str, page_number : int, chunk_index :int)->str:
    source_name = Path(source_file).stem
    source_name = source_name.lower().replace(" ","_")
    chunk_id = f"{source_name}_page_{page_number}_chunk{chunk_index}"
    return chunk_id

def create_document_chunks(
    file_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    document_id: str | None = None,
    file_hash: str | None = None,
) -> list[DocumentChunk]:
    file_path = Path(file_path)

    chunks = split_text_into_chunks(
        file_path=file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    document_chunks = []

    for chunk_index, chunk in enumerate(chunks):
        source_file = Path(chunk.metadata.get("source", file_path)).name
        page_number = int(chunk.metadata.get("page", 0)) + 1

        if document_id is not None:
            chunk_id = generate_chunk_id(document_id, chunk_index)
        else:
            chunk_id = create_chunk_id(
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk_index,
            )

        document_chunk = DocumentChunk(
            chunk_id=chunk_id,
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk_index,
            text=chunk.page_content,
            document_type=file_path.suffix.lower().replace(".", ""),
            created_at=datetime.now().isoformat(),
            document_id=document_id,
            file_hash=file_hash,
        )

        document_chunks.append(document_chunk)

    return document_chunks

def save_chunks(document_chunks :list[DocumentChunk],output_path :str|Path)->None:
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True,exist_ok=True)

    chunk_as_dict =[]

    for chunk in document_chunks:
        chunk_dict = asdict(chunk)
        chunk_as_dict.append(chunk_dict)
    
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(chunk_as_dict,f,indent=4,ensure_ascii=False)


def build_vectorsearch(document_chunks:DocumentChunk,persist_directory :str|Path):
    persist_directory = Path(persist_directory)

    langchain_document = []

    for chunk in document_chunks:
        meta_data={
            "chunk_id" : chunk.chunk_id,
            "source_file" : chunk.source_file,
            "page_number" : chunk.page_number,
            "chunk_index" : chunk.chunk_index,
            "text" : chunk.text,
            "document_type" : chunk.document_type,
            "created_at" : chunk.created_at,
            "document_id" :chunk.document_id,
            "file_hash": chunk.file_hash
        }

        document = Document(
            page_content=chunk.text,
            metadata = meta_data
        )

        langchain_document.append(document)
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=langchain_document,
        embedding=embeddings_model,
        persist_directory=str(persist_directory)
    )
    return vectorstore


def main():
    file_path = "data/raw/sample.pdf"

    document_chunks = create_document_chunks(
        file_path=file_path,
        chunk_size=800,
        chunk_overlap=150,
        document_id=None,
        file_hash=None
    )

    save_chunks(
        document_chunks=document_chunks,
        output_path="data/processed/chunks.json"
    )

    build_vectorsearch(document_chunks=document_chunks,
                       persist_directory="vectorstore/Chroma")

    print(f"Saved {len(document_chunks)} chunks.")
    print("chunks saved to chunks.json")
    print("vectorstore saved in vectorstore/Chroma")
    print("Injeestion successful")
if __name__ == "__main__":
    main()
