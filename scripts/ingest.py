from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import json
import re
import tempfile

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.schemas import DocumentChunk
from app.utils import generate_chunk_id

HEADING_PATTERN = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+\S.*"
    r"|"
    r"[A-Z][A-Z0-9\s/&(),'-]{2,}"
    r")$"
)

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

def is_heading(text: str)->bool:
    text = text.strip()

    if not text:
        return False
    return bool(HEADING_PATTERN.match(text))

def get_heading_level(text :str)->int:
    text = text.strip()
    text_match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", text.strip())

    if not text:
        return 0

    if text_match:
        return len(text_match.group(1).split("."))
    if is_heading(text) and text.isupper():
        return 1
    return 0

def update_heading_path(heading_path :list[str],
                        heading :str,
                        level: int) ->list[str]:
    if level <=0:
        return heading_path
    new_path = heading_path[:level-1]
    new_path.append(heading)
    return new_path

def extract_sections(text :str)->list[dict]:
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n",text)
        if block.strip()
    ]

    sections=[]
    heading_path = []
    para =[]

    for block in blocks:
        if is_heading(block):
            if para: #just making sure that the previous para is saved
                sections.append({
                    "heading_path":heading_path.copy(),
                    "text":"\n\n".join(para)
                })
                para=[]

            level = get_heading_level(block)
            heading_path = update_heading_path(
                heading_path,
                block,
                level
            )
        else:
            para.append(block)

    if para:
        sections.append({
            "heading_path":heading_path.copy(),
            "text":"\n\n".join(para)
        })
    return sections

def split_oversized_chunks(section: dict,max_chars :int):

    paragraphs = [paragraph.strip()
                  for paragraph in section["text"].split("\n\n")
                  if paragraph.strip()
                  ]
    chunks=[]
    current_paragraphs =[]

    if max_chars<=0:
        raise ValueError("max chars should be greater than 0")

    for para in paragraphs:
        candidate = "\n\n".join(current_paragraphs+[para])

        if current_paragraphs and len(candidate)>max_chars:
            chunks.append({
                "heading_path":section['heading_path'].copy(),
                "text":"\n\n".join(current_paragraphs)
            })
            current_paragraphs =[para]
        else:
            current_paragraphs.append(para)

    if current_paragraphs:
        chunks.append({
            "heading_path" : section['heading_path'].copy(),
            "text": "\n\n".join(current_paragraphs)
        })
    return chunks

def create_structured_chunks(text :str, max_chars :int):
    text = text.strip()

    sections = extract_sections(text)
    chunk_list =[]

    for section in sections:
        chunks =split_oversized_chunks(section,max_chars)

        for chunk in chunks:
            chunk["context"] = " > ".join(chunk['heading_path'])
            chunk_list.append(chunk)
    return chunk_list

def split_text_into_chunks(file_path: str|Path, chunk_size :int,
                           chunk_overlap :int)->list[Document]:

    del chunk_overlap
    file_path =Path(file_path)
    docs = load_documents(file_path)
    output_docs = []

    for doc in docs:
        chunks = create_structured_chunks(
            text = doc.page_content,
            max_chars=chunk_size)

        for chunk in chunks:
            context = chunk['context']
            content= (
                f"{context}\n\n{chunk['text']}"
                if context
                else chunk['text']
            )

            metadata = doc.metadata.copy()
            metadata.update({
                'heading_path':context,
                "context":context,
                "section_title":(
                    chunk['heading_path'][-1]
                    if chunk["heading_path"]
                    else ""
                ),
            })

            output_docs.append(
                Document(
                    page_content=content,
                    metadata = metadata,
                )
            )
    return output_docs


def create_chunk_id(source_file : str, page_number : int, chunk_index :int)->str:
    source_name = Path(source_file).stem
    source_name = source_name.lower().replace(" ","_")
    chunk_id = f"{source_name}_page_{page_number}_chunk{chunk_index}"
    return chunk_id

def create_document_chunks(
    file_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    user_id: str,
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
            user_id= user_id,
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk_index,
            text=chunk.page_content,
            document_type=file_path.suffix.lower().replace(".", ""),
            created_at=datetime.now().isoformat(),
            document_id=document_id,
            file_hash=file_hash,
            metadata= chunk.metadata
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
            "user_id":chunk.user_id,
            "source_file" : chunk.source_file,
            "page_number" : chunk.page_number,
            "chunk_index" : chunk.chunk_index,
            "text" : chunk.text,
            "document_type" : chunk.document_type,
            "created_at" : chunk.created_at,
            "document_id" :chunk.document_id,
            "file_hash": chunk.file_hash,
            "context":(chunk.metadata or {}).get("context",""),
            "heading_path":(chunk.metadata or  {}).get("heading_path",""),
            "section_title":(chunk.metadata or {}).get("section_title","")
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
        user_id="dev_user",
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
##################################################################

#testing theading path

def test_heading_path():
    path=[]

    path = update_heading_path(path,"1. Introduction",1)
    assert path ==["1. Introduction"]

    path = update_heading_path(path,"1.1 motivation",2)
    assert path ==["1. Introduction","1.1 motivation"]

    print("All heading path tests passed!!")

def test_extract_sections():
    sample_text = """
1. Introduction

This is the introduction paragraph.

1.1 Motivation

This is the motivation paragraph.

1.2 Objectives

This is the objectives paragraph.

2. Methods

This is the methods paragraph.
"""

    sections = extract_sections(sample_text)

    assert len(sections) == 4

    assert sections[0]["heading_path"] == ["1. Introduction"]
    assert sections[0]["text"] == "This is the introduction paragraph."

    assert sections[1]["heading_path"] == [
        "1. Introduction",
        "1.1 Motivation",
    ]
    assert sections[1]["text"] == "This is the motivation paragraph."

    assert sections[2]["heading_path"] == [
        "1. Introduction",
        "1.2 Objectives",
    ]
    assert sections[2]["text"] == "This is the objectives paragraph."

    assert sections[3]["heading_path"] == ["2. Methods"]
    assert sections[3]["text"] == "This is the methods paragraph."

    print("All extract_sections tests passed!")

def test_split_oversized_chunks():
    section = {
        "heading_path": ["1. Introduction", "1.1 Motivation"],
        "text": (
            "This is the first paragraph.\n\n"
            "This is the second paragraph.\n\n"
            "This is the third paragraph."
        ),
    }

    max_chars = 40
    chunks = split_oversized_chunks(section, max_chars)

    assert len(chunks) == 3

    expected_paragraphs = [
        "This is the first paragraph.",
        "This is the second paragraph.",
        "This is the third paragraph.",
    ]

    assert [chunk["text"] for chunk in chunks] == expected_paragraphs

    for chunk in chunks:
        assert chunk["heading_path"] == [
            "1. Introduction",
            "1.1 Motivation",
        ]
        assert len(chunk["text"]) <= max_chars

    print("All oversized-chunk tests passed!")

def test_create_structured_chunks():
    sample_text = """
1. Introduction

This is the introduction paragraph.

1.1 Motivation

This is the first motivation paragraph.

This is the second motivation paragraph.

This is the third motivation paragraph.
"""

    chunks = create_structured_chunks(sample_text, max_chars=45)

    assert isinstance(chunks, list)
    assert len(chunks) == 4

    expected_texts = [
        "This is the introduction paragraph.",
        "This is the first motivation paragraph.",
        "This is the second motivation paragraph.",
        "This is the third motivation paragraph.",
    ]

    assert [chunk["text"] for chunk in chunks] == expected_texts

    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert "heading_path" in chunk
        assert "context" in chunk
        assert "text" in chunk
        assert chunk["context"] == " > ".join(chunk["heading_path"])
        assert len(chunk["text"]) <= 45

    assert chunks[0]["heading_path"] == ["1. Introduction"]

    for chunk in chunks[1:]:
        assert chunk["heading_path"] == [
            "1. Introduction",
            "1.1 Motivation",
        ]

    print("All structured-chunk tests passed!")

def test_split_text_into_chunks():
    sample_text = """
1. Introduction

This is the introduction paragraph.

1.1 Motivation

This is the motivation paragraph.
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "sample.txt"
        file_path.write_text(sample_text, encoding="utf-8")

        documents = split_text_into_chunks(
            file_path=file_path,
            chunk_size=800,
            chunk_overlap=0,
        )

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

        assert documents[0].page_content == (
            "1. Introduction\n\n"
            "This is the introduction paragraph."
        )

        assert documents[1].page_content == (
            "1. Introduction > 1.1 Motivation\n\n"
            "This is the motivation paragraph."
        )

        assert documents[0].metadata["heading_path"] == "1. Introduction"
        assert documents[0].metadata["context"] == "1. Introduction"
        assert documents[0].metadata["section_title"] == "1. Introduction"

        assert documents[1].metadata["heading_path"] == (
            "1. Introduction > 1.1 Motivation"
        )
        assert documents[1].metadata["context"] == (
            "1. Introduction > 1.1 Motivation"
        )
        assert documents[1].metadata["section_title"] == "1.1 Motivation"

    print("All split-text tests passed!")


def test_create_document_chunks():
    sample_text = """
1. Introduction

This is the introduction paragraph.

1.1 Motivation

This is the motivation paragraph.
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "sample.txt"
        file_path.write_text(sample_text, encoding="utf-8")

        chunks = create_document_chunks(
            file_path=file_path,
            chunk_size=800,
            chunk_overlap=0,
            user_id="test-user",
            document_id="doc_test123",
            file_hash="hash_test123",
        )

        assert len(chunks) == 2
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

        assert chunks[0].chunk_id == "doc_test123_chunk_0"
        assert chunks[1].chunk_id == "doc_test123_chunk_1"
        assert chunks[0].chunk_id != chunks[1].chunk_id

        for chunk in chunks:
            assert chunk.source_file == "sample.txt"
            assert chunk.page_number == 1
            assert chunk.document_id == "doc_test123"
            assert chunk.file_hash == "hash_test123"

            assert chunk.metadata is not None
            assert "heading_path" in chunk.metadata
            assert "context" in chunk.metadata
            assert "section_title" in chunk.metadata

        assert chunks[0].text == (
            "1. Introduction\n\n"
            "This is the introduction paragraph."
        )

        assert chunks[1].text == (
            "1. Introduction > 1.1 Motivation\n\n"
            "This is the motivation paragraph."
        )

        assert chunks[1].metadata["section_title"] == "1.1 Motivation"

    print("All document-chunk tests passed!")

if __name__ == "__main__":
    #main()
    test_create_document_chunks()
