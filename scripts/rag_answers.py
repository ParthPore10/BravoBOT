from pathlib import Path
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = Path("vectorstore/Chroma")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"

def load_vectors():
    persist_dir = Path(PERSIST_DIR)

    if not persist_dir.exists():
        raise FileNotFoundError("The given directory path doesnt exist. Enter a valid path")
    
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDINGS_MODEL
    )

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    return vectorstore

def retrieve(query : str, top_k :int=3):
    vectorstore = load_vectors()

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k
    )
    return results

def format_context(results):
    context_blocks =[]

    for rank, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata

        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")

        block = f"""
[Source {rank}]
File: {source_file}
Page: {page_number}
Chunk ID: {chunk_id}
Score: {score}

Content:
{doc.page_content}
"""
        context_blocks.append(block)

    return "\n".join(context_blocks)


def build_prompt(query: str, context: str):
    prompt = f"""
You are a helpful local RAG assistant.

Use ONLY the provided context to answer the user's question.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

When you use information from the context, cite the source number like [Source 1] or [Source 2].

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt

def generate_answer(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]

def print_answer(answer: str, results):
    print("\n========== ANSWER ==========")
    print(answer)

    print("\n========== SOURCES ==========")

    for rank, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata

        print(f"\nRaw metadata for Source {rank}:")
        print(metadata)

        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")

        print(f"\n[Source {rank}]")
        print(f"File: {source_file}")
        print(f"Page: {page_number}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Score: {score}")

def main():
    query = input("Enter your query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    results = retrieve(query=query, top_k=3)

    context = format_context(results)

    prompt = build_prompt(query=query, context=context)

    answer = generate_answer(prompt)

    print_answer(answer=answer, results=results)

if __name__ == "__main__":
    main()