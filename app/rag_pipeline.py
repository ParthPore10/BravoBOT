import json
import time

import requests
from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL, LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL
from app.hybrid_retriever import hybrid_search
from app.rerank import reranker

GEMINI_CLIENT = None

def get_result_text(result):
    doc = result.get("doc")
    chunk = result.get("chunk")

    if doc is not None:
        return doc.page_content

    return chunk["text"]

def get_result_metadata(result):
    doc = result.get("doc")
    chunk = result.get("chunk")

    if doc is not None:
        return doc.metadata

    return chunk

def format_context(results):
    context_blocks = []

    for rank, result in enumerate(results, start=1):
        text = get_result_text(result)
        metadata = get_result_metadata(result)

        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")

        block = f"""
[Source {rank}]
File: {source_file}
Page: {page_number}
Chunk ID: {chunk_id}

Content:
{text}
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


def generate_answer_with_ollama(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 400,
            "temperature": 0.2,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


def get_gemini_client():
    global GEMINI_CLIENT

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    if GEMINI_CLIENT is None:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)

    return GEMINI_CLIENT


def generate_answer_with_gemini(prompt: str):
    client = get_gemini_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=400,
        ),
    )

    return response.text


def stream_answer_with_ollama(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": 400,
            "temperature": 0.2,
        },
    }

    with requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
        stream=True,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))
            token = data.get("response")

            if token:
                yield token


def stream_answer_with_gemini(prompt: str):
    client = get_gemini_client()

    response_stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=400,
        ),
    )

    for chunk in response_stream:
        token = getattr(chunk, "text", None)

        if token:
            yield token


def generate_answer(prompt: str):
    if LLM_PROVIDER == "ollama":
        return generate_answer_with_ollama(prompt)

    try:
        return generate_answer_with_gemini(prompt)
    except Exception as exc:
        message = str(exc)

        if "429" in message or "503" in message or "RESOURCE_EXHAUSTED" in message or "UNAVAILABLE" in message:
            print("Gemini unavailable or quota-limited. Falling back to Ollama.")
            return generate_answer_with_ollama(prompt)

        raise


def stream_answer(prompt: str):
    if LLM_PROVIDER == "ollama":
        yield from stream_answer_with_ollama(prompt)
        return

    try:
        yield from stream_answer_with_gemini(prompt)
    except Exception as exc:
        message = str(exc)

        if "429" in message or "503" in message or "RESOURCE_EXHAUSTED" in message or "UNAVAILABLE" in message:
            print("Gemini stream unavailable or quota-limited. Falling back to Ollama.")
            yield from stream_answer_with_ollama(prompt)
            return

        raise

def dedupe_results(results):
    seen = set()
    unique_results = []

    for result in results:
        chunk_id = result.get("chunk_id")

        if chunk_id in seen:
            continue
        
        seen.add(chunk_id)
        unique_results.append(result)
    return unique_results

def build_rag_prompt(query: str, candidate_k: int = 5, final_k: int = 3):
    start = time.perf_counter()

    hybrid_results = hybrid_search(
        query=query,
        top_k=candidate_k
    )
    print(f"hybrid_search: {time.perf_counter() - start:.2f}s")

    dedupe_start = time.perf_counter()
    hybrid_results = dedupe_results(hybrid_results)
    print(f"dedupe hybrid: {time.perf_counter() - dedupe_start:.2f}s")

    rerank_start = time.perf_counter()
    reranked_results = reranker(
        query=query,
        results=hybrid_results,
        top_k=final_k
    )
    print(f"reranker: {time.perf_counter() - rerank_start:.2f}s")

    dedupe_rerank_start = time.perf_counter()
    reranked_results = dedupe_results(reranked_results)
    print(f"dedupe reranked: {time.perf_counter() - dedupe_rerank_start:.2f}s")

    context_start = time.perf_counter()
    context = format_context(reranked_results)
    print(f"format_context: {time.perf_counter() - context_start:.2f}s")

    prompt = build_prompt(
        query=query,
        context=context
    )

    return prompt, reranked_results, start

def build_normal_chat_prompt(query: str):
    return f"""
You are BravoBOT.
Answer the user's question clearly and concisely.

Question:
{query}

Answer:
"""
def normal_chat(query: str):
    prompt = build_normal_chat_prompt(query)
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": []
    }

def stream_normal_chat(query: str):
    prompt = build_normal_chat_prompt(query)

    for token in stream_answer(prompt):
        yield {
            "type": "token",
            "text": token,
        }

    yield {
        "type": "sources",
        "sources": [],
    }
    
def answer_query(query: str, candidate_k: int = 5, final_k: int = 3):
    prompt, reranked_results, start = build_rag_prompt(
        query=query,
        candidate_k=candidate_k,
        final_k=final_k,
    )

    llm_start = time.perf_counter()
    answer = generate_answer(prompt)
    print(f"{LLM_PROVIDER}: {time.perf_counter() - llm_start:.2f}s")
    print(f"total answer_query: {time.perf_counter() - start:.2f}s")

    return {
        "answer": answer,
        "sources": reranked_results
    }


def stream_answer_query(query: str, candidate_k: int = 5, final_k: int = 3):
    prompt, reranked_results, start = build_rag_prompt(
        query=query,
        candidate_k=candidate_k,
        final_k=final_k,
    )

    llm_start = time.perf_counter()

    for token in stream_answer(prompt):
        yield {
            "type": "token",
            "text": token,
        }

    print(f"{LLM_PROVIDER} stream: {time.perf_counter() - llm_start:.2f}s")
    print(f"total stream_answer_query: {time.perf_counter() - start:.2f}s")

    yield {
        "type": "sources",
        "sources": reranked_results,
    }
