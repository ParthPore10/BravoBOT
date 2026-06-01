from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi

CHUNK_PATH = Path("data/processed/chunks.json")

def load_chunks():
    json_path = Path(CHUNK_PATH)

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

def search_bm25(query :str, top_k : int=3):
    chunks = load_chunks()
    bm25 =build_bm25(chunks)

    tokenize_query = tokenize(query)
    scores =bm25.get_scores(tokenize_query)

    ranked_index = sorted(
        range(len(scores)),
        key= lambda x: scores[x],
        reverse=True
    )

    results = []

    for index in ranked_index[:top_k]:
        results.append(
            {
                "chunk": chunks[index],
                "score": scores[index]
            }
        )
    return results

def print_results(results):
    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        score = result["score"]

        text_preview = chunk["text"][:300].replace("\n", " ")

        print(f"\nRank {rank}")
        print(f"Score: {score}")
        print(f"Source: {chunk.get('source_file', 'unknown')}")
        print(f"Page: {chunk.get('page_number', 'unknown')}")
        print(f"Chunk ID: {chunk.get('chunk_id', 'unknown')}")
        print(f"Text preview: {text_preview}...")

def main():
    query = input("Enter your query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    results = search_bm25(query=query, top_k=3)

    print_results(results)

if __name__=="__main__":
    main()
