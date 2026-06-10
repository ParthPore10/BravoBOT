import os

from app.retriever import search_bm25

def print_results(results):
    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        score = result["bm25_score"]

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

    results = search_bm25(
        query=query,
        user_id=os.getenv("EVAL_USER_ID", "user-a"),
        top_k=3,
    )

    print_results(results)

if __name__=="__main__":
    main()
