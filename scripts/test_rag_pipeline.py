import os

from app.rag_pipeline import answer_query


def print_answer(response):
    answer = response["answer"]
    sources = response["sources"]

    print("\n========== ANSWER ==========")
    print(answer)

    print("\n========== SOURCES ==========")

    for rank, result in enumerate(sources, start=1):
        doc = result.get("doc")
        chunk = result.get("chunk")

        if doc is not None:
            metadata = doc.metadata
        else:
            metadata = chunk

        print(f"\n[Source {rank}]")
        print(f"File: {metadata.get('source_file', 'unknown')}")
        print(f"Page: {metadata.get('page_number', 'unknown')}")
        print(f"Chunk ID: {metadata.get('chunk_id', 'unknown')}")
        print(f"Rerank Score: {result.get('rerank_score')}")
        print(f"RRF Score: {result.get('rrf_score')}")
        print(f"Dense Rank: {result.get('dense_rank')}")
        print(f"BM25 Rank: {result.get('bm25_rank')}")


def main():
    query = input("Enter your query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    response = answer_query(
        query=query,
        user_id=os.getenv("EVAL_USER_ID", "user-a"),
        candidate_k=5,
        final_k=3
    )

    print_answer(response)


if __name__ == "__main__":
    main()
