from app.hybrid_retriever import hybrid_search
from app.rerank import reranker


def print_results(results):
    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        doc = result.get("doc")
        chunk = result.get("chunk")

        if doc is not None:
            metadata = doc.metadata
            text = doc.page_content
        else:
            metadata = chunk
            text = chunk["text"]

        print(f"\nRank {rank}")
        print(f"Rerank Score: {result.get('rerank_score')}")
        print(f"RRF Score: {result.get('rrf_score')}")
        print(f"Dense Rank: {result.get('dense_rank')}")
        print(f"BM25 Rank: {result.get('bm25_rank')}")
        print(f"Source: {metadata.get('source_file', 'unknown')}")
        print(f"Page: {metadata.get('page_number', 'unknown')}")
        print(f"Chunk ID: {metadata.get('chunk_id', 'unknown')}")
        print(f"Text preview: {text[:300].replace(chr(10), ' ')}...")


def main():
    query = input("Enter your query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    hybrid_results = hybrid_search(query=query, top_k=5)

    reranked_results = reranker(
        query=query,
        results=hybrid_results,
        top_k=3
    )

    print_results(reranked_results)


if __name__ == "__main__":
    main()