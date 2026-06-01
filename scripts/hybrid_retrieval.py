from app.hybrid_retriever import hybrid_search
from app.config import RRF_K

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
        print(f"RRF Score: {result['rrf_score']}")
        print(f"Dense Rank: {result.get('dense_rank')}")
        print(f"Dense Score: {result.get('dense_score')}")
        print(f"BM25 Rank: {result.get('bm25_rank')}")
        print(f"BM25 Score: {result.get('bm25_score')}")
        print(f"Source: {metadata.get('source_file', 'unknown')}")
        print(f"Page: {metadata.get('page_number', 'unknown')}")
        print(f"Chunk ID: {metadata.get('chunk_id', 'unknown')}")
        print(f"Text preview: {text[:300].replace(chr(10), ' ')}...")


def main():
    query = input("Enter your query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    results = hybrid_search(query=query, top_k=5)
    print_results(results)


if __name__ == "__main__":
    main()