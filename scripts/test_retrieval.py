import os

from app.retriever import dense_search

def print_results(results)->None:
    if not results:
        print("No results found")
    
    for rank, result in enumerate(results,start=1):
        docs = result["doc"]
        score = result["dense_score"]
        metadata = docs.metadata

        source_file = metadata.get("source_file","unknown")
        page_number = metadata.get("page_number","unknown")
        chunk_id = metadata.get("chunk_id","unknown")
        
        text_preview = docs.page_content[:300].replace("\n"," ")

        print(f"\nRank: {rank}")
        print(f"Score: {score}")
        print(f"Source: {source_file}")
        print(f"Page: {page_number}")
        print(f"Chunk ID: {chunk_id}")
        print(f"text_preview: {text_preview}.....")

def main():
    query = input(
        "Enter Your Query : "
    ).strip()

    if not query:
        print("Query cannot be empty")
        return
    
    top_k =3

    results = dense_search(
        query=query,
        user_id=os.getenv("EVAL_USER_ID", "user-a"),
        top_k=top_k,
    )
    print_results(results)

if __name__ =="__main__":
    main()
