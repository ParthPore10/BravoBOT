from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = Path("vectorstore/Chroma")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

def print_results(results : retrieve)->None:
    if not results:
        print("No results found")
    
    for rank, (docs,score) in enumerate(results,start=1):
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

    results = retrieve(query=query, top_k=top_k)
    print_results(results)

if __name__ =="__main__":
    main()