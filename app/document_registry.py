from pathlib import Path
import json

DOCUMENT_REGISTRY = 'data/processed/document_registry.json'

def load_registry():
    doc_registry = Path(DOCUMENT_REGISTRY)
    doc_registry.parent.mkdir(parents=True,exist_ok=True)
    
    if not doc_registry.exists():
        return []
    
    
    with open(doc_registry,'r',encoding='utf-8') as f:
        return json.load(f)
    

def save_registry(registry):
    registry_path = Path(DOCUMENT_REGISTRY)

    registry_path.parent.mkdir(parents=True,exist_ok=True)

    with open(registry_path,'w',encoding='utf-8') as f:
        json.dump(registry,f,indent=4,ensure_ascii=False)

def find_document_by_hash(file_hash :str,
                          user_id:str):

    documents = load_registry()

    for document in documents:
        if (
            document.get("file_hash") == file_hash
            and document.get("user_id") == user_id
        ):
            return document
    return None

def add_document_record(document_record :dict[str,object]):
    documents = load_registry()

    documents.append(document_record)
    save_registry(documents)
    return document_record

def remove_documents_for_user(user_id: str) -> None:
    documents = load_registry()
    remaining_documents = [
        document
        for document in documents
        if document.get("user_id") != user_id
    ]
    save_registry(remaining_documents)

def update_document_status(document_id:str,status:str,num_chunks :int=None, num_pages :int =None):
    documents = load_registry()

    for document in documents:
        if document['document_id'] == document_id:
            document['status'] = status
        
            if num_chunks is not None:
                document['num_chunks'] = num_chunks
            if num_pages is not None:
                document['num_pages'] = num_pages
    
            save_registry(documents)
            return document
    return None
