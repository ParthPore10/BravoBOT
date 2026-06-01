from pathlib import Path
import hashlib
import uuid

def compute_file_hash(file_path :str|Path)->str:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError("Enter a vslid file path")
    
    h = hashlib.sha256()

    with open(file_path,"rb")as f:
        for block in iter(lambda :f.read(1024*1024),b""):
            h.update(block)
    
    return h.hexdigest()

def generate_document_id()->str:
    return f"doc_{uuid.uuid4().hex}"

def generate_chunk_id(document_id :str, chunk_index :int)->str:
    return f"{document_id}_chunk_{chunk_index}"
