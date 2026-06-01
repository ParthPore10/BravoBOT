from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER =None

def load_reranker():
    global RERANKER

    if RERANKER is None:
        RERANKER = CrossEncoder(RERANKER_MODEL)
    
    return RERANKER

def get_result_text(results):
    doc = results.get("doc")
    chunk = results.get("chunk")

    if doc is not None:
        return doc.page_content
    return chunk['text']

def reranker(query :str, results, top_k :int=5):
    if not results:
        return[]
    
    reranker = load_reranker()
    pairs= []
    
    for result in results:
        text = get_result_text(result)
        pairs.append((query,text))
    
    scores = reranker.predict(pairs)
    reranked =[]

    for result,score in zip(results,scores):
        result["reranked_scores"] = float(score)
        reranked.append(result)
    
    reranked = sorted(
        reranked,
        key=lambda x: x['reranked_scores'],
        reverse=True
    )
    return reranked[:top_k]