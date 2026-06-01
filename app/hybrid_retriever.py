from app.config import RRF_K
from app.retriever import dense_search, search_bm25

def rrf(rank:int)->float:
    return 1/(RRF_K+rank)

def hybrid_search(query :str, top_k : int=5):
    dense_results = dense_search(query=query,top_k = top_k)
    bm25_results = search_bm25(query=query,top_k=top_k)

    fused={}

    for item in dense_results:
        chunk_id = item['chunk_id']

        if chunk_id not in fused:
            fused[chunk_id]={
                "chunk_id" :chunk_id,
                "doc":item["doc"],
                "dense_score":item["dense_score"],
                "dense_rank":item["dense_rank"],
                "bm25_score":None,
                "bm25_rank":None,
                "rrf_score":0.0
            }
        fused[chunk_id]["rrf_score"] += rrf(item['dense_rank'])

    for item in bm25_results:
        chunk_id = item["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id]={
                "chunk_id" :chunk_id,
                "doc":None,
                "chunk":item["chunk"],
                "dense_score":None,
                "dense_rank":None,
                "bm25_score":item['bm25_score'],
                "bm25_rank":item['bm25_rank'],
                "rrf_score":0.0
            }
        fused[chunk_id]['bm25_score'] = item['bm25_score']
        fused[chunk_id]['bm25_rank'] = item['bm25_rank']
        fused[chunk_id]['rrf_score'] += rrf(item['bm25_rank'])
    
    ranked_results =sorted(
        fused.values(),
        key = lambda x: x['rrf_score'],
        reverse=True
    )
    return ranked_results[:top_k]