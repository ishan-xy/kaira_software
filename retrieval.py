import numpy as np
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-base')

def get_top_k_chunks(model, query: str, embeddings: np.ndarray, chunks: np.ndarray, top_k: int = 5):
    
    instruction = "Represent this sentence for searching relevant passages: "
    query_embedding = model.encode(
        [instruction + query],
        normalize_embeddings=True
    )
    
    scores = query_embedding @ embeddings.T
    
    candidate_k = min(50, len(chunks)) 
    
    top_candidate_indices = np.argsort(scores[0])[::-1][:candidate_k]
    
    candidate_chunks = [chunks[idx] for idx in top_candidate_indices]

    rerank_pairs = [[query, chunk] for chunk in candidate_chunks]

    rerank_scores = reranker.predict(rerank_pairs)

    scored_chunks = list(zip(rerank_scores, candidate_chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    results = [chunk for score, chunk in scored_chunks[:top_k]]
    
    return results