import numpy as np

def get_top_k_chunks(model, query: str, embeddings: np.ndarray, chunks: np.ndarray, top_k: int = 5):
    """Get top k most relevant chunks for query"""
    
    # Encode query with instruction
    instruction = "Represent this sentence for searching relevant passages: "
    query_embedding = model.encode(
        [instruction + query],
        normalize_embeddings=True
    )
    
    # Calculate similarity
    scores = query_embedding @ embeddings.T
    
    # Get top k indices
    top_indices = np.argsort(scores[0])[::-1][:top_k]
    
    # Return top chunks with scores
    results = []
    for idx in top_indices:
        results.append(chunks[idx])
    
    return results

