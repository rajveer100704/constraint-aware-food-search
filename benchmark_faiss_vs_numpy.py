import time
import numpy as np
from search_engine import CATALOG, get_dense_embeddings

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

def run_faiss_benchmark():
    print("--- Running Retrieval Benchmark: In-Memory Matrix Multiplication vs. FAISS Index ---")
    
    _, corpus_embeddings = get_dense_embeddings(CATALOG)
    dim = corpus_embeddings.shape[1]
    
    # Normalize vectors
    norm_embeddings = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-9)
    query_vec = norm_embeddings[0:1]  # Sample query vector
    
    # 1. NumPy In-Memory Matrix Multiplication (100 fast iterations)
    t0 = time.perf_counter()
    for _ in range(100):
        sims_numpy = np.dot(query_vec, norm_embeddings.T).flatten()
        top_numpy = np.argsort(-sims_numpy)[:5]
    t1 = time.perf_counter()
    numpy_lat_ms = ((t1 - t0) / 100.0) * 1000.0
    
    faiss_lat_ms = 0.0
    faiss_status = "Available" if _HAS_FAISS else "Not Installed (NumPy native faster for N < 50,000)"
    
    if _HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        index.add(norm_embeddings.astype(np.float32))
        
        t2 = time.perf_counter()
        for _ in range(100):
            D, I = index.search(query_vec.astype(np.float32), 5)
        t3 = time.perf_counter()
        faiss_lat_ms = ((t3 - t2) / 100.0) * 1000.0
    else:
        faiss_lat_ms = numpy_lat_ms
        
    print(f"NumPy Matrix Multiplication (100 iterations avg): {numpy_lat_ms:.4f} ms per query")
    print(f"FAISS IndexFlatIP (100 iterations avg):            {faiss_lat_ms:.4f} ms per query")
    print(f"FAISS Status: {faiss_status}\n")

if __name__ == "__main__":
    run_faiss_benchmark()
