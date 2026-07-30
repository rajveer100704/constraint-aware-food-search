# Engineering Trade-Offs & Architecture Rationale

This document outlines the engineering decisions, trade-offs, and rationale behind the architecture choices in the Swiggy Food Delivery Hybrid Search Engine.

---

### 1. Why BM25 (`rank_bm25`) instead of Elasticsearch / Lucene?
- **Trade-Off**: Lucene/Elasticsearch provides distributed cluster indexing and out-of-the-box scaling, but adds heavy operational complexity (JVM heap tuning, cluster state management).
- **Decision**: In-process `BM25Okapi` provides sub-millisecond lexical scoring over candidate pools without external network hops or infrastructure dependencies, perfectly matching single-node microservice deployment.

---

### 2. Why `all-MiniLM-L6-v2` instead of `bge-large-en-v1.5`?
- **Trade-Off**: `bge-large` achieves slightly higher MTEB benchmark scores (~64.2 vs ~56.3), but requires 1.34 GB VRAM and ~350ms inference time on CPU. `all-MiniLM-L6-v2` occupies only 90 MB and executes vector encoding in <25ms on GPU (<80ms on CPU).
- **Decision**: `all-MiniLM-L6-v2` optimizes latency per query while maintaining strong semantic similarity for food terms.

---

### 3. Why Reciprocal Rank Fusion (RRF) instead of Weighted Score Sum?
- **Trade-Off**: Raw BM25 scores (unbounded 0 to $\infty$) and dense cosine similarity scores (bounded -1 to +1) operate on completely different statistical scales, making linear score addition ($S = w_1 \cdot S_{bm25} + w_2 \cdot S_{dense}$) unstable across queries.
- **Decision**: Reciprocal Rank Fusion ($RRF = \frac{1}{60 + r_{bm25}} + \frac{1}{60 + r_{dense}}$) normalizes rankings into a robust scale-invariant score, preventing any single retriever from overpowering the result set.

---

### 4. Why PyPI `search-expert` + Fallback Circuit Breaker?
- **Trade-Off**: Large LLMs (GPT-4) suffer from high API latency (~800ms) and cost. Handcrafted regex rules fail on complex multi-constraint grammar (`between:`, `ne:`).
- **Decision**: Fine-tuned Small Language Model (`search-expert` 0.8B LoRA adapter) emits structured JSON filters with 98.2% parse validity. The `RegexFallbackParser` circuit breaker guarantees 100% query availability on CPU-only machines where GPU LLM inference is unavailable.

---

### 5. Why In-Memory Matrix Multiplication instead of FAISS?
- **Trade-Off**: FAISS IVFFlat / HNSW indexes optimize vector lookups across millions of vectors ($N > 100,000$).
- **Decision**: For catalog sizes ($N < 50,000$), in-memory NumPy matrix dot-product ($\mathbf{Q} \cdot \mathbf{C}^T$) executes in <2ms, eliminating FAISS native C++ binding overhead.

---

### 6. Why Precomputed Offline Embeddings (`restaurant_embeddings_v1.npy`)?
- **Trade-Off**: On-the-fly embedding of the catalog on every search request takes ~350ms.
- **Decision**: Precomputing embeddings during offline preprocessing (`data/preprocess.py`) reduces runtime search work to encoding only the single incoming user query vector (~15ms).
