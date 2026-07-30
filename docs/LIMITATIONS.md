# ⚠️ System Limitations & Out-of-Scope Design Choices

This document explicitly outlines the technical boundaries, assumptions, and intentional limitations of the Constraint-Aware Hybrid Food Search Engine repository.

---

### 1. Offline Evaluation Benchmark
- **Limitation**: Evaluation metrics (Recall@5, Precision@5, NDCG@5) are computed over a 200-query offline benchmark dataset (`data/evaluation_set_v2.json`, `SEED=42`).
- **Production Reality**: True search effectiveness requires online A/B testing measuring Search-to-Order Conversion Rate, Click-Through Rate (CTR), and Zero-Result Rate (ZRR) across live user traffic streams.

---

### 2. Simulated Click Log Supervision
- **Limitation**: The Learning-to-Rank GBDT model (`HistGradientBoostingRegressor`) is trained on simulated query click logs (`data/query_click_logs.csv`).
- **Production Reality**: While the click generator models **position bias** ($P(\text{click}) \propto \text{rank}^{-0.85}$) and **query abandonment (10%)**, real production click logs capture complex multi-session user behaviors, seasonal order patterns, and real-time merchant surge pricing.

---

### 3. Sample Dish / Menu Item Extensions
- **Limitation**: Dish-level menu records (e.g. *Chicken Dum Biryani*, *Paneer Butter Masala*) are synthetic sample extensions added to enrich the 8-restaurant source catalog.
- **Production Reality**: Commercial food delivery platforms index hundreds of millions of dynamic menu items, requiring real-time merchant inventory sync.

---

### 4. Single-Node In-Memory Vector Operations
- **Limitation**: Candidate dense retrieval uses in-memory NumPy matrix dot-product ($\mathbf{Q} \cdot \mathbf{C}^T$), which executes in <2ms for $N < 50,000$ items.
- **Production Reality**: Scaling to 40,000,000 restaurants requires geographic index partitioning and distributed FAISS HNSW / Milvus cluster sharding, as documented in **[docs/PRODUCTION.md](docs/PRODUCTION.md)**.

---

### 5. Non-Personalized Search Serving
- **Limitation**: The current search engine ranks candidates based on query text intent, hard filter constraints, restaurant quality, and dish matching without incorporating real-time user preference embeddings in the serving path.
- **Production Reality**: Production search systems blend user historical order affinities (e.g. vegetarian preference, price sensitivity, past restaurant order frequency) into the reranking stage.
