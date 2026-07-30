# Experimental Results Log & Benchmark Progression

This document tracks the experimental progression of the Swiggy Food Delivery Search Engine across dataset versions, query benchmark sizes, and model configurations.

> **Note on Benchmark Comparison**: The early 15-query benchmark served as an initial sanity check on exact-match terms and should not be compared directly with the more diverse, multi-intent 200-query benchmark dataset (`evaluation_set_v2.json`).

---

### Benchmark Phase 1: Initial Lexical Retrieval Sanity Check
- **Dataset**: `evaluation_set_v1.json` (15 manually curated sanity-check queries)
- **Pipeline Config**: Lexical Retrieval Baseline (No Dense Embeddings, No RRF, No LTR Ranker)
- **Measured Metrics**:
  - `BM25Okapi`: Recall@5 = **0.9667**, NDCG@5 = **0.9496**
  - `TfidfVectorizer`: Recall@5 = **0.9333**, NDCG@5 = **0.9087**
- **Observations**: BM25's term frequency saturation and document length normalization prevent long restaurant descriptions from artificially overpowering lexical rankings.
- **Decision**: Adopt BM25Okapi as the core lexical retriever ([ADR 0001](docs/ADR/0001-bm25-vs-tfidf.md)).

---

### Benchmark Phase 2: Hybrid Retrieval & Constraint Compliance
- **Dataset**: 50-query constraint benchmark
- **Pipeline Config**: Lexical BM25 + Dense Vector (`all-MiniLM-L6-v2`) + Reciprocal Rank Fusion ($RRF = \frac{1}{60 + r_{bm25}} + \frac{1}{60 + r_{dense}}$)
- **Measured Metrics**:
  - BM25 Only: High keyword precision, fails on zero lexical overlap ("spicy noodles" $\rightarrow$ "Schezwan Hakka Noodles").
  - Dense Only: High semantic recall, fails on strict price cutoffs.
  - Hybrid RRF: Achieves **0.9500 Recall@5** and **100% Hard Constraint Compliance**.
- **Observations**: Scale-invariant rank fusion normalizes score distributions across lexical and dense retrievers.
- **Decision**: Adopt Hybrid RRF retrieval stack ([ADR 0003](docs/ADR/0003-hybrid-retrieval-and-ranking-config.md)).

---

### Benchmark Phase 3: Final Production Stack (Current Repository State)
- **Dataset**: `evaluation_set_v2.json` (200 categorized queries across 10 intent buckets, `SEED=42`)
- **Supervision**: `query_click_logs_v1.csv` (893 simulated user click logs with position decay bias $P(\text{click}) \sim \text{rank}^{-0.85}$ and 10% query abandonment, `SEED=42`)
- **Pipeline Config**: PyPI `search_expert` SLM + Fallback Circuit Breaker + Precomputed Dense Matrix (`restaurant_embeddings_v1.npy`) + BM25 RRF + `HistGradientBoostingRegressor` GBDT Ranker + LRU Query Cache.
- **Single Source of Truth Metrics (`metrics.json`)**:
  - **Total Benchmark Queries**: `200`
  - **Recall@5**: `0.7425`
  - **NDCG@5**: `0.6853`
  - **Constraint Satisfaction**: `100.0%`
  - **Mean Latency (Cold)**: `69.96 ms`
  - **Warm LRU Cache Latency**: `0.45 ms` (**99.3% Latency Reduction**)
  - **P50 SLA Latency**: `28.06 ms`
  - **P95 SLA Latency**: `41.32 ms`
- **Observations**: The 200-query dataset tests complex multi-intent, typo, negation, and boundary constraints, providing a realistic evaluation distribution.
- **Decision**: Freeze current architecture and maintain single source of truth across all repository reports.
