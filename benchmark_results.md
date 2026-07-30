# Comparative Retrieval & Cache Benchmark Report

Comparative evaluation over 200 categorized food search queries:

| Retrieval Pipeline | Recall@5 | NDCG@5 | Cold Latency (ms) | Warm Cache Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 Lexical Only** | 0.9675 | 0.8864 | 0.15 ms | 0.15 ms |
| **Hybrid RRF + GBDT Ranker** | **0.7425** | **0.6853** | **71.09 ms** | **29.06 ms** |

## Cache Effectiveness Experiment

- **Cold Query Latency (Mean)**: `71.09 ms`
- **Warm Query Latency (LRU Cache Hit)**: `29.06 ms`
- **Measured Latency Reduction**: **59.1%**
