# Measured Performance & Latency Profiling Report

Derived from empirical latency measurements across 50 benchmark queries:

| Pipeline Stage | Measured Latency | Latency Share (%) | Optimization Notes |
| :--- | :--- | :--- | :--- |
| **Query Parsing** | 0.05 ms | 0.1% | Hybrid SLM parser with sub-ms fallback |
| **Hard Filtering** | 0.01 ms | 0.0% | Pre-filtering candidate pool |
| **BM25 Retrieval** | 0.19 ms | 0.3% | Lexical scoring over catalog |
| **Query Embedding & Vector Search** | 60.25 ms | 98.8% | Query vector encoding + matrix dot-product |
| **LTR Feature Ranking** | 0.46 ms | 0.8% | Multi-feature scoring & explainability |
| **Total Query Latency (Mean)** | **60.97 ms** | **100.0%** | End-to-end average per search query |

## Latency Percentiles (SLA Profile)

- **P50 Latency**: `22.45 ms`
- **P90 Latency**: `24.95 ms`
- **P95 Latency**: `26.02 ms`
- **P99 Latency**: `29.02 ms`
