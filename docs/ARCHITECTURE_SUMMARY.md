# Architecture & Scope Summary

This document provides a concise 1-page overview of the implemented search engine components versus future production scaling blueprints.

---

## 1. Implemented System Scope vs. Future Blueprints

| Component Area | Local Implemented Reality | Future Production Blueprint (`docs/PRODUCTION.md`) |
| :--- | :--- | :--- |
| **Primary Domain** | **Constraint-Aware Food Search & Ranking** | Multi-tenant Search & Personalization Engine |
| **Query Parsing** | PyPI `search_expert` (0.8B SLM) + `RegexFallbackParser` | Triton / ONNX Inference Cluster |
| **Pre-Filtering** | Pre-filter hard price cutoffs & diet exclusions | Geographic Regional Candidate Partitions |
| **Candidate Retrieval** | In-Memory BM25 + Dense Vector Matrix (`MiniLM-L6-v2`) | Distributed Lucene + FAISS HNSW Vector Shards |
| **Rank Fusion** | Reciprocal Rank Fusion ($RRF = \frac{1}{60+r_b} + \frac{1}{60+r_d}$) | Multi-Retriever RRF Aggregator |
| **Learning-to-Rank** | `HistGradientBoostingRegressor` GBDT Ranker | Pairwise LambdaMART / Deep Learning Ranker |
| **Supervision** | Versioned simulated click logs with position bias | Real-time click-through & order conversion streams |
| **Recommendation** | Discussed only as a scaling extension | Two-Tower User-Restaurant Personalization |

---

## 2. Key Architectural Guarantees

- **100% Hard Filter Compliance**: Hard constraint rules (price cutoffs, dietary exclusions) execute prior to scoring, ensuring zero non-compliant results reach the candidate pool.
- **Circuit Breaker Availability**: If GPU LLM inference is unavailable or fails, `RegexFallbackParser` guarantees sub-millisecond query execution without downtime.
- **Scale-Invariant Retrieval**: Reciprocal Rank Fusion prevents raw BM25 lexical scores or dense cosine similarity scores from overpowering candidate selection.
- **Reproducibility**: Every model artifact, offline embedding matrix, and metric report is regenerated deterministically via `python build.py` (`SEED=42`).
