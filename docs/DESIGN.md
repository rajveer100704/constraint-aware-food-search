# System Architecture & Design Specification (v3.1.0)

## Overview

The **Swiggy Food Delivery Hybrid Search Engine** is a constraint-aware retrieval and recommendation system combining:
1. PyPI `search-expert` (Qwen3.5-0.8B LoRA SLM) structured query extraction with regex circuit breaker.
2. Hard pre-filtering for 100% constraint satisfaction.
3. Dense Vector Embeddings (`sentence-transformers/all-MiniLM-L6-v2`) + BM25 Lexical Retrieval fused via Reciprocal Rank Fusion (RRF).
4. Menu & Dish-Item level search index matching.
5. Learning-to-Rank (LTR) GBDT model scoring over 7 normalized features.

## Architecture Flow Diagram

```
User Query ("vegetarian thali under 250")
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Hybrid Query Parser (search_expert -> Fallback)     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Hard Pre-Filtering Engine (Price, Rating, Veg, Ex)   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Reciprocal Rank Fusion (BM25 + Dense MiniLM)         │
│    RRF = 1/(60 + r_bm25) + 1/(60 + r_dense)            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Dish Matching & GBDT Feature Reranking              │
└─────────────────────────────────────────────────────────┘
```

## Explicit Non-Goals & Engineering Honesty

| Non-Goal | Reason for Exclusion |
| :--- | :--- |
| **Kubernetes / Distributed Mesh** | Single Docker container minimizes deployment overhead for portfolio scope. |
| **Invented SLA Targets** | Latency is measured empirically (`avg_latency_ms: 404ms` on CPU), not guessed. |
| **Production Target Labels** | LTR training target labels are generated transparently and logged as synthetic targets for pipeline demonstration. |
