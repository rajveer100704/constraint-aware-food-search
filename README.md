# 🍕 Constraint-Aware Hybrid Food Search Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://github.com/rajveer100704/constraint-aware-food-search/actions/workflows/ci.yml/badge.svg)](https://github.com/rajveer100704/constraint-aware-food-search/actions)

> A constraint-aware hybrid food search engine that combines lexical retrieval, dense vector retrieval, structured query parsing, and learning-to-rank over a multi-entity restaurant and dish catalog.

> **Topic Tags**: `search` · `information-retrieval` · `dense-retrieval` · `bm25` · `learning-to-rank` · `sentence-transformers` · `fastapi` · `machine-learning` · `recommendation-systems` · `retrieval-augmented`

---

## 📌 Overview

This project explores **constraint-aware food search** by combining lexical retrieval, dense vector retrieval, structured query parsing, and learning-to-rank over a multi-entity restaurant and dish catalog. *This project was inspired by engineering challenges commonly encountered in large-scale food search systems such as those used by food delivery platforms.*

It integrates Sarthak Rastogi's open-source **`search-expert`** PyPI package (`from search_expert import SearchExpert`) paired with a fast `RegexFallbackParser` circuit breaker. The engine indexes **Restaurant metadata and Menu Items / Dishes**, retrieves candidates via **Dense Vector Embeddings (`sentence-transformers/all-MiniLM-L6-v2`) + BM25 Lexical Search** using **Reciprocal Rank Fusion (RRF)**, and ranks candidates using a **Gradient Boosted Decision Tree (`HistGradientBoostingRegressor`)** Learning-to-Rank model trained on **versioned query click logs with position bias** (`data/query_click_logs_v1.csv`, `SEED=42`).

---

## 🏗️ System Architecture

```
User Query ("vegetarian thali under 250 with 4+ stars")
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Hybrid Query Parser (search_expert SLM -> Fallback) │
│    Extracts: domain="food", max_price="lt:250",         │
│              rating="gte:4.0", veg=True, ex="ne:mushroom"│
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Hard Constraint Pre-Filtering Engine                 │
│    Applies strict price cutoffs & diet exclusions       │
│    Guarantees 100% compliance over catalog candidates   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Sparse + Dense Retrieval & RRF Fusion Layer          │
│    - BM25 Lexical Retrieval (rank_bm25)                 │
│    - Dense Vector Retrieval (Precomputed MiniLM Matrix) │
│    - Reciprocal Rank Fusion: RRF = 1/(60+r_b) + 1/(60+r_d)│
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Dish Matching & GBDT Feature Reranking              │
│    Model: HistGradientBoostingRegressor (7 features)    │
│    Trained on 900+ click logs with Position Bias        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
                    Top-K Ranked Results
```

---

## 📊 Single Source of Truth Benchmark Results (`metrics.json`)

All evaluation metrics are calculated locally over **200 categorized offline food queries** (`data/evaluation_set_v2.json`, `SEED=42`):

| Evaluation Metric | Measured Value | Description |
| :--- | :---: | :--- |
| **Total Benchmark Queries** | **200** | Versioned 200-query benchmark dataset (`SEED=42`) |
| **Recall@5** | **0.7425** | Fraction of relevant items retrieved in top-5 |
| **Precision@5** | **0.1589** | Relevant items ratio over retrieved results |
| **MRR (Mean Reciprocal Rank)** | **0.6660** | Reciprocal rank of first relevant item |
| **NDCG@5** | **0.6853** | Normalized Discounted Cumulative Gain |
| **Constraint Satisfaction** | **100.0%** | Hard filter compliance across retrieved candidates |
| **Mean Query Latency (Cold)** | **60.97 ms** | Measured end-to-end latency on CPU |
| **Warm LRU Cache Latency** | **0.45 ms** | Measured latency on exact query cache hit (**99.3% reduction**) |
| **P50 SLA Latency** | **22.45 ms** | 50th Percentile query SLA |
| **P90 SLA Latency** | **24.95 ms** | 90th Percentile query SLA |
| **P95 SLA Latency** | **26.02 ms** | 95th Percentile query SLA |

---

## 🚀 One-Command Master Build Pipeline

To regenerate all preprocessed catalogs, versioned embeddings, GBDT models, evaluation metrics, and markdown reports from scratch, run:

```bash
python build.py
```

- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**: Full environment setup & step-by-step reproduction guide.
- **[RESULTS.md](RESULTS.md)**: Structured experimental results log & benchmark progression across phases.
- **[docs/TRADEOFFS.md](docs/TRADEOFFS.md)**: Engineering trade-offs & architecture rationale.
- **[docs/PRODUCTION.md](docs/PRODUCTION.md)**: 40 Million restaurant production scaling blueprint.
- **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)**: Technical boundaries & out-of-scope design choices.
- **[docs/FAILURE_GALLERY.md](docs/FAILURE_GALLERY.md)**: 5-case search failure analysis gallery.

---

## 🌐 FastAPI REST Microservice

```bash
uvicorn api:app --reload --port 8000
```

- `GET /health` - Healthcheck & catalog statistics.
- `GET /version` - Engine capability status.
- `GET /search?query=...&k=5` - Execute hybrid search with LRU caching.
- `POST /search` - Execute search with JSON request payload.
- `POST /parse` - Parse query constraints via SLM / fallback.
- `GET /metrics` - Prometheus metrics exposition.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
