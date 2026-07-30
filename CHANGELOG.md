# Changelog

All notable changes to the Constraint-Aware Hybrid Food Search Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-30

### Added
- **Hybrid Retrieval Stack**: BM25 lexical search (`rank_bm25`) + Dense vector retrieval (`sentence-transformers/all-MiniLM-L6-v2`) combined via Reciprocal Rank Fusion (RRF).
- **Query Parser Circuit Breaker**: Real PyPI `search_expert` Small Language Model integration paired with `RegexFallbackParser` for sub-ms resilience.
- **Offline Embedding Persistence**: Versioned dense vector array (`data/restaurant_embeddings_v1.npy`) and checksum metadata (`data/embedding_metadata.json`).
- **Learning-to-Rank GBDT Model**: `HistGradientBoostingRegressor` trained on simulated query click logs (`data/query_click_logs_v1.csv`, `SEED=42`) incorporating exponential position bias ($P(\text{click}) \sim \text{rank}^{-0.85}$).
- **Evaluation Benchmark Suite**: 200 categorized evaluation queries across 10 intent buckets (`data/evaluation_set_v2.json`, `SEED=42`).
- **Automated Build Pipeline**: One-command build script (`python build.py`) executing preprocessing, click-log generation, model training, evaluation, profiling, benchmarks, and pytest.
- **Architecture Documentation**: Comprehensive documentation suite (`docs/DESIGN.md`, `docs/TRADEOFFS.md`, `docs/LIMITATIONS.md`, `docs/PRODUCTION.md`, `docs/FAILURE_GALLERY.md`, `docs/ARCHITECTURE_SUMMARY.md`, `RESULTS.md`, `REPRODUCIBILITY.md`).
- **FastAPI REST Microservice**: REST endpoints (`GET /health`, `GET /version`, `GET /search`, `POST /search`, `POST /parse`, `GET /metrics`) with LRU caching.
