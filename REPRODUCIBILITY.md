# Reproducibility Guide

This document provides step-by-step instructions to rebuild the entire repository pipeline, models, vector embeddings, and evaluation reports from raw data.

---

## 1. Prerequisites & Environment Setup

- **Python Version**: Python 3.9+ (Tested on Python 3.11 & 3.14)
- **Dependencies**: Listed in `requirements.txt` and `pyproject.toml`

```bash
# Clone repository
git clone https://github.com/rajveer/swiggy-hybrid-search-engine.git
cd swiggy-hybrid-search-engine

# Install dependencies
pip install -r requirements.txt
```

---

## 2. One-Command Master Build Pipeline

To regenerate all preprocessed catalogs, versioned embeddings, GBDT models, evaluation metrics, and markdown reports from scratch, execute:

```bash
python build.py
```

### Execution Lifecycle

`build.py` runs the following sequential pipeline:
1. `data/preprocess.py` $\rightarrow$ Generates `processed_catalog.json`, `restaurant_embeddings_v1.npy`, and `embedding_metadata.json`.
2. `train_ltr.py` $\rightarrow$ Logs 7 feature vectors, generates target labels, and trains `models/ltr_model.pkl` (`HistGradientBoostingRegressor`).
3. `evaluate.py` $\rightarrow$ Computes offline benchmark metrics, measures stage latencies, and writes `metrics.json`, `performance.json`, `performance.md`, `benchmark_results.md`, `failure_analysis.md`, and `docs/evaluation.md`.
4. `ablation.py` $\rightarrow$ Runs component ablation matrix generating `ablation_results.md`.
5. `benchmark_bm25_vs_tfidf.py` $\rightarrow$ Runs lexical baseline comparison.
6. `pytest` $\rightarrow$ Runs full unit & integration test suite (22 tests).

---

## 3. Expected Artifact Outputs

Upon successful completion of `python build.py`, the following files will be populated:

| Output Artifact | Description |
| :--- | :--- |
| `data/restaurant_embeddings_v1.npy` | Precomputed 384-dimensional dense vector embeddings matrix |
| `data/embedding_metadata.json` | Model ID, dimension, checksum, and creation timestamp |
| `models/ltr_model.pkl` | Serialized GBDT Ranker model |
| `metrics.json` | Structured evaluation metrics (Recall@5, Precision@5, MRR, NDCG@5, Constraint Sat) |
| `performance.json` | Granular stage latencies in milliseconds |
| `performance.md` | Derived latency profiling report with stage percentages |
| `docs/evaluation.md` | Methodology, data provenance, and benchmark reports |

---

## 4. Running the FastAPI Microservice

```bash
uvicorn api:app --reload --port 8000
```
- Healthcheck: `http://localhost:8000/health`
- Version & Capabilities: `http://localhost:8000/version`
- Interactive API Docs: `http://localhost:8000/docs`
