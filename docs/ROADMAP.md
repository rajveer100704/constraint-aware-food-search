# 10-Phase Project Roadmap & Implementation Status

## Implementation Matrix

| Phase | Milestone | Status | Key Deliverable |
| :---: | :--- | :---: | :--- |
| **1** | **Dense Vector Retrieval + RRF** | ✅ Done | Fused `sentence-transformers/all-MiniLM-L6-v2` + BM25 via Reciprocal Rank Fusion |
| **2** | **PyPI search-expert Integration** | ✅ Done | Imported `from search_expert import SearchExpert` with fallback circuit breaker |
| **3** | **Menu & Dish-Level Search** | ✅ Done | Catalog extended with dish items (*"Chicken Tikka Pizza"*, *"Paneer Butter Masala"*) |
| **4** | **Learning-to-Rank (LTR)** | ✅ Done | 7-feature vector logging & GBDT Ranker model (`models/ltr_model.pkl`) |
| **5** | **Multi-Metric Evaluation** | ✅ Done | Calculated Recall, Precision, MRR, NDCG, Constraint Satisfaction, & 5 Failure Buckets |
| **6** | **Experiment Tracking** | ✅ Done | One-command CLI (`python evaluate.py`) generating `metrics.json` & markdown reports |
| **7** | **Production Engineering** | ✅ Done | `Dockerfile`, `docker-compose.yml`, and GitHub Actions CI workflow |
| **8** | **API Refinement** | ✅ Done | FastAPI REST service (`GET /search`, `POST /search`, `POST /parse`, `GET /metrics`) |
| **9** | **Observability** | ✅ Done | Prometheus metrics exposition (`search_requests_total`, `search_latency_ms_avg`) |
| **10** | **YAML Config & Schemas** | ✅ Done | `config.yaml`, `schemas.py`, `performance.md`, and updated ADRs (0001–0005) |
