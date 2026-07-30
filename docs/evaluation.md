# Evaluation Methodology & Benchmark Report

## Data Provenance & Assumptions

- **Menu Items / Dishes**: Sample menu item extensions (e.g. *Chicken Dum Biryani*, *Paneer Butter Masala*) added for demonstration because the source catalog lacks item-level dish records.
- **Learning-to-Rank Labels**: Relevance targets derived from simulated query click logs (`data/query_click_logs_v1.csv`, `SEED=42`), modeling user clicks, add-to-cart, order actions, and exponential position decay bias.
- **Evaluation Set**: Evaluated over a versioned offline benchmark (`data/evaluation_set_v2.json`, `SEED=42`) of 200 diverse queries across 10 intent categories (Lexical, Semantic, Mixed Constraints, Typos, Negations, Price Limits, Cuisine, Location, Dish Names, Multi-Intent).

## Single Source of Truth Benchmark Results (`metrics.json`)

| Metric | Measured Value | Description |
| :--- | :---: | :--- |
| **Total Benchmark Queries** | **200** | Versioned 200-query benchmark dataset (`SEED=42`) |
| **Recall@5** | **0.7425** | Fraction of relevant items retrieved in top-5 |
| **Precision@5** | **0.1589** | Relevant items ratio over top-5 |
| **MRR (Mean Reciprocal Rank)** | **0.6660** | Reciprocal rank of first relevant item |
| **NDCG@5** | **0.6853** | Normalized Discounted Cumulative Gain |
| **Constraint Satisfaction** | **100.0%** | Compliance with hard filter constraints |
| **Mean Total Latency** | **60.97 ms** | End-to-end search runtime |
| **P50 SLA Latency** | **22.45 ms** | 50th Percentile query SLA |
| **P90 SLA Latency** | **24.95 ms** | 90th Percentile query SLA |
| **P95 SLA Latency** | **26.02 ms** | 95th Percentile query SLA |
| **P99 SLA Latency** | **29.02 ms** | 99th Percentile query SLA |

## Hybrid Query Parser Resilience Narrative

- **Primary Parser**: PyPI `search_expert` fine-tuned Small Language Model (Qwen3.5-0.8B LoRA).
- **Fallback Parser**: Rule-based `RegexFallbackParser` for sub-ms execution.
- **Resilience Pattern**: Operates seamlessly on CPU-only machines where GPU LLM inference is unavailable.
