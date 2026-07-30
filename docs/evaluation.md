# Evaluation Methodology & Benchmark Report

## Data Provenance & Assumptions

- **Menu Items / Dishes**: Sample menu item extensions (e.g. *Chicken Dum Biryani*, *Paneer Butter Masala*) added for demonstration because the source catalog lacks item-level dish records.
- **Learning-to-Rank Labels**: Relevance targets derived from simulated query click logs (`data/query_click_logs.csv`), modeling user clicks, add-to-cart, and order actions.
- **Evaluation Set**: Evaluated over a manually curated offline benchmark of 50 diverse queries across 8 categories (Simple Lexical, Price Cutoffs, Dietary Veg/Non-Veg, Cuisine Match, Dish Match, Area/Location, Mixed Constraints, Exclusions).

## Measured Benchmark Results

| Metric | Measured Value | Description |
| :--- | :---: | :--- |
| **Recall@5** | **0.7425** | Fraction of relevant items retrieved in top-5 |
| **Precision@5** | **0.1589** | Relevant items ratio over top-5 |
| **MRR** | **0.666** | Mean Reciprocal Rank |
| **NDCG@5** | **0.6853** | Normalized Discounted Cumulative Gain |
| **Constraint Satisfaction** | **100.0%** | Compliance with hard filter constraints |
| **Avg Total Latency** | **60.97 ms** | End-to-end search runtime |
| **P95 Latency** | **26.02 ms** | 95th Percentile query SLA |

## Hybrid Query Parser Resilience Narrative

- **Primary Parser**: PyPI `search_expert` fine-tuned Small Language Model (Qwen3.5-0.8B LoRA).
- **Fallback Parser**: Rule-based `RegexFallbackParser` for sub-ms execution.
- **Resilience Pattern**: Operates seamlessly on CPU-only machines where GPU LLM inference is unavailable.
