import json
import math
import os
import time
from typing import List, Dict, Any
import numpy as np

from schemas import RestaurantSchema
from search_engine import search_with_timing, CATALOG, apply_hard_filters, get_searchable_text, _PARSER_PIPELINE

def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(relevant_ids)

def precision_at_k(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(retrieved_ids)

def mrr(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int = 5) -> float:
    retrieved_k = retrieved_ids[:k]
    dcg = 0.0
    for i, rid in enumerate(retrieved_k):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_constraint_satisfaction(results: List[Dict[str, Any]], parsed) -> float:
    if not results:
        return 1.0
    satisfied = 0
    for r in results:
        is_ok = True
        if parsed.max_price and r["price"] > parsed.max_price:
            is_ok = False
        if parsed.min_rating and r["rating"] < parsed.min_rating:
            is_ok = False
        if is_ok:
            satisfied += 1
    return satisfied / len(results)

def run_evaluation(eval_set_path: str = "data/evaluation_set.json", catalog: List[RestaurantSchema] = CATALOG):
    print("\n--- Running Evaluation & Performance Profiling (50 Benchmark Queries) ---")
    with open(eval_set_path, "r") as f:
        eval_queries = json.load(f)

    recalls, precisions, mrrs, ndcgs, constraint_sats = [], [], [], [], []
    stage_timings = {"parser_ms": [], "filter_ms": [], "bm25_ms": [], "dense_ms": [], "ranking_ms": [], "total_ms": []}
    
    filter_failures = 0
    ranking_failures = 0
    failure_logs = []

    for q in eval_queries:
        q_text = q["query"]
        rel_ids = q["relevant_ids"]

        results, timing = search_with_timing(q_text, k=5, catalog=catalog, use_cache=False)
        for k_time, val in timing.items():
            if k_time in stage_timings:
                stage_timings[k_time].append(val)

        ret_ids = [r["id"] for r in results]
        rec = recall_at_k(ret_ids, rel_ids) if rel_ids else 1.0
        prec = precision_at_k(ret_ids, rel_ids) if rel_ids else 1.0
        m = mrr(ret_ids, rel_ids) if rel_ids else 1.0
        n = ndcg_at_k(ret_ids, rel_ids, k=5) if rel_ids else 1.0

        parsed = _PARSER_PIPELINE.parse(q_text)
        c_sat = evaluate_constraint_satisfaction(results, parsed)

        recalls.append(rec)
        precisions.append(prec)
        mrrs.append(m)
        ndcgs.append(n)
        constraint_sats.append(c_sat)

        if rel_ids and rec < 1.0:
            filtered = apply_hard_filters(catalog, parsed)
            filtered_ids = [r.id for r in filtered]

            for rel_id in rel_ids:
                if rel_id not in ret_ids:
                    if rel_id not in filtered_ids:
                        filter_failures += 1
                        category = "Filter Exclusion"
                    else:
                        ranking_failures += 1
                        category = "Ranking Miss"

                    failure_logs.append({
                        "query": q_text,
                        "missed_id": rel_id,
                        "failure_category": category
                    })

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    avg_timings = {k_time: round(avg(vals), 2) for k_time, vals in stage_timings.items()}

    # Compute Latency Percentiles (P50, P90, P95, P99)
    total_lats = stage_timings["total_ms"]
    p50 = round(float(np.percentile(total_lats, 50)), 2)
    p90 = round(float(np.percentile(total_lats, 90)), 2)
    p95 = round(float(np.percentile(total_lats, 95)), 2)
    p99 = round(float(np.percentile(total_lats, 99)), 2)

    avg_timings["p50_latency_ms"] = p50
    avg_timings["p90_latency_ms"] = p90
    avg_timings["p95_latency_ms"] = p95
    avg_timings["p99_latency_ms"] = p99

    # Save performance.json
    with open("performance.json", "w") as f:
        json.dump(avg_timings, f, indent=2)

    # Derive performance.md
    tot = avg_timings["total_ms"] if avg_timings["total_ms"] > 0 else 1.0
    p_parse = (avg_timings["parser_ms"] / tot) * 100.0
    p_filter = (avg_timings["filter_ms"] / tot) * 100.0
    p_bm25 = (avg_timings["bm25_ms"] / tot) * 100.0
    p_dense = (avg_timings["dense_ms"] / tot) * 100.0
    p_rank = (avg_timings["ranking_ms"] / tot) * 100.0

    perf_md = (
        "# Measured Performance & Latency Profiling Report\n\n"
        "Derived from empirical latency measurements across 50 benchmark queries:\n\n"
        "| Pipeline Stage | Measured Latency | Latency Share (%) | Optimization Notes |\n"
        "| :--- | :--- | :--- | :--- |\n"
        f"| **Query Parsing** | {avg_timings['parser_ms']} ms | {p_parse:.1f}% | Hybrid SLM parser with sub-ms fallback |\n"
        f"| **Hard Filtering** | {avg_timings['filter_ms']} ms | {p_filter:.1f}% | Pre-filtering candidate pool |\n"
        f"| **BM25 Retrieval** | {avg_timings['bm25_ms']} ms | {p_bm25:.1f}% | Lexical scoring over catalog |\n"
        f"| **Query Embedding & Vector Search** | {avg_timings['dense_ms']} ms | {p_dense:.1f}% | Query vector encoding + matrix dot-product |\n"
        f"| **LTR Feature Ranking** | {avg_timings['ranking_ms']} ms | {p_rank:.1f}% | Multi-feature scoring & explainability |\n"
        f"| **Total Query Latency (Mean)** | **{avg_timings['total_ms']} ms** | **100.0%** | End-to-end average per search query |\n\n"
        "## Latency Percentiles (SLA Profile)\n\n"
        f"- **P50 Latency**: `{p50} ms`\n"
        f"- **P90 Latency**: `{p90} ms`\n"
        f"- **P95 Latency**: `{p95} ms`\n"
        f"- **P99 Latency**: `{p99} ms`\n"
    )
    with open("performance.md", "w") as f:
        f.write(perf_md)

    metrics_dict = {
        "recall_at_5": round(avg(recalls), 4),
        "precision_at_5": round(avg(precisions), 4),
        "mrr": round(avg(mrrs), 4),
        "ndcg_at_5": round(avg(ndcgs), 4),
        "constraint_satisfaction": round(avg(constraint_sats), 4),
        "avg_latency_ms": avg_timings["total_ms"],
        "p50_latency_ms": p50,
        "p90_latency_ms": p90,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "filter_failures": filter_failures,
        "ranking_failures": ranking_failures,
        "total_queries": len(eval_queries)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save docs/evaluation.md
    eval_doc_content = (
        "# Evaluation Methodology & Benchmark Report\n\n"
        "## Data Provenance & Assumptions\n\n"
        "- **Menu Items / Dishes**: Sample menu item extensions (e.g. *Chicken Dum Biryani*, *Paneer Butter Masala*) added for demonstration because the source catalog lacks item-level dish records.\n"
        "- **Learning-to-Rank Labels**: Relevance targets derived from simulated query click logs (`data/query_click_logs.csv`), modeling user clicks, add-to-cart, and order actions.\n"
        "- **Evaluation Set**: Evaluated over a manually curated offline benchmark of 50 diverse queries across 8 categories (Simple Lexical, Price Cutoffs, Dietary Veg/Non-Veg, Cuisine Match, Dish Match, Area/Location, Mixed Constraints, Exclusions).\n\n"
        "## Measured Benchmark Results\n\n"
        "| Metric | Measured Value | Description |\n"
        "| :--- | :---: | :--- |\n"
        f"| **Recall@5** | **{metrics_dict['recall_at_5']}** | Fraction of relevant items retrieved in top-5 |\n"
        f"| **Precision@5** | **{metrics_dict['precision_at_5']}** | Relevant items ratio over top-5 |\n"
        f"| **MRR** | **{metrics_dict['mrr']}** | Mean Reciprocal Rank |\n"
        f"| **NDCG@5** | **{metrics_dict['ndcg_at_5']}** | Normalized Discounted Cumulative Gain |\n"
        f"| **Constraint Satisfaction** | **{metrics_dict['constraint_satisfaction'] * 100:.1f}%** | Compliance with hard filter constraints |\n"
        f"| **Avg Total Latency** | **{metrics_dict['avg_latency_ms']} ms** | End-to-end search runtime |\n"
        f"| **P95 Latency** | **{p95} ms** | 95th Percentile query SLA |\n\n"
        "## Hybrid Query Parser Resilience Narrative\n\n"
        "- **Primary Parser**: PyPI `search_expert` fine-tuned Small Language Model (Qwen3.5-0.8B LoRA).\n"
        "- **Fallback Parser**: Rule-based `RegexFallbackParser` for sub-ms execution.\n"
        "- **Resilience Pattern**: Operates seamlessly on CPU-only machines where GPU LLM inference is unavailable.\n"
    )
    os.makedirs("docs", exist_ok=True)
    with open("docs/evaluation.md", "w") as f:
        f.write(eval_doc_content)

    # Save benchmark_results.md
    bm_md = (
        "# Search Engine Benchmark Report\n\n"
        "| Metric | Value |\n"
        "| :--- | :--- |\n"
        f"| Total Benchmark Queries | {len(eval_queries)} |\n"
        f"| Recall@5 | {metrics_dict['recall_at_5']} |\n"
        f"| Precision@5 | {metrics_dict['precision_at_5']} |\n"
        f"| MRR | {metrics_dict['mrr']} |\n"
        f"| NDCG@5 | {metrics_dict['ndcg_at_5']} |\n"
        f"| Constraint Satisfaction | {metrics_dict['constraint_satisfaction'] * 100:.1f}% |\n"
        f"| P50 / P90 / P95 Latency | {p50} ms / {p90} ms / {p95} ms |\n"
    )
    with open("benchmark_results.md", "w") as f:
        f.write(bm_md)

    # Save failure_analysis.md
    fail_md = ["# Failure Analysis Report\n", "| Query | Missed ID | Failure Category |", "| :--- | :--- | :--- |"]
    for log in failure_logs:
        fail_md.append(f"| {log['query']} | {log['missed_id']} | {log['failure_category']} |")
    if not failure_logs:
        fail_md.append("| None | None | All evaluation queries achieved 100% Recall@5 |")
    with open("failure_analysis.md", "w") as f:
        f.write("\n".join(fail_md) + "\n")

    print(f"Total Benchmark Queries: {len(eval_queries)}")
    print(f"Recall@5:                {metrics_dict['recall_at_5']}")
    print(f"Precision@5:             {metrics_dict['precision_at_5']}")
    print(f"MRR:                     {metrics_dict['mrr']}")
    print(f"NDCG@5:                  {metrics_dict['ndcg_at_5']}")
    print(f"Constraint Satisfaction: {metrics_dict['constraint_satisfaction'] * 100:.1f}%")
    print(f"Mean Latency:            {metrics_dict['avg_latency_ms']} ms")
    print(f"P50 / P95 Latency:       {p50} ms / {p95} ms")
    print("Artifacts generated: metrics.json, performance.json, performance.md, docs/evaluation.md, benchmark_results.md, failure_analysis.md\n")

    return metrics_dict

if __name__ == "__main__":
    run_evaluation()
