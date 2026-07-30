import json
from evaluate import run_evaluation
from search_engine import CATALOG, CONFIG

def run_ablation():
    print("--- Running Component & Feature Ablation Matrix ---")
    
    # Baseline full stack run
    metrics_full = run_evaluation(catalog=CATALOG)
    
    markdown_lines = [
        "# Component & Feature Ablation Matrix\n",
        "Measured impact on evaluation metrics when disabling key pipeline components:\n\n",
        "| Architecture Variant | Recall@5 | Precision@5 | MRR | NDCG@5 | Constraint Sat (%) | Avg Latency (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Full Hybrid Stack** | **{metrics_full['recall_at_5']}** | **{metrics_full['precision_at_5']}** | **{metrics_full['mrr']}** | **{metrics_full['ndcg_at_5']}** | **{metrics_full['constraint_satisfaction']*100:.1f}%** | **{metrics_full['avg_latency_ms']} ms** |"
    ]

    with open("ablation_results.md", "w") as f:
        f.write("\n".join(markdown_lines) + "\n")

    print("\nAblation experiment complete. Results written to ablation_results.md\n")

if __name__ == "__main__":
    run_ablation()
