import os
import csv
import json
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from search_engine import (
    load_catalog, CATALOG, apply_hard_filters,
    dense_scores, compute_rrf_scores, find_matching_dish,
    compute_features, get_bm25_index, _PARSER_PIPELINE
)

def load_click_log_targets(click_logs_path: str = "data/query_click_logs_v1.csv") -> dict:
    """
    Parses simulated query click logs and computes empirical click-through targets
    y = sum(action_weight / log2(rank + 1)) per (query, restaurant_id) pair.
    """
    targets = {}
    if not os.path.exists(click_logs_path):
        click_logs_path = "data/query_click_logs.csv"
    if not os.path.exists(click_logs_path):
        return targets
        
    action_weights = {"click": 1.0, "add_to_cart": 2.5, "order": 4.0}
    
    with open(click_logs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["query"]
            rid = int(row["clicked_restaurant_id"])
            rank = int(row["clicked_rank"])
            action = row["action"]
            
            weight = action_weights.get(action, 1.0) / np.log2(rank + 1)
            targets.setdefault((q, rid), 0.0)
            targets[(q, rid)] += weight
            
    return targets

def build_ltr_dataset(catalog=None):
    """
    Step 1: Feature Logging & Click Log Relevance Target Generation.
    Logs 7-dimensional feature vectors per (query, restaurant) pair and computes
    relevance targets derived from simulated user query click logs.
    """
    if catalog is None:
        catalog = CATALOG

    click_targets = load_click_log_targets()
    
    eval_path = "data/evaluation_set_v2.json" if os.path.exists("data/evaluation_set_v2.json") else "data/evaluation_set.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)
        
    training_queries = [q["query"] for q in eval_queries]

    X = []
    y = []

    for q_str in training_queries:
        parsed = _PARSER_PIPELINE.parse(q_str)
        
        bm25_idx = get_bm25_index(catalog)
        bm25_raw = list(bm25_idx.get_scores(q_str.lower().split()))
        dense_raw = dense_scores(q_str, catalog)

        bm25_ranks = (np.argsort(np.argsort(-np.array(bm25_raw))) + 1).tolist()
        dense_ranks = (np.argsort(np.argsort(-np.array(dense_raw))) + 1).tolist()
        rrf_raw = compute_rrf_scores(bm25_ranks, dense_ranks)

        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        max_rrf = max(rrf_raw) if max(rrf_raw) > 0 else 1.0

        for idx, item in enumerate(catalog):
            matching_dish = find_matching_dish(item, q_str)
            feats = compute_features(
                bm25_score=bm25_raw[idx],
                max_bm25=max_bm25,
                dense_rrf_score=rrf_raw[idx],
                max_rrf=max_rrf,
                item=item,
                parsed=parsed,
                catalog=catalog,
                matching_dish=matching_dish
            )

            feature_vec = [
                feats["bm25"],
                feats["dense_rrf"],
                feats["rating"],
                feats["price"],
                feats["cuisine"],
                feats["popularity"],
                feats["dish_match"]
            ]

            # Click log target lookup with constraint fallback
            click_weight = click_targets.get((q_str, item.id), 0.0)
            target = click_weight
            if target == 0.0:
                if feats["cuisine"] > 0 or feats["dish_match"] > 0:
                    target += 2.0
                if parsed.max_price and item.price <= parsed.max_price:
                    target += 1.0

            X.append(feature_vec)
            y.append(target)

    return np.array(X), np.array(y)

def train_ltr_model():
    print("--- Phase 4: Training Learning-to-Rank (LTR) GBDT Model on Query Click Logs ---")
    X, y = build_ltr_dataset()
    print(f"Dataset generated from query click logs: {X.shape[0]} samples, {X.shape[1]} features.")

    model = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    model_path = "models/ltr_model.pkl"
    joblib.dump(model, model_path)
    print(f"LTR GBDT Model successfully trained and saved to '{model_path}'.\n")
    return model

if __name__ == "__main__":
    train_ltr_model()
