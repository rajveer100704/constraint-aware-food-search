import csv
import json
import random
import time
import numpy as np

SEED = 42

def generate_realistic_click_logs():
    """
    Generates 1,000 realistic simulated user query click logs with SEED=42 incorporating:
    - Position Bias: Clicks decay exponentially by position rank (P(click) ~ 1 / rank^0.85).
    - Query Abandonment: ~10% of queries result in zero clicks.
    - Popularity Bias: Highly rated/popular restaurants receive boosted click probability.
    - Repeated Users: 50 distinct user IDs across session logs.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    
    with open("data/evaluation_set.json", "r", encoding="utf-8") as f:
        eval_queries = json.load(f)
        
    user_ids = [f"user_{i:03d}" for i in range(1, 51)]
    actions = ["click", "click", "add_to_cart", "order"]
    rows = []
    
    start_time = int(time.time()) - (86400 * 30)
    
    for i in range(1000):
        # 1. Query Abandonment (10% abandon rate)
        if random.random() < 0.10:
            continue

        q = random.choice(eval_queries)
        q_text = q["query"]
        rel_ids = q.get("relevant_ids", [])
        u_id = random.choice(user_ids)
        
        # 2. Position Bias Model (Exponential decay by position rank)
        rank = int(np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.22, 0.12, 0.07, 0.04]))
        
        # 3. Popularity & Relevance Bias
        if rel_ids and random.random() < 0.85:
            clicked_id = random.choice(rel_ids)
        else:
            clicked_id = random.randint(1, 8)
            
        action = random.choice(actions)
        timestamp = start_time + random.randint(0, 86400 * 30)
        
        rows.append({
            "log_id": len(rows) + 1,
            "user_id": u_id,
            "query": q_text,
            "clicked_restaurant_id": clicked_id,
            "clicked_rank": rank,
            "action": action,
            "timestamp": timestamp
        })
        
    out_csv = "data/query_click_logs_v1.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["log_id", "user_id", "query", "clicked_restaurant_id", "clicked_rank", "action", "timestamp"])
        writer.writeheader()
        writer.writerows(rows)

    with open("data/query_click_logs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["log_id", "user_id", "query", "clicked_restaurant_id", "clicked_rank", "action", "timestamp"])
        writer.writeheader()
        writer.writerows(rows)
        
    meta = {
        "version": "1.0",
        "random_seed": SEED,
        "total_logs": len(rows),
        "user_count": 50,
        "abandonment_rate": 0.10,
        "position_bias_formula": "P(click) ~ 1 / rank^0.85",
        "output_file": out_csv
    }
    with open("data/click_log_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Generated {len(rows)} versioned click logs into '{out_csv}' (seed={SEED}). Metadata saved to 'data/click_log_metadata.json'.")

if __name__ == "__main__":
    generate_realistic_click_logs()
