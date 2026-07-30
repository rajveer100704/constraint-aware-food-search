import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from search_engine import CATALOG, get_bm25_index, get_searchable_text
from evaluate import recall_at_k, precision_at_k, mrr, ndcg_at_k

def run_benchmark():
    print("--- Running Retrieval Method Comparison: BM25 vs. TF-IDF Baseline ---")
    
    eval_path = "data/evaluation_set_v2.json" if os.path.exists("data/evaluation_set_v2.json") else "data/evaluation_set.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    eval_queries = data.get("queries", data) if isinstance(data, dict) else data
        
    corpus = [get_searchable_text(r) for r in CATALOG]
    vectorizer = TfidfVectorizer().fit(corpus)
    tfidf_matrix = vectorizer.transform(corpus)
    
    bm25 = get_bm25_index(CATALOG)
    
    bm25_recalls, bm25_precisions, bm25_mrrs, bm25_ndcgs = [], [], [], []
    tfidf_recalls, tfidf_precisions, tfidf_mrrs, tfidf_ndcgs = [], [], [], []
    
    for q in eval_queries:
        q_text = q["query"]
        rel_ids = q["relevant_ids"]
        
        # 1. BM25 Scores
        b_scores = bm25.get_scores(q_text.lower().split())
        b_ranked = [r.id for r, s in sorted(zip(CATALOG, b_scores), key=lambda x: x[1], reverse=True)[:5]]
        
        bm25_recalls.append(recall_at_k(b_ranked, rel_ids))
        bm25_precisions.append(precision_at_k(b_ranked, rel_ids))
        bm25_mrrs.append(mrr(b_ranked, rel_ids))
        bm25_ndcgs.append(ndcg_at_k(b_ranked, rel_ids, k=5))
        
        # 2. TF-IDF Scores
        q_vec = vectorizer.transform([q_text.lower()])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        t_ranked = [r.id for r, s in sorted(zip(CATALOG, sims), key=lambda x: x[1], reverse=True)[:5]]
        
        tfidf_recalls.append(recall_at_k(t_ranked, rel_ids))
        tfidf_precisions.append(precision_at_k(t_ranked, rel_ids))
        tfidf_mrrs.append(mrr(t_ranked, rel_ids))
        tfidf_ndcgs.append(ndcg_at_k(t_ranked, rel_ids, k=5))
        
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    
    markdown_content = (
        "# Retrieval Method Comparison Report\n\n"
        "Comparative evaluation of retrieval algorithms over 200 categorized food search queries (`data/evaluation_set_v2.json`):\n\n"
        "| Retrieval Strategy | Recall@5 | Precision@5 | MRR | NDCG@5 |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        f"| **BM25 Lexical** | **{avg(bm25_recalls):.4f}** | **{avg(bm25_precisions):.4f}** | **{avg(bm25_mrrs):.4f}** | **{avg(bm25_ndcgs):.4f}** |\n"
        f"| TF-IDF Baseline | {avg(tfidf_recalls):.4f} | {avg(tfidf_precisions):.4f} | {avg(tfidf_mrrs):.4f} | {avg(tfidf_ndcgs):.4f} |\n"
    )
    
    with open("benchmark_results.md", "w") as f:
        f.write(markdown_content)
        
    print(f"Benchmark complete across {len(eval_queries)} queries. Report saved to benchmark_results.md\n")

if __name__ == "__main__":
    run_benchmark()
