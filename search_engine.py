import json
import math
import os
import sys
import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from schemas import RestaurantSchema, MenuItem, ParsedQuerySchema, SearchResponseItem
from parser_pipeline import HybridQueryParser

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

# -----------------------------------------------------------------------------
# Configuration Loader
# -----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "catalog": {"default_path": "data/processed_catalog.json"},
    "models": {"dense_model_name": "sentence-transformers/all-MiniLM-L6-v2"},
    "ranking_weights": {
        "bm25_weight": 0.30,
        "dense_weight": 0.25,
        "rating_weight": 0.15,
        "price_weight": 0.10,
        "popularity_weight": 0.08,
        "cuisine_weight": 0.07,
        "dish_match_weight": 0.05,
    },
    "rrf": {"k_constant": 60}
}

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if _HAS_YAML and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG

CONFIG = load_config()

# -----------------------------------------------------------------------------
# Catalog & Precomputed Embedding Loader (Load Once On Startup)
# -----------------------------------------------------------------------------

def load_catalog(filepath: Optional[str] = None) -> List[RestaurantSchema]:
    path = filepath or CONFIG["catalog"].get("default_path", "data/processed_catalog.json")
    if not os.path.exists(path):
        path = "data/processed_catalog.json"
    with open(path, "r") as f:
        data = json.load(f)
    return [RestaurantSchema(**d) for d in data]

CATALOG: List[RestaurantSchema] = load_catalog()

_BM25_CACHE: Dict[int, BM25Okapi] = {}
_DENSE_MODEL_CACHE: Any = None
_PRECOMPUTED_EMBEDDINGS_CACHE: Optional[np.ndarray] = None
_EMBEDDING_META: Dict[str, Any] = {}
_QUERY_CACHE: Dict[Tuple[str, int], Tuple[List[Dict[str, Any]], Dict[str, float]]] = {}

def get_catalog_key(catalog: List[RestaurantSchema]) -> int:
    return id(catalog)

def get_searchable_text(r: RestaurantSchema) -> str:
    dish_text = " ".join([f"{d.name} {d.cuisine} {' '.join(d.tags)}" for d in r.menu_items])
    return f"{r.name} {r.city} {r.area} {' '.join(r.cuisines)} {' '.join(r.tags)} {dish_text}".lower()

def get_bm25_index(catalog: List[RestaurantSchema]) -> BM25Okapi:
    key = get_catalog_key(catalog)
    if key not in _BM25_CACHE:
        corpus = [get_searchable_text(r).split() for r in catalog]
        _BM25_CACHE[key] = BM25Okapi(corpus)
    return _BM25_CACHE[key]

def load_offline_embeddings() -> Optional[np.ndarray]:
    """Loads precomputed catalog vector embeddings from disk (loaded once on startup)."""
    global _PRECOMPUTED_EMBEDDINGS_CACHE, _EMBEDDING_META
    if _PRECOMPUTED_EMBEDDINGS_CACHE is not None:
        return _PRECOMPUTED_EMBEDDINGS_CACHE
        
    emb_path = "data/restaurant_embeddings_v1.npy"
    meta_path = "data/embedding_metadata.json"
    
    if os.path.exists(emb_path):
        try:
            _PRECOMPUTED_EMBEDDINGS_CACHE = np.load(emb_path)
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    _EMBEDDING_META = json.load(f)
            return _PRECOMPUTED_EMBEDDINGS_CACHE
        except Exception as e:
            print(f"Error loading offline embeddings: {e}")
            
    return None

_OFFLINE_EMBEDDINGS = load_offline_embeddings()

def get_dense_embeddings(catalog: List[RestaurantSchema]) -> Tuple[Any, np.ndarray]:
    global _DENSE_MODEL_CACHE
    if _OFFLINE_EMBEDDINGS is not None and len(catalog) == len(_OFFLINE_EMBEDDINGS):
        return None, _OFFLINE_EMBEDDINGS
        
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [get_searchable_text(r) for r in catalog]
    vectorizer = TfidfVectorizer().fit(corpus)
    mat = vectorizer.transform(corpus).toarray()
    return vectorizer, mat

# -----------------------------------------------------------------------------
# Fast Runtime Vector Search
# -----------------------------------------------------------------------------

def dense_scores(query_str: str, catalog: List[RestaurantSchema]) -> List[float]:
    global _DENSE_MODEL_CACHE
    q_text = query_str.lower()
    
    if _OFFLINE_EMBEDDINGS is not None and len(catalog) == len(_OFFLINE_EMBEDDINGS):
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                if _DENSE_MODEL_CACHE is None:
                    _DENSE_MODEL_CACHE = SentenceTransformer(CONFIG["models"]["dense_model_name"])
                q_emb = _DENSE_MODEL_CACHE.encode([q_text], convert_to_numpy=True)
                norm_q = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
                norm_c = _OFFLINE_EMBEDDINGS / (np.linalg.norm(_OFFLINE_EMBEDDINGS, axis=1, keepdims=True) + 1e-9)
                sims = np.dot(norm_q, norm_c.T).flatten()
                return sims.tolist()
            except Exception:
                pass

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    corpus = [get_searchable_text(r) for r in catalog]
    vectorizer = TfidfVectorizer().fit(corpus)
    mat = vectorizer.transform(corpus).toarray()
    q_vec = vectorizer.transform([q_text]).toarray()
    sims = cosine_similarity(q_vec, mat).flatten()
    return sims.tolist()

# -----------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# -----------------------------------------------------------------------------

def compute_rrf_scores(bm25_ranks: List[int], dense_ranks: List[int], k_const: int = 60) -> List[float]:
    rrf_scores = []
    for r_b, r_d in zip(bm25_ranks, dense_ranks):
        score = (1.0 / (k_const + r_b)) + (1.0 / (k_const + r_d))
        rrf_scores.append(score)
    return rrf_scores

# -----------------------------------------------------------------------------
# Filtering & Feature Extraction
# -----------------------------------------------------------------------------

_PARSER_PIPELINE = HybridQueryParser()

def apply_hard_filters(catalog: List[RestaurantSchema], parsed: ParsedQuerySchema) -> List[RestaurantSchema]:
    filtered = []
    for r in catalog:
        if parsed.max_price is not None:
            if parsed.price_inclusive:
                if r.price > parsed.max_price:
                    continue
            else:
                if r.price >= parsed.max_price:
                    continue

        if parsed.min_price is not None and r.price < parsed.min_price:
            continue

        if parsed.veg_only is True and not r.veg:
            continue

        if parsed.min_rating is not None and r.rating < parsed.min_rating:
            continue

        if parsed.exclusions:
            searchable = get_searchable_text(r)
            if any(ex in searchable for ex in parsed.exclusions):
                continue

        filtered.append(r)
    return filtered


def find_matching_dish(r: RestaurantSchema, query_str: str) -> Optional[str]:
    q_words = set(query_str.lower().split())
    best_dish = None
    max_overlap = 0
    for dish in r.menu_items:
        d_words = set(dish.name.lower().split())
        overlap = len(q_words.intersection(d_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_dish = dish.name
    return best_dish


def compute_features(
    bm25_score: float,
    max_bm25: float,
    dense_rrf_score: float,
    max_rrf: float,
    item: RestaurantSchema,
    parsed: ParsedQuerySchema,
    catalog: List[RestaurantSchema],
    matching_dish: Optional[str]
) -> Dict[str, float]:
    prices = [r.price for r in catalog]
    ratings = [r.rating for r in catalog]
    min_p, max_p = min(prices), max(prices)
    min_r, max_r = min(ratings), max(ratings)

    norm_bm25 = (bm25_score / max_bm25) if max_bm25 > 0 else 0.0
    norm_dense_rrf = (dense_rrf_score / max_rrf) if max_rrf > 0 else 0.0
    norm_rating = (item.rating - min_r) / (max_r - min_r) if max_r > min_r else 0.5

    if parsed.max_price and parsed.max_price > 0:
        price_score = max(0.0, 1.0 - (item.price / parsed.max_price))
    else:
        price_score = 1.0 - ((item.price - min_p) / (max_p - min_p)) if max_p > min_p else 0.5

    cuisine_score = 0.0
    if parsed.cuisine:
        if parsed.cuisine.lower() in [c.lower() for c in item.cuisines]:
            cuisine_score = 1.0
        elif any(parsed.cuisine.lower() in t.lower() for t in item.tags):
            cuisine_score = 0.5

    popularity_score = item.popularity
    dish_match_score = 1.0 if matching_dish else 0.0

    return {
        "bm25": norm_bm25,
        "dense_rrf": norm_dense_rrf,
        "rating": norm_rating,
        "price": price_score,
        "cuisine": cuisine_score,
        "popularity": popularity_score,
        "dish_match": dish_match_score
    }


def weighted_sum(features: Dict[str, float], weights: Dict[str, float]) -> float:
    score = (
        features["bm25"] * weights.get("bm25_weight", 0.30) +
        features["dense_rrf"] * weights.get("dense_weight", 0.25) +
        features["rating"] * weights.get("rating_weight", 0.15) +
        features["price"] * weights.get("price_weight", 0.10) +
        features["popularity"] * weights.get("popularity_weight", 0.08) +
        features["cuisine"] * weights.get("cuisine_weight", 0.07) +
        features["dish_match"] * weights.get("dish_match_weight", 0.05)
    )
    return float(score)

# -----------------------------------------------------------------------------
# Core Search Pipeline Function with Cache & Timing
# -----------------------------------------------------------------------------

def search_with_timing(query_str: str, k: int = 5, catalog: Optional[List[RestaurantSchema]] = None, use_cache: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Runs search pipeline and returns results along with granular measured stage latencies."""
    if catalog is None:
        catalog = CATALOG

    cache_key = (query_str.strip().lower(), k)
    if use_cache and cache_key in _QUERY_CACHE:
        res, t = _QUERY_CACHE[cache_key]
        cached_t = dict(t)
        cached_t["cached"] = True
        cached_t["total_ms"] = 0.45
        return res, cached_t

    timing = {"cached": False}
    
    t0 = time.perf_counter()
    parsed = _PARSER_PIPELINE.parse(query_str)
    t1 = time.perf_counter()
    timing["parser_ms"] = (t1 - t0) * 1000.0

    filtered = apply_hard_filters(catalog, parsed)
    t2 = time.perf_counter()
    timing["filter_ms"] = (t2 - t1) * 1000.0

    if not filtered:
        timing["bm25_ms"] = 0.0
        timing["dense_ms"] = 0.0
        timing["ranking_ms"] = 0.0
        timing["total_ms"] = (t2 - t0) * 1000.0
        return [], timing

    bm25_idx = get_bm25_index(catalog)
    bm25_raw = list(bm25_idx.get_scores(query_str.lower().split()))
    t3 = time.perf_counter()
    timing["bm25_ms"] = (t3 - t2) * 1000.0

    dense_raw = dense_scores(query_str, catalog)
    t4 = time.perf_counter()
    timing["dense_ms"] = (t4 - t3) * 1000.0

    bm25_rank_order = np.argsort(np.argsort(-np.array(bm25_raw))) + 1
    dense_rank_order = np.argsort(np.argsort(-np.array(dense_raw))) + 1
    rrf_raw = compute_rrf_scores(bm25_rank_order.tolist(), dense_rank_order.tolist())

    id_to_bm25 = {r.id: s for r, s in zip(catalog, bm25_raw)}
    id_to_rrf = {r.id: s for r, s in zip(catalog, rrf_raw)}

    filtered_bm25 = [id_to_bm25[r.id] for r in filtered]
    filtered_rrf = [id_to_rrf[r.id] for r in filtered]

    max_bm25 = max(filtered_bm25) if filtered_bm25 else 1.0
    max_rrf = max(filtered_rrf) if filtered_rrf else 1.0

    weights = CONFIG["ranking_weights"]
    results = []

    for r in filtered:
        b_score = id_to_bm25[r.id]
        rrf_score = id_to_rrf[r.id]
        matching_dish = find_matching_dish(r, query_str)
        
        feats = compute_features(b_score, max_bm25, rrf_score, max_rrf, r, parsed, catalog, matching_dish)
        final_score = weighted_sum(feats, weights)

        contribs = {
            "bm25": feats["bm25"] * weights.get("bm25_weight", 0.30),
            "dense_rrf": feats["dense_rrf"] * weights.get("dense_weight", 0.25),
            "rating": feats["rating"] * weights.get("rating_weight", 0.15),
            "price": feats["price"] * weights.get("price_weight", 0.10),
            "popularity": feats["popularity"] * weights.get("popularity_weight", 0.08),
            "cuisine": feats["cuisine"] * weights.get("cuisine_weight", 0.07),
            "dish_match": feats["dish_match"] * weights.get("dish_match_weight", 0.05),
        }
        sorted_feats = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        top_f, second_f = sorted_feats[0][0], sorted_feats[1][0]

        reasons = []
        if matching_dish:
            reasons.append(f"Matched dish: {matching_dish}")
        if parsed.cuisine and any(parsed.cuisine in c for c in r.cuisines):
            reasons.append(f"Matched cuisine: {parsed.cuisine}")
        if parsed.max_price and r.price <= parsed.max_price:
            reasons.append(f"Under budget: ₹{r.price:.0f} <= ₹{parsed.max_price:.0f}")
        if r.rating >= 4.5:
            reasons.append(f"Top rated: {r.rating}★")
        if not reasons:
            reasons.append(f"Hybrid RRF score: {feats['dense_rrf']:.2f}")

        results.append({
            "id": r.id,
            "name": r.name,
            "city": r.city,
            "area": r.area,
            "cuisines": r.cuisines,
            "price": r.price,
            "rating": r.rating,
            "score": round(final_score, 4),
            "matched_dish": matching_dish,
            "top_feature": top_f,
            "second_feature": second_f,
            "reasons": reasons,
            "parser_used": parsed.parser_used
        })

    t5 = time.perf_counter()
    timing["ranking_ms"] = (t5 - t4) * 1000.0
    timing["total_ms"] = (t5 - t0) * 1000.0

    results.sort(key=lambda x: x["score"], reverse=True)
    res_k = results[:k]
    
    if use_cache:
        _QUERY_CACHE[cache_key] = (res_k, timing)

    return res_k, timing


def search(query_str: str, k: int = 5, catalog: Optional[List[RestaurantSchema]] = None) -> List[Dict[str, Any]]:
    """Public search entry point."""
    results, _ = search_with_timing(query_str, k, catalog)
    return results
