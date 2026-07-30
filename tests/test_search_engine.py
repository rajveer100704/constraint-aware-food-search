import json
import pytest
from schemas import RestaurantSchema, ParsedQuerySchema
from search_engine import (
    search, apply_hard_filters, CATALOG,
    load_catalog, compute_features, weighted_sum, CONFIG, _PARSER_PIPELINE
)
from evaluate import recall_at_k, precision_at_k, mrr, ndcg_at_k

def test_parse_query_pipeline():
    parsed = _PARSER_PIPELINE.parse("pizza under 300")
    assert parsed.max_price == 300.0
    assert parsed.price_inclusive is False

def test_hard_filter_excludes_over_budget():
    parsed = ParsedQuerySchema(raw_query="test", max_price=200.0)
    filtered = apply_hard_filters(CATALOG, parsed)
    assert all(r.price < 200.0 for r in filtered)

def test_hard_filter_veg_only():
    parsed = ParsedQuerySchema(raw_query="test", veg_only=True)
    filtered = apply_hard_filters(CATALOG, parsed)
    assert all(r.veg for r in filtered)

def test_search_returns_top_result():
    results = search("vegetarian thali under 250")
    assert len(results) > 0
    assert results[0]["id"] == 1  # Green Leaf

def test_search_results_dish_matching():
    results = search("chicken biryani", k=3)
    assert len(results) > 0
    assert any(r["matched_dish"] is not None for r in results)

def test_recall_at_k():
    assert recall_at_k([1, 2], [1]) == 1.0
    assert recall_at_k([2], [1]) == 0.0

def test_precision_at_k():
    assert precision_at_k([1, 2], [1]) == 0.5
    assert precision_at_k([1], [1]) == 1.0

def test_mrr():
    assert mrr([2, 1], [1]) == 0.5
    assert mrr([1, 2], [1]) == 1.0

def test_ndcg_at_k_perfect_order():
    assert ndcg_at_k([1, 2], [1], k=2) == 1.0

def test_load_catalog_default():
    cat = load_catalog(None)
    assert len(cat) == len(CATALOG)

def test_load_catalog_from_json(tmp_path):
    json_file = tmp_path / "custom_catalog.json"
    data = [{
        "id": 101, "name": "Test Cafe", "city": "bangalore", "area": "koramangala",
        "cuisines": ["italian"], "price": 200.0, "rating": 4.5, "veg": True,
        "popularity": 0.8, "delivery_time_mins": 25, "tags": ["pasta"],
        "menu_items": [{"id": 1001, "name": "Pasta Carbonara", "price": 200.0, "veg": True, "cuisine": "italian", "category": "main", "tags": []}]
    }]
    json_file.write_text(json.dumps(data))
    cat = load_catalog(str(json_file))
    assert len(cat) == 1
    assert cat[0].name == "Test Cafe"
