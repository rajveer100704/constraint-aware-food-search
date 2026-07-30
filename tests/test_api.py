import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_healthcheck_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["catalog_size"] > 0

def test_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "3.1.0"
    assert data["dense_retrieval"] is True

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "search_requests_total" in response.text

def test_get_search_endpoint():
    response = client.get("/search?query=veg%20thali%20under%20250&k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "veg thali under 250"
    assert data["total_results"] > 0
    assert len(data["results"]) <= 3
    assert data["results"][0]["id"] == 1

def test_post_search_endpoint():
    payload = {
        "query": "biryani",
        "k": 5,
        "max_price": 300.0,
        "min_rating": 4.0
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] > 0
    assert all(r["price"] <= 300.0 for r in data["results"])
    assert all(r["rating"] >= 4.0 for r in data["results"])

def test_post_parse_endpoint():
    response = client.post("/parse?query=pizza%20under%20400")
    assert response.status_code == 200
    data = response.json()
    assert data["raw_query"] == "pizza under 400"
    assert data["max_price"] == 400.0
