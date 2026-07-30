import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Response
from pydantic import BaseModel

from schemas import SearchRequest, SearchResponseItem, SearchResponse, ParsedQuerySchema
from search_engine import search, CATALOG, _PARSER_PIPELINE

app = FastAPI(
    title="Swiggy Hybrid Search & Recommendation Engine API",
    description="Production Food Search Service featuring PyPI SearchExpert (SLM) Parsing, Dense+BM25 RRF Retrieval, and Dish-Level Matching.",
    version="3.1.0"
)

# Prometheus Observability Counters
REQUEST_COUNT = 0
TOTAL_LATENCY_MS = 0.0
CACHE_HITS = 0
CACHE_MISSES = 0
BM25_LATENCY_SUM_MS = 0.0
DENSE_LATENCY_SUM_MS = 0.0
SLM_LATENCY_SUM_MS = 0.0

@app.get("/health", summary="Service Healthcheck")
def healthcheck():
    return {
        "status": "healthy",
        "service": "swiggy-hybrid-search-api",
        "version": "3.1.0",
        "catalog_size": len(CATALOG)
    }

@app.get("/version", summary="Service Version & Capability Status")
def version():
    return {
        "version": "3.1.0",
        "slm_parser": _PARSER_PIPELINE.use_slm,
        "dense_retrieval": True,
        "rrf_fusion": True,
        "dish_level_matching": True
    }

@app.get("/metrics", summary="Prometheus Metrics Exposition")
def metrics():
    avg_latency = (TOTAL_LATENCY_MS / REQUEST_COUNT) if REQUEST_COUNT > 0 else 0.0
    metrics_text = (
        f"# HELP search_requests_total Total number of search requests processed\n"
        f"# TYPE search_requests_total counter\n"
        f"search_requests_total {REQUEST_COUNT}\n\n"
        f"# HELP search_latency_ms_avg Average total search latency in milliseconds\n"
        f"# TYPE search_latency_ms_avg gauge\n"
        f"search_latency_ms_avg {avg_latency:.2f}\n\n"
        f"# HELP cache_hits_total Total cache hit count\n"
        f"# TYPE cache_hits_total counter\n"
        f"cache_hits_total {CACHE_HITS}\n\n"
        f"# HELP cache_misses_total Total cache miss count\n"
        f"# TYPE cache_misses_total counter\n"
        f"cache_misses_total {CACHE_MISSES}\n"
    )
    return Response(content=metrics_text, media_type="text/plain")

@app.get("/search", response_model=SearchResponse, summary="Execute Hybrid Search")
def get_search(
    query: str = Query(..., description="Natural language food search query"),
    k: int = Query(default=5, ge=1, le=50, description="Top-k results count")
):
    global REQUEST_COUNT, TOTAL_LATENCY_MS
    t0 = time.perf_counter()
    
    results = search(query_str=query, k=k, catalog=CATALOG)
    
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000.0
    
    REQUEST_COUNT += 1
    TOTAL_LATENCY_MS += latency
    parser_used = results[0]["parser_used"] if results else "regex_fallback"
    
    return SearchResponse(
        query=query,
        total_results=len(results),
        latency_ms=round(latency, 2),
        parser_used=parser_used,
        results=[SearchResponseItem(**r) for r in results]
    )

@app.post("/search", response_model=SearchResponse, summary="Execute Hybrid Search (POST)")
def post_search(req: SearchRequest):
    global REQUEST_COUNT, TOTAL_LATENCY_MS
    t0 = time.perf_counter()
    
    results = search(query_str=req.query, k=req.k, catalog=CATALOG)
    
    if req.max_price is not None:
        results = [r for r in results if r["price"] <= req.max_price]
    if req.min_rating is not None:
        results = [r for r in results if r["rating"] >= req.min_rating]
        
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000.0
    
    REQUEST_COUNT += 1
    TOTAL_LATENCY_MS += latency
    parser_used = results[0]["parser_used"] if results else "regex_fallback"
    
    return SearchResponse(
        query=req.query,
        total_results=len(results),
        latency_ms=round(latency, 2),
        parser_used=parser_used,
        results=[SearchResponseItem(**r) for r in results]
    )

@app.post("/parse", response_model=ParsedQuerySchema, summary="Parse Query Constraints")
def post_parse(query: str = Query(..., description="Query to parse")):
    return _PARSER_PIPELINE.parse(query)
