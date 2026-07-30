# 🚀 Scaling to Production: 40M Restaurants & 500M Menu Items

This document outlines the system architecture, infrastructure design, and scaling strategy for scaling the Swiggy Food Delivery Hybrid Search Engine to production scale (**40,000,000+ restaurants and 500,000,000+ dish menu items**).

---

## 1. High-Scale Production System Architecture

```
User Search Request ("spicy chicken biryani under 300 in koramangala")
                               │
                               ▼
                    ┌─────────────────────┐
                    │ API Gateway & WAF   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Exact Query Cache   │ ──(Hit: <1ms)──► Return Top-K
                    │  (Redis Cluster)    │
                    └──────────┬──────────┘
                               │ (Miss)
                               ▼
             ┌───────────────────────────────────┐
             │ Query Processing Microservice     │
             │ - SLM Query Parser (Triton / ONNX)│
             │ - Feature Store Query Enrichment  │
             └─────────────────┬─────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌─────────────────────────┐           ┌─────────────────────────┐
│ Sparse Retrieval Cluster│           │ Dense Retrieval Cluster │
│ (Elasticsearch/Lucene)  │           │ (FAISS HNSW Vector DB)  │
│ - Lexical BM25 Scoring  │           │ - Distributed ANN Index │
└───────────┬─────────────┘           └───────────┬─────────────┘
            │ Candidate Pool Top-500              │ Candidate Pool Top-500
            └──────────────────┬──────────────────┘
                               │
                               ▼
             ┌───────────────────────────────────┐
             │ Candidate Aggregation & RRF Fusion│
             │ RRF = 1/(60 + r_b) + 1/(60 + r_d) │
             └─────────────────┬─────────────────┘
                               │ Merged Top-100 Candidates
                               ▼
             ┌───────────────────────────────────┐
             │ Distributed Feature Store (Feast) │
             │ Fetches Real-Time User & Rest Feats│
             └─────────────────┬─────────────────┘
                               │ Enriched Feature Vectors
                               ▼
             ┌───────────────────────────────────┐
             │ Heavy GBDT / LambdaMART Reranker │
             │ (Triton GPU Inference Cluster)    │
             └─────────────────┬─────────────────┘
                               │ Top-20 Ranked Items
                               ▼
                     Search Response API
```

---

## 2. Distributed Scale Components

### A. Two-Tower Candidate Retrieval Model
- **Query Tower**: Real-time transformer model (ONNX Runtime / TensorRT) encoding user query + user context (location, historical preferences) into a 128-dimensional embedding in <5ms.
- **Item/Restaurant Tower**: Offline Spark streaming pipeline computing embeddings for 500M menu items and updating vector indexes hourly.

### B. Distributed Vector Indexing (FAISS HNSW / Milvus Cluster)
- For $N > 100,000$, brute-force matrix multiplication is replaced by **FAISS HNSW (Hierarchical Navigable Small World)** graphs indexed by geographic region/city partitions (e.g. Bangalore, Mumbai, Delhi).
- Queries are routed strictly to local regional shards, capping target search spaces to <50,000 active items per shard.

### C. Real-Time Feature Store (Redis + Feast)
- Real-time signals (e.g., current restaurant preparation time, delivery delay, real-time availability, live surge pricing) are updated asynchronously into a Redis feature store and joined in <2ms before LTR scoring.

### D. Multi-Stage Ranking Funnel
1. **Candidate Retrieval (Recall Stage)**: BM25 + FAISS Dense Vector Retrieval fetches Top-500 candidates.
2. **First-Pass Filter & RRF Fusion**: Applies hard geo-fencing, dietary compliance, and RRF fusion to reduce candidates to Top-100.
3. **Heavy LTR Reranking (Precision Stage)**: Pairwise LambdaMART GBDT / Deep Learning Ranker ranks Top-100 candidates down to Top-20 for presentation.

---

## 3. Production SLAs, Monitoring & A/B Testing

- **Latency SLA**: P50 < 30ms, P95 < 80ms, P99 < 150ms.
- **Monitoring Metrics**:
  - Search Click-Through Rate (CTR)
  - Search Conversion Rate (Search $\rightarrow$ Order)
  - Zero-Result Rate (ZRR)
  - Parser Fallback Rate (%)
  - Vector Cache Hit Ratio (%)
- **Online A/B Testing**: Traffic split via Istio service mesh comparing control (BM25 baseline) vs treatment (Hybrid RRF + GBDT ranker).
