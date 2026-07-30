# Hybrid Dense-Lexical Retrieval Implementation Plan

## Goal
Upgrade the retrieval layer from pure BM25 lexical search to a hybrid **Dense Vector Search (Sentence-Transformers + FAISS) + BM25 Lexical Search** system using Reciprocal Rank Fusion (RRF).

## Proposed Architecture

```
                       User Query
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    BM25 Lexical Search         Dense Vector Search
    (Corpus Keyword Match)     (Sentence-Transformers)
             │                           │
             ▼                           ▼
    BM25 Rank List             Dense Rank List
             │                           │
             └─────────────┬─────────────┘
                           ▼
             Reciprocal Rank Fusion (RRF)
               RRF_Score = 1/(60 + r_bm25) + 1/(60 + r_dense)
                           │
                           ▼
               Surviving Filter Candidates
                           │
                           ▼
             Multi-Feature Linear Reranker
```

## Reciprocal Rank Fusion Formula

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $k = 60$, and $r_m(d)$ is the rank of document $d$ in retriever $m$.

## Acceptance Criteria
1. Handles vocabulary mismatch (e.g., matching query "spicy noodles" to "Schezwan Hakka Noodles" with zero exact token overlap).
2. Maintains overall search latency < 50ms.
