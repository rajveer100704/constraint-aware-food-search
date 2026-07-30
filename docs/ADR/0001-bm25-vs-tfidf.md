# ADR 0001: Selecting BM25 over TF-IDF for Lexical Retrieval

## Context
We needed a lexical retrieval algorithm to rank restaurant catalog items against natural language search queries.

## Options Considered
1. Standard TF-IDF Cosine Similarity
2. BM25Okapi Probabilistic Retrieval

## Decision
We selected **BM25Okapi**.

## Rationale
- BM25 incorporates document length normalization and term frequency saturation, preventing long restaurant descriptions from dominating search results.
- In offline benchmark evaluations, BM25 achieved superior NDCG@5 (0.9250 vs 0.9087) compared to TF-IDF.
