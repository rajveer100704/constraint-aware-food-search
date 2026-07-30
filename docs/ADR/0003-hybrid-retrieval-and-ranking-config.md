# ADR 0003: Multi-Feature Linear Reranking vs. Complex ML Models

## Context
After hard filtering, candidates must be scored using multiple signals (relevance, rating, price, popularity).

## Decision
We implemented a configurable linear weighted scoring model (`RankingConfig`).

## Rationale
- Explainability: Allows explicit tracking of feature contributions (`top_feature`, `second_feature`).
- Zero Cold Start: Does not depend on click logs or user tracking data.
- Performance: Computes scores in < 1ms across candidate sets.
