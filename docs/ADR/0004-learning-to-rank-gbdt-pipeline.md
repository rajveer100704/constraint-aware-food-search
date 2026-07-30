# ADR 0004: Learning-to-Rank (LTR) GBDT Pipeline & Transparent Dataset Logging

## Context
Fixed linear feature weights (`0.30 BM25 + 0.25 Dense + ...`) cannot model non-linear feature interactions (e.g. high rating + high dish match vs low price).

## Decision
We implemented a Learning-to-Rank (LTR) dataset logging and Gradient Boosted Decision Tree (`HistGradientBoostingRegressor` / `LightGBM`) training pipeline in `train_ltr.py`.

## Rationale & Engineering Honesty
- **Data-First LTR**: 7-dimensional feature vectors are explicitly logged per candidate pair.
- **Transparent Relevance Targets**: Training labels are calculated transparently using exact constraint compliance and normalized rating, explicitly documented in code and logs as demonstration targets.
- **Model Artifact**: Serialized GBDT model (`models/ltr_model.pkl`) enables instant scoring.
