# ADR 0005: Menu and Dish-Item Level Search Indexing Architecture

## Context
Food delivery users search for specific dishes (*"Hyderabadi Dum Biryani"*, *"Paneer Butter Masala"*, *"Garlic Bread"*) rather than restaurant names (*"Dominos"*). Restaurant-level metadata alone fails to capture dish relevance.

## Decision
We extended the catalog schema to index **Menu Items / Dishes** under each restaurant, enabling dish-level token and dense vector embedding matching alongside restaurant metadata.

## Benefits
- **Dish Exact Match Feature**: Boosts candidates where specific dish names match user queries.
- **Explainability**: Search responses populate `matched_dish` (e.g. `Matched dish: Chicken Dum Biryani`).
- **Higher Retrieval Recall**: Catches specific food cravings with zero vocabulary mismatch.
