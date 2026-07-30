# 🔍 Search Engine Failure Case Gallery

This document details 5 specific query edge cases, analyzing raw retrieval results, root causes, and architectural remedies.

---

### Case 1: Complex Multi-Word Dish Synonym
- **Query**: `"north indian platter"`
- **Expected Match**: Restaurant ID 1 (*Punjab Grill*) serving *North Indian Thali*
- **Observed Behavior**: BM25 lexical retriever gave low score due to word mismatch ("platter" vs "thali"). Dense vector retriever retrieved *Punjab Grill* at Rank 2.
- **Root Cause**: BM25 exact token matching fails when user terminology differs from catalog dish descriptions.
- **Remedy / Fix**: Dense Vector retrieval + Reciprocal Rank Fusion (RRF) elevates *Punjab Grill* to Rank 1 via semantic similarity.

---

### Case 2: Out-of-Vocabulary Typo Handling
- **Query**: `"chiken biryani"`
- **Expected Match**: Restaurant ID 3 (*Meghana Foods*) & Restaurant ID 4 (*Empire Restaurant*)
- **Observed Behavior**: BM25 failed on token `"chiken"`. Fallback parser matched `"biryani"`.
- **Root Cause**: Raw BM25 tokenizer lacks character n-gram fuzzy edit distance matching.
- **Remedy / Fix**: Dense vector model (`all-MiniLM-L6-v2`) handles minor character typos gracefully in embedding space. In production, query spell-correction (SymSpell / Levenshtein) precedes candidate retrieval.

---

### Case 3: Complex Numeric Range Constraint
- **Query**: `"biryani between 200 and 300"`
- **Expected Match**: Restaurant ID 4 (*Empire Restaurant*, price ₹280)
- **Observed Behavior**: Regex parser captured `max_price=300` but missed explicit `min_price=200`.
- **Root Cause**: Simple regex extraction patterns struggle with dual boundary conditions (`between: X and Y`).
- **Remedy / Fix**: Fine-tuned Small Language Model (`search-expert` LoRA) extracts `min_price: 200` and `max_price: 300` into structured JSON parameters.

---

### Case 4: Negation / Dietary Exclusion
- **Query**: `"biryani without mutton"`
- **Expected Match**: Restaurant ID 3 (*Meghana Foods* - Hyderabadi Chicken Biryani)
- **Observed Behavior**: BM25 retrieved Restaurant ID 4 (*Empire Restaurant*) because "mutton" appeared in its menu tags!
- **Root Cause**: Naive keyword search matches excluded terms positively.
- **Remedy / Fix**: Hard Constraint Filter Engine parses `exclusions: ["mutton"]` and strips any restaurant/dish containing "mutton" from the candidate pool prior to ranking.

---

### Case 5: Dish Name Conflict vs. Restaurant Name
- **Query**: `"empire"`
- **Expected Match**: Restaurant ID 4 (*Empire Restaurant*)
- **Observed Behavior**: High score for *Empire Restaurant*, but also matched menu items named "Empire Special Biryani".
- **Root Cause**: Field weighting ambiguity between restaurant entity names and dish names.
- **Remedy / Fix**: Configurable field boosting (`restaurant_name_weight: 2.0`, `dish_name_weight: 1.0`) prioritizes direct entity matches over menu item matches.
