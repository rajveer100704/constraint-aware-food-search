import json
import random

SEED = 42

def generate_200_queries():
    random.seed(SEED)
    
    cuisines = ["North Indian", "South Indian", "Chinese", "Italian", "Fast Food", "Desserts", "Biryani"]
    areas = ["Koramangala", "HSR Layout", "Indiranagar", "Jayanagar", "BTM Layout", "Whitefield"]
    dishes = [
        ("paneer butter masala", 1), ("chicken tikka pizza", 2), ("hyderabadi chicken biryani", 3),
        ("mutton dum biryani", 4), ("schezwan hakka noodles", 5), ("veg cheese burger", 6),
        ("masala dosa", 7), ("keshari rasgulla", 8), ("chicken dum biryani", 3), ("steamed idli sambar", 7),
        ("chicken fried rice", 5), ("margherita pizza", 2), ("peri peri french fries", 6), ("mishti doi", 8)
    ]
    
    queries = []
    qid = 1

    # 1. Dish Names (20 queries)
    for name, rid in dishes + [("dal makhani", 1), ("garlic naan", 1), ("tandoori chicken", 3), ("spring rolls", 5), ("gulab jamun", 8), ("cold coffee", 6)]:
        queries.append({
            "id": qid,
            "query": name,
            "relevant_ids": [rid],
            "category": "Dish Names",
            "description": f"Exact dish lookup: {name}"
        })
        qid += 1

    # 2. Price Limits (25 queries)
    prices = [150, 200, 250, 300, 350, 400, 500]
    for p in prices:
        for c in ["veg food", "biryani", "pizza", "chinese", "south indian"]:
            rid = 7 if "south" in c else (3 if "biryani" in c else (2 if "pizza" in c else (5 if "chinese" in c else 1)))
            queries.append({
                "id": qid,
                "query": f"{c} under {p}",
                "relevant_ids": [rid],
                "category": "Price Limits",
                "description": f"Price constraint: {c} <= {p}"
            })
            qid += 1

    # 3. Location / Area (25 queries)
    for a in areas:
        for food in ["biryani", "dosa", "burger", "chinese"]:
            rid = 4 if "biryani" in food else (7 if "dosa" in food else (6 if "burger" in food else 5))
            queries.append({
                "id": qid,
                "query": f"{food} in {a}",
                "relevant_ids": [rid],
                "category": "Location",
                "description": f"Geographic area filter: {food} in {a}"
            })
            qid += 1

    # 4. Cuisine (20 queries)
    for c in cuisines:
        for suffix in ["food", "restaurant", "dishes"]:
            rid = 1 if "North" in c else (7 if "South" in c else (5 if "Chinese" in c else (2 if "Italian" in c else (6 if "Fast" in c else (8 if "Dessert" in c else 3)))))
            queries.append({
                "id": qid,
                "query": f"{c} {suffix}",
                "relevant_ids": [rid],
                "category": "Cuisine",
                "description": f"Cuisine tag match: {c}"
            })
            qid += 1

    # 5. Mixed Constraints (30 queries)
    for p in [200, 250, 300, 400]:
        for a in ["koramangala", "hsr layout", "indiranagar"]:
            for f in ["biryani", "pizza", "thali"]:
                rid = 4 if f == "biryani" else (2 if f == "pizza" else 1)
                queries.append({
                    "id": qid,
                    "query": f"pure veg {f} in {a} under {p}",
                    "relevant_ids": [rid],
                    "category": "Mixed Constraints",
                    "description": f"Multi-constraint: veg + {f} in {a} under {p}"
                })
                qid += 1

    # 6. Typos (20 queries)
    typos = [
        ("chiken biryani", [3]), ("pisa under 300", [2]), ("biryany in hsr", [3]),
        ("paner butter masala", [1]), ("chese burger", [6]), ("noddles chinese", [5]),
        ("dosa south indian", [7]), ("sweet dessert", [8]), ("hyderabdi biryani", [3]),
        ("shezwan hakka noodles", [5]), ("mutton dum biryany", [4]), ("margherita pisa", [2]),
        ("kesari rasgulla", [8]), ("peri peri fry", [6]), ("steamed idly", [7]),
        ("north indan thali", [1]), ("bengali sweet", [8]), ("fast fud burger", [6]),
        ("indiranagar chinese", [5]), ("indial thali veg", [1])
    ]
    for typo, rids in typos:
        queries.append({
            "id": qid,
            "query": typo,
            "relevant_ids": rids,
            "category": "Typos",
            "description": f"Typo robustness check: {typo}"
        })
        qid += 1

    # 7. Semantic Queries (20 queries)
    semantics = [
        ("spicy noodles", [5]), ("sweet dessert platter", [8]), ("crispy breakfast", [7]),
        ("cheesy meal", [2, 6]), ("rich gravy curry", [1]), ("authentic hyderabadi taste", [3]),
        ("quick bite burger", [6]), ("traditional north meal", [1]), ("hot spicy mutton", [4]),
        ("bengali sweets special", [8]), ("light south indian breakfast", [7]), ("crunchy french fries", [6]),
        ("rich creamy paneer", [1]), ("family biryani bucket", [3, 4]), ("loaded pizza slice", [2]),
        ("spicy schezwan gravy", [5]), ("refreshing cold dessert", [8]), ("hearty thali meal", [1]),
        ("classic woodfired pizza", [2]), ("authentic sourdough crust", [2])
    ]
    for sem, rids in semantics:
        queries.append({
            "id": qid,
            "query": sem,
            "relevant_ids": rids,
            "category": "Semantic",
            "description": f"Semantic vector search: {sem}"
        })
        qid += 1

    # 8. Negations / Exclusions (15 queries)
    negations = [
        ("biryani without mutton", [3]), ("pizza without chicken", [2]), ("noodles without eggs", [5]),
        ("burger without onion", [6]), ("thali without garlic", [1]), ("desserts without sugar", [8]),
        ("biryani without chicken", [4]), ("pizza without cheese", [2]), ("dosa without potato", [7]),
        ("fried rice without pork", [5]), ("curry without dairy", [1]), ("burger without mayo", [6]),
        ("sweets without dairy", [8]), ("thali without rice", [1]), ("biryani without rice", [3])
    ]
    for neg, rids in negations:
        queries.append({
            "id": qid,
            "query": neg,
            "relevant_ids": rids,
            "category": "Negations",
            "description": f"Exclusion filter: {neg}"
        })
        qid += 1

    # 9. Lexical Exact Matches (15 queries)
    lexicals = [
        ("Punjab Grill", [1]), ("Olio Pizza", [2]), ("Meghana Foods", [3]),
        ("Empire Restaurant", [4]), ("Mainland China", [5]), ("Truffles", [6]),
        ("CTR Shri Sagar", [7]), ("KC Das Sweets", [8]), ("Koramangala Punjab Grill", [1]),
        ("Jayanagar Truffles", [6]), ("Indiranagar Mainland China", [5]), ("BTM Layout CTR", [7]),
        ("Whitefield KC Das", [8]), ("HSR Layout Meghana", [3]), ("Church Street Empire", [4])
    ]
    for lex, rids in lexicals:
        queries.append({
            "id": qid,
            "query": lex,
            "relevant_ids": rids,
            "category": "Lexical",
            "description": f"Exact entity match: {lex}"
        })
        qid += 1

    # 10. Multi-Intent (10 queries)
    multi_intents = [
        ("top rated spicy biryani in koramangala under 300", [4]),
        ("cheap veg pizza in indiranagar with high rating", [2]),
        ("best fast food burger in jayanagar under 200", [6]),
        ("authentic south indian dosa in btm layout under 150", [7]),
        ("bengali sweets in whitefield under 250 top rated", [8]),
        ("chinese schezwan noodles in indiranagar under 300", [5]),
        ("north indian veg thali in koramangala under 250", [1]),
        ("spicy mutton biryani in hsr layout under 400", [4]),
        ("italian cheese pizza under 500 in indiranagar", [2]),
        ("hyderabadi chicken biryani in hsr layout under 350", [3])
    ]
    for mi, rids in multi_intents:
        queries.append({
            "id": qid,
            "query": mi,
            "relevant_ids": rids,
            "category": "Multi-Intent",
            "description": f"Multi-intent complex query: {mi}"
        })
        qid += 1

    queries_200 = queries[:200]
    
    # Save versioned dataset & metadata
    dataset_dict = {
        "version": "2.0",
        "total_queries": len(queries_200),
        "random_seed": SEED,
        "queries": queries_200
    }
    
    with open("data/evaluation_set_v2.json", "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2)
        
    with open("data/evaluation_set.json", "w", encoding="utf-8") as f:
        json.dump(queries_200, f, indent=2)

    print(f"Successfully generated versioned dataset 'data/evaluation_set_v2.json' ({len(queries_200)} queries, seed={SEED}).")

if __name__ == "__main__":
    generate_200_queries()
