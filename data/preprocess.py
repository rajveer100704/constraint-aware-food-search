import json
import os
import re
import hashlib
import time
import numpy as np

# Attempt loading sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

def clean_cuisine_token(cuisine_str: str) -> list[str]:
    """Cleans raw cuisine string, normalizing commas, double-spaces, and plurals."""
    if not cuisine_str:
        return []
    normalized = re.sub(r'\s{2,}', ', ', cuisine_str.strip())
    raw_tokens = [c.strip().lower() for c in normalized.split(',') if c.strip()]
    cleaned_tokens = []
    for token in raw_tokens:
        if token == "pizzas":
            token = "pizza"
        elif token == "burgers":
            token = "burger"
        cleaned_tokens.append(token)
    return cleaned_tokens

def calculate_cuisine_medians(records: list[dict]) -> dict[str, float]:
    """Calculates median price per cuisine across catalog."""
    cuisine_prices = {}
    for item in records:
        price = item.get("price", 250)
        cuisines = item.get("cuisines", [])
        if isinstance(cuisines, str):
            cuisines = clean_cuisine_token(cuisines)
        for c in cuisines:
            cuisine_prices.setdefault(c, []).append(price)
            
    medians = {}
    for c, prices in cuisine_prices.items():
        sorted_p = sorted(prices)
        n = len(sorted_p)
        mid = n // 2
        medians[c] = sorted_p[mid] if n % 2 != 0 else (sorted_p[mid - 1] + sorted_p[mid]) / 2.0
    return medians

def get_searchable_text_from_dict(r: dict) -> str:
    cuisines = r.get("cuisines", [])
    if isinstance(cuisines, list):
        cuisines_str = " ".join(cuisines)
    else:
        cuisines_str = str(cuisines)
        
    tags = r.get("tags", [])
    if isinstance(tags, list):
        tags_str = " ".join(tags)
    else:
        tags_str = str(tags)
        
    dish_text = ""
    for d in r.get("menu_items", []):
        d_name = d.get("name", "")
        d_cuisine = d.get("cuisine", "")
        d_tags = " ".join(d.get("tags", []))
        dish_text += f" {d_name} {d_cuisine} {d_tags}"
        
    return f"{r.get('name', '')} {r.get('city', '')} {r.get('area', '')} {cuisines_str} {tags_str} {dish_text}".strip().lower()

def generate_offline_embeddings(catalog_records: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Offline Embedding Persistence Pipeline:
    Precomputes vector embeddings for catalog items and saves versioned artifact
    'data/restaurant_embeddings_v1.npy' alongside 'data/embedding_metadata.json'.
    """
    corpus = [get_searchable_text_from_dict(r) for r in catalog_records]
    catalog_bytes = json.dumps(catalog_records, sort_keys=True).encode("utf-8")
    checksum = hashlib.sha256(catalog_bytes).hexdigest()[:12]
    
    emb_dimension = 384
    embeddings = None
    
    if _HAS_SENTENCE_TRANSFORMERS:
        try:
            print(f"Generating offline embeddings via {model_name}...")
            model = SentenceTransformer(model_name)
            embeddings = model.encode(corpus, convert_to_numpy=True)
            emb_dimension = embeddings.shape[1]
        except Exception as e:
            print(f"SentenceTransformer fallback during embedding generation: {e}")
            
    if embeddings is None:
        # TF-IDF Cosine Embedding fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer().fit(corpus)
        embeddings = vectorizer.transform(corpus).toarray()
        emb_dimension = embeddings.shape[1]
        
    emb_path = "data/restaurant_embeddings_v1.npy"
    np.save(emb_path, embeddings)
    
    meta_path = "data/embedding_metadata.json"
    metadata = {
        "model_name": model_name,
        "embedding_dimension": int(emb_dimension),
        "total_items": len(catalog_records),
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "catalog_checksum": checksum,
        "embedding_file": emb_path
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Offline embeddings successfully saved to '{emb_path}' ({embeddings.shape}).")
    print(f"Embedding metadata saved to '{meta_path}'.")

def preprocess_catalog(raw_records: list[dict]) -> tuple[list[dict], dict]:
    """Processes raw restaurant records into normalized catalog with dish menu items."""
    processed = []
    all_cuisines = set()
    
    for item in raw_records:
        cuisines_raw = item.get("cuisines", item.get("cuisine", ""))
        if isinstance(cuisines_raw, list):
            cuisines = cuisines_raw
        else:
            cuisines = clean_cuisine_token(str(cuisines_raw))
            
        all_cuisines.update(cuisines)
        
        tags_raw = item.get("tags", [])
        if isinstance(tags_raw, str):
            tags = [t.strip().lower() for t in tags_raw.split(',') if t.strip()]
        else:
            tags = [t.lower() for t in tags_raw]
            
        menu_items_raw = item.get("menu_items", [])
        formatted_menu = []
        for dish in menu_items_raw:
            formatted_menu.append({
                "id": int(dish.get("id", 0)),
                "name": str(dish["name"]),
                "price": float(dish.get("price", item.get("price", 200))),
                "veg": bool(dish.get("veg", item.get("veg", True))),
                "cuisine": str(dish.get("cuisine", cuisines[0] if cuisines else "general")).lower(),
                "category": str(dish.get("category", "Main Course")).lower(),
                "tags": [t.lower() for t in dish.get("tags", [])]
            })
            
        rec = {
            "id": int(item["id"]),
            "name": str(item["name"]),
            "city": str(item.get("city", "Bangalore")).lower(),
            "area": str(item.get("area", "")).lower(),
            "cuisines": cuisines,
            "price": float(item.get("price", 250)),
            "rating": float(item.get("rating", 4.0)),
            "veg": bool(item.get("veg", False)),
            "popularity": float(item.get("popularity", 0.5)),
            "delivery_time_mins": int(item.get("delivery_time_mins", 30)),
            "tags": tags,
            "menu_items": formatted_menu
        }
        processed.append(rec)
        
    medians = calculate_cuisine_medians(processed)
    
    metadata = {
        "version": "3.2.0",
        "total_items": len(processed),
        "total_cuisines": len(all_cuisines),
        "total_dishes": sum(len(r["menu_items"]) for r in processed),
        "cuisine_median_prices": medians,
        "supported_cities": list(set(r["city"] for r in processed)),
    }
    
    return processed, metadata

if __name__ == "__main__":
    sample_raw = [
        {
            "id": 1, "name": "Green Leaf", "city": "Bangalore", "area": "Koramangala",
            "cuisines": "North Indian, Thalis, South Indian", "price": 220, "rating": 4.6, "veg": True, "popularity": 0.9, "delivery_time_mins": 25,
            "tags": ["thali", "north indian", "paneer"],
            "menu_items": [
                {"id": 101, "name": "Special Veg Thali", "price": 220, "veg": True, "cuisine": "thalis", "category": "Thalis", "tags": ["thali", "paneer", "roti", "dal"]},
                {"id": 102, "name": "Paneer Butter Masala", "price": 190, "veg": True, "cuisine": "north indian", "category": "Main Course", "tags": ["paneer", "curry"]}
            ]
        },
        {
            "id": 2, "name": "Pizza Corner", "city": "Bangalore", "area": "Indiranagar",
            "cuisines": "Pizzas, Italian", "price": 399, "rating": 4.2, "veg": False, "popularity": 0.7, "delivery_time_mins": 35,
            "tags": ["pizza", "pasta", "cheese"],
            "menu_items": [
                {"id": 201, "name": "Chicken Tikka Pizza", "price": 420, "veg": False, "cuisine": "pizza", "category": "Pizzas", "tags": ["chicken", "pizza", "cheese"]},
                {"id": 202, "name": "Margherita Pizza", "price": 340, "veg": True, "cuisine": "pizza", "category": "Pizzas", "tags": ["cheese", "pizza", "tomato"]}
            ]
        },
        {
            "id": 3, "name": "Spice Garden", "city": "Bangalore", "area": "HSR Layout",
            "cuisines": "Biryani, Mughlai", "price": 350, "rating": 4.5, "veg": False, "popularity": 0.85, "delivery_time_mins": 30,
            "tags": ["biryani", "chicken", "spicy"],
            "menu_items": [
                {"id": 301, "name": "Hyderabadi Chicken Biryani", "price": 350, "veg": False, "cuisine": "biryani", "category": "Biryani", "tags": ["chicken", "biryani", "spicy"]},
                {"id": 302, "name": "Chicken Tikka Kebab", "price": 280, "veg": False, "cuisine": "mughlai", "category": "Starters", "tags": ["kebab", "chicken"]}
            ]
        },
        {
            "id": 4, "name": "Royal Biryani", "city": "Bangalore", "area": "Koramangala",
            "cuisines": "Biryani, Hyderabadi", "price": 280, "rating": 4.3, "veg": False, "popularity": 0.8, "delivery_time_mins": 28,
            "tags": ["biryani", "mutton", "kebab"],
            "menu_items": [
                {"id": 401, "name": "Mutton Dum Biryani", "price": 320, "veg": False, "cuisine": "biryani", "category": "Biryani", "tags": ["mutton", "biryani"]},
                {"id": 402, "name": "Chicken Dum Biryani", "price": 280, "veg": False, "cuisine": "biryani", "category": "Biryani", "tags": ["chicken", "biryani"]}
            ]
        },
        {
            "id": 5, "name": "Dragon Express", "city": "Bangalore", "area": "Indiranagar",
            "cuisines": "Chinese, Asian", "price": 260, "rating": 4.1, "veg": False, "popularity": 0.75, "delivery_time_mins": 32,
            "tags": ["noodles", "fried rice", "dim sum"],
            "menu_items": [
                {"id": 501, "name": "Schezwan Hakka Noodles", "price": 240, "veg": True, "cuisine": "chinese", "category": "Noodles", "tags": ["noodles", "spicy"]},
                {"id": 502, "name": "Chicken Fried Rice", "price": 260, "veg": False, "cuisine": "chinese", "category": "Rice", "tags": ["chicken", "rice"]}
            ]
        },
        {
            "id": 6, "name": "Burger House", "city": "Bangalore", "area": "Jayanagar",
            "cuisines": "American, Fast Food", "price": 180, "rating": 4.4, "veg": True, "popularity": 0.88, "delivery_time_mins": 20,
            "tags": ["burger", "fries", "shake"],
            "menu_items": [
                {"id": 601, "name": "Veg Cheese Burger", "price": 180, "veg": True, "cuisine": "american", "category": "Burgers", "tags": ["burger", "cheese"]},
                {"id": 602, "name": "Peri Peri French Fries", "price": 120, "veg": True, "cuisine": "fast food", "category": "Sides", "tags": ["fries", "spicy"]}
            ]
        },
        {
            "id": 7, "name": "South Tiffin House", "city": "Bangalore", "area": "BTM Layout",
            "cuisines": "South Indian, Dosa", "price": 120, "rating": 4.7, "veg": True, "popularity": 0.95, "delivery_time_mins": 18,
            "tags": ["dosa", "idli", "filter coffee"],
            "menu_items": [
                {"id": 701, "name": "Masala Dosa", "price": 110, "veg": True, "cuisine": "south indian", "category": "Tiffin", "tags": ["dosa", "potato"]},
                {"id": 702, "name": "Steamed Idli Sambar", "price": 80, "veg": True, "cuisine": "south indian", "category": "Tiffin", "tags": ["idli", "sambar"]}
            ]
        },
        {
            "id": 8, "name": "Sweet Bengal", "city": "Bangalore", "area": "Whitefield",
            "cuisines": "Desserts, Bengali", "price": 150, "rating": 4.6, "veg": True, "popularity": 0.65, "delivery_time_mins": 22,
            "tags": ["sweets", "rasgulla", "mishti doi"],
            "menu_items": [
                {"id": 801, "name": "Keshari Rasgulla", "price": 140, "veg": True, "cuisine": "desserts", "category": "Sweets", "tags": ["sweet", "rasgulla"]},
                {"id": 802, "name": "Mishti Doi", "price": 120, "veg": True, "cuisine": "bengali", "category": "Desserts", "tags": ["sweet", "curd"]}
            ]
        }
    ]
    
    proc, meta = preprocess_catalog(sample_raw)
    with open("data/processed_catalog.json", "w") as f:
        json.dump(proc, f, indent=2)
    with open("data/processed_catalog.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Preprocessed {len(proc)} restaurants with {meta['total_dishes']} dishes.")
    
    # Generate versioned offline embeddings
    generate_offline_embeddings(proc)
