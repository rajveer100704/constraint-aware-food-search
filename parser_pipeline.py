import re
import time
from typing import Optional, Dict, Any, List
from schemas import ParsedQuerySchema

# Attempt real PyPI search-expert package import (gracefully fallback if CUDA GPU unavailable for Unsloth)
try:
    from search_expert import SearchExpert, ModelFormat
    _HAS_SEARCH_EXPERT_PKG = True
except Exception:
    _HAS_SEARCH_EXPERT_PKG = False

class RegexFallbackParser:
    """Rule-based regex fallback query parser for resilient offline query processing."""
    
    def parse(self, query_str: str) -> ParsedQuerySchema:
        q_lower = query_str.strip().lower()
        parsed = ParsedQuerySchema(raw_query=query_str, parser_used="regex_fallback")
        
        # 1. Price extraction
        between_match = re.search(r'between\s+(?:₹|rs\.?|usd|\$)?\s*(\d+)\s+and\s+(?:₹|rs\.?|usd|\$)?\s*(\d+)', q_lower)
        if between_match:
            parsed.min_price = float(between_match.group(1))
            parsed.max_price = float(between_match.group(2))
        else:
            under_match = re.search(r'(?:under|below|less than|max|up to)\s+(?:₹|rs\.?|usd|\$)?\s*(\d+)', q_lower)
            if under_match:
                parsed.max_price = float(under_match.group(1))
                parsed.price_inclusive = False
            elif "cheap" in q_lower:
                parsed.max_price = 300.0
                parsed.price_inclusive = True

        # 2. Rating extraction
        rating_match = re.search(r'(\d(?:\.\d)?)\s*\+?\s*(?:star|stars|rating)', q_lower)
        if rating_match:
            parsed.min_rating = float(rating_match.group(1))
        elif "top rated" in q_lower or "best rated" in q_lower:
            parsed.min_rating = 4.5

        # 3. Veg / Non-Veg extraction
        if re.search(r'\b(pure veg|vegetarian|veg only|veg)\b', q_lower) and not re.search(r'\b(non-veg|non veg)\b', q_lower):
            parsed.veg_only = True
        elif re.search(r'\b(non-veg|non veg|chicken|mutton)\b', q_lower):
            parsed.veg_only = False

        # 4. Exclusions
        exclusion_match = re.search(r'(?:no|without|except|excluding)\s+([a-zA-Z\s,]+)', q_lower)
        if exclusion_match:
            ex_items = [x.strip() for x in re.split(r'and|,|\s+', exclusion_match.group(1)) if x.strip()]
            parsed.exclusions = [item for item in ex_items if item not in ["food", "under", "in", "with"]]

        # 5. Cuisine & Area
        cuisines_list = ["thali", "pizza", "biryani", "chinese", "burger", "dosa", "south indian", "north indian", "italian", "desserts", "sweets", "american", "mughlai"]
        for c in cuisines_list:
            if c in q_lower:
                parsed.cuisine = c
                break

        areas_list = ["koramangala", "indiranagar", "hsr layout", "jayanagar", "btm layout", "whitefield"]
        for a in areas_list:
            if a in q_lower:
                parsed.area = a
                break

        return parsed


class HybridQueryParser:
    """
    Production Hybrid Query Parser.
    Attempts fine-tuned SearchExpert (SLM) parsing first via PyPI package,
    and falls back to RegexFallbackParser if SLM is unavailable or fails.
    """
    def __init__(self, use_slm: bool = True):
        self.use_slm = use_slm and _HAS_SEARCH_EXPERT_PKG
        self.slm_expert = None
        self.regex_fallback = RegexFallbackParser()
        
        if self.use_slm:
            try:
                # Instantiates PyPI search_expert JSON adapter
                self.slm_expert = SearchExpert(model_format=ModelFormat.JSON)
            except Exception:
                self.use_slm = False

    def parse(self, query_str: str) -> ParsedQuerySchema:
        t0 = time.perf_counter()
        
        if self.use_slm and self.slm_expert is not None:
            try:
                res = self.slm_expert.parse(query_str)
                parsed = self._map_slm_result(query_str, res)
                t1 = time.perf_counter()
                parsed.parsing_latency_ms = (t1 - t0) * 1000.0
                parsed.parser_used = "search_expert"
                return parsed
            except Exception:
                pass  # Gracefully drop to regex fallback on SLA breach or parse error
                
        # Regex Fallback path
        parsed = self.regex_fallback.parse(query_str)
        t1 = time.perf_counter()
        parsed.parsing_latency_ms = (t1 - t0) * 1000.0
        return parsed

    def _map_slm_result(self, raw_query: str, res: Any) -> ParsedQuerySchema:
        """Maps search_expert ParseResult fields to internal ParsedQuerySchema."""
        fields = getattr(res, "fields", {})
        parsed = ParsedQuerySchema(raw_query=raw_query, parser_used="search_expert")
        
        # Extract numeric constraints
        if hasattr(res, "get_numeric_constraint"):
            price_c = res.get_numeric_constraint("price")
            if price_c:
                op = price_c.get("operator")
                if op in ("lt", "lte", "approx"):
                    parsed.max_price = price_c.get("value")
                elif op == "gte":
                    parsed.min_price = price_c.get("value")
                elif op == "between":
                    parsed.min_price = price_c.get("value")
                    parsed.max_price = price_c.get("value_hi")
                    
            rating_c = res.get_numeric_constraint("rating")
            if rating_c and rating_c.get("operator") == "gte":
                parsed.min_rating = rating_c.get("value")
                
        if hasattr(res, "get_exclusions"):
            parsed.exclusions = res.get_exclusions("exclusions")
            
        parsed.cuisine = fields.get("cuisine")
        parsed.area = fields.get("area")
        
        diet = fields.get("diet")
        if diet == "veg":
            parsed.veg_only = True
        elif diet == "non-veg":
            parsed.veg_only = False
            
        return parsed
