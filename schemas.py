from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class MenuItem(BaseModel):
    id: int
    name: str
    price: float
    veg: bool
    cuisine: str
    category: str
    tags: List[str] = Field(default_factory=list)

class RestaurantSchema(BaseModel):
    id: int
    name: str
    city: str
    area: str
    cuisines: List[str]
    price: float
    rating: float
    veg: bool
    popularity: float
    delivery_time_mins: int
    tags: List[str] = Field(default_factory=list)
    menu_items: List[MenuItem] = Field(default_factory=list)

class ParsedQuerySchema(BaseModel):
    raw_query: str
    domain: str = "food"
    cuisine: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    price_inclusive: bool = False
    veg_only: Optional[bool] = None
    min_rating: Optional[float] = None
    area: Optional[str] = None
    city: Optional[str] = None
    exclusions: List[str] = Field(default_factory=list)
    expanded_terms: List[str] = Field(default_factory=list)
    parser_used: str = "search_expert"  # 'search_expert' or 'regex_fallback'
    parsing_latency_ms: float = 0.0

class SearchRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "vegetarian thali under 250"})
    k: int = Field(default=5, ge=1, le=50)
    max_price: Optional[float] = Field(default=None, json_schema_extra={"example": 250.0})
    min_rating: Optional[float] = Field(default=None, json_schema_extra={"example": 4.0})
    veg_only: Optional[bool] = Field(default=None, json_schema_extra={"example": True})

class SearchResponseItem(BaseModel):
    id: int
    name: str
    city: str
    area: str
    cuisines: List[str]
    price: float
    rating: float
    score: float
    matched_dish: Optional[str] = None
    top_feature: str
    second_feature: str
    reasons: List[str]

class SearchResponse(BaseModel):
    query: str
    total_results: int
    latency_ms: float
    parser_used: str
    results: List[SearchResponseItem]

class EvaluationMetricsSchema(BaseModel):
    recall_at_5: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    constraint_satisfaction: float
    avg_latency_ms: float
    filter_failures: int
    ranking_failures: int
