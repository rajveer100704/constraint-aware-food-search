import pytest
from parser_pipeline import HybridQueryParser
from schemas import ParsedQuerySchema

@pytest.fixture
def parser():
    return HybridQueryParser()

def test_hybrid_parse_price_under(parser):
    res: ParsedQuerySchema = parser.parse("biryani under 300")
    assert res.max_price == 300.0

def test_hybrid_parse_rating_gte(parser):
    res: ParsedQuerySchema = parser.parse("pizza with 4+ stars")
    assert res.min_rating == 4.0

def test_hybrid_parse_exclusions(parser):
    res: ParsedQuerySchema = parser.parse("pizza without mushroom")
    assert "mushroom" in res.exclusions

def test_hybrid_parse_dietary_veg(parser):
    res: ParsedQuerySchema = parser.parse("vegetarian south indian food")
    assert res.veg_only is True

def test_hybrid_parse_dietary_non_veg(parser):
    res: ParsedQuerySchema = parser.parse("non-veg chicken biryani")
    assert res.veg_only is False
