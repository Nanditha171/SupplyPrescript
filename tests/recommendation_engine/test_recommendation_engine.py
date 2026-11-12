"""Tests for Recommendation Engine (Day 11)."""

import pytest
from src.recommendation_engine.schemas import RecommendationOption
from src.recommendation_engine.ranking import rank_recommendations
from src.recommendation_engine.explanation import generate_strategy_explanation


@pytest.fixture
def sample_options():
    return [
        RecommendationOption(
            recommendation_code="ALT_A",
            strategy_type="AIR_FREIGHT",
            strategy_description="Air freight",
            estimated_cost=15000.0,
            estimated_delay_days=2.0,
            estimated_risk=0.08,
            service_level=1.0,
            rank=0,
            is_recommended=False,
            advantages=[],
            disadvantages=[],
            reasoning="",
        ),
        RecommendationOption(
            recommendation_code="ALT_B",
            strategy_type="SECONDARY_SUPPLIER",
            strategy_description="Secondary supplier",
            estimated_cost=138050.0,
            estimated_delay_days=7.0,
            estimated_risk=0.15,
            service_level=1.0,
            rank=0,
            is_recommended=False,
            advantages=[],
            disadvantages=[],
            reasoning="",
        ),
        RecommendationOption(
            recommendation_code="ALT_C",
            strategy_type="DELAY_LAUNCH",
            strategy_description="Delay product launch",
            estimated_cost=21800.0,
            estimated_delay_days=14.0,
            estimated_risk=0.35,
            service_level=0.95,
            rank=0,
            is_recommended=False,
            advantages=[],
            disadvantages=[],
            reasoning="",
        ),
    ]


def test_ranking_min_cost(sample_options):
    ranked = rank_recommendations(sample_options, ranking_mode="MIN_COST")
    assert ranked[0].strategy_type == "AIR_FREIGHT"
    assert ranked[0].rank == 1
    assert ranked[0].is_recommended is True
    assert ranked[1].strategy_type == "DELAY_LAUNCH"
    assert ranked[2].strategy_type == "SECONDARY_SUPPLIER"


def test_ranking_min_risk(sample_options):
    ranked = rank_recommendations(sample_options, ranking_mode="MIN_RISK")
    assert ranked[0].strategy_type == "AIR_FREIGHT"
    assert ranked[1].strategy_type == "SECONDARY_SUPPLIER"
    assert ranked[2].strategy_type == "DELAY_LAUNCH"


def test_ranking_min_delay(sample_options):
    ranked = rank_recommendations(sample_options, ranking_mode="MIN_DELAY")
    assert ranked[0].strategy_type == "AIR_FREIGHT"  # 2 days
    assert ranked[1].strategy_type == "SECONDARY_SUPPLIER"  # 7 days
    assert ranked[2].strategy_type == "DELAY_LAUNCH"  # 14 days


def test_ranking_balanced(sample_options):
    ranked = rank_recommendations(sample_options, ranking_mode="BALANCED")
    assert len(ranked) == 3
    assert ranked[0].rank == 1
    assert ranked[0].is_recommended is True


def test_explanation_generation():
    desc, adv, disadv, reason = generate_strategy_explanation(
        strategy_type="AIR_FREIGHT",
        estimated_cost=15000.0,
        estimated_delay_days=2.0,
        estimated_risk=0.08,
        service_level=1.0,
        product_name="Microchip X1",
    )
    assert "air" in desc.lower()
    assert len(adv) > 0
    assert len(disadv) > 0
    assert "2.0" in reason

