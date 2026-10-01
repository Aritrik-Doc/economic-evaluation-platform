import math

import pytest

from model.economics import Strategy, evaluate_two_strategies


def test_base_case_calculations():
    comparator = Strategy("Standard care", cost=10_000, effect=4.0)
    intervention = Strategy("New treatment", cost=15_000, effect=4.3)

    result = evaluate_two_strategies(comparator, intervention, willingness_to_pay=30_000)

    assert result.incremental_cost == pytest.approx(5_000)
    assert result.incremental_effect == pytest.approx(0.3)
    assert result.icer == pytest.approx(16_666.6666667)
    assert result.comparator_nmb == pytest.approx(110_000)
    assert result.intervention_nmb == pytest.approx(114_000)
    assert result.incremental_nmb == pytest.approx(4_000)
    assert result.status == "more_costly_more_effective"
    assert result.preferred_by_nmb == "New treatment"


def test_dominant_intervention():
    comparator = Strategy("Comparator", 10_000, 2.0)
    intervention = Strategy("Intervention", 9_000, 2.2)

    result = evaluate_two_strategies(comparator, intervention, 30_000)

    assert result.status == "dominant"
    assert result.incremental_cost == -1_000
    assert result.incremental_effect == pytest.approx(0.2)
    assert result.icer == pytest.approx(-5_000)
    assert result.incremental_nmb > 0


def test_dominated_intervention():
    comparator = Strategy("Comparator", 10_000, 2.0)
    intervention = Strategy("Intervention", 12_000, 1.8)

    result = evaluate_two_strategies(comparator, intervention, 30_000)

    assert result.status == "dominated"
    assert result.incremental_nmb < 0


def test_equal_effect_has_no_icer():
    comparator = Strategy("Comparator", 10_000, 2.0)
    intervention = Strategy("Intervention", 12_000, 2.0)

    result = evaluate_two_strategies(comparator, intervention, 30_000)

    assert result.icer is None
    assert result.status == "equal_effect_more_costly"


def test_equal_cost_more_effective():
    comparator = Strategy("Comparator", 10_000, 2.0)
    intervention = Strategy("Intervention", 10_000, 2.5)

    result = evaluate_two_strategies(comparator, intervention, 30_000)

    assert result.status == "equal_cost_more_effective"
    assert result.icer == 0
    assert result.incremental_nmb > 0


def test_less_costly_less_effective_can_still_have_positive_inmb():
    comparator = Strategy("Comparator", 10_000, 2.0)
    intervention = Strategy("Intervention", 5_000, 1.9)

    result = evaluate_two_strategies(comparator, intervention, 30_000)

    assert result.status == "less_costly_less_effective"
    assert result.incremental_nmb == pytest.approx(2_000)
    assert result.preferred_by_nmb == "Intervention"


def test_zero_difference_is_tie():
    comparator = Strategy("Comparator", 10_000, 2.0)
    intervention = Strategy("Intervention", 10_000, 2.0)

    result = evaluate_two_strategies(comparator, intervention, 30_000)

    assert result.status == "no_difference"
    assert result.icer is None
    assert result.incremental_nmb == 0
    assert result.preferred_by_nmb == "Tie"


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_non_finite_values_rejected(invalid):
    with pytest.raises(ValueError):
        Strategy("Bad", invalid, 1.0)


def test_empty_strategy_name_rejected():
    with pytest.raises(ValueError):
        Strategy("   ", 100, 1.0)


def test_negative_wtp_rejected():
    comparator = Strategy("Comparator", 10_000, 2.0)
    intervention = Strategy("Intervention", 12_000, 2.2)

    with pytest.raises(ValueError):
        evaluate_two_strategies(comparator, intervention, -1)
