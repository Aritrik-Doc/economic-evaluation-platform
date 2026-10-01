import math

import pytest

from model.economics import (
    OUTCOME_MEASURES,
    Strategy,
    evaluate_two_strategies,
    fully_incremental_analysis,
)


def test_outcome_measure_catalog():
    assert set(OUTCOME_MEASURES) == {"QALY", "LYG", "DALY_AVERTED"}
    assert OUTCOME_MEASURES["DALY_AVERTED"].label == "DALYs averted"


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


def test_multi_strategy_incremental_frontier():
    strategies = [
        Strategy("A", 10_000, 1.0),
        Strategy("B", 12_000, 1.2),
        Strategy("C", 15_000, 1.5),
    ]

    result = fully_incremental_analysis(strategies, 30_000)
    frontier = result.efficient_frontier

    assert [r.strategy.name for r in frontier] == ["A", "B", "C"]
    assert frontier[0].icer is None
    assert frontier[1].icer == pytest.approx(10_000)
    assert frontier[1].compared_with == "A"
    assert frontier[2].icer == pytest.approx(10_000)
    assert frontier[2].compared_with == "B"


def test_strong_dominance_in_multi_strategy_analysis():
    strategies = [
        Strategy("A", 10_000, 1.0),
        Strategy("B", 14_000, 1.1),
        Strategy("C", 13_000, 1.2),
    ]

    result = fully_incremental_analysis(strategies, 30_000)
    status = {r.strategy.name: r.status for r in result.rows}

    assert status["B"] == "strongly_dominated"
    assert status["A"] == "efficient"
    assert status["C"] == "efficient"


def test_extended_dominance_in_multi_strategy_analysis():
    strategies = [
        Strategy("A", 10_000, 1.0),
        Strategy("B", 14_000, 1.2),
        Strategy("C", 16_000, 1.4),
    ]

    result = fully_incremental_analysis(strategies, 30_000)
    rows = {r.strategy.name: r for r in result.rows}

    assert rows["B"].status == "extendedly_dominated"
    assert rows["C"].status == "efficient"
    assert rows["C"].compared_with == "A"
    assert rows["C"].icer == pytest.approx(15_000)


def test_nmb_preference_across_multiple_strategies():
    strategies = [
        Strategy("A", 10_000, 1.0),
        Strategy("B", 13_000, 1.2),
        Strategy("C", 20_000, 1.5),
    ]

    result = fully_incremental_analysis(strategies, 30_000)

    assert result.preferred_by_nmb == ("C",)


def test_equal_effect_more_costly_is_strongly_dominated():
    strategies = [
        Strategy("A", 10_000, 1.0),
        Strategy("B", 12_000, 1.0),
        Strategy("C", 15_000, 1.5),
    ]

    result = fully_incremental_analysis(strategies, 30_000)
    status = {r.strategy.name: r.status for r in result.rows}
    assert status["B"] == "strongly_dominated"


def test_duplicate_coordinates_rejected():
    with pytest.raises(ValueError, match="identical cost and effect"):
        fully_incremental_analysis(
            [Strategy("A", 10_000, 1.0), Strategy("B", 10_000, 1.0)],
            30_000,
        )


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        fully_incremental_analysis(
            [Strategy("A", 10_000, 1.0), Strategy("A", 12_000, 1.2)],
            30_000,
        )


def test_at_least_two_strategies_required():
    with pytest.raises(ValueError, match="At least two"):
        fully_incremental_analysis([Strategy("A", 10_000, 1.0)], 30_000)


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
