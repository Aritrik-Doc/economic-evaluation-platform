import math

import pytest

from model.budget_impact import (
    AnnualCostInput,
    BudgetImpactDefinition,
    BudgetImpactValidationError,
    BudgetIntervention,
    PopulationYear,
    TreatmentMixShare,
    apply_simple_scenario,
    reallocate_target_share,
    run_budget_impact,
    top_down_eligible_population,
)


def _definition():
    interventions = (
        BudgetIntervention("current", "Current treatment"),
        BudgetIntervention("new", "New intervention"),
    )
    population = (
        PopulationYear(1, 1000, 100_000),
        PopulationYear(2, 1100, 101_000),
    )
    mix = (
        TreatmentMixShare("current", 1, "current", 1.0),
        TreatmentMixShare("current", 1, "new", 0.0),
        TreatmentMixShare("future", 1, "current", 0.8),
        TreatmentMixShare("future", 1, "new", 0.2),
        TreatmentMixShare("current", 2, "current", 1.0),
        TreatmentMixShare("current", 2, "new", 0.0),
        TreatmentMixShare("future", 2, "current", 0.6),
        TreatmentMixShare("future", 2, "new", 0.4),
    )
    costs = []
    for year in (1, 2):
        costs.extend(
            [
                AnnualCostInput("current", year, "acquisition", 100.0),
                AnnualCostInput("current", year, "monitoring", 20.0),
                AnnualCostInput("new", year, "acquisition", 200.0),
                AnnualCostInput("new", year, "monitoring", 10.0),
            ]
        )
    return BudgetImpactDefinition(
        interventions=interventions,
        population=population,
        treatment_mix=mix,
        costs=tuple(costs),
    )


def test_top_down_population_funnel():
    value = top_down_eligible_population(1_000_000, 0.10, 0.80, 0.50, 0.75)
    assert value == pytest.approx(30_000)


def test_budget_impact_basic_annual_and_cumulative_results():
    result = run_budget_impact(_definition())
    y1, y2 = result.years

    assert y1.current_cost == pytest.approx(120_000)
    assert y1.future_cost == pytest.approx(138_000)
    assert y1.net_budget_impact == pytest.approx(18_000)
    assert y1.cumulative_budget_impact == pytest.approx(18_000)
    assert y1.pmpm_budget_impact == pytest.approx(18_000 / (100_000 * 12))

    assert y2.current_cost == pytest.approx(132_000)
    assert y2.future_cost == pytest.approx(171_600)
    assert y2.net_budget_impact == pytest.approx(39_600)
    assert y2.cumulative_budget_impact == pytest.approx(57_600)
    assert result.cumulative_budget_impact == pytest.approx(57_600)


def test_cost_categories_can_be_included_or_excluded():
    definition = _definition()
    acquisition_only = BudgetImpactDefinition(
        interventions=definition.interventions,
        population=definition.population,
        treatment_mix=definition.treatment_mix,
        costs=definition.costs,
        included_cost_categories=("acquisition",),
    )
    result = run_budget_impact(acquisition_only)
    assert result.years[0].current_cost == pytest.approx(100_000)
    assert result.years[0].future_cost == pytest.approx(120_000)
    assert result.years[0].net_budget_impact == pytest.approx(20_000)


def test_treatment_mix_must_sum_to_one():
    definition = _definition()
    broken = list(definition.treatment_mix)
    broken[2] = TreatmentMixShare("future", 1, "current", 0.7)
    invalid = BudgetImpactDefinition(
        interventions=definition.interventions,
        population=definition.population,
        treatment_mix=tuple(broken),
        costs=definition.costs,
    )
    with pytest.raises(BudgetImpactValidationError, match="sum"):
        run_budget_impact(invalid)


def test_reallocate_target_share_preserves_total_and_relative_other_mix():
    shares = {"a": 0.5, "b": 0.3, "new": 0.2}
    adjusted = reallocate_target_share(shares, "new", 0.4)
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert adjusted["new"] == pytest.approx(0.4)
    assert adjusted["a"] / adjusted["b"] == pytest.approx(0.5 / 0.3)


def test_simple_scenario_changes_population_cost_and_uptake():
    definition = _definition()
    scenario = apply_simple_scenario(
        definition,
        population_multiplier=1.1,
        target_intervention_id="new",
        uptake_multiplier=1.5,
        target_cost_multiplier=0.9,
    )
    result = run_budget_impact(scenario)

    assert scenario.population[0].eligible_population == pytest.approx(1100)
    future_y1 = {
        row.intervention_id: row.share
        for row in scenario.treatment_mix
        if row.scenario == "future" and row.year == 1
    }
    assert future_y1["new"] == pytest.approx(0.3)
    assert sum(future_y1.values()) == pytest.approx(1.0)
    new_cost = next(
        row.cost_per_treated_person
        for row in scenario.costs
        if row.intervention_id == "new" and row.year == 1 and row.category == "acquisition"
    )
    assert new_cost == pytest.approx(180.0)
    assert math.isfinite(result.cumulative_budget_impact)


def test_population_cannot_exceed_covered_lives():
    with pytest.raises(ValueError, match="cannot exceed"):
        PopulationYear(1, 2000, 1000)
