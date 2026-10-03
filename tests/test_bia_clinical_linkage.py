from __future__ import annotations

import pytest

from model.bia_clinical_linkage import (
    ClinicalCostProfile,
    ClinicalLinkageError,
    project_markov_cost_profiles,
    run_linked_budget_impact,
)
from model.budget_impact import (
    AnnualCostInput,
    BudgetImpactDefinition,
    BudgetIntervention,
    PopulationYear,
    TreatmentMixShare,
)
from model.markov import (
    CohortMarkovDefinition,
    InitialStateAllocation,
    MarkovState,
    MarkovStrategyDefinition,
    StateReward,
    TransitionProbability,
)
from model.schema import AssumptionSpec, EvidenceSource, Parameter, UncertaintySpec


def _parameter(pid: str, value: float, category: str = "cost") -> Parameter:
    kwargs = {}
    if category == "cost":
        kwargs.update(currency="GBP", price_year=2026, cost_bearers=("health_system",))
    return Parameter(
        id=pid,
        label=pid,
        value=value,
        unit="currency/year" if category == "cost" else "probability",
        category=category,
        source=EvidenceSource(citation="test", source_type="user_assumption"),
        assumption=AssumptionSpec(statement="test", rationale="test"),
        uncertainty=UncertaintySpec(kind="none", rationale="test"),
        **kwargs,
    )


def _simple_markov(cycle_length: float = 1.0) -> tuple[CohortMarkovDefinition, tuple[Parameter, ...]]:
    states = (
        MarkovState("alive", "Alive"),
        MarkovState("dead", "Dead", absorbing=True),
    )
    transitions = (
        TransitionProbability("alive", "dead", "p_death", "direct"),
        TransitionProbability("alive", "alive", None, "residual"),
    )
    strategies = (
        MarkovStrategyDefinition(
            "a",
            "A",
            (InitialStateAllocation("alive", 1.0),),
            transitions,
            state_rewards=(StateReward("alive", "disease_cost_a", "cost", "per_year"),),
        ),
        MarkovStrategyDefinition(
            "b",
            "B",
            (InitialStateAllocation("alive", 1.0),),
            transitions,
            state_rewards=(
                StateReward("alive", "disease_cost_b", "cost", "per_year"),
                StateReward("alive", "drug_cost_b", "cost", "per_year"),
            ),
        ),
    )
    model = CohortMarkovDefinition(
        states=states,
        strategies=strategies,
        cycle_length_years=cycle_length,
        max_cycles=int(round(3 / cycle_length)),
        state_accrual_timing="start",
    )
    parameters = (
        _parameter("p_death", 0.0, "clinical"),
        _parameter("disease_cost_a", 100.0),
        _parameter("disease_cost_b", 80.0),
        _parameter("drug_cost_b", 1000.0),
    )
    return model, parameters


def _bia() -> BudgetImpactDefinition:
    interventions = (
        BudgetIntervention("current", "Current"),
        BudgetIntervention("new", "New"),
    )
    population = (
        PopulationYear(1, 100.0, 1000.0),
        PopulationYear(2, 100.0, 1000.0),
    )
    mix = (
        TreatmentMixShare("current", 1, "current", 1.0),
        TreatmentMixShare("current", 1, "new", 0.0),
        TreatmentMixShare("future", 1, "current", 0.5),
        TreatmentMixShare("future", 1, "new", 0.5),
        TreatmentMixShare("current", 2, "current", 1.0),
        TreatmentMixShare("current", 2, "new", 0.0),
        TreatmentMixShare("future", 2, "current", 0.0),
        TreatmentMixShare("future", 2, "new", 1.0),
    )
    costs = []
    for intervention_id, acquisition, disease in (
        ("current", 10.0, 999.0),
        ("new", 20.0, 999.0),
    ):
        for year in (1, 2):
            costs.extend(
                [
                    AnnualCostInput(intervention_id, year, "acquisition", acquisition),
                    AnnualCostInput(intervention_id, year, "disease_management", disease),
                ]
            )
    return BudgetImpactDefinition(
        interventions=interventions,
        population=population,
        treatment_mix=mix,
        costs=tuple(costs),
        included_cost_categories=("acquisition", "disease_management"),
    )


def test_markov_projection_can_exclude_drug_cost_to_avoid_double_counting():
    model, parameters = _simple_markov()
    profiles = project_markov_cost_profiles(
        model,
        parameters,
        selected_cost_parameter_ids=("disease_cost_a", "disease_cost_b"),
        horizon_years=2,
    )
    by_id = {profile.strategy_id: profile for profile in profiles}
    assert by_id["a"].annual_cost_per_patient == pytest.approx((100.0, 100.0))
    assert by_id["b"].annual_cost_per_patient == pytest.approx((80.0, 80.0))
    assert "drug_cost_b" not in by_id["b"].included_parameter_ids


def test_linked_bia_replaces_direct_disease_management_and_stacks_cohorts():
    profiles = (
        ClinicalCostProfile("a", "A", (100.0, 50.0), "cohort_markov", ("disease_a",)),
        ClinicalCostProfile("b", "B", (200.0, 100.0), "cohort_markov", ("disease_b",)),
    )
    result = run_linked_budget_impact(
        _bia(),
        profiles,
        intervention_to_strategy={"current": "a", "new": "b"},
    )

    y1, y2 = result.years
    # Direct acquisition only: current 100*10=1000; future 50*10 + 50*20=1500.
    # Clinical Y1: current 100*100=10000; future 50*100 + 50*200=15000.
    assert y1.current_cost == pytest.approx(11000.0)
    assert y1.future_cost == pytest.approx(16500.0)
    assert y1.net_budget_impact == pytest.approx(5500.0)

    # Year 2 includes follow-up costs from Year-1 cohorts plus costs for Year-2 starts.
    # Current: direct 1000 + prior 100*50 + new starts 100*100 = 16000.
    # Future: direct 2000 + prior (50*50 + 50*100) + new starts 100*200 = 29500.
    assert y2.current_cost == pytest.approx(16000.0)
    assert y2.future_cost == pytest.approx(29500.0)
    assert y2.net_budget_impact == pytest.approx(13500.0)
    assert result.cumulative_budget_impact == pytest.approx(19000.0)


def test_linked_bia_requires_complete_strategy_mapping():
    profiles = (
        ClinicalCostProfile("a", "A", (100.0, 50.0), "cohort_markov", ("disease_a",)),
    )
    with pytest.raises(ClinicalLinkageError, match="Every BIA intervention"):
        run_linked_budget_impact(
            _bia(),
            profiles,
            intervention_to_strategy={"current": "a"},
        )


def test_linked_bia_requires_new_treatment_start_basis():
    profiles = (
        ClinicalCostProfile("a", "A", (100.0, 50.0), "cohort_markov", ("disease_a",)),
        ClinicalCostProfile("b", "B", (100.0, 50.0), "cohort_markov", ("disease_b",)),
    )
    with pytest.raises(ClinicalLinkageError, match="new-treatment-start"):
        run_linked_budget_impact(
            _bia(),
            profiles,
            intervention_to_strategy={"current": "a", "new": "b"},
            population_basis="annual_snapshot",
        )


def test_projection_rejects_cycle_length_that_does_not_divide_year():
    model, parameters = _simple_markov(cycle_length=0.4)
    with pytest.raises(ClinicalLinkageError, match="divide one year exactly"):
        project_markov_cost_profiles(
            model,
            parameters,
            selected_cost_parameter_ids=("disease_cost_a", "disease_cost_b"),
            horizon_years=2,
        )
