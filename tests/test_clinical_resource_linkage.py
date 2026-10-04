import pytest

from model.clinical_resource_linkage import (
    StateResourceMapping,
    TreeResourceMapping,
    combine_resource_requirements,
    profiles_to_requirements,
    project_decision_tree_resource_profiles,
    project_markov_resource_profiles,
)
from model.decision_tree import ChanceNode, DecisionTreeDefinition, StrategyRoot, TerminalNode, TreeBranch
from model.markov import CohortMarkovDefinition, InitialStateAllocation, MarkovState, MarkovStrategyDefinition
from model.resource_capacity import ResourceRequirement
from model.schema import AssumptionSpec, EvidenceSource, Parameter, UncertaintySpec


SOURCE = EvidenceSource(citation="Regression source", source_type="user_assumption")
ASSUMPTION = AssumptionSpec("Regression assumption", "Exercises resource linkage.")


def _parameter(pid, value, category="resource_use"):
    return Parameter(
        id=pid,
        label=pid,
        value=value,
        unit="hours/year" if category == "resource_use" else "probability",
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=UncertaintySpec(kind="none", rationale="Fixed regression input."),
    )


def test_markov_state_resource_profile_uses_state_occupancy_and_selected_cycle():
    states = (MarkovState("stable", "Stable", absorbing=True), MarkovState("other", "Other", absorbing=True))
    strategies = (
        MarkovStrategyDefinition("A", "A", (InitialStateAllocation("stable", 1.0),), ()),
        MarkovStrategyDefinition("B", "B", (InitialStateAllocation("other", 1.0),), ()),
    )
    model = CohortMarkovDefinition(
        states=states,
        strategies=strategies,
        cycle_length_years=0.5,
        max_cycles=4,
        state_accrual_timing="start",
    )
    parameters = (_parameter("staff_hours", 4.0),)
    mappings = (
        StateResourceMapping("A", "stable", "staff", "staff_hours", "per_year"),
        StateResourceMapping("B", "stable", "staff", "staff_hours", "per_year"),
    )
    profiles = project_markov_resource_profiles(model, parameters, mappings=mappings, horizon_years=2)
    by_strategy = {profile.strategy_id: profile for profile in profiles}
    assert by_strategy["A"].annual_units_per_patient == pytest.approx((4.0, 4.0))
    assert by_strategy["B"].annual_units_per_patient == pytest.approx((0.0, 0.0))


def test_decision_tree_resource_profile_is_probability_weighted_and_timed():
    parameters = (
        _parameter("p", 0.25, "clinical"),
        _parameter("scan", 2.0),
    )
    tree = DecisionTreeDefinition(
        strategy_roots=(StrategyRoot("A", "root_a"), StrategyRoot("B", "root_b")),
        chance_nodes=(
            ChanceNode(
                "root_a",
                "Root A",
                (
                    TreeBranch("event", "p", "a_event", "direct"),
                    TreeBranch("none", "p", "a_none", "complement"),
                ),
            ),
            ChanceNode(
                "root_b",
                "Root B",
                (
                    TreeBranch("event", "p", "b_event", "direct"),
                    TreeBranch("none", "p", "b_none", "complement"),
                ),
            ),
        ),
        terminal_nodes=(
            TerminalNode("a_event", "A event"),
            TerminalNode("a_none", "A none"),
            TerminalNode("b_event", "B event"),
            TerminalNode("b_none", "B none"),
        ),
    )
    mappings = (
        TreeResourceMapping("A", "a_event", "scan_resource", "scan", 1.5),
        TreeResourceMapping("B", "b_event", "scan_resource", "scan", 1.5),
    )
    profiles = project_decision_tree_resource_profiles(
        tree,
        parameters,
        mappings=mappings,
        horizon_years=2,
        strategy_names={"A": "A", "B": "B"},
    )
    for profile in profiles:
        assert profile.annual_units_per_patient == pytest.approx((0.0, 0.5))


def test_profiles_convert_to_capacity_requirements_and_hybrid_adds_manual_use():
    states = (MarkovState("stable", "Stable", absorbing=True), MarkovState("other", "Other", absorbing=True))
    model = CohortMarkovDefinition(
        states=states,
        strategies=(
            MarkovStrategyDefinition("A", "A", (InitialStateAllocation("stable", 1.0),), ()),
            MarkovStrategyDefinition("B", "B", (InitialStateAllocation("stable", 1.0),), ()),
        ),
        cycle_length_years=1.0,
        max_cycles=2,
        state_accrual_timing="start",
    )
    profiles = project_markov_resource_profiles(
        model,
        (_parameter("hours", 3.0),),
        mappings=(
            StateResourceMapping("A", "stable", "staff", "hours"),
            StateResourceMapping("B", "stable", "staff", "hours"),
        ),
        horizon_years=2,
    )
    linked = profiles_to_requirements(
        profiles,
        intervention_to_strategy={"old": "A", "new": "B"},
        horizon_years=2,
    )
    combined = combine_resource_requirements(
        (ResourceRequirement("new", "staff", 1, 1.0),),
        linked,
    )
    lookup = {(row.intervention_id, row.resource_id, row.period): row.units_per_person for row in combined}
    assert lookup[("new", "staff", 1)] == pytest.approx(4.0)
    assert lookup[("old", "staff", 2)] == pytest.approx(3.0)
