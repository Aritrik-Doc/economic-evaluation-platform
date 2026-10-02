import numpy as np
import pytest

from model.markov import InitialStateAllocation, MarkovState, StateReward
from model.schema import (
    AssumptionSpec,
    DistributionSpec,
    EvidenceSource,
    Parameter,
    UncertaintySpec,
)
from model.semi_markov import DynamicTransition, SemiMarkovDefinition, SemiMarkovStrategyDefinition
from model.semi_markov_psa import run_semi_markov_psa
from model.semi_markov_sensitivity import tornado_semi_markov_inmb
from model.transition_dynamics import ParameterBand, PiecewiseParameterSchedule


SOURCE = EvidenceSource(citation="Illustrative source", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(statement="Test assumption.", rationale="Unit-test construction.")


def parameter(pid, value, category="clinical", uncertainty=None):
    if uncertainty is None:
        uncertainty = UncertaintySpec(kind="none", rationale="Fixed test parameter.")
    kwargs = {}
    if category == "cost":
        kwargs = {"currency": "GBP", "price_year": 2026, "cost_bearers": ("health_system",)}
    return Parameter(
        id=pid,
        label=pid,
        value=value,
        unit="unit",
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=uncertainty,
        **kwargs,
    )


def schedule(pid):
    return PiecewiseParameterSchedule("state_time", (ParameterBand(0.0, pid, None),))


def build_model():
    uncertain_p = parameter(
        "p_death_a",
        0.20,
        uncertainty=UncertaintySpec(
            kind="range_and_distribution",
            rationale="Illustrative uncertainty.",
            lower=0.10,
            upper=0.30,
            distribution=DistributionSpec("beta", (("alpha", 20.0), ("beta", 80.0))),
        ),
    )
    parameters = (
        uncertain_p,
        parameter("p_death_b", 0.25),
        parameter("cost_a", 100.0, "cost"),
        parameter("cost_b", 80.0, "cost"),
        parameter("utility", 1.0, "utility"),
    )
    states = (MarkovState("alive", "Alive"), MarkovState("dead", "Dead", absorbing=True))

    def strategy(sid, probability_parameter, cost_parameter):
        return SemiMarkovStrategyDefinition(
            strategy_id=sid,
            label=sid,
            initial_distribution=(InitialStateAllocation("alive", 1.0),),
            transitions=(DynamicTransition("alive", "dead", schedule(probability_parameter)),),
            state_rewards=(
                StateReward("alive", cost_parameter, "cost", "per_year"),
                StateReward("alive", "utility", "outcome", "per_year"),
            ),
        )

    model = SemiMarkovDefinition(
        states=states,
        strategies=(strategy("A", "p_death_a", "cost_a"), strategy("B", "p_death_b", "cost_b")),
        cycle_length_years=1.0,
        max_cycles=5,
        state_accrual_timing="start",
    )
    return model, parameters


def test_tornado_reruns_complete_semi_markov_model():
    model, parameters = build_model()
    rows = tornado_semi_markov_inmb(
        model,
        parameters,
        intervention_id="A",
        comparator_id="B",
        willingness_to_pay=1000.0,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.parameter_id == "p_death_a"
    assert row.low_inmb != pytest.approx(row.high_inmb)
    assert row.impact > 0


def test_semi_markov_psa_is_seed_reproducible_and_returns_strategy_draws():
    model, parameters = build_model()
    first = run_semi_markov_psa(model, parameters, iterations=50, seed=123)
    second = run_semi_markov_psa(model, parameters, iterations=50, seed=123)
    assert first.strategy_ids == ("A", "B")
    assert np.array_equal(first.parameter_draws["p_death_a"], second.parameter_draws["p_death_a"])
    assert np.array_equal(first.costs["A"], second.costs["A"])
    assert np.array_equal(first.outcomes["A"], second.outcomes["A"])
    assert np.std(first.outcomes["A"]) > 0
