import pytest

from model.markov import (
    CohortMarkovDefinition,
    InitialStateAllocation,
    MarkovState,
    MarkovStrategyDefinition,
    StateReward,
    TransitionProbability,
)
from model.markov_sensitivity import tornado_markov_inmb
from model.schema import (
    AssumptionSpec,
    DeterministicUncertaintySpec,
    EvidenceSource,
    Parameter,
    ProbabilisticUncertaintySpec,
)


SOURCE = EvidenceSource(citation="Illustrative", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(statement="Illustrative", rationale="Unit test")
NO_PSA = ProbabilisticUncertaintySpec(enabled=False, rationale="Not represented in PSA.")


def parameter(id, value, category="clinical", *, dsa=None):
    kwargs = {}
    if category == "cost":
        kwargs = {"currency": "GBP", "price_year": 2026, "cost_bearers": ("health_system",)}
    return Parameter(
        id=id,
        label=id,
        value=value,
        unit="unit",
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        dsa=dsa or DeterministicUncertaintySpec(enabled=False, rationale="Not represented in DSA."),
        psa=NO_PSA,
        **kwargs,
    )


def test_tornado_uses_split_dsa_ranges():
    states = (MarkovState("alive", "Alive"), MarkovState("dead", "Dead", absorbing=True))
    initial = (InitialStateAllocation("alive", 1.0),)

    def strategy(sid, death_parameter, annual_cost):
        return MarkovStrategyDefinition(
            sid,
            sid,
            initial,
            (
                TransitionProbability("alive", "dead", death_parameter),
                TransitionProbability("alive", "alive", probability_mode="residual"),
            ),
            state_rewards=(
                StateReward("alive", annual_cost, "cost", "per_year"),
                StateReward("alive", "utility", "outcome", "per_year"),
            ),
        )

    model = CohortMarkovDefinition(
        states=states,
        strategies=(strategy("A", "p_a", "cost_a"), strategy("B", "p_b", "cost_b")),
        cycle_length_years=1.0,
        max_cycles=5,
    )
    parameters = (
        parameter(
            "p_a",
            0.10,
            dsa=DeterministicUncertaintySpec(enabled=True, rationale="Interval", lower=0.05, upper=0.20),
        ),
        parameter("p_b", 0.15),
        parameter("cost_a", 100, "cost"),
        parameter("cost_b", 80, "cost"),
        parameter("utility", 1.0, "utility"),
    )
    rows = tornado_markov_inmb(
        model,
        parameters,
        intervention_id="A",
        comparator_id="B",
        willingness_to_pay=20_000,
        included_cost_bearers=("health_system",),
    )
    assert len(rows) == 1
    assert rows[0].parameter_id == "p_a"
    assert rows[0].low_inmb != pytest.approx(rows[0].high_inmb)
