import numpy as np
import pytest

from model.markov import (
    CohortMarkovDefinition,
    InitialStateAllocation,
    MarkovState,
    MarkovStrategyDefinition,
    StateReward,
    TransitionProbability,
)
from model.markov_psa import run_markov_psa
from model.psa import ceac, incremental_plane
from model.schema import (
    AssumptionSpec,
    DeterministicUncertaintySpec,
    DistributionSpec,
    EvidenceSource,
    Parameter,
    ProbabilisticUncertaintySpec,
)


SOURCE = EvidenceSource(citation="Illustrative", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(statement="Illustrative", rationale="Unit test")
NO_DSA = DeterministicUncertaintySpec(enabled=False, rationale="Not represented in DSA.")


def parameter(id, value, category="clinical", *, psa=None):
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
        dsa=NO_DSA,
        psa=psa or ProbabilisticUncertaintySpec(enabled=False, rationale="Not represented in PSA."),
        **kwargs,
    )


def model_and_parameters():
    states = (MarkovState("alive", "Alive"), MarkovState("dead", "Dead", absorbing=True))
    initial = (InitialStateAllocation("alive", 1.0),)

    def strategy(sid, p, cost):
        return MarkovStrategyDefinition(
            sid,
            sid,
            initial,
            (
                TransitionProbability("alive", "dead", p),
                TransitionProbability("alive", "alive", probability_mode="residual"),
            ),
            state_rewards=(
                StateReward("alive", cost, "cost", "per_year"),
                StateReward("alive", "utility", "outcome", "per_year"),
            ),
        )

    model = CohortMarkovDefinition(
        states=states,
        strategies=(strategy("A", "p_a", "cost_a"), strategy("B", "p_b", "cost_b")),
        cycle_length_years=1.0,
        max_cycles=8,
    )
    beta = lambda alpha, beta: ProbabilisticUncertaintySpec(
        enabled=True,
        rationale="Sampling uncertainty",
        distribution=DistributionSpec("beta", (("alpha", alpha), ("beta", beta))),
    )
    parameters = (
        parameter("p_a", 0.10, psa=beta(10, 90)),
        parameter("p_b", 0.15, psa=beta(15, 85)),
        parameter("cost_a", 100, "cost"),
        parameter("cost_b", 80, "cost"),
        parameter("utility", 1.0, "utility"),
    )
    return model, parameters


def test_markov_psa_is_reproducible():
    model, parameters = model_and_parameters()
    a = run_markov_psa(model, parameters, iterations=50, seed=123)
    b = run_markov_psa(model, parameters, iterations=50, seed=123)
    assert np.array_equal(a.parameter_draws["p_a"], b.parameter_draws["p_a"])
    assert np.array_equal(a.costs["A"], b.costs["A"])
    assert np.array_equal(a.outcomes["B"], b.outcomes["B"])


def test_markov_psa_outputs_feed_common_ce_plane_and_ceac():
    model, parameters = model_and_parameters()
    result = run_markov_psa(model, parameters, iterations=100, seed=7)
    delta_effect, delta_cost = incremental_plane(
        result,
        intervention_id="A",
        comparator_id="B",
    )
    assert delta_effect.shape == (100,)
    assert delta_cost.shape == (100,)
    curve = ceac(result, [0, 20_000])
    assert curve.thresholds.tolist() == [0.0, 20_000.0]
    for i in range(2):
        total = sum(values[i] for values in curve.probabilities.values())
        assert total == pytest.approx(1.0)
