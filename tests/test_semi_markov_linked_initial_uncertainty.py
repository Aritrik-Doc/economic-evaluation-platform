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
from model.semi_markov import SemiMarkovDefinition, SemiMarkovStrategyDefinition
from model.semi_markov_psa import run_semi_markov_psa
from model.semi_markov_sensitivity import tornado_semi_markov_inmb


SOURCE = EvidenceSource(citation="Regression-test source", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(
    statement="Regression-test assumption.",
    rationale="Exercises Advanced Markov parameter-linked starting cohorts.",
)


def _parameter(pid, value, category="clinical", uncertainty=None):
    if uncertainty is None:
        uncertainty = UncertaintySpec(kind="none", rationale="Fixed regression input.")
    return Parameter(
        id=pid,
        label=pid,
        value=value,
        unit="utility" if category == "utility" else "probability",
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=uncertainty,
    )


def _model():
    parameters = (
        _parameter(
            "p_cure",
            0.80,
            uncertainty=UncertaintySpec(
                kind="range_and_distribution",
                rationale="Cure uncertainty.",
                lower=0.60,
                upper=0.95,
                distribution=DistributionSpec("beta", (("alpha", 80.0), ("beta", 20.0))),
            ),
        ),
        _parameter(
            "u_cured",
            0.90,
            "utility",
            uncertainty=UncertaintySpec(
                kind="range",
                rationale="Utility uncertainty.",
                lower=0.75,
                upper=0.98,
            ),
        ),
        _parameter("u_uncured", 0.55, "utility"),
    )
    states = (
        MarkovState("cured", "Cured", absorbing=True),
        MarkovState("uncured", "Uncured", absorbing=True),
    )
    rewards = (
        StateReward("cured", "u_cured", "outcome", "per_year"),
        StateReward("uncured", "u_uncured", "outcome", "per_year"),
    )
    comparator = SemiMarkovStrategyDefinition(
        "comparator",
        "Comparator",
        (
            InitialStateAllocation("cured", 0.30),
            InitialStateAllocation("uncured", 0.70),
        ),
        (),
        state_rewards=rewards,
    )
    intervention = SemiMarkovStrategyDefinition(
        "intervention",
        "Intervention",
        (
            InitialStateAllocation(
                "cured",
                proportion_parameter_id="p_cure",
                proportion_mode="direct",
            ),
            InitialStateAllocation(
                "uncured",
                proportion_parameter_id="p_cure",
                proportion_mode="complement",
            ),
        ),
        (),
        state_rewards=rewards,
    )
    return (
        SemiMarkovDefinition(
            states=states,
            strategies=(comparator, intervention),
            cycle_length_years=0.5,
            max_cycles=4,
            state_accrual_timing="start",
        ),
        parameters,
    )


def test_advanced_markov_cure_and_utility_dsa_are_not_flat():
    model, parameters = _model()
    rows = tornado_semi_markov_inmb(
        model,
        parameters,
        intervention_id="intervention",
        comparator_id="comparator",
        willingness_to_pay=1000.0,
    )
    impacts = {row.parameter_id: row for row in rows}
    assert model.cycle_length_years == pytest.approx(0.5)
    assert model.max_cycles == 4
    assert impacts["p_cure"].low_inmb != pytest.approx(impacts["p_cure"].high_inmb)
    assert impacts["u_cured"].low_inmb != pytest.approx(impacts["u_cured"].high_inmb)


def test_advanced_markov_cure_psa_propagates_through_linked_starting_cohort():
    model, parameters = _model()
    result = run_semi_markov_psa(model, parameters, iterations=80, seed=23)
    assert np.std(result.parameter_draws["p_cure"]) > 0
    assert np.std(result.outcomes["intervention"]) > 0
    assert np.std(result.outcomes["comparator"]) == pytest.approx(0.0)
