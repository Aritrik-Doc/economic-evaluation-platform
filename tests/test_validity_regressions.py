import numpy as np
import pytest

from model.markov import (
    CohortMarkovDefinition,
    InitialStateAllocation,
    MarkovState,
    MarkovStrategyDefinition,
    StateReward,
)
from model.markov_psa import run_markov_psa
from model.markov_sensitivity import tornado_markov_inmb
from model.schema import (
    AssumptionSpec,
    DistributionSpec,
    EvidenceSource,
    Parameter,
    UncertaintySpec,
)
from model.semi_markov import (
    BackgroundMortalityRule,
    DynamicTransition,
    SemiMarkovDefinition,
    SemiMarkovStrategyDefinition,
    SemiMarkovValidationError,
    run_semi_markov,
)
from model.semi_markov_builder import compile_dynamic_transitions
from model.transition_dynamics import (
    AgeSpecificMortalityTable,
    ParameterBand,
    PiecewiseParameterSchedule,
    competing_rates_to_probabilities,
    probability_to_rate,
)


SOURCE = EvidenceSource(citation="Regression-test source", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(
    statement="Regression-test assumption.",
    rationale="Exercises linked model inputs and analysis settings.",
)


def parameter(pid, value, category="clinical", uncertainty=None):
    if uncertainty is None:
        uncertainty = UncertaintySpec(kind="none", rationale="Fixed regression input.")
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


def linked_initial_model():
    p_cure = parameter(
        "p_cure",
        0.80,
        uncertainty=UncertaintySpec(
            kind="range_and_distribution",
            rationale="Cure-rate uncertainty.",
            lower=0.60,
            upper=0.95,
            distribution=DistributionSpec("beta", (("alpha", 80.0), ("beta", 20.0))),
        ),
    )
    u_cured = parameter(
        "u_cured",
        0.90,
        "utility",
        uncertainty=UncertaintySpec(
            kind="range",
            rationale="Utility uncertainty.",
            lower=0.75,
            upper=0.98,
        ),
    )
    parameters = (
        p_cure,
        u_cured,
        parameter("u_uncured", 0.55, "utility"),
    )
    states = (
        MarkovState("cured", "Cured", absorbing=True),
        MarkovState("uncured", "Uncured", absorbing=True),
    )
    rewards = (
        StateReward("cured", "u_cured", "outcome", "per_year"),
        StateReward("uncured", "u_uncured", "outcome", "per_year"),
    )
    comparator = MarkovStrategyDefinition(
        strategy_id="comparator",
        label="Comparator",
        initial_distribution=(
            InitialStateAllocation("cured", 0.30),
            InitialStateAllocation("uncured", 0.70),
        ),
        transitions=(),
        state_rewards=rewards,
    )
    intervention = MarkovStrategyDefinition(
        strategy_id="intervention",
        label="Intervention",
        initial_distribution=(
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
        transitions=(),
        state_rewards=rewards,
    )
    model = CohortMarkovDefinition(
        states=states,
        strategies=(comparator, intervention),
        cycle_length_years=0.5,
        max_cycles=4,
        state_accrual_timing="start",
    )
    return model, parameters


def test_cure_and_utility_dsa_change_inmb_with_selected_half_year_cycle_model():
    model, parameters = linked_initial_model()
    rows = tornado_markov_inmb(
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
    assert impacts["p_cure"].impact > 0
    assert impacts["u_cured"].impact > 0


def test_cure_psa_changes_outcomes_through_parameter_linked_starting_cohort():
    model, parameters = linked_initial_model()
    result = run_markov_psa(model, parameters, iterations=80, seed=17)
    assert np.std(result.parameter_draws["p_cure"]) > 0
    assert np.std(result.outcomes["intervention"]) > 0
    assert np.std(result.outcomes["comparator"]) == pytest.approx(0.0)


def _schedule(pid):
    return PiecewiseParameterSchedule("model_time", (ParameterBand(0.0, pid, None),))


def test_explicit_probability_to_rate_can_compete_with_background_mortality():
    parameters = (parameter("p_progress", 0.20),)
    states = (
        MarkovState("alive", "Alive"),
        MarkovState("progressed", "Progressed", absorbing=True),
        MarkovState("dead", "Dead", absorbing=True),
    )
    transition = DynamicTransition(
        "alive",
        "progressed",
        _schedule("p_progress"),
        input_type="probability_to_rate",
        source_interval_years=1.0,
    )
    mortality = BackgroundMortalityRule(
        destination_state="dead",
        mortality_table=AgeSpecificMortalityTable(((60, 0.10), (61, 0.10))),
        initial_age=60.0,
        applicable_states=("alive",),
    )
    strategy = lambda sid: SemiMarkovStrategyDefinition(
        sid,
        sid,
        (InitialStateAllocation("alive", 1.0),),
        (transition,),
        background_mortality=mortality,
    )
    model = SemiMarkovDefinition(
        states=states,
        strategies=(strategy("A"), strategy("B")),
        cycle_length_years=0.5,
        max_cycles=1,
    )
    result = run_semi_markov(model, parameters).strategies[0]
    expected = competing_rates_to_probabilities(
        {
            "progressed": probability_to_rate(0.20, 1.0),
            "dead": probability_to_rate(1 - (1 - 0.10) ** 0.5, 0.5),
        },
        0.5,
    )
    assert result.trace[1][1] == pytest.approx(expected.destination_probabilities["progressed"])
    assert result.trace[1][2] == pytest.approx(expected.destination_probabilities["dead"])


def test_plain_probability_with_background_mortality_remains_rejected():
    parameters = (parameter("p_progress", 0.20),)
    states = (
        MarkovState("alive", "Alive"),
        MarkovState("progressed", "Progressed", absorbing=True),
        MarkovState("dead", "Dead", absorbing=True),
    )
    transition = DynamicTransition("alive", "progressed", _schedule("p_progress"))
    mortality = BackgroundMortalityRule(
        destination_state="dead",
        mortality_table=AgeSpecificMortalityTable(((60, 0.10), (61, 0.10))),
        initial_age=60.0,
        applicable_states=("alive",),
    )
    strategy = lambda sid: SemiMarkovStrategyDefinition(
        sid,
        sid,
        (InitialStateAllocation("alive", 1.0),),
        (transition,),
        background_mortality=mortality,
    )
    model = SemiMarkovDefinition(
        states=states,
        strategies=(strategy("A"), strategy("B")),
        cycle_length_years=1.0,
        max_cycles=1,
    )
    with pytest.raises(SemiMarkovValidationError, match="explicitly choose probability_to_rate"):
        run_semi_markov(model, parameters)


def test_advanced_transition_import_accepts_standard_alias_and_drops_residual_self_stay():
    rows = [
        {
            "strategy_id": "A",
            "origin_state": "alive",
            "destination_state": "event",
            "probability_parameter_id": "p_event",
            "probability_mode": "direct",
        },
        {
            "strategy_id": "A",
            "origin_state": "alive",
            "destination_state": "alive",
            "probability_mode": "residual",
            "probability_parameter_id": "",
        },
    ]
    compiled = compile_dynamic_transitions(rows)
    assert len(compiled["A"]) == 1
    transition = compiled["A"][0]
    assert transition.input_type == "probability"
    assert transition.schedule.parameter_id(model_time=0.0, state_time=0.0) == "p_event"
