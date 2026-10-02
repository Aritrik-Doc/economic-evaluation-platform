import numpy as np
import pytest

from model.markov import InitialStateAllocation, MarkovState, StateReward
from model.schema import AssumptionSpec, EvidenceSource, Parameter, UncertaintySpec
from model.semi_markov import (
    BackgroundMortalityRule,
    DynamicTransition,
    SemiMarkovDefinition,
    SemiMarkovStrategyDefinition,
    SemiMarkovValidationError,
    run_semi_markov,
)
from model.transition_dynamics import (
    AgeSpecificMortalityTable,
    ParameterBand,
    PiecewiseParameterSchedule,
    competing_rates_to_probabilities,
    probability_to_rate,
)


SOURCE = EvidenceSource(citation="Illustrative test source", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(
    statement="Illustrative unit-test assumption.",
    rationale="Required for deterministic test construction.",
)
NO_UNCERTAINTY = UncertaintySpec(kind="none", rationale="Fixed unit-test value.")


def parameter(pid, value, category="clinical"):
    kwargs = {}
    if category == "cost":
        kwargs = {
            "currency": "GBP",
            "price_year": 2026,
            "cost_bearers": ("health_system",),
        }
    return Parameter(
        id=pid,
        label=pid,
        value=value,
        unit="unit",
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=NO_UNCERTAINTY,
        **kwargs,
    )


def schedule(parameter_id, *, basis="state_time"):
    return PiecewiseParameterSchedule(
        basis=basis,
        bands=(ParameterBand(0.0, parameter_id, None),),
    )


def strategy(strategy_id, transitions, *, state_rewards=(), mortality=None):
    return SemiMarkovStrategyDefinition(
        strategy_id=strategy_id,
        label=strategy_id,
        initial_distribution=(InitialStateAllocation("alive", 1.0),),
        transitions=tuple(transitions),
        state_rewards=tuple(state_rewards),
        background_mortality=mortality,
    )


def test_state_time_schedule_implements_tunnel_memory():
    parameters = (parameter("p_early", 0.1), parameter("p_late", 0.5))
    states = (MarkovState("alive", "Alive"), MarkovState("dead", "Dead", absorbing=True))
    transition = DynamicTransition(
        "alive",
        "dead",
        PiecewiseParameterSchedule(
            basis="state_time",
            bands=(
                ParameterBand(0.0, "p_early", 1.0),
                ParameterBand(1.0, "p_late", None),
            ),
        ),
    )
    model = SemiMarkovDefinition(
        states=states,
        strategies=(strategy("A", (transition,)), strategy("B", (transition,))),
        cycle_length_years=1.0,
        max_cycles=3,
        state_accrual_timing="start",
    )
    result = run_semi_markov(model, parameters)
    trace = result.strategies[0].trace
    assert trace[1] == pytest.approx((0.9, 0.1))
    assert trace[2] == pytest.approx((0.45, 0.55))
    assert trace[3] == pytest.approx((0.225, 0.775))
    assert result.strategies[0].mean_state_time_trace[2][0] == pytest.approx(2.0)


def test_model_time_and_state_time_are_not_conflated():
    parameters = (
        parameter("p_enter_sick", 0.5),
        parameter("p_zero", 0.0),
        parameter("p_one", 1.0),
        parameter("p_dead_healthy", 0.0),
    )
    states = (
        MarkovState("alive", "Healthy"),
        MarkovState("sick", "Sick"),
        MarkovState("dead", "Dead", absorbing=True),
    )
    healthy_exit = DynamicTransition("alive", "sick", schedule("p_enter_sick"))

    state_time_exit = DynamicTransition(
        "sick",
        "dead",
        PiecewiseParameterSchedule(
            basis="state_time",
            bands=(ParameterBand(0.0, "p_zero", 1.0), ParameterBand(1.0, "p_one", None)),
        ),
    )
    model_time_exit = DynamicTransition(
        "sick",
        "dead",
        PiecewiseParameterSchedule(
            basis="model_time",
            bands=(ParameterBand(0.0, "p_zero", 1.0), ParameterBand(1.0, "p_one", None)),
        ),
    )

    def make_strategy(sid, sick_exit):
        return SemiMarkovStrategyDefinition(
            strategy_id=sid,
            label=sid,
            initial_distribution=(InitialStateAllocation("alive", 1.0),),
            transitions=(healthy_exit, sick_exit),
        )

    model = SemiMarkovDefinition(
        states=states,
        strategies=(make_strategy("state_time", state_time_exit), make_strategy("model_time", model_time_exit)),
        cycle_length_years=1.0,
        max_cycles=2,
    )
    result = {row.strategy_id: row for row in run_semi_markov(model, parameters).strategies}
    # At t=1, 50% newly entered sick. Under state-time logic they have spent
    # zero completed cycles there and survive the second cycle. Under model-time
    # logic the calendar/model time has crossed year 1, so they die in cycle 2.
    assert result["state_time"].trace[2][1] == pytest.approx(0.75)
    assert result["state_time"].trace[2][2] == pytest.approx(0.0)
    assert result["model_time"].trace[2][1] == pytest.approx(0.25)
    assert result["model_time"].trace[2][2] == pytest.approx(0.5)


def test_rate_based_disease_exit_and_background_mortality_compete_jointly():
    parameters = (parameter("event_rate", 0.1), parameter("smr", 1.0))
    states = (
        MarkovState("alive", "Alive"),
        MarkovState("event", "Event", absorbing=True),
        MarkovState("dead", "Dead", absorbing=True),
    )
    mortality = BackgroundMortalityRule(
        destination_state="dead",
        mortality_table=AgeSpecificMortalityTable(((60, 0.10), (61, 0.10))),
        initial_age=60.0,
        applicable_states=("alive",),
        smr_parameter_id="smr",
    )
    event_transition = DynamicTransition(
        "alive",
        "event",
        schedule("event_rate"),
        input_type="rate",
    )
    model = SemiMarkovDefinition(
        states=states,
        strategies=(
            strategy("A", (event_transition,), mortality=mortality),
            strategy("B", (event_transition,), mortality=mortality),
        ),
        cycle_length_years=1.0,
        max_cycles=1,
    )
    result = run_semi_markov(model, parameters).strategies[0]
    mortality_rate = probability_to_rate(0.10, 1.0)
    expected = competing_rates_to_probabilities({"event": 0.1, "dead": mortality_rate}, 1.0)
    assert result.trace[1][0] == pytest.approx(expected.stay_probability)
    assert result.trace[1][1] == pytest.approx(expected.destination_probabilities["event"])
    assert result.trace[1][2] == pytest.approx(expected.destination_probabilities["dead"])


def test_background_mortality_is_not_heuristically_mixed_with_probability_rows():
    parameters = (parameter("p_event", 0.1),)
    states = (
        MarkovState("alive", "Alive"),
        MarkovState("event", "Event", absorbing=True),
        MarkovState("dead", "Dead", absorbing=True),
    )
    mortality = BackgroundMortalityRule(
        destination_state="dead",
        mortality_table=AgeSpecificMortalityTable(((60, 0.1), (61, 0.1))),
        initial_age=60,
        applicable_states=("alive",),
    )
    probability_exit = DynamicTransition("alive", "event", schedule("p_event"))
    model = SemiMarkovDefinition(
        states=states,
        strategies=(
            strategy("A", (probability_exit,), mortality=mortality),
            strategy("B", (probability_exit,), mortality=mortality),
        ),
        cycle_length_years=1.0,
        max_cycles=1,
    )
    with pytest.raises(SemiMarkovValidationError, match="rate-based exits"):
        run_semi_markov(model, parameters)


def test_state_rewards_and_discounting_work_with_tenure_tracking():
    parameters = (
        parameter("p_death", 0.0),
        parameter("cost_alive", 100.0, "cost"),
        parameter("utility_alive", 0.8, "utility"),
    )
    states = (MarkovState("alive", "Alive"), MarkovState("dead", "Dead", absorbing=True))
    death = DynamicTransition("alive", "dead", schedule("p_death"))
    rewards = (
        StateReward("alive", "cost_alive", "cost", "per_year"),
        StateReward("alive", "utility_alive", "outcome", "per_year"),
    )
    model = SemiMarkovDefinition(
        states=states,
        strategies=(strategy("A", (death,), state_rewards=rewards), strategy("B", (death,), state_rewards=rewards)),
        cycle_length_years=1.0,
        max_cycles=2,
        state_accrual_timing="start",
    )
    result = run_semi_markov(model, parameters, cost_discount_rate=0.10, outcome_discount_rate=0.0).strategies[0]
    assert result.expected_cost == pytest.approx(100.0 + 100.0 / 1.1)
    assert result.expected_outcome == pytest.approx(1.6)


def test_probability_rows_cannot_sum_above_one():
    parameters = (parameter("p_a", 0.7), parameter("p_b", 0.5))
    states = (
        MarkovState("alive", "Alive"),
        MarkovState("event", "Event", absorbing=True),
        MarkovState("dead", "Dead", absorbing=True),
    )
    exits = (
        DynamicTransition("alive", "event", schedule("p_a")),
        DynamicTransition("alive", "dead", schedule("p_b")),
    )
    model = SemiMarkovDefinition(
        states=states,
        strategies=(strategy("A", exits), strategy("B", exits)),
        cycle_length_years=1.0,
        max_cycles=1,
    )
    with pytest.raises(SemiMarkovValidationError, match="exceeding 1"):
        run_semi_markov(model, parameters)


def test_overrides_propagate_to_dynamic_schedules_for_sensitivity_analysis():
    parameters = (parameter("p_death", 0.1),)
    states = (MarkovState("alive", "Alive"), MarkovState("dead", "Dead", absorbing=True))
    death = DynamicTransition("alive", "dead", schedule("p_death"))
    model = SemiMarkovDefinition(
        states=states,
        strategies=(strategy("A", (death,)), strategy("B", (death,))),
        cycle_length_years=1.0,
        max_cycles=1,
    )
    result = run_semi_markov(model, parameters, overrides={"p_death": 0.4})
    assert result.strategies[0].trace[1] == pytest.approx((0.6, 0.4))
