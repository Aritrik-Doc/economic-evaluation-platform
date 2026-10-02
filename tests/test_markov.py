import pytest

from model.markov import (
    CohortMarkovDefinition,
    InitialStateAllocation,
    MarkovState,
    MarkovStrategyDefinition,
    MarkovValidationError,
    StateReward,
    TransitionProbability,
    TransitionReward,
    run_cohort_markov,
    validate_cohort_markov,
)
from model.schema import AssumptionSpec, EvidenceSource, Parameter, UncertaintySpec


SOURCE = EvidenceSource(citation="Illustrative source", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(
    statement="Illustrative test assumption.",
    rationale="Required for deterministic unit test.",
)
NO_UNCERTAINTY = UncertaintySpec(kind="none", rationale="Unit test uses fixed values.")


def parameter(
    id: str,
    value: float,
    category: str = "clinical",
    *,
    cost_bearers: tuple[str, ...] = (),
) -> Parameter:
    kwargs = {}
    if category == "cost":
        kwargs = {
            "currency": "GBP",
            "price_year": 2026,
            "cost_bearers": cost_bearers or ("health_system",),
        }
    return Parameter(
        id=id,
        label=id,
        value=value,
        unit="unit",
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=NO_UNCERTAINTY,
        **kwargs,
    )


def simple_model(*, timing="half_cycle", termination="fixed_cycles", max_cycles=1, threshold=1e-6):
    states = (
        MarkovState("healthy", "Healthy"),
        MarkovState("sick", "Sick"),
        MarkovState("dead", "Dead", absorbing=True),
    )
    initial = (InitialStateAllocation("healthy", 1.0),)

    def strategy(strategy_id: str, p_sick: str):
        return MarkovStrategyDefinition(
            strategy_id=strategy_id,
            label=strategy_id,
            initial_distribution=initial,
            transitions=(
                TransitionProbability("healthy", "sick", p_sick),
                TransitionProbability("healthy", "dead", "p_h_dead"),
                TransitionProbability("healthy", "healthy", probability_mode="residual"),
                TransitionProbability("sick", "dead", "p_s_dead"),
                TransitionProbability("sick", "sick", probability_mode="residual"),
            ),
            state_rewards=(
                StateReward("healthy", "cost_healthy", "cost", "per_year"),
                StateReward("sick", "cost_sick", "cost", "per_year"),
                StateReward("healthy", "utility_healthy", "outcome", "per_year"),
                StateReward("sick", "utility_sick", "outcome", "per_year"),
            ),
        )

    model = CohortMarkovDefinition(
        states=states,
        strategies=(strategy("A", "p_h_sick_a"), strategy("B", "p_h_sick_b")),
        cycle_length_years=1.0,
        max_cycles=max_cycles,
        state_accrual_timing=timing,
        termination_mode=termination,
        depletion_threshold=threshold,
    )
    parameters = (
        parameter("p_h_sick_a", 0.10),
        parameter("p_h_sick_b", 0.05),
        parameter("p_h_dead", 0.05),
        parameter("p_s_dead", 0.20),
        parameter("cost_healthy", 100.0, "cost"),
        parameter("cost_sick", 500.0, "cost"),
        parameter("utility_healthy", 1.0, "utility"),
        parameter("utility_sick", 0.6, "utility"),
    )
    return model, parameters


def test_one_cycle_trace_and_half_cycle_rewards():
    model, parameters = simple_model()
    result = run_cohort_markov(model, parameters)
    a = result.strategies[0]
    assert a.trace[0] == pytest.approx((1.0, 0.0, 0.0))
    assert a.trace[1] == pytest.approx((0.85, 0.10, 0.05))
    assert a.expected_cost == pytest.approx(117.5)
    assert a.expected_outcome == pytest.approx(0.955)


def test_start_and_end_state_accrual_are_explicit():
    start_model, parameters = simple_model(timing="start")
    end_model, _ = simple_model(timing="end")
    start = run_cohort_markov(start_model, parameters).strategies[0]
    end = run_cohort_markov(end_model, parameters).strategies[0]
    assert start.expected_cost == pytest.approx(100.0)
    assert start.expected_outcome == pytest.approx(1.0)
    assert end.expected_cost == pytest.approx(135.0)
    assert end.expected_outcome == pytest.approx(0.91)


def test_transition_event_rewards_use_expected_flows():
    model, parameters = simple_model()
    parameters = parameters + (parameter("progression_cost", 1000.0, "cost"),)
    a, b = model.strategies
    a = MarkovStrategyDefinition(
        strategy_id=a.strategy_id,
        label=a.label,
        initial_distribution=a.initial_distribution,
        transitions=a.transitions,
        state_rewards=a.state_rewards,
        transition_rewards=(TransitionReward("healthy", "sick", "progression_cost", "cost"),),
    )
    model = CohortMarkovDefinition(
        states=model.states,
        strategies=(a, b),
        cycle_length_years=model.cycle_length_years,
        max_cycles=model.max_cycles,
    )
    result = run_cohort_markov(model, parameters)
    assert result.strategies[0].expected_cost == pytest.approx(217.5)


def test_parameter_override_recalculates_transition_matrix():
    model, parameters = simple_model()
    result = run_cohort_markov(model, parameters, overrides={"p_h_sick_a": 0.20})
    assert result.strategies[0].trace[1] == pytest.approx((0.75, 0.20, 0.05))


def test_residual_transition_rejects_probability_sum_above_one():
    model, parameters = simple_model()
    with pytest.raises(MarkovValidationError, match="exceed 1"):
        run_cohort_markov(
            model,
            parameters,
            overrides={"p_h_sick_a": 0.98, "p_h_dead": 0.05},
        )


def test_initial_distribution_must_sum_to_one():
    model, parameters = simple_model()
    a = model.strategies[0]
    bad_a = MarkovStrategyDefinition(
        strategy_id=a.strategy_id,
        label=a.label,
        initial_distribution=(InitialStateAllocation("healthy", 0.9),),
        transitions=a.transitions,
        state_rewards=a.state_rewards,
    )
    bad = CohortMarkovDefinition(
        states=model.states,
        strategies=(bad_a, model.strategies[1]),
        cycle_length_years=1.0,
        max_cycles=1,
    )
    with pytest.raises(MarkovValidationError, match="Initial distribution"):
        validate_cohort_markov(bad, parameters)


def test_absorbing_state_may_omit_explicit_self_transition():
    model, parameters = simple_model()
    validate_cohort_markov(model, parameters)
    result = run_cohort_markov(model, parameters)
    assert result.strategies[0].trace[1][2] == pytest.approx(0.05)


def test_absorbing_state_cannot_leave():
    model, parameters = simple_model()
    parameters = parameters + (parameter("p_dead_leave", 0.1),)
    a = model.strategies[0]
    bad_a = MarkovStrategyDefinition(
        strategy_id=a.strategy_id,
        label=a.label,
        initial_distribution=a.initial_distribution,
        transitions=a.transitions
        + (
            TransitionProbability("dead", "healthy", "p_dead_leave"),
            TransitionProbability("dead", "dead", probability_mode="residual"),
        ),
        state_rewards=a.state_rewards,
    )
    bad = CohortMarkovDefinition(
        states=model.states,
        strategies=(bad_a, model.strategies[1]),
        cycle_length_years=1.0,
        max_cycles=1,
    )
    with pytest.raises(MarkovValidationError, match="Absorbing state"):
        validate_cohort_markov(bad, parameters)


def test_cost_perspective_filters_state_rewards():
    model, parameters = simple_model(timing="start")
    parameters = tuple(p for p in parameters if p.id != "cost_healthy") + (
        parameter("cost_healthy", 100.0, "cost", cost_bearers=("patient",)),
    )
    result = run_cohort_markov(
        model,
        parameters,
        included_cost_bearers=("health_system",),
    )
    assert result.strategies[0].expected_cost == pytest.approx(0.0)


def test_discounting_uses_state_accrual_time():
    model, parameters = simple_model(timing="end")
    result = run_cohort_markov(
        model,
        parameters,
        cost_discount_rate=0.10,
        outcome_discount_rate=0.10,
    )
    assert result.strategies[0].expected_cost == pytest.approx(135.0 / 1.1)
    assert result.strategies[0].expected_outcome == pytest.approx(0.91 / 1.1)


def test_per_year_rewards_scale_with_cycle_length():
    model, parameters = simple_model(timing="start", max_cycles=1)
    quarterly = CohortMarkovDefinition(
        states=model.states,
        strategies=model.strategies,
        cycle_length_years=0.25,
        max_cycles=1,
        state_accrual_timing="start",
    )
    result = run_cohort_markov(quarterly, parameters)
    assert result.strategies[0].expected_cost == pytest.approx(25.0)
    assert result.strategies[0].expected_outcome == pytest.approx(0.25)


def test_cohort_depletion_stops_before_max_cycles():
    states = (MarkovState("alive", "Alive"), MarkovState("dead", "Dead", absorbing=True))
    initial = (InitialStateAllocation("alive", 1.0),)
    strategy = lambda sid: MarkovStrategyDefinition(
        sid,
        sid,
        initial,
        (
            TransitionProbability("alive", "dead", "p_die"),
            TransitionProbability("alive", "alive", probability_mode="residual"),
        ),
    )
    model = CohortMarkovDefinition(
        states=states,
        strategies=(strategy("A"), strategy("B")),
        cycle_length_years=1.0,
        max_cycles=100,
        termination_mode="cohort_depletion",
        depletion_threshold=0.01,
    )
    result = run_cohort_markov(model, (parameter("p_die", 0.5),))
    assert result.strategies[0].stopped_early is True
    assert result.strategies[0].cycles_run < 100
    assert result.strategies[0].trace[-1][0] <= 0.01


def test_depletion_requires_absorbing_state():
    states = (MarkovState("a", "A"), MarkovState("b", "B"))
    initial = (InitialStateAllocation("a", 1.0),)
    transitions = (
        TransitionProbability("a", "b", "p"),
        TransitionProbability("a", "a", probability_mode="residual"),
        TransitionProbability("b", "b", probability_mode="residual"),
    )
    strategy = lambda sid: MarkovStrategyDefinition(sid, sid, initial, transitions)
    model = CohortMarkovDefinition(
        states=states,
        strategies=(strategy("A"), strategy("B")),
        cycle_length_years=1.0,
        max_cycles=10,
        termination_mode="cohort_depletion",
    )
    with pytest.raises(MarkovValidationError, match="absorbing"):
        validate_cohort_markov(model, (parameter("p", 0.1),))
