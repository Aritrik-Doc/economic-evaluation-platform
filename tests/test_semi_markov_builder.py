import pytest

from model.semi_markov import run_semi_markov
from model.semi_markov_builder import compile_semi_markov_tables
from model.tree_builder import BuilderValidationError


def parameter_row(pid, value, category="clinical"):
    row = {
        "id": pid,
        "label": pid,
        "value": value,
        "unit": "unit",
        "category": category,
        "source_citation": "Illustrative source",
        "source_type": "user_assumption",
        "publication_year": None,
        "source_url": "",
        "source_details": "",
        "assumption": "Illustrative test assumption.",
        "assumption_rationale": "Required for test model.",
        "dsa_enabled": False,
        "dsa_rationale": "Not represented in DSA.",
        "psa_enabled": False,
        "psa_rationale": "Not represented in PSA.",
        "notes": "",
    }
    if category == "cost":
        row.update(currency="GBP", price_year=2026, cost_bearers="health_system")
    return row


def tables():
    parameters = [
        parameter_row("rate_a", 0.1),
        parameter_row("rate_b", 0.2),
        parameter_row("smr", 1.0),
        parameter_row("cost_alive", 100, "cost"),
        parameter_row("utility_alive", 0.8, "utility"),
    ]
    states = [
        {"state_id": "alive", "state_name": "Alive", "absorbing": False},
        {"state_id": "dead", "state_name": "Dead", "absorbing": True},
    ]
    strategies = [
        {"strategy_id": "A", "strategy_name": "A"},
        {"strategy_id": "B", "strategy_name": "B"},
    ]
    initial = [
        {"strategy_id": "A", "state_id": "alive", "proportion": 1.0},
        {"strategy_id": "B", "state_id": "alive", "proportion": 1.0},
    ]
    transitions = [
        {"strategy_id": "A", "origin_state": "alive", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "model_time", "start_time": 0.0, "end_time": 1.0, "parameter_id": "rate_a"},
        {"strategy_id": "A", "origin_state": "alive", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "model_time", "start_time": 1.0, "end_time": None, "parameter_id": "rate_b"},
        {"strategy_id": "B", "origin_state": "alive", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": None, "parameter_id": "rate_a"},
    ]
    state_rewards = [
        {"strategy_id": sid, "state_id": "alive", "parameter_id": "cost_alive", "reward_type": "cost", "accrual": "per_year"}
        for sid in ("A", "B")
    ] + [
        {"strategy_id": sid, "state_id": "alive", "parameter_id": "utility_alive", "reward_type": "outcome", "accrual": "per_year"}
        for sid in ("A", "B")
    ]
    mortality_table = []
    mortality_rules = []
    return parameters, states, strategies, initial, transitions, state_rewards, [], mortality_table, mortality_rules


def test_compiler_builds_time_varying_schedule_and_runs():
    compiled = compile_semi_markov_tables(
        *tables(), cycle_length_years=1.0, max_cycles=2
    )
    assert compiled.model.strategies[0].transitions[0].schedule.basis == "model_time"
    assert len(compiled.model.strategies[0].transitions[0].schedule.bands) == 2
    result = run_semi_markov(compiled.model, compiled.parameters)
    assert len(result.strategies) == 2
    assert result.strategies[0].trace[2][0] < result.strategies[1].trace[2][0]


def test_compiler_builds_age_mortality_rule():
    data = list(tables())
    data[4] = [
        {"strategy_id": sid, "origin_state": "alive", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": None, "parameter_id": "rate_a"}
        for sid in ("A", "B")
    ]
    data[7] = [
        {"age": 60, "annual_probability": 0.01},
        {"age": 61, "annual_probability": 0.02},
        {"age": 62, "annual_probability": 0.03},
    ]
    # Mortality cannot duplicate the explicit death destination, so add a
    # different absorbing event destination to the disease transition.
    data[1] = [
        {"state_id": "alive", "state_name": "Alive", "absorbing": False},
        {"state_id": "event", "state_name": "Event", "absorbing": True},
        {"state_id": "dead", "state_name": "Dead", "absorbing": True},
    ]
    for row in data[4]:
        row["destination_state"] = "event"
    data[8] = [
        {"strategy_id": sid, "destination_state": "dead", "initial_age": 60.0, "applicable_states": "alive", "smr_parameter_id": "smr"}
        for sid in ("A", "B")
    ]
    compiled = compile_semi_markov_tables(*data, cycle_length_years=1.0, max_cycles=2)
    mortality = compiled.model.strategies[0].background_mortality
    assert mortality is not None
    assert mortality.initial_age == 60.0
    assert mortality.mortality_table.maximum_age == 62


def test_overlapping_schedule_bands_are_rejected():
    data = list(tables())
    data[4].append(
        {"strategy_id": "A", "origin_state": "alive", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "model_time", "start_time": 0.5, "end_time": 1.5, "parameter_id": "rate_b"}
    )
    with pytest.raises(BuilderValidationError, match="overlap"):
        compile_semi_markov_tables(*data, cycle_length_years=1.0, max_cycles=2)
