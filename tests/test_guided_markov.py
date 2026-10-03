import pytest

from model.guided_markov import (
    GuidedMarkovError,
    add_dynamic_transition,
    add_state,
    add_strategy,
    add_transition,
    delete_state,
    delete_strategy,
    set_initial_distribution,
)


def test_add_state_and_strategy():
    states = add_state([], name="Stable")
    assert states == [{"state_id": "stable", "state_name": "Stable", "absorbing": False}]
    strategies = add_strategy([], name="Standard care")
    assert strategies[0]["strategy_id"] == "standard_care"


def test_duplicate_state_id_rejected():
    with pytest.raises(GuidedMarkovError, match="already exists"):
        add_state([{"state_id": "stable", "state_name": "Stable", "absorbing": False}], name="Another", state_id="stable")


def test_initial_distribution_replaces_strategy_rows_and_requires_sum_one():
    initial = [
        {"strategy_id": "A", "state_id": "old", "proportion": 1.0},
        {"strategy_id": "B", "state_id": "stable", "proportion": 1.0},
    ]
    updated = set_initial_distribution(
        initial,
        strategy_id="A",
        allocations={"stable": 0.75, "progressed": 0.25},
    )
    a_rows = [row for row in updated if row["strategy_id"] == "A"]
    assert {row["state_id"] for row in a_rows} == {"stable", "progressed"}
    with pytest.raises(GuidedMarkovError, match="sum to 1"):
        set_initial_distribution(initial, strategy_id="A", allocations={"stable": 0.8})


def test_standard_transition_rules():
    transitions = add_transition(
        [],
        strategy_id="A",
        origin_state="stable",
        destination_state="progressed",
        parameter_id="p_prog",
        probability_mode="direct",
    )
    transitions = add_transition(
        transitions,
        strategy_id="A",
        origin_state="stable",
        destination_state="stable",
        probability_mode="residual",
    )
    assert transitions[1]["probability_parameter_id"] == ""
    with pytest.raises(GuidedMarkovError, match="Only one residual"):
        add_transition(
            transitions,
            strategy_id="A",
            origin_state="stable",
            destination_state="dead",
            probability_mode="residual",
        )


def test_dynamic_transition_normalises_rate_mode():
    rows = add_dynamic_transition(
        [],
        strategy_id="A",
        origin_state="stable",
        destination_state="progressed",
        input_type="rate",
        parameter_id="h_prog",
        time_basis="state_time",
        start_time=0,
        end_time=2,
        probability_mode="complement",
    )
    assert rows[0]["probability_mode"] == "direct"
    assert rows[0]["time_basis"] == "state_time"


def test_delete_state_cascades_related_rows():
    result = delete_state(
        "progressed",
        states=[
            {"state_id": "stable"},
            {"state_id": "progressed"},
            {"state_id": "dead"},
        ],
        initial=[{"strategy_id": "A", "state_id": "progressed", "proportion": 1.0}],
        transitions=[
            {"strategy_id": "A", "origin_state": "stable", "destination_state": "progressed"},
            {"strategy_id": "A", "origin_state": "progressed", "destination_state": "dead"},
        ],
        state_rewards=[{"strategy_id": "A", "state_id": "progressed"}],
        transition_rewards=[{"strategy_id": "A", "origin_state": "progressed", "destination_state": "dead"}],
        mortality_rules=[
            {"strategy_id": "A", "destination_state": "dead", "applicable_states": "stable, progressed"}
        ],
    )
    assert [row["state_id"] for row in result["states"]] == ["stable", "dead"]
    assert not result["transitions"]
    assert result["mortality_rules"][0]["applicable_states"] == "stable"


def test_delete_strategy_cascades_related_rows():
    result = delete_strategy(
        "A",
        strategies=[{"strategy_id": "A"}, {"strategy_id": "B"}, {"strategy_id": "C"}],
        initial=[{"strategy_id": "A"}, {"strategy_id": "B"}],
        transitions=[{"strategy_id": "A"}, {"strategy_id": "B"}],
        state_rewards=[{"strategy_id": "A"}],
        transition_rewards=[{"strategy_id": "A"}],
        mortality_rules=[{"strategy_id": "A"}, {"strategy_id": "B"}],
    )
    assert [row["strategy_id"] for row in result["strategies"]] == ["B", "C"]
    assert all(row["strategy_id"] != "A" for row in result["transitions"])
