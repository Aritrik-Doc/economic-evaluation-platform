import pytest

from model.markov_reproducibility import (
    MarkovReproducibilityError,
    markov_run_fingerprint,
    validate_analysis_currency,
)


def test_cost_currency_mismatch_is_blocked():
    rows = [
        {"id": "cost_a", "category": "cost", "currency": "GBP"},
        {"id": "utility", "category": "utility", "currency": ""},
    ]
    with pytest.raises(MarkovReproducibilityError, match="Automatic FX conversion"):
        validate_analysis_currency(rows, "INR")


def test_matching_cost_currency_passes():
    validate_analysis_currency(
        [{"id": "cost_a", "category": "cost", "currency": "GBP"}],
        "GBP",
    )


def test_run_fingerprint_changes_when_model_changes():
    common = dict(
        state_rows=[{"state_id": "alive"}, {"state_id": "dead"}],
        strategy_rows=[{"strategy_id": "A"}, {"strategy_id": "B"}],
        initial_rows=[{"strategy_id": "A", "state_id": "alive", "proportion": 1.0}],
        transition_rows=[],
        state_reward_rows=[],
        transition_reward_rows=[],
        settings={"cycle_length_years": 1.0, "max_cycles": 10},
    )
    a = markov_run_fingerprint(
        parameter_rows=[{"id": "p", "value": 0.1}],
        **common,
    )
    b = markov_run_fingerprint(
        parameter_rows=[{"id": "p", "value": 0.2}],
        **common,
    )
    assert a != b


def test_run_fingerprint_is_stable_for_same_content():
    kwargs = dict(
        parameter_rows=[{"id": "p", "value": 0.1}],
        state_rows=[{"state_id": "alive"}],
        strategy_rows=[{"strategy_id": "A"}],
        initial_rows=[],
        transition_rows=[],
        state_reward_rows=[],
        transition_reward_rows=[],
        settings={"max_cycles": 5},
    )
    assert markov_run_fingerprint(**kwargs) == markov_run_fingerprint(**kwargs)
