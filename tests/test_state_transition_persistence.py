import json

import pytest

from model.state_transition_persistence import (
    STATE_TRANSITION_SCHEMA_VERSION,
    StateTransitionPersistenceError,
    build_state_transition_audit_record,
    build_state_transition_bundle,
    compile_loaded_state_transition_bundle,
    load_state_transition_bundle,
    state_transition_audit_json,
    state_transition_bundle_json,
)


def parameter_rows():
    base = {
        "label": "Probability",
        "unit": "probability",
        "category": "clinical",
        "source_citation": "Illustrative",
        "source_type": "user_assumption",
        "publication_year": None,
        "source_url": "",
        "source_details": "",
        "assumption": "Illustrative assumption",
        "assumption_rationale": "Test",
        "dsa_enabled": False,
        "dsa_lower": None,
        "dsa_upper": None,
        "dsa_rationale": "Not represented",
        "psa_enabled": False,
        "psa_rationale": "Not represented",
        "distribution_family": "",
        "distribution_parameters": "",
        "correlation_group": "",
        "notes": "",
    }
    rows = []
    for pid, value in (("p_a", 0.1), ("p_b", 0.05), ("h_a", 0.1), ("h_b", 0.05), ("smr", 1.0)):
        row = dict(base)
        row.update(id=pid, value=value)
        rows.append(row)
    utility = dict(base)
    utility.update(id="u_alive", label="Alive utility", value=0.8, unit="utility", category="utility")
    rows.append(utility)
    cost = dict(base)
    cost.update(
        id="c_alive",
        label="Alive cost",
        value=100,
        unit="GBP/year",
        category="cost",
        currency="GBP",
        price_year=2026,
        cost_bearers="health_system",
    )
    rows.append(cost)
    return rows


def methods():
    return {
        "reference_case_code": "CUSTOM",
        "outcome_code": "QALY",
        "currency_code": "GBP",
        "threshold": 30000.0,
        "perspective_label": "Healthcare payer",
        "included_cost_bearers": ["health_system"],
        "cost_discount_rate": 0.035,
        "outcome_discount_rate": 0.035,
    }


def engine():
    return {
        "cycle_length_years": 1.0,
        "max_cycles": 10,
        "state_accrual_timing": "half_cycle",
        "transition_reward_timing": "mid_cycle",
        "termination_mode": "fixed_cycles",
        "depletion_threshold": 0.0001,
    }


def common_structure():
    states = [
        {"state_id": "alive", "state_name": "Alive", "absorbing": False},
        {"state_id": "dead", "state_name": "Dead", "absorbing": True},
    ]
    strategies = [
        {"strategy_id": "A", "strategy_name": "Treatment A"},
        {"strategy_id": "B", "strategy_name": "Treatment B"},
    ]
    initial = [
        {"strategy_id": "A", "state_id": "alive", "proportion": 1.0},
        {"strategy_id": "B", "state_id": "alive", "proportion": 1.0},
    ]
    rewards = [
        {"strategy_id": sid, "state_id": "alive", "parameter_id": "u_alive", "reward_type": "outcome", "accrual": "per_year"}
        for sid in ("A", "B")
    ]
    return states, strategies, initial, rewards


def cohort_bundle():
    states, strategies, initial, rewards = common_structure()
    transitions = [
        {"strategy_id": "A", "origin_state": "alive", "destination_state": "dead", "probability_parameter_id": "p_a", "probability_mode": "direct"},
        {"strategy_id": "A", "origin_state": "alive", "destination_state": "alive", "probability_parameter_id": "", "probability_mode": "residual"},
        {"strategy_id": "B", "origin_state": "alive", "destination_state": "dead", "probability_parameter_id": "p_b", "probability_mode": "direct"},
        {"strategy_id": "B", "origin_state": "alive", "destination_state": "alive", "probability_parameter_id": "", "probability_mode": "residual"},
    ]
    return build_state_transition_bundle(
        model_type="cohort_markov",
        model_name="Cohort example",
        methods=methods(),
        engine=engine(),
        parameter_rows=parameter_rows(),
        state_rows=states,
        strategy_rows=strategies,
        initial_rows=initial,
        transition_rows=transitions,
        state_reward_rows=rewards,
        transition_reward_rows=[],
        author="Tester",
    )


def semi_bundle():
    states, strategies, initial, rewards = common_structure()
    transitions = [
        {"strategy_id": "A", "origin_state": "alive", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": None, "parameter_id": "h_a"},
        {"strategy_id": "B", "origin_state": "alive", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "model_time", "start_time": 0.0, "end_time": None, "parameter_id": "h_b"},
    ]
    return build_state_transition_bundle(
        model_type="semi_markov",
        model_name="Semi-Markov example",
        methods=methods(),
        engine=engine(),
        parameter_rows=parameter_rows(),
        state_rows=states,
        strategy_rows=strategies,
        initial_rows=initial,
        transition_rows=transitions,
        state_reward_rows=rewards,
        transition_reward_rows=[],
        author="Tester",
    )


def test_cohort_markov_round_trip_recompiles():
    bundle = cohort_bundle()
    assert bundle["schema_version"] == STATE_TRANSITION_SCHEMA_VERSION
    loaded = load_state_transition_bundle(state_transition_bundle_json(bundle))
    compiled = compile_loaded_state_transition_bundle(loaded)
    assert loaded["model_type"] == "cohort_markov"
    assert len(compiled.model.strategies) == 2


def test_semi_markov_round_trip_preserves_time_basis():
    bundle = semi_bundle()
    loaded = load_state_transition_bundle(state_transition_bundle_json(bundle))
    compiled = compile_loaded_state_transition_bundle(loaded)
    assert loaded["model_type"] == "semi_markov"
    assert compiled.model.strategies[0].transitions[0].schedule.basis == "state_time"
    assert compiled.model.strategies[1].transitions[0].schedule.basis == "model_time"


def test_tampered_bundle_is_rejected_by_hash():
    bundle = cohort_bundle()
    data = json.loads(state_transition_bundle_json(bundle))
    data["model"]["parameters"][0]["value"] = 0.9
    with pytest.raises(StateTransitionPersistenceError, match="content hash"):
        load_state_transition_bundle(json.dumps(data))


def test_loader_rejects_unsupported_schema_version():
    data = cohort_bundle()
    data["schema_version"] = "999"
    with pytest.raises(StateTransitionPersistenceError, match="Unsupported"):
        load_state_transition_bundle(json.dumps(data))


def test_homogeneous_bundle_rejects_advanced_mortality_rules():
    states, strategies, initial, rewards = common_structure()
    transitions = [
        {"strategy_id": "A", "origin_state": "alive", "destination_state": "dead", "probability_parameter_id": "p_a", "probability_mode": "direct"},
        {"strategy_id": "A", "origin_state": "alive", "destination_state": "alive", "probability_parameter_id": "", "probability_mode": "residual"},
        {"strategy_id": "B", "origin_state": "alive", "destination_state": "dead", "probability_parameter_id": "p_b", "probability_mode": "direct"},
        {"strategy_id": "B", "origin_state": "alive", "destination_state": "alive", "probability_parameter_id": "", "probability_mode": "residual"},
    ]
    with pytest.raises(StateTransitionPersistenceError, match="cannot contain advanced mortality"):
        build_state_transition_bundle(
            model_type="cohort_markov",
            model_name="Bad",
            methods=methods(),
            engine=engine(),
            parameter_rows=parameter_rows(),
            state_rows=states,
            strategy_rows=strategies,
            initial_rows=initial,
            transition_rows=transitions,
            state_reward_rows=rewards,
            transition_reward_rows=[],
            mortality_table_rows=[{"age": 60, "annual_probability": 0.01}],
        )


def test_audit_embeds_model_snapshot_hash_and_run_settings():
    bundle = semi_bundle()
    record = build_state_transition_audit_record(
        bundle,
        analysis_type="base_case",
        run_settings={"seed": None, "threshold": 30000},
        results={"A": {"cost": 1000, "effect": 2.0}},
        warnings=("Illustrative model",),
    )
    assert record["model_type"] == "semi_markov"
    assert record["model_hash_sha256"] == bundle["content_hash_sha256"]
    assert record["model_snapshot"]["model_name"] == "Semi-Markov example"
    assert json.loads(state_transition_audit_json([record]))["records"][0]["run_id"] == record["run_id"]
