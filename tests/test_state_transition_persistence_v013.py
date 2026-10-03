from model.state_transition_persistence import (
    build_state_transition_bundle,
    compile_loaded_state_transition_bundle,
    load_state_transition_bundle,
    state_transition_bundle_json,
)


def _row(pid, value):
    return {
        "id": pid,
        "label": pid,
        "value": value,
        "unit": "probability",
        "category": "clinical",
        "source_citation": "Regression evidence",
        "source_type": "user_assumption",
        "publication_year": None,
        "source_url": "",
        "source_details": "",
        "assumption": "Regression assumption",
        "assumption_rationale": "Tests persistence of v0.13 model semantics.",
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


def test_bundle_round_trip_preserves_parameter_linked_initial_allocation():
    bundle = build_state_transition_bundle(
        model_type="cohort_markov",
        model_name="Linked-start regression",
        methods={
            "reference_case_code": "CUSTOM",
            "outcome_code": "QALY",
            "currency_code": "GBP",
            "threshold": 30000.0,
            "perspective_label": "Healthcare payer",
            "included_cost_bearers": ["health_system"],
            "cost_discount_rate": 0.035,
            "outcome_discount_rate": 0.035,
        },
        engine={
            "cycle_length_years": 0.5,
            "max_cycles": 20,
            "state_accrual_timing": "half_cycle",
            "transition_reward_timing": "mid_cycle",
            "termination_mode": "fixed_cycles",
            "depletion_threshold": 0.0001,
        },
        parameter_rows=[_row("p_cure", 0.8)],
        state_rows=[
            {"state_id": "cured", "state_name": "Cured", "absorbing": True},
            {"state_id": "uncured", "state_name": "Uncured", "absorbing": True},
        ],
        strategy_rows=[
            {"strategy_id": "A", "strategy_name": "A"},
            {"strategy_id": "B", "strategy_name": "B"},
        ],
        initial_rows=[
            {"strategy_id": sid, "state_id": "cured", "proportion": None, "proportion_mode": "direct", "proportion_parameter_id": "p_cure"}
            for sid in ("A", "B")
        ] + [
            {"strategy_id": sid, "state_id": "uncured", "proportion": None, "proportion_mode": "complement", "proportion_parameter_id": "p_cure"}
            for sid in ("A", "B")
        ],
        transition_rows=[],
        state_reward_rows=[],
        transition_reward_rows=[],
    )
    loaded = load_state_transition_bundle(state_transition_bundle_json(bundle))
    compiled = compile_loaded_state_transition_bundle(loaded)
    assert loaded["engine"]["cycle_length_years"] == 0.5
    allocations = compiled.model.strategies[0].initial_distribution
    assert allocations[0].proportion_mode == "direct"
    assert allocations[0].proportion_parameter_id == "p_cure"
    assert allocations[1].proportion_mode == "complement"


def test_semi_markov_bundle_preserves_probability_to_rate_source_interval():
    bundle = build_state_transition_bundle(
        model_type="semi_markov",
        model_name="Converted probability regression",
        methods={
            "reference_case_code": "CUSTOM",
            "outcome_code": "QALY",
            "currency_code": "GBP",
            "threshold": 30000.0,
            "perspective_label": "Healthcare payer",
            "included_cost_bearers": ["health_system"],
            "cost_discount_rate": 0.035,
            "outcome_discount_rate": 0.035,
        },
        engine={
            "cycle_length_years": 0.25,
            "max_cycles": 12,
            "state_accrual_timing": "half_cycle",
            "transition_reward_timing": "mid_cycle",
            "termination_mode": "fixed_cycles",
            "depletion_threshold": 0.0001,
        },
        parameter_rows=[_row("p_event", 0.2)],
        state_rows=[
            {"state_id": "alive", "state_name": "Alive", "absorbing": False},
            {"state_id": "event", "state_name": "Event", "absorbing": True},
        ],
        strategy_rows=[
            {"strategy_id": "A", "strategy_name": "A"},
            {"strategy_id": "B", "strategy_name": "B"},
        ],
        initial_rows=[
            {"strategy_id": "A", "state_id": "alive", "proportion": 1.0},
            {"strategy_id": "B", "state_id": "alive", "proportion": 1.0},
        ],
        transition_rows=[
            {
                "strategy_id": sid,
                "origin_state": "alive",
                "destination_state": "event",
                "input_type": "probability_to_rate",
                "probability_mode": "direct",
                "time_basis": "model_time",
                "start_time": 0.0,
                "end_time": None,
                "parameter_id": "p_event",
                "source_interval_years": 1.0,
            }
            for sid in ("A", "B")
        ],
        state_reward_rows=[],
        transition_reward_rows=[],
    )
    loaded = load_state_transition_bundle(state_transition_bundle_json(bundle))
    compiled = compile_loaded_state_transition_bundle(loaded)
    transition = compiled.model.strategies[0].transitions[0]
    assert loaded["engine"]["cycle_length_years"] == 0.25
    assert transition.input_type == "probability_to_rate"
    assert transition.source_interval_years == 1.0
