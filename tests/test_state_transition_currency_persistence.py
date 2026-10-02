import pytest

from model.state_transition_persistence import StateTransitionPersistenceError, build_state_transition_bundle


def test_persisted_model_rejects_cost_currency_mismatch():
    parameters = [
        {
            "id": "p_death",
            "label": "Death probability",
            "value": 0.1,
            "unit": "probability",
            "category": "clinical",
            "source_citation": "Illustrative",
            "source_type": "user_assumption",
            "publication_year": None,
            "source_url": "",
            "source_details": "",
            "assumption": "Illustrative",
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
        },
        {
            "id": "cost_alive",
            "label": "Alive cost",
            "value": 1000,
            "unit": "INR/year",
            "category": "cost",
            "currency": "INR",
            "price_year": 2026,
            "cost_bearers": "health_system",
            "source_citation": "Illustrative",
            "source_type": "user_assumption",
            "publication_year": None,
            "source_url": "",
            "source_details": "",
            "assumption": "Illustrative",
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
        },
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
    transitions = []
    for sid in ("A", "B"):
        transitions += [
            {"strategy_id": sid, "origin_state": "alive", "destination_state": "dead", "probability_parameter_id": "p_death", "probability_mode": "direct"},
            {"strategy_id": sid, "origin_state": "alive", "destination_state": "alive", "probability_parameter_id": "", "probability_mode": "residual"},
        ]

    with pytest.raises(StateTransitionPersistenceError, match="selected analysis currency is GBP"):
        build_state_transition_bundle(
            model_type="cohort_markov",
            model_name="Currency mismatch",
            methods={
                "reference_case_code": "CUSTOM",
                "outcome_code": "QALY",
                "currency_code": "GBP",
                "threshold": 30000.0,
                "perspective_label": "Payer",
                "included_cost_bearers": ["health_system"],
                "cost_discount_rate": 0.035,
                "outcome_discount_rate": 0.035,
            },
            engine={
                "cycle_length_years": 1.0,
                "max_cycles": 10,
                "state_accrual_timing": "half_cycle",
                "transition_reward_timing": "mid_cycle",
                "termination_mode": "fixed_cycles",
                "depletion_threshold": 0.0001,
            },
            parameter_rows=parameters,
            state_rows=states,
            strategy_rows=strategies,
            initial_rows=initial,
            transition_rows=transitions,
            state_reward_rows=[],
            transition_reward_rows=[],
        )
