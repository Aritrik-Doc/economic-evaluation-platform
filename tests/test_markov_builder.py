from model.markov_builder import compile_markov_tables, markov_structure_to_dot


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
    for pid, value in (("p_a", 0.1), ("p_b", 0.05)):
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


def test_compile_markov_tables():
    compiled = compile_markov_tables(
        parameter_rows(),
        [
            {"state_id": "alive", "state_name": "Alive", "absorbing": False},
            {"state_id": "dead", "state_name": "Dead", "absorbing": True},
        ],
        [
            {"strategy_id": "A", "strategy_name": "Treatment A"},
            {"strategy_id": "B", "strategy_name": "Treatment B"},
        ],
        [
            {"strategy_id": "A", "state_id": "alive", "proportion": 1.0},
            {"strategy_id": "B", "state_id": "alive", "proportion": 1.0},
        ],
        [
            {"strategy_id": "A", "origin_state": "alive", "destination_state": "dead", "probability_parameter_id": "p_a", "probability_mode": "direct"},
            {"strategy_id": "A", "origin_state": "alive", "destination_state": "alive", "probability_parameter_id": "", "probability_mode": "residual"},
            {"strategy_id": "B", "origin_state": "alive", "destination_state": "dead", "probability_parameter_id": "p_b", "probability_mode": "direct"},
            {"strategy_id": "B", "origin_state": "alive", "destination_state": "alive", "probability_parameter_id": "", "probability_mode": "residual"},
        ],
        [
            {"strategy_id": "A", "state_id": "alive", "parameter_id": "u_alive", "reward_type": "outcome", "accrual": "per_year"},
            {"strategy_id": "B", "state_id": "alive", "parameter_id": "u_alive", "reward_type": "outcome", "accrual": "per_year"},
        ],
        [],
        cycle_length_years=1.0,
        max_cycles=10,
    )
    assert len(compiled.model.states) == 2
    assert compiled.strategy_names["A"] == "Treatment A"
    assert compiled.model.strategies[0].transitions[1].probability_mode == "residual"


def test_dot_contains_strategy_specific_edges():
    dot = markov_structure_to_dot(
        [
            {"state_id": "alive", "state_name": "Alive", "absorbing": False},
            {"state_id": "dead", "state_name": "Dead", "absorbing": True},
        ],
        [
            {"strategy_id": "A", "origin_state": "alive", "destination_state": "dead", "probability_parameter_id": "p_a", "probability_mode": "direct"},
            {"strategy_id": "B", "origin_state": "alive", "destination_state": "dead", "probability_parameter_id": "p_b", "probability_mode": "direct"},
        ],
        strategy_id="A",
    )
    assert "p_a" in dot
    assert "p_b" not in dot
    assert "doublecircle" in dot
