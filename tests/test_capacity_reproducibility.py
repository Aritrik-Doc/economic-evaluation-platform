from model.capacity_reproducibility import (
    capacity_result_is_current,
    capacity_source_fingerprint,
)


def test_shared_population_change_invalidates_capacity_result():
    state = {
        "rc_population_source": "Shared Population & Uptake",
        "rc_requirement_source": "Manual",
        "rc_capacity_total_staff_1": 100.0,
        "pu_options": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}],
        "pu_population_rows": [{"year": 1, "eligible_population": 1000.0, "covered_lives": None}],
        "pu_uptake_rows": [
            {"scenario": "future", "year": 1, "intervention_id": "A", "share": 0.5},
            {"scenario": "future", "year": 1, "intervention_id": "B", "share": 0.5},
        ],
        "pu_population_basis": "annual_eligible_population",
    }
    context = {
        "population_source": "Shared Population & Uptake",
        "requirement_source": "Manual",
        "clinical_model_type": None,
    }
    fingerprint = capacity_source_fingerprint(
        state,
        population_source=context["population_source"],
        requirement_source=context["requirement_source"],
    )
    assert capacity_result_is_current(state, context, fingerprint)
    state["pu_population_rows"] = [
        {"year": 1, "eligible_population": 1200.0, "covered_lives": None}
    ]
    assert not capacity_result_is_current(state, context, fingerprint)


def test_capacity_local_display_change_does_not_invalidate_cross_page_result():
    state = {
        "rc_result_resource": "staff",
        "pu_options": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}],
        "pu_population_rows": [{"year": 1, "eligible_population": 1000.0, "covered_lives": None}],
        "pu_uptake_rows": [],
        "pu_population_basis": "annual_eligible_population",
    }
    context = {
        "population_source": "Shared Population & Uptake",
        "requirement_source": "Manual",
        "clinical_model_type": None,
    }
    fingerprint = capacity_source_fingerprint(
        state,
        population_source=context["population_source"],
        requirement_source=context["requirement_source"],
    )
    state["rc_result_resource"] = "bed_days"
    assert capacity_result_is_current(state, context, fingerprint)


def test_linked_clinical_model_change_invalidates_capacity_result_but_psa_output_does_not():
    state = {
        "pu_options": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}],
        "pu_population_rows": [{"year": 1, "eligible_population": 1000.0, "covered_lives": None}],
        "pu_uptake_rows": [],
        "pu_population_basis": "new_treatment_starts",
        "markov_parameters": [{"id": "staff_hours", "value": 2.0}],
        "mk_cycle_months": 6,
        "markov_psa_result": "irrelevant stochastic output",
    }
    context = {
        "population_source": "Shared Population & Uptake",
        "requirement_source": "Linked clinical model",
        "clinical_model_type": "Cohort Markov",
    }
    fingerprint = capacity_source_fingerprint(
        state,
        population_source=context["population_source"],
        requirement_source=context["requirement_source"],
        clinical_model_type=context["clinical_model_type"],
    )
    state["markov_psa_result"] = "different PSA result"
    assert capacity_result_is_current(state, context, fingerprint)
    state["markov_parameters"] = [{"id": "staff_hours", "value": 3.0}]
    assert not capacity_result_is_current(state, context, fingerprint)


def test_bia_cost_only_change_conservatively_invalidates_bia_sourced_capacity():
    state = {
        "bia_eligible_3_0": 1000.0,
        "bia_share_future_1_new": 0.3,
        "bia_cost_inputs": {"new": {"acquisition": 100.0}},
    }
    context = {
        "population_source": "Budget Impact Analysis",
        "requirement_source": "Manual",
        "clinical_model_type": None,
    }
    fingerprint = capacity_source_fingerprint(
        state,
        population_source=context["population_source"],
        requirement_source=context["requirement_source"],
    )
    state["bia_cost_inputs"] = {"new": {"acquisition": 150.0}}
    assert not capacity_result_is_current(state, context, fingerprint)
