import pytest

from ui.bia_context import BIAContextError, bia_definition_from_session


def _session(valid=True):
    return {
        "bia_context_valid": valid,
        "bia_profile": "CUSTOM",
        "bia_horizon_custom": 2,
        "bia_currency_custom": "GBP",
        "bia_interventions": [
            {"id": "a", "name": "Current option"},
            {"id": "b", "name": "New option"},
        ],
        "bia_population_rows": [
            {"year": 1, "eligible_population": 1000.0, "covered_lives": 10000.0},
            {"year": 2, "eligible_population": 1100.0, "covered_lives": 10500.0},
        ],
        "bia_treatment_mix_rows": [
            {"scenario": scenario, "year": year, "intervention_id": intervention, "share": share}
            for year in (1, 2)
            for scenario, values in (
                ("current", {"a": 1.0, "b": 0.0}),
                ("future", {"a": 0.5, "b": 0.5}),
            )
            for intervention, share in values.items()
        ],
        "bia_cost_inputs": {
            "a": {"acquisition": 100.0, "annual_change": 0.0},
            "b": {"acquisition": 200.0, "annual_change": 0.0},
        },
        "bia_included_categories": ["acquisition"],
    }


def test_valid_bia_context_rebuilds_current_definition():
    definition, currency, horizon = bia_definition_from_session(_session())
    assert currency == "GBP"
    assert horizon == 2
    assert [row.eligible_population for row in definition.population] == pytest.approx([1000.0, 1100.0])
    assert len(definition.treatment_mix) == 8
    acquisition = [row for row in definition.costs if row.category == "acquisition" and row.intervention_id == "b"]
    assert [row.cost_per_treated_person for row in acquisition] == pytest.approx([200.0, 200.0])


def test_invalid_bia_context_is_rejected_instead_of_reusing_stale_state():
    with pytest.raises(BIAContextError, match="valid analysis"):
        bia_definition_from_session(_session(valid=False))


def test_horizon_mismatch_is_rejected():
    session = _session()
    session["bia_horizon_custom"] = 3
    with pytest.raises(BIAContextError, match="horizon"):
        bia_definition_from_session(session)
