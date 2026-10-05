from model.transparency import parameter_transparency_items, transparency_summary


def base_row(**overrides):
    row = {
        "id": "p1",
        "label": "Probability of response",
        "value": 0.7,
        "category": "clinical",
        "source_citation": "Smith et al. 2025",
        "assumption": "Applies equally across strategies.",
        "assumption_rationale": "No evidence of effect modification.",
        "dsa_enabled": True,
        "dsa_lower": 0.6,
        "dsa_upper": 0.8,
        "dsa_rationale": "95% confidence interval.",
        "psa_enabled": True,
        "distribution_family": "beta",
        "distribution_parameters": {"alpha": 70, "beta": 30},
        "psa_rationale": "Probability bounded on 0 to 1.",
    }
    row.update(overrides)
    return row


def test_complete_parameter_is_documentationally_complete():
    items = parameter_transparency_items(base_row())
    assert {item.status for item in items} == {"complete"}


def test_missing_source_is_flagged_without_quality_scoring():
    summary = transparency_summary([base_row(source_citation="")])
    assert summary.total_parameters == 1
    assert summary.incomplete_parameters == 1
    assert summary.complete_parameters == 0
    assert summary.headline == "Documentation needs attention"


def test_illustrative_placeholder_is_provisional():
    summary = transparency_summary([
        base_row(source_citation="Illustrative input — replace with evidence")
    ])
    assert summary.provisional_parameters == 1
    assert summary.incomplete_parameters == 0


def test_parameter_library_default_instructions_are_provisional():
    summary = transparency_summary([
        base_row(
            source_citation="User-entered parameter — add the evidence source before substantive use",
            assumption="User-entered model parameter.",
            assumption_rationale="Complete the modelling rationale before substantive use.",
            dsa_enabled=False,
            psa_enabled=False,
        )
    ])
    assert summary.provisional_parameters == 1
    assert summary.complete_parameters == 0


def test_explicit_user_assumption_can_be_documentationally_complete():
    summary = transparency_summary([
        base_row(
            source_citation="User assumption documented in the modelling protocol",
            assumption="Treatment effect is held constant after the observed period.",
            assumption_rationale="No longer-term comparative evidence was identified; tested in DSA.",
            dsa_enabled=False,
            psa_enabled=False,
        )
    ])
    assert summary.complete_parameters == 1


def test_cost_parameter_requires_currency_price_year_and_bearer():
    summary = transparency_summary([
        base_row(
            category="cost",
            currency="GBP",
            price_year=2026,
            cost_bearers="health_system",
        )
    ])
    assert summary.complete_parameters == 1

    incomplete = transparency_summary([
        base_row(category="cost", currency="", price_year=None, cost_bearers="")
    ])
    assert incomplete.incomplete_parameters == 1


def test_dsa_and_psa_fields_checked_only_when_enabled():
    summary = transparency_summary([
        base_row(
            dsa_enabled=False,
            dsa_lower=None,
            dsa_upper=None,
            dsa_rationale="",
            psa_enabled=False,
            distribution_family="",
            distribution_parameters={},
            psa_rationale="",
        )
    ])
    assert summary.complete_parameters == 1
