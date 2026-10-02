import pytest

from model.reference_cases import REFERENCE_CASES, custom_reference_case


def test_recognised_reference_cases_present():
    assert set(REFERENCE_CASES) == {"NICE_TA", "HTAIN_2023"}


def test_nice_current_methods_profile():
    nice = REFERENCE_CASES["NICE_TA"]
    assert nice.preferred_outcome_code == "QALY"
    assert nice.perspective.code == "NHS_PSS"
    assert nice.cost_discount_rate == pytest.approx(0.035)
    assert nice.outcome_discount_rate == pytest.approx(0.035)
    assert nice.threshold_range is not None
    assert nice.threshold_range.lower == 25_000
    assert nice.threshold_range.upper == 35_000
    assert nice.analysis_currency == "GBP"


def test_indian_reference_case_profile():
    india = REFERENCE_CASES["HTAIN_2023"]
    assert india.preferred_outcome_code == "QALY"
    assert india.outcome_status("DALY_AVERTED") == "conditional"
    assert india.perspective.code == "ABRIDGED_SOCIETAL"
    assert india.cost_discount_rate == pytest.approx(0.03)
    assert india.outcome_discount_rate == pytest.approx(0.03)
    assert india.threshold_range is None
    assert india.analysis_currency == "INR"


def test_custom_reference_case_requires_explicit_methods():
    profile = custom_reference_case(
        name="Local HTA",
        perspective_label="Hospital payer",
        preferred_outcome_code="QALY",
        cost_discount_rate=0.04,
        outcome_discount_rate=0.03,
        time_horizon_rule="10 years",
        analysis_currency="USD",
        threshold=50_000,
    )
    assert profile.code == "CUSTOM"
    assert profile.threshold_range.lower == 50_000
    assert profile.perspective.label == "Hospital payer"
