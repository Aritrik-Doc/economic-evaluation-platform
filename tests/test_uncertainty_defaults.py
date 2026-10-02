from model.schema import AssumptionSpec, EvidenceSource, Parameter, UncertaintySpec
from model.uncertainty_defaults import suggest_distribution


SOURCE = EvidenceSource(citation="Test source", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(statement="Test", rationale="Test")
UNCERTAINTY = UncertaintySpec(kind="none", rationale="Test")


def parameter(id, label, value, unit, category, **kwargs):
    return Parameter(
        id=id,
        label=label,
        value=value,
        unit=unit,
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=UNCERTAINTY,
        **kwargs,
    )


def test_probability_suggests_beta():
    p = parameter("p_event", "Probability of event", 0.2, "proportion", "clinical")
    assert suggest_distribution(p).family == "beta"


def test_positive_cost_suggests_gamma():
    p = parameter(
        "cost",
        "Hospital cost",
        1000,
        "GBP",
        "cost",
        currency="GBP",
        price_year=2026,
        cost_bearers=("health_system",),
    )
    assert suggest_distribution(p).family == "gamma"


def test_hazard_ratio_suggests_lognormal():
    p = parameter("hr", "Hazard ratio", 0.75, "ratio", "clinical")
    assert suggest_distribution(p).family == "lognormal"


def test_bounded_utility_suggests_beta_with_caution():
    p = parameter("u", "Utility", 0.7, "utility", "utility")
    suggestion = suggest_distribution(p)
    assert suggestion.family == "beta"
    assert "below 0" in suggestion.caution


def test_negative_utility_does_not_suggest_beta():
    p = parameter("u", "Utility", -0.1, "utility", "utility")
    assert suggest_distribution(p).family == "normal"
