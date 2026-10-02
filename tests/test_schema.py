from datetime import date

import pytest

from model.schema import (
    AssumptionSpec,
    DistributionSpec,
    EvidenceSource,
    FXConversionSpec,
    Parameter,
    UncertaintySpec,
)


SOURCE = EvidenceSource(
    citation="Example study",
    source_type="observational_study",
    publication_year=2025,
)
ASSUMPTION = AssumptionSpec(
    statement="No extrapolation beyond observed period.",
    rationale="Example test assumption.",
)


def test_cost_parameter_requires_currency_price_year_and_bearer():
    with pytest.raises(ValueError, match="currency"):
        Parameter(
            id="drug_cost",
            label="Drug cost",
            value=100,
            unit="per patient",
            category="cost",
            source=SOURCE,
            assumption=ASSUMPTION,
            uncertainty=UncertaintySpec(kind="none", rationale="Fixed tariff."),
        )


def test_cost_parameter_carries_auditable_metadata():
    fx_source = EvidenceSource(citation="FX provider", source_type="database")
    parameter = Parameter(
        id="drug_cost",
        label="Drug cost",
        value=8_000,
        unit="per patient",
        category="cost",
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=UncertaintySpec(
            kind="distribution",
            rationale="Sampling uncertainty in cost.",
            distribution=DistributionSpec("gamma", (("shape", 4.0), ("scale", 2000.0))),
        ),
        currency="INR",
        price_year=2026,
        cost_bearers=("health_system",),
        fx_conversion=FXConversionSpec(
            from_currency="USD",
            to_currency="INR",
            rate=90.0,
            rate_date=date(2026, 10, 1),
            source=fx_source,
        ),
    )
    assert parameter.price_year == 2026
    assert parameter.fx_conversion.rate == 90.0


def test_uncertainty_must_be_explicit_even_when_none():
    with pytest.raises(ValueError, match="rationale"):
        UncertaintySpec(kind="none", rationale="")
