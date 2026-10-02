from model.schema import (
    AssumptionSpec,
    DeterministicUncertaintySpec,
    DistributionSpec,
    EvidenceSource,
    Parameter,
    ProbabilisticUncertaintySpec,
)
from model.tree_builder import compile_parameter_rows


SOURCE = EvidenceSource(citation="Test evidence", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(statement="Test assumption", rationale="Unit test")


def test_parameter_can_enable_dsa_and_psa_together():
    parameter = Parameter(
        id="p_event",
        label="Event probability",
        value=0.4,
        unit="probability",
        category="clinical",
        source=SOURCE,
        assumption=ASSUMPTION,
        dsa=DeterministicUncertaintySpec(
            enabled=True,
            rationale="Plausible deterministic range",
            lower=0.25,
            upper=0.55,
        ),
        psa=ProbabilisticUncertaintySpec(
            enabled=True,
            rationale="Sampling uncertainty",
            distribution=DistributionSpec("beta", (("alpha", 4.0), ("beta", 6.0))),
        ),
    )

    assert parameter.dsa.enabled is True
    assert parameter.dsa.lower == 0.25
    assert parameter.psa.enabled is True
    assert parameter.psa.distribution.family == "beta"


def test_builder_compiles_split_dsa_and_psa_from_same_row():
    rows = [
        {
            "id": "p_event",
            "label": "Event probability",
            "value": 0.4,
            "unit": "probability",
            "category": "clinical",
            "source_type": "user_assumption",
            "source_citation": "Test evidence",
            "source_url": "",
            "publication_year": 2026,
            "source_details": "",
            "assumption": "Test assumption",
            "assumption_rationale": "Unit test",
            "dsa_enabled": True,
            "dsa_lower": 0.25,
            "dsa_upper": 0.55,
            "dsa_rationale": "Plausible deterministic range",
            "psa_enabled": True,
            "psa_rationale": "Sampling uncertainty",
            "distribution_family": "beta",
            "distribution_parameters": {"alpha": 4.0, "beta": 6.0},
            "correlation_group": "",
            "currency": "",
            "price_year": None,
            "cost_bearers": "",
            "notes": "",
        }
    ]

    parameter = compile_parameter_rows(rows)[0]
    assert parameter.dsa.enabled is True
    assert parameter.psa.enabled is True
    assert parameter.psa.distribution.parameters == (("alpha", 4.0), ("beta", 6.0))
