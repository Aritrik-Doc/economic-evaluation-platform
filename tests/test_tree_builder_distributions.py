import pytest

from model.tree_builder import BuilderValidationError, compile_parameter_rows, parse_distribution_parameters


def row():
    return {
        "id":"p_event","label":"Probability of event","value":0.2,"unit":"proportion","category":"clinical",
        "source_type":"randomised_trial","source_citation":"Example trial","source_url":"","publication_year":2025,
        "assumption":"Applies to model population","assumption_rationale":"Test",
        "uncertainty_kind":"distribution","uncertainty_rationale":"Sampling uncertainty",
        "lower":None,"upper":None,"distribution_family":"beta","distribution_parameters":"alpha=20,beta=80",
        "correlation_group":"","currency":"","price_year":None,"cost_bearers":"","notes":"",
    }


def test_distribution_parameters_parse():
    assert parse_distribution_parameters("alpha=20; beta=80") == (("alpha", 20.0), ("beta", 80.0))


def test_builder_compiles_distribution_uncertainty():
    parameter = compile_parameter_rows([row()])[0]
    assert parameter.uncertainty.kind == "distribution"
    assert parameter.uncertainty.distribution.family == "beta"
    assert dict(parameter.uncertainty.distribution.parameters) == {"alpha":20.0,"beta":80.0}


def test_distribution_requires_named_parameters():
    bad=row(); bad["distribution_parameters"]="20,80"
    with pytest.raises(BuilderValidationError, match="name=value"):
        compile_parameter_rows([bad])
