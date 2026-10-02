import pytest

from model.parameterisation import (
    ParameterisationError,
    beta_from_mean_se,
    canonical_distribution_parameters,
    gamma_from_mean_sd,
    migrate_parameter_row,
)


def test_beta_mean_se_round_trip_mean():
    params = beta_from_mean_se(0.7, 0.05)
    mean = params["alpha"] / (params["alpha"] + params["beta"])
    assert mean == pytest.approx(0.7)


def test_gamma_mean_sd_parameterisation():
    params = gamma_from_mean_sd(100.0, 20.0)
    assert params["shape"] * params["scale"] == pytest.approx(100.0)
    assert (params["shape"] * params["scale"] ** 2) ** 0.5 == pytest.approx(20.0)


def test_normal_ci_converts_to_positive_sd():
    params = canonical_distribution_parameters(
        "normal",
        "Estimate + 95% CI",
        {"estimate": 2.0, "lower_ci": 1.6, "upper_ci": 2.4},
    )
    assert params["mean"] == 2.0
    assert params["sd"] > 0


def test_beta_rejects_impossible_mean_se():
    with pytest.raises(ParameterisationError):
        beta_from_mean_se(0.5, 1.0)


def test_legacy_range_row_migrates_to_dsa_only():
    row = migrate_parameter_row(
        {
            "uncertainty_kind": "range",
            "uncertainty_rationale": "Evidence interval",
            "lower": 0.2,
            "upper": 0.8,
        }
    )
    assert row["dsa_enabled"] is True
    assert row["psa_enabled"] is False
    assert row["dsa_lower"] == 0.2


def test_legacy_distribution_row_migrates_to_psa_only():
    row = migrate_parameter_row(
        {
            "uncertainty_kind": "distribution",
            "uncertainty_rationale": "Sampling uncertainty",
            "distribution_family": "beta",
            "distribution_parameters": "alpha=2,beta=3",
        }
    )
    assert row["dsa_enabled"] is False
    assert row["psa_enabled"] is True
    assert row["distribution_parameters"] == {"alpha": 2.0, "beta": 3.0}
