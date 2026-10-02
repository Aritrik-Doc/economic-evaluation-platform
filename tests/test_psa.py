import numpy as np
import pytest

from model.decision_tree import ChanceNode, DecisionTreeDefinition, StrategyRoot, TerminalNode, TreeBranch
from model.psa import (
    PSAConfigurationError,
    ceac,
    incremental_plane,
    pairwise_probability_cost_effective,
    psa_configuration_warnings,
    run_tree_psa,
    sample_distribution,
)
from model.schema import AssumptionSpec, DistributionSpec, EvidenceSource, Parameter, UncertaintySpec


SOURCE = EvidenceSource(citation="Test source", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(statement="Test", rationale="Test")
FIXED = UncertaintySpec(kind="none", rationale="Fixed for test")


def param(id, value, category="clinical", uncertainty=FIXED, **kwargs):
    return Parameter(
        id=id,
        label=id,
        value=value,
        unit="unit",
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=uncertainty,
        **kwargs,
    )


def example_model():
    beta_uncertainty = UncertaintySpec(
        kind="distribution",
        rationale="Illustrative beta uncertainty",
        distribution=DistributionSpec("beta", (("alpha", 8.0), ("beta", 2.0))),
    )
    parameters = (
        param("p_success", 0.8, uncertainty=beta_uncertainty),
        param("cost_a", 1000, "cost", currency="GBP", price_year=2026, cost_bearers=("health_system",)),
        param("cost_b", 600, "cost", currency="GBP", price_year=2026, cost_bearers=("health_system",)),
        param("success_qaly", 2.0, "utility"),
        param("failure_qaly", 1.0, "utility"),
        param("b_qaly", 1.4, "utility"),
    )
    tree = DecisionTreeDefinition(
        strategy_roots=(StrategyRoot("A", "a_root"), StrategyRoot("B", "b_terminal")),
        chance_nodes=(
            ChanceNode(
                "a_root",
                "A outcome",
                (
                    TreeBranch("Success", "p_success", "success", "direct"),
                    TreeBranch("Failure", "p_success", "failure", "complement"),
                ),
                cost_parameter_ids=("cost_a",),
            ),
        ),
        terminal_nodes=(
            TerminalNode("success", "Success", outcome_parameter_ids=("success_qaly",)),
            TerminalNode("failure", "Failure", outcome_parameter_ids=("failure_qaly",)),
            TerminalNode("b_terminal", "B", cost_parameter_ids=("cost_b",), outcome_parameter_ids=("b_qaly",)),
        ),
    )
    return tree, parameters


def dirichlet_model():
    def d_uncertainty(alpha):
        return UncertaintySpec(
            kind="distribution",
            rationale="Joint competing probability uncertainty",
            distribution=DistributionSpec("dirichlet", (("alpha", float(alpha)),)),
            correlation_group="outcome_simplex",
        )

    parameters = (
        param("p_good", 0.5, uncertainty=d_uncertainty(5)),
        param("p_mid", 0.3, uncertainty=d_uncertainty(3)),
        param("p_bad", 0.2, uncertainty=d_uncertainty(2)),
        param("good_qaly", 2.0, "utility"),
        param("mid_qaly", 1.0, "utility"),
        param("bad_qaly", 0.0, "utility"),
        param("b_qaly", 0.8, "utility"),
    )
    tree = DecisionTreeDefinition(
        strategy_roots=(StrategyRoot("A", "root"), StrategyRoot("B", "b")),
        chance_nodes=(
            ChanceNode(
                "root",
                "Three competing outcomes",
                (
                    TreeBranch("Good", "p_good", "good"),
                    TreeBranch("Intermediate", "p_mid", "mid"),
                    TreeBranch("Bad", "p_bad", "bad"),
                ),
            ),
        ),
        terminal_nodes=(
            TerminalNode("good", "Good", outcome_parameter_ids=("good_qaly",)),
            TerminalNode("mid", "Intermediate", outcome_parameter_ids=("mid_qaly",)),
            TerminalNode("bad", "Bad", outcome_parameter_ids=("bad_qaly",)),
            TerminalNode("b", "B", outcome_parameter_ids=("b_qaly",)),
        ),
    )
    return tree, parameters


def test_sample_beta_stays_bounded():
    rng = np.random.default_rng(1)
    dist = DistributionSpec("beta", (("alpha", 2.0), ("beta", 3.0)))
    draws = [sample_distribution(dist, rng) for _ in range(100)]
    assert all(0 <= draw <= 1 for draw in draws)


def test_scalar_dirichlet_sampling_is_rejected():
    rng = np.random.default_rng(1)
    dist = DistributionSpec("dirichlet", (("alpha", 2.0),))
    with pytest.raises(PSAConfigurationError, match="sampled jointly"):
        sample_distribution(dist, rng)


def test_psa_is_reproducible_with_seed():
    tree, parameters = example_model()
    a = run_tree_psa(tree, parameters, iterations=50, seed=123)
    b = run_tree_psa(tree, parameters, iterations=50, seed=123)
    assert np.array_equal(a.parameter_draws["p_success"], b.parameter_draws["p_success"])
    assert np.array_equal(a.costs["A"], b.costs["A"])
    assert np.array_equal(a.outcomes["A"], b.outcomes["A"])


def test_dirichlet_group_is_joint_and_sums_to_one():
    tree, parameters = dirichlet_model()
    result = run_tree_psa(tree, parameters, iterations=200, seed=77)
    total = result.parameter_draws["p_good"] + result.parameter_draws["p_mid"] + result.parameter_draws["p_bad"]
    assert np.allclose(total, 1.0)
    assert np.all(result.parameter_draws["p_good"] >= 0)
    assert np.all(result.parameter_draws["p_mid"] >= 0)
    assert np.all(result.parameter_draws["p_bad"] >= 0)
    assert result.warnings == ()


def test_dirichlet_group_is_reproducible():
    tree, parameters = dirichlet_model()
    a = run_tree_psa(tree, parameters, iterations=30, seed=9)
    b = run_tree_psa(tree, parameters, iterations=30, seed=9)
    for pid in ("p_good", "p_mid", "p_bad"):
        assert np.array_equal(a.parameter_draws[pid], b.parameter_draws[pid])


def test_incremental_plane_and_ceac_shapes():
    tree, parameters = example_model()
    result = run_tree_psa(tree, parameters, iterations=100, seed=7)
    de, dc = incremental_plane(result, intervention_id="A", comparator_id="B")
    assert de.shape == (100,)
    assert dc.shape == (100,)

    curve = ceac(result, [0, 1000, 10000])
    assert curve.thresholds.shape == (3,)
    for index in range(3):
        assert sum(curve.probabilities[sid][index] for sid in result.strategy_ids) == pytest.approx(1.0)


def test_pairwise_probability_cost_effective_is_probability():
    tree, parameters = example_model()
    result = run_tree_psa(tree, parameters, iterations=100, seed=8)
    probability = pairwise_probability_cost_effective(
        result,
        intervention_id="A",
        comparator_id="B",
        willingness_to_pay=30000,
    )
    assert 0 <= probability <= 1


def test_declared_non_dirichlet_correlation_runs_with_warning():
    tree, parameters = example_model()
    grouped = []
    for parameter in parameters:
        if parameter.id == "p_success":
            grouped.append(
                Parameter(
                    id=parameter.id,
                    label=parameter.label,
                    value=parameter.value,
                    unit=parameter.unit,
                    category=parameter.category,
                    source=parameter.source,
                    assumption=parameter.assumption,
                    uncertainty=UncertaintySpec(
                        kind="distribution",
                        rationale="Grouped",
                        distribution=parameter.uncertainty.distribution,
                        correlation_group="g1",
                    ),
                )
            )
        else:
            grouped.append(parameter)
    second = param(
        "another_p",
        0.5,
        uncertainty=UncertaintySpec(
            kind="distribution",
            rationale="Grouped",
            distribution=DistributionSpec("beta", (("alpha", 5.0), ("beta", 5.0))),
            correlation_group="g1",
        ),
    )
    configured = tuple(grouped) + (second,)
    warnings = psa_configuration_warnings(configured)
    assert len(warnings) == 1
    assert "sampled independently" in warnings[0]

    result = run_tree_psa(tree, configured, iterations=10, seed=1)
    assert result.iterations == 10
    assert result.warnings == warnings


def test_dirichlet_cannot_be_mixed_with_other_families_in_group():
    tree, parameters = example_model()
    d = param(
        "d",
        0.5,
        uncertainty=UncertaintySpec(
            kind="distribution",
            rationale="Joint",
            distribution=DistributionSpec("dirichlet", (("alpha", 2.0),)),
            correlation_group="mixed",
        ),
    )
    b = param(
        "b",
        0.5,
        uncertainty=UncertaintySpec(
            kind="distribution",
            rationale="Joint",
            distribution=DistributionSpec("beta", (("alpha", 2.0), ("beta", 2.0))),
            correlation_group="mixed",
        ),
    )
    with pytest.raises(PSAConfigurationError, match="mixes Dirichlet"):
        run_tree_psa(tree, parameters + (d, b), iterations=10, seed=1)
