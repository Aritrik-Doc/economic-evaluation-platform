import pytest

from model.decision_tree import (
    ChanceNode,
    DecisionTreeDefinition,
    DecisionTreeValidationError,
    StrategyRoot,
    TerminalNode,
    TreeBranch,
    run_decision_tree,
    validate_decision_tree,
)
from model.schema import AssumptionSpec, EvidenceSource, Parameter, UncertaintySpec


SOURCE = EvidenceSource(
    citation="Illustrative source",
    source_type="user_assumption",
)
ASSUMPTION = AssumptionSpec(
    statement="Illustrative test assumption.",
    rationale="Required for deterministic unit test.",
)
NO_UNCERTAINTY = UncertaintySpec(
    kind="none",
    rationale="Unit test uses fixed values.",
)


def parameter(
    id: str,
    value: float,
    category: str = "clinical",
    *,
    cost_bearers: tuple[str, ...] = (),
) -> Parameter:
    kwargs = {}
    if category == "cost":
        kwargs = {
            "currency": "GBP",
            "price_year": 2026,
            "cost_bearers": cost_bearers or ("health_system",),
        }
    return Parameter(
        id=id,
        label=id,
        value=value,
        unit="unit",
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=NO_UNCERTAINTY,
        **kwargs,
    )


def simple_tree():
    parameters = (
        parameter("p_success_a", 0.8),
        parameter("p_failure_a", 0.2),
        parameter("p_success_b", 0.6),
        parameter("p_failure_b", 0.4),
        parameter("cost_a", 1000, "cost"),
        parameter("cost_b", 500, "cost"),
        parameter("cost_success", 100, "cost"),
        parameter("cost_failure", 1000, "cost"),
        parameter("qaly_success", 2.0, "utility"),
        parameter("qaly_failure", 1.0, "utility"),
    )
    tree = DecisionTreeDefinition(
        strategy_roots=(
            StrategyRoot("A", "a_root"),
            StrategyRoot("B", "b_root"),
        ),
        chance_nodes=(
            ChanceNode(
                "a_root",
                "Outcome under A",
                (
                    TreeBranch("Success", "p_success_a", "success"),
                    TreeBranch("Failure", "p_failure_a", "failure"),
                ),
                cost_parameter_ids=("cost_a",),
            ),
            ChanceNode(
                "b_root",
                "Outcome under B",
                (
                    TreeBranch("Success", "p_success_b", "success"),
                    TreeBranch("Failure", "p_failure_b", "failure"),
                ),
                cost_parameter_ids=("cost_b",),
            ),
        ),
        terminal_nodes=(
            TerminalNode(
                "success",
                "Success",
                cost_parameter_ids=("cost_success",),
                outcome_parameter_ids=("qaly_success",),
            ),
            TerminalNode(
                "failure",
                "Failure",
                cost_parameter_ids=("cost_failure",),
                outcome_parameter_ids=("qaly_failure",),
            ),
        ),
    )
    return tree, parameters


def test_expected_values_by_strategy():
    tree, parameters = simple_tree()
    result = run_decision_tree(tree, parameters)
    rows = {row.strategy_id: row for row in result.strategies}

    assert rows["A"].expected_cost == pytest.approx(1280)
    assert rows["A"].expected_outcome == pytest.approx(1.8)
    assert rows["B"].expected_cost == pytest.approx(960)
    assert rows["B"].expected_outcome == pytest.approx(1.6)


def test_parameter_overrides_support_sensitivity_analysis():
    tree, parameters = simple_tree()
    result = run_decision_tree(
        tree,
        parameters,
        overrides={"p_success_a": 0.5, "p_failure_a": 0.5},
    )
    row = next(item for item in result.strategies if item.strategy_id == "A")
    assert row.expected_cost == pytest.approx(1550)
    assert row.expected_outcome == pytest.approx(1.5)


def test_probabilities_must_sum_to_one():
    tree, parameters = simple_tree()
    with pytest.raises(DecisionTreeValidationError, match="sum to"):
        validate_decision_tree(
            tree,
            parameters,
            overrides={"p_success_a": 0.7},
        )


def test_probability_must_be_between_zero_and_one():
    tree, parameters = simple_tree()
    with pytest.raises(DecisionTreeValidationError, match="between 0 and 1"):
        validate_decision_tree(
            tree,
            parameters,
            overrides={"p_success_a": 1.1, "p_failure_a": -0.1},
        )


def test_undefined_child_is_rejected():
    tree, parameters = simple_tree()
    bad = DecisionTreeDefinition(
        strategy_roots=tree.strategy_roots,
        chance_nodes=(
            ChanceNode(
                "a_root",
                "Bad",
                (
                    TreeBranch("Success", "p_success_a", "missing"),
                    TreeBranch("Failure", "p_failure_a", "failure"),
                ),
            ),
            tree.chance_nodes[1],
        ),
        terminal_nodes=tree.terminal_nodes,
    )
    with pytest.raises(DecisionTreeValidationError, match="undefined child"):
        validate_decision_tree(bad, parameters)


def test_cycle_is_rejected():
    parameters = (
        parameter("p1", 0.5),
        parameter("p2", 0.5),
        parameter("p3", 0.5),
        parameter("p4", 0.5),
    )
    tree = DecisionTreeDefinition(
        strategy_roots=(StrategyRoot("A", "n1"), StrategyRoot("B", "end")),
        chance_nodes=(
            ChanceNode(
                "n1",
                "Cycle",
                (
                    TreeBranch("Loop", "p1", "n2"),
                    TreeBranch("End", "p2", "end"),
                ),
            ),
            ChanceNode(
                "n2",
                "Cycle 2",
                (
                    TreeBranch("Loop", "p3", "n1"),
                    TreeBranch("End", "p4", "end"),
                ),
            ),
        ),
        terminal_nodes=(TerminalNode("end", "End"),),
    )
    with pytest.raises(DecisionTreeValidationError, match="cycle"):
        validate_decision_tree(tree, parameters)


def test_cost_perspective_filters_bearers():
    parameters = (
        parameter("payer_cost", 1000, "cost", cost_bearers=("health_system",)),
        parameter(
            "patient_cost",
            300,
            "cost",
            cost_bearers=("patient_direct_non_medical",),
        ),
    )
    tree = DecisionTreeDefinition(
        strategy_roots=(StrategyRoot("A", "a"), StrategyRoot("B", "b")),
        chance_nodes=(),
        terminal_nodes=(
            TerminalNode("a", "A", cost_parameter_ids=("payer_cost", "patient_cost")),
            TerminalNode("b", "B"),
        ),
    )

    payer = run_decision_tree(
        tree,
        parameters,
        included_cost_bearers=("health_system",),
    )
    societal = run_decision_tree(
        tree,
        parameters,
        included_cost_bearers=("health_system", "patient_direct_non_medical"),
    )

    assert payer.strategies[0].expected_cost == 1000
    assert societal.strategies[0].expected_cost == 1300


def test_non_cost_reward_cannot_be_used_as_cost():
    tree, parameters = simple_tree()
    bad = DecisionTreeDefinition(
        strategy_roots=tree.strategy_roots,
        chance_nodes=tree.chance_nodes,
        terminal_nodes=(
            TerminalNode(
                "success",
                "Success",
                cost_parameter_ids=("qaly_success",),
            ),
            tree.terminal_nodes[1],
        ),
    )
    with pytest.raises(DecisionTreeValidationError, match="must reference a cost"):
        validate_decision_tree(bad, parameters)


def test_unknown_override_rejected():
    tree, parameters = simple_tree()
    with pytest.raises(DecisionTreeValidationError, match="Unknown parameter"):
        run_decision_tree(tree, parameters, overrides={"missing": 1.0})
