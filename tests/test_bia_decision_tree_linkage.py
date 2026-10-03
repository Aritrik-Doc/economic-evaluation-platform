import pytest

from model.bia_decision_tree_linkage import (
    available_tree_cost_parameters,
    project_decision_tree_cost_profiles,
)
from model.bia_clinical_linkage import ClinicalLinkageError
from model.decision_tree import (
    ChanceNode,
    DecisionTreeDefinition,
    StrategyRoot,
    TerminalNode,
    TimedReward,
    TreeBranch,
)
from model.schema import AssumptionSpec, EvidenceSource, Parameter, UncertaintySpec


SOURCE = EvidenceSource(citation="test", source_type="user_assumption")
ASSUMPTION = AssumptionSpec(statement="test", rationale="test")
NO_UNCERTAINTY = UncertaintySpec(kind="none", rationale="test")


def parameter(pid: str, value: float, category: str = "clinical") -> Parameter:
    kwargs = {}
    if category == "cost":
        kwargs = {
            "currency": "GBP",
            "price_year": 2026,
            "cost_bearers": ("health_system",),
        }
    return Parameter(
        id=pid,
        label=pid,
        value=value,
        unit="unit",
        category=category,
        source=SOURCE,
        assumption=ASSUMPTION,
        uncertainty=NO_UNCERTAINTY,
        **kwargs,
    )


def tree_and_parameters():
    parameters = (
        parameter("p", 0.75),
        parameter("root_cost", 100.0, "cost"),
        parameter("success_cost", 200.0, "cost"),
        parameter("failure_cost", 1000.0, "cost"),
        parameter("later_cost", 50.0, "cost"),
    )
    tree = DecisionTreeDefinition(
        strategy_roots=(
            StrategyRoot("new", "root"),
            StrategyRoot("current", "current_end"),
        ),
        chance_nodes=(
            ChanceNode(
                "root",
                "Outcome",
                (
                    TreeBranch("Success", "p", "success"),
                    TreeBranch("Failure", "p", "failure", probability_mode="complement"),
                ),
                cost_rewards=(TimedReward("root_cost", 0.0),),
            ),
        ),
        terminal_nodes=(
            TerminalNode(
                "success",
                "Success",
                cost_rewards=(TimedReward("success_cost", 1.0), TimedReward("later_cost", 2.2)),
            ),
            TerminalNode(
                "failure",
                "Failure",
                cost_rewards=(TimedReward("failure_cost", 1.0),),
            ),
            TerminalNode("current_end", "Current"),
        ),
    )
    return tree, parameters


def test_available_tree_cost_parameters_lists_reward_parameters():
    tree, _ = tree_and_parameters()
    assert available_tree_cost_parameters(tree) == (
        "failure_cost",
        "later_cost",
        "root_cost",
        "success_cost",
    )


def test_decision_tree_projection_preserves_expected_path_probabilities_and_timing():
    tree, parameters = tree_and_parameters()
    profiles = project_decision_tree_cost_profiles(
        tree,
        parameters,
        selected_cost_parameter_ids=("root_cost", "success_cost", "failure_cost", "later_cost"),
        horizon_years=3,
        strategy_names={"new": "New", "current": "Current"},
    )
    by_id = {profile.strategy_id: profile for profile in profiles}

    # Year 1 includes t=0 root cost and t=1 terminal rewards.
    # 100 + 0.75*200 + 0.25*1000 = 500.
    assert by_id["new"].annual_cost_per_patient[0] == pytest.approx(500.0)
    # t=2.2 falls in Budget Year 3.
    assert by_id["new"].annual_cost_per_patient[1] == pytest.approx(0.0)
    assert by_id["new"].annual_cost_per_patient[2] == pytest.approx(37.5)
    assert by_id["current"].annual_cost_per_patient == pytest.approx((0.0, 0.0, 0.0))


def test_decision_tree_projection_only_imports_selected_costs():
    tree, parameters = tree_and_parameters()
    profiles = project_decision_tree_cost_profiles(
        tree,
        parameters,
        selected_cost_parameter_ids=("failure_cost",),
        horizon_years=2,
    )
    new = next(profile for profile in profiles if profile.strategy_id == "new")
    assert new.annual_cost_per_patient == pytest.approx((250.0, 0.0))


def test_decision_tree_projection_requires_explicit_cost_selection():
    tree, parameters = tree_and_parameters()
    with pytest.raises(ClinicalLinkageError, match="Select at least one"):
        project_decision_tree_cost_profiles(
            tree,
            parameters,
            selected_cost_parameter_ids=(),
            horizon_years=2,
        )
