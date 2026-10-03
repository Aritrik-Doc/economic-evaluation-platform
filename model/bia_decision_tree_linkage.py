"""Decision-tree adapter for Budget Impact Analysis clinical linkage."""

from __future__ import annotations

from math import ceil
from typing import Mapping, Sequence

from model.bia_clinical_linkage import ClinicalCostProfile, ClinicalLinkageError
from model.decision_tree import (
    ChanceNode,
    DecisionTreeDefinition,
    TerminalNode,
    TimedReward,
    validate_decision_tree,
)
from model.schema import Parameter


def _cost_rewards(node: ChanceNode | TerminalNode) -> tuple[TimedReward, ...]:
    legacy = tuple(TimedReward(parameter_id, 0.0) for parameter_id in node.cost_parameter_ids)
    return legacy + node.cost_rewards


def available_tree_cost_parameters(tree: DecisionTreeDefinition) -> tuple[str, ...]:
    ids = {
        reward.parameter_id
        for node in (*tree.chance_nodes, *tree.terminal_nodes)
        for reward in _cost_rewards(node)
    }
    return tuple(sorted(ids))


def _budget_year(time_years: float) -> int:
    """Map an absolute reward time to an annual cash-flow bucket.

    Budget Year 1 covers model time 0 through 1 year inclusive; a reward just
    after year 1 falls in Budget Year 2. This makes t=0 initiation costs and
    t=1 end-of-first-year costs both visible in the first annual budget period.
    """
    if time_years <= 0:
        return 1
    return max(1, int(ceil(time_years - 1e-12)))


def project_decision_tree_cost_profiles(
    tree: DecisionTreeDefinition,
    parameters: Sequence[Parameter],
    *,
    selected_cost_parameter_ids: Sequence[str],
    horizon_years: int,
    strategy_names: Mapping[str, str] | None = None,
    included_cost_bearers: Sequence[str] | None = None,
    overrides: Mapping[str, float] | None = None,
) -> tuple[ClinicalCostProfile, ...]:
    if horizon_years < 1:
        raise ClinicalLinkageError("BIA linkage requires at least one annual budget period.")

    validate_decision_tree(
        tree,
        parameters,
        overrides=overrides,
        cost_discount_rate=0.0,
        outcome_discount_rate=0.0,
    )

    parameter_map = {parameter.id: parameter for parameter in parameters}
    referenced = set(available_tree_cost_parameters(tree))
    selected = tuple(
        dict.fromkeys(str(item).strip() for item in selected_cost_parameter_ids if str(item).strip())
    )
    if not selected:
        raise ClinicalLinkageError(
            "Select at least one decision-tree cost parameter to import. This explicit selection prevents double counting."
        )
    unknown = set(selected) - referenced
    if unknown:
        raise ClinicalLinkageError(
            "Selected cost parameter(s) are not used as decision-tree cost rewards: "
            + ", ".join(sorted(unknown))
        )
    for parameter_id in selected:
        parameter = parameter_map.get(parameter_id)
        if parameter is None:
            raise ClinicalLinkageError(f"Selected cost parameter '{parameter_id}' is undefined.")
        if parameter.category != "cost":
            raise ClinicalLinkageError(f"Selected parameter '{parameter_id}' is not a cost parameter.")

    values = {
        parameter.id: float((overrides or {}).get(parameter.id, parameter.value))
        for parameter in parameters
    }
    included_bearers = set(included_cost_bearers) if included_cost_bearers is not None else None
    selected_set = set(selected)
    chance = {node.id: node for node in tree.chance_nodes}
    terminal = {node.id: node for node in tree.terminal_nodes}

    def cost_value(parameter_id: str) -> float:
        parameter = parameter_map[parameter_id]
        if included_bearers is not None and not set(parameter.cost_bearers).intersection(included_bearers):
            return 0.0
        return values[parameter_id]

    def branch_probability(branch) -> float:
        value = values[branch.probability_parameter_id]
        return value if branch.probability_mode == "direct" else 1.0 - value

    names = dict(strategy_names or {})
    profiles = []
    for root in tree.strategy_roots:
        annual = [0.0] * horizon_years

        def visit(node_id: str, reach_probability: float) -> None:
            node = terminal.get(node_id) or chance.get(node_id)
            if node is None:
                raise ClinicalLinkageError(f"Decision tree references unknown node '{node_id}'.")

            for reward in _cost_rewards(node):
                if reward.parameter_id not in selected_set:
                    continue
                year = _budget_year(reward.time_years)
                if year <= horizon_years:
                    annual[year - 1] += reach_probability * cost_value(reward.parameter_id)

            if isinstance(node, ChanceNode):
                for branch in node.branches:
                    visit(
                        branch.child_node_id,
                        reach_probability * branch_probability(branch),
                    )

        visit(root.root_node_id, 1.0)
        profiles.append(
            ClinicalCostProfile(
                strategy_id=root.strategy_id,
                strategy_name=names.get(root.strategy_id, root.strategy_id),
                annual_cost_per_patient=tuple(annual),
                source_model_type="decision_tree",
                included_parameter_ids=selected,
            )
        )

    return tuple(profiles)
