"""Constrained row-level operations used by the guided decision-tree UI.

These helpers intentionally do not permit arbitrary graph edits. They create
strategies, add a branch and its child together, and only delete safe leaf
nodes. The analytical compiler remains the final validation gate.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Sequence


class GuidedTreeError(ValueError):
    pass


def slugify(value: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or fallback


def unique_id(base: str, existing: Sequence[str]) -> str:
    existing_set = set(existing)
    if base not in existing_set:
        return base
    number = 2
    while f"{base}_{number}" in existing_set:
        number += 1
    return f"{base}_{number}"


def add_strategy(
    strategy_rows: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    *,
    strategy_name: str,
    first_event_label: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not strategy_name.strip():
        raise GuidedTreeError("Strategy name is required.")
    strategies = [dict(row) for row in deepcopy(strategy_rows)]
    nodes = [dict(row) for row in deepcopy(node_rows)]
    existing_strategy_ids = [str(row.get("strategy_id", "")) for row in strategies]
    existing_node_ids = [str(row.get("id", "")) for row in nodes]
    sid = unique_id(slugify(strategy_name, "strategy"), existing_strategy_ids)
    root_id = unique_id(f"{sid}_root", existing_node_ids)
    label = (first_event_label or f"What happens after {strategy_name}?").strip()
    strategies.append({"strategy_id": sid, "strategy_name": strategy_name.strip(), "root_node_id": root_id})
    nodes.append({"id": root_id, "label": label, "type": "chance", "cost_rewards": "", "outcome_rewards": ""})
    return strategies, nodes


def add_branch_with_child(
    node_rows: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
    *,
    parent_id: str,
    branch_label: str,
    probability_parameter_id: str,
    probability_mode: str,
    child_label: str,
    child_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    nodes = [dict(row) for row in deepcopy(node_rows)]
    branches = [dict(row) for row in deepcopy(branch_rows)]
    node_map = {str(row.get("id")): row for row in nodes}
    if parent_id not in node_map:
        raise GuidedTreeError("Select a defined parent node.")
    if str(node_map[parent_id].get("type", "")).lower() != "chance":
        raise GuidedTreeError("Branches can only be added to chance-event nodes.")
    if not branch_label.strip() or not child_label.strip() or not probability_parameter_id.strip():
        raise GuidedTreeError("Branch label, probability parameter and child label are required.")
    probability_mode = probability_mode.strip().lower()
    if probability_mode not in {"direct", "complement"}:
        raise GuidedTreeError("Probability mode must be direct or complement.")
    child_type = child_type.strip().lower()
    if child_type not in {"chance", "terminal"}:
        raise GuidedTreeError("Child type must be chance or terminal.")
    existing_ids = list(node_map)
    child_id = unique_id(slugify(child_label, "node"), existing_ids)
    nodes.append({"id": child_id, "label": child_label.strip(), "type": child_type, "cost_rewards": "", "outcome_rewards": ""})
    branches.append(
        {
            "from_node": parent_id,
            "label": branch_label.strip(),
            "probability_parameter_id": probability_parameter_id.strip(),
            "probability_mode": probability_mode,
            "to_node": child_id,
        }
    )
    return nodes, branches, child_id


def update_node(
    node_rows: Sequence[Mapping[str, Any]],
    node_id: str,
    *,
    label: str | None = None,
    cost_rewards: str | None = None,
    outcome_rewards: str | None = None,
) -> list[dict[str, Any]]:
    nodes = [dict(row) for row in deepcopy(node_rows)]
    matched = False
    for row in nodes:
        if str(row.get("id")) != node_id:
            continue
        matched = True
        if label is not None:
            if not label.strip():
                raise GuidedTreeError("Node label cannot be empty.")
            row["label"] = label.strip()
        if cost_rewards is not None:
            row["cost_rewards"] = cost_rewards.strip()
        if outcome_rewards is not None:
            row["outcome_rewards"] = outcome_rewards.strip()
    if not matched:
        raise GuidedTreeError("Selected node was not found.")
    return nodes


def append_reward(
    node_rows: Sequence[Mapping[str, Any]],
    node_id: str,
    *,
    parameter_id: str,
    time_years: float,
    reward_kind: str,
) -> list[dict[str, Any]]:
    if time_years < 0:
        raise GuidedTreeError("Reward time cannot be negative.")
    if reward_kind not in {"cost", "outcome"}:
        raise GuidedTreeError("Reward kind must be cost or outcome.")
    nodes = [dict(row) for row in deepcopy(node_rows)]
    field = "cost_rewards" if reward_kind == "cost" else "outcome_rewards"
    token = f"{parameter_id}@{time_years:g}"
    for row in nodes:
        if str(row.get("id")) == node_id:
            current = str(row.get(field) or "").strip()
            tokens = [item.strip() for item in current.split(",") if item.strip()]
            if token not in tokens:
                tokens.append(token)
            row[field] = ", ".join(tokens)
            return nodes
    raise GuidedTreeError("Selected node was not found.")


def delete_leaf_node(
    strategy_rows: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
    node_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root_ids = {str(row.get("root_node_id")) for row in strategy_rows}
    if node_id in root_ids:
        raise GuidedTreeError("A strategy root cannot be deleted. Delete or restructure the strategy instead.")
    if any(str(row.get("from_node")) == node_id for row in branch_rows):
        raise GuidedTreeError("Only leaf nodes can be deleted in guided mode.")
    if not any(str(row.get("id")) == node_id for row in node_rows):
        raise GuidedTreeError("Selected node was not found.")
    nodes = [dict(row) for row in node_rows if str(row.get("id")) != node_id]
    branches = [dict(row) for row in branch_rows if str(row.get("to_node")) != node_id]
    return nodes, branches
