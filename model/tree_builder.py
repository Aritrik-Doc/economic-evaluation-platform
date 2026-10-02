"""Pure helpers for compiling hybrid decision-tree builder tables."""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any, Mapping, Sequence

from model.decision_tree import (
    ChanceNode,
    DecisionTreeDefinition,
    StrategyRoot,
    TerminalNode,
    TimedReward,
    TreeBranch,
)
from model.schema import AssumptionSpec, EvidenceSource, Parameter, UncertaintySpec


class BuilderValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CompiledDecisionTree:
    tree: DecisionTreeDefinition
    parameters: tuple[Parameter, ...]
    strategy_names: Mapping[str, str]


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and isnan(value):
        return True
    return not str(value).strip()


def _text(value: Any, field: str) -> str:
    if _blank(value):
        raise BuilderValidationError(f"{field} is required.")
    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    return None if _blank(value) else str(value).strip()


def _float(value: Any, field: str) -> float:
    if _blank(value):
        raise BuilderValidationError(f"{field} is required.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BuilderValidationError(f"{field} must be numeric.") from exc


def _optional_float(value: Any) -> float | None:
    if _blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BuilderValidationError("Optional numeric value is invalid.") from exc


def _optional_int(value: Any) -> int | None:
    if _blank(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BuilderValidationError("Optional integer value is invalid.") from exc
    if not number.is_integer():
        raise BuilderValidationError("Expected a whole-number year.")
    return int(number)


def parse_id_list(value: Any) -> tuple[str, ...]:
    if _blank(value):
        return ()
    text = str(value).replace(";", ",")
    return tuple(item.strip() for item in text.split(",") if item.strip())


def parse_reward_list(value: Any) -> tuple[TimedReward, ...]:
    """Parse ``parameter@time`` tokens; missing @time means t=0 years."""
    rewards: list[TimedReward] = []
    for token in parse_id_list(value):
        if "@" in token:
            parameter_id, time_text = token.rsplit("@", 1)
            parameter_id = parameter_id.strip()
            try:
                time_years = float(time_text.strip())
            except ValueError as exc:
                raise BuilderValidationError(
                    f"Reward '{token}' has an invalid time. Use parameter@years."
                ) from exc
        else:
            parameter_id = token.strip()
            time_years = 0.0
        try:
            rewards.append(TimedReward(parameter_id, time_years))
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc
    return tuple(rewards)


def compile_parameter_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Parameter, ...]:
    parameters: list[Parameter] = []
    for index, row in enumerate(rows, start=1):
        if _blank(row.get("id")):
            continue
        parameter_id = _text(row.get("id"), f"Parameter row {index}: id")
        category = _text(row.get("category"), f"Parameter '{parameter_id}': category")
        source = EvidenceSource(
            citation=_text(row.get("source_citation"), f"Parameter '{parameter_id}': source_citation"),
            source_type=_text(row.get("source_type"), f"Parameter '{parameter_id}': source_type"),
            url=_optional_text(row.get("source_url")),
            publication_year=_optional_int(row.get("publication_year")),
            details=_optional_text(row.get("source_details")) or "",
        )
        assumption = AssumptionSpec(
            statement=_text(row.get("assumption"), f"Parameter '{parameter_id}': assumption"),
            rationale=_text(row.get("assumption_rationale"), f"Parameter '{parameter_id}': assumption_rationale"),
        )
        uncertainty = UncertaintySpec(
            kind=_text(row.get("uncertainty_kind"), f"Parameter '{parameter_id}': uncertainty_kind"),
            rationale=_text(row.get("uncertainty_rationale"), f"Parameter '{parameter_id}': uncertainty_rationale"),
            lower=_optional_float(row.get("lower")),
            upper=_optional_float(row.get("upper")),
        )
        kwargs: dict[str, Any] = {}
        if category == "cost":
            kwargs.update(
                currency=_text(row.get("currency"), f"Cost parameter '{parameter_id}': currency"),
                price_year=_optional_int(row.get("price_year")),
                cost_bearers=parse_id_list(row.get("cost_bearers")),
            )
        try:
            parameters.append(
                Parameter(
                    id=parameter_id,
                    label=_text(row.get("label"), f"Parameter '{parameter_id}': label"),
                    value=_float(row.get("value"), f"Parameter '{parameter_id}': value"),
                    unit=_text(row.get("unit"), f"Parameter '{parameter_id}': unit"),
                    category=category,
                    source=source,
                    assumption=assumption,
                    uncertainty=uncertainty,
                    notes=_optional_text(row.get("notes")) or "",
                    **kwargs,
                )
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc
    ids = [parameter.id for parameter in parameters]
    if not parameters:
        raise BuilderValidationError("At least one parameter is required.")
    if len(ids) != len(set(ids)):
        raise BuilderValidationError("Parameter ids must be unique.")
    return tuple(parameters)


def compile_structure_rows(strategy_rows, node_rows, branch_rows):
    roots: list[StrategyRoot] = []
    strategy_names: dict[str, str] = {}
    for index, row in enumerate(strategy_rows, start=1):
        if _blank(row.get("strategy_id")):
            continue
        sid = _text(row.get("strategy_id"), f"Strategy row {index}: strategy_id")
        name = _text(row.get("strategy_name"), f"Strategy '{sid}': strategy_name")
        root = _text(row.get("root_node_id"), f"Strategy '{sid}': root_node_id")
        if sid in strategy_names:
            raise BuilderValidationError("Strategy ids must be unique.")
        if name in strategy_names.values():
            raise BuilderValidationError("Strategy names must be unique.")
        strategy_names[sid] = name
        roots.append(StrategyRoot(sid, root))

    outgoing: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(branch_rows, start=1):
        if _blank(row.get("from_node")):
            continue
        origin = _text(row.get("from_node"), f"Branch row {index}: from_node")
        outgoing.setdefault(origin, []).append(row)

    chance_nodes: list[ChanceNode] = []
    terminal_nodes: list[TerminalNode] = []
    node_types: dict[str, str] = {}
    for index, row in enumerate(node_rows, start=1):
        if _blank(row.get("id")):
            continue
        node_id = _text(row.get("id"), f"Node row {index}: id")
        if node_id in node_types:
            raise BuilderValidationError("Node ids must be globally unique.")
        node_type = _text(row.get("type"), f"Node '{node_id}': type").lower()
        node_types[node_id] = node_type
        label = _text(row.get("label"), f"Node '{node_id}': label")
        legacy_costs = parse_id_list(row.get("cost_parameter_ids"))
        legacy_outcomes = parse_id_list(row.get("outcome_parameter_ids"))
        cost_rewards = parse_reward_list(row.get("cost_rewards"))
        outcome_rewards = parse_reward_list(row.get("outcome_rewards"))

        if node_type == "chance":
            branches = []
            for b_index, branch in enumerate(outgoing.get(node_id, ()), start=1):
                branches.append(
                    TreeBranch(
                        label=_text(branch.get("label"), f"Branch {b_index} from '{node_id}': label"),
                        probability_parameter_id=_text(branch.get("probability_parameter_id"), f"Branch '{node_id}': probability_parameter_id"),
                        child_node_id=_text(branch.get("to_node"), f"Branch '{node_id}': to_node"),
                        probability_mode=(
                            _optional_text(branch.get("probability_mode")) or "direct"
                        ).lower(),
                    )
                )
            try:
                chance_nodes.append(
                    ChanceNode(
                        node_id, label, tuple(branches),
                        cost_parameter_ids=legacy_costs,
                        outcome_parameter_ids=legacy_outcomes,
                        cost_rewards=cost_rewards,
                        outcome_rewards=outcome_rewards,
                    )
                )
            except ValueError as exc:
                raise BuilderValidationError(str(exc)) from exc
        elif node_type == "terminal":
            if outgoing.get(node_id):
                raise BuilderValidationError(f"Terminal node '{node_id}' cannot have outgoing branches.")
            terminal_nodes.append(
                TerminalNode(
                    node_id, label,
                    cost_parameter_ids=legacy_costs,
                    outcome_parameter_ids=legacy_outcomes,
                    cost_rewards=cost_rewards,
                    outcome_rewards=outcome_rewards,
                )
            )
        else:
            raise BuilderValidationError(f"Node '{node_id}' type must be 'chance' or 'terminal'.")

    unknown_origins = set(outgoing) - set(node_types)
    if unknown_origins:
        raise BuilderValidationError(
            "Branches originate from undefined node(s): " + ", ".join(sorted(unknown_origins)) + "."
        )
    try:
        tree = DecisionTreeDefinition(tuple(roots), tuple(chance_nodes), tuple(terminal_nodes))
    except ValueError as exc:
        raise BuilderValidationError(str(exc)) from exc
    return tree, strategy_names


def compile_builder_tables(parameter_rows, strategy_rows, node_rows, branch_rows):
    parameters = compile_parameter_rows(parameter_rows)
    tree, strategy_names = compile_structure_rows(strategy_rows, node_rows, branch_rows)
    return CompiledDecisionTree(tree, parameters, strategy_names)


def structure_to_dot(strategy_rows, node_rows, branch_rows) -> str:
    import json
    def q(value: str) -> str:
        return json.dumps(str(value))
    lines = ["digraph DecisionTree {", 'rankdir="LR";', 'graph [pad="0.2", nodesep="0.35", ranksep="0.55"];', 'node [fontname="Arial"];']
    for row in node_rows:
        if _blank(row.get("id")):
            continue
        node_id = str(row.get("id")).strip()
        label = str(row.get("label") or node_id).strip()
        node_type = str(row.get("type") or "").strip().lower()
        shape = "circle" if node_type == "chance" else "doublecircle" if node_type == "terminal" else "box"
        rewards = []
        if not _blank(row.get("cost_rewards")):
            rewards.append(f"C: {row.get('cost_rewards')}")
        if not _blank(row.get("outcome_rewards")):
            rewards.append(f"E: {row.get('outcome_rewards')}")
        full_label = label if not rewards else label + "\n" + "\n".join(rewards)
        lines.append(f"{q(node_id)} [label={q(full_label)}, shape={q(shape)}];")
    for row in strategy_rows:
        if _blank(row.get("strategy_id")) or _blank(row.get("root_node_id")):
            continue
        sid = str(row.get("strategy_id")).strip(); name = str(row.get("strategy_name") or sid).strip(); root = str(row.get("root_node_id")).strip(); visual_id=f"strategy::{sid}"
        lines.append(f"{q(visual_id)} [label={q(name)}, shape=\"box\"];")
        lines.append(f"{q(visual_id)} -> {q(root)};")
    for row in branch_rows:
        if _blank(row.get("from_node")) or _blank(row.get("to_node")):
            continue
        origin=str(row.get("from_node")).strip(); child=str(row.get("to_node")).strip(); label=str(row.get("label") or "").strip(); probability=str(row.get("probability_parameter_id") or "").strip(); mode=str(row.get("probability_mode") or "direct").strip().lower()
        prob_label = f"1 - {probability}" if mode == "complement" else probability
        edge_label = label if not probability else f"{label} [{prob_label}]"
        lines.append(f"{q(origin)} -> {q(child)} [label={q(edge_label)}];")
    lines.append("}")
    return "\n".join(lines)
