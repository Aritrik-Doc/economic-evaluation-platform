"""Pure helpers for compiling the hybrid decision-tree builder tables.

The Streamlit UI edits simple row-oriented tables. This module converts those
rows into the auditable v0.3 schema and deterministic v0.4 decision-tree engine,
so parsing/validation is testable without Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any, Mapping, Sequence

from model.decision_tree import (
    ChanceNode,
    DecisionTreeDefinition,
    StrategyRoot,
    TerminalNode,
    TreeBranch,
)
from model.schema import AssumptionSpec, EvidenceSource, Parameter, UncertaintySpec


class BuilderValidationError(ValueError):
    """Raised when an editable builder table cannot be compiled safely."""


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
    """Parse comma- or semicolon-separated ids, removing empty items."""
    if _blank(value):
        return ()
    text = str(value).replace(";", ",")
    return tuple(item.strip() for item in text.split(",") if item.strip())


def compile_parameter_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Parameter, ...]:
    parameters: list[Parameter] = []

    for index, row in enumerate(rows, start=1):
        if _blank(row.get("id")):
            continue

        parameter_id = _text(row.get("id"), f"Parameter row {index}: id")
        category = _text(row.get("category"), f"Parameter '{parameter_id}': category")
        source_type = _text(row.get("source_type"), f"Parameter '{parameter_id}': source_type")
        uncertainty_kind = _text(row.get("uncertainty_kind"), f"Parameter '{parameter_id}': uncertainty_kind")

        source = EvidenceSource(
            citation=_text(row.get("source_citation"), f"Parameter '{parameter_id}': source_citation"),
            source_type=source_type,
            url=_optional_text(row.get("source_url")),
            publication_year=_optional_int(row.get("publication_year")),
            details=_optional_text(row.get("source_details")) or "",
        )
        assumption = AssumptionSpec(
            statement=_text(row.get("assumption"), f"Parameter '{parameter_id}': assumption"),
            rationale=_text(row.get("assumption_rationale"), f"Parameter '{parameter_id}': assumption_rationale"),
        )
        uncertainty = UncertaintySpec(
            kind=uncertainty_kind,
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


def compile_structure_rows(
    strategy_rows: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
) -> tuple[DecisionTreeDefinition, Mapping[str, str]]:
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
        costs = parse_id_list(row.get("cost_parameter_ids"))
        outcomes = parse_id_list(row.get("outcome_parameter_ids"))

        if node_type == "chance":
            branches = []
            for b_index, branch in enumerate(outgoing.get(node_id, ()), start=1):
                branches.append(
                    TreeBranch(
                        label=_text(branch.get("label"), f"Branch {b_index} from '{node_id}': label"),
                        probability_parameter_id=_text(branch.get("probability_parameter_id"), f"Branch '{node_id}': probability_parameter_id"),
                        child_node_id=_text(branch.get("to_node"), f"Branch '{node_id}': to_node"),
                    )
                )
            try:
                chance_nodes.append(
                    ChanceNode(
                        id=node_id,
                        label=label,
                        branches=tuple(branches),
                        cost_parameter_ids=costs,
                        outcome_parameter_ids=outcomes,
                    )
                )
            except ValueError as exc:
                raise BuilderValidationError(str(exc)) from exc
        elif node_type == "terminal":
            if outgoing.get(node_id):
                raise BuilderValidationError(f"Terminal node '{node_id}' cannot have outgoing branches.")
            terminal_nodes.append(
                TerminalNode(
                    id=node_id,
                    label=label,
                    cost_parameter_ids=costs,
                    outcome_parameter_ids=outcomes,
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
        tree = DecisionTreeDefinition(
            strategy_roots=tuple(roots),
            chance_nodes=tuple(chance_nodes),
            terminal_nodes=tuple(terminal_nodes),
        )
    except ValueError as exc:
        raise BuilderValidationError(str(exc)) from exc
    return tree, strategy_names


def compile_builder_tables(
    parameter_rows: Sequence[Mapping[str, Any]],
    strategy_rows: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
) -> CompiledDecisionTree:
    parameters = compile_parameter_rows(parameter_rows)
    tree, strategy_names = compile_structure_rows(strategy_rows, node_rows, branch_rows)
    return CompiledDecisionTree(tree=tree, parameters=parameters, strategy_names=strategy_names)


def structure_to_dot(
    strategy_rows: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Render even an incomplete structure as Graphviz DOT for live preview."""
    import json

    def q(value: str) -> str:
        return json.dumps(str(value))

    lines = [
        "digraph DecisionTree {",
        'rankdir="LR";',
        'graph [pad="0.2", nodesep="0.35", ranksep="0.55"];',
        'node [fontname="Arial"];',
    ]

    for row in node_rows:
        if _blank(row.get("id")):
            continue
        node_id = str(row.get("id")).strip()
        label = str(row.get("label") or node_id).strip()
        node_type = str(row.get("type") or "").strip().lower()
        shape = "circle" if node_type == "chance" else "box"
        if node_type == "terminal":
            shape = "doublecircle"
        lines.append(f"{q(node_id)} [label={q(label)}, shape={q(shape)}];")

    for row in strategy_rows:
        if _blank(row.get("strategy_id")) or _blank(row.get("root_node_id")):
            continue
        sid = str(row.get("strategy_id")).strip()
        name = str(row.get("strategy_name") or sid).strip()
        root = str(row.get("root_node_id")).strip()
        visual_id = f"strategy::{sid}"
        lines.append(f"{q(visual_id)} [label={q(name)}, shape=\"box\"];")
        lines.append(f"{q(visual_id)} -> {q(root)};")

    for row in branch_rows:
        if _blank(row.get("from_node")) or _blank(row.get("to_node")):
            continue
        origin = str(row.get("from_node")).strip()
        child = str(row.get("to_node")).strip()
        label = str(row.get("label") or "").strip()
        probability = str(row.get("probability_parameter_id") or "").strip()
        edge_label = label if not probability else f"{label} [{probability}]"
        lines.append(f"{q(origin)} -> {q(child)} [label={q(edge_label)}];")

    lines.append("}")
    return "\n".join(lines)
