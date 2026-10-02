"""Decision-tree model engine for deterministic health-economic evaluation.

The engine is deliberately model-agnostic with respect to jurisdiction and UI.
It consumes the v0.3 parameter schema, resolves branch probabilities and node
rewards from parameter ids, validates the tree, and returns expected cost and
health outcome totals by strategy.

All primary economic outcomes are assumed to be oriented so that larger values
mean greater health benefit (QALYs gained, life-years gained, DALYs averted).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Mapping, Sequence

from model.schema import Parameter


@dataclass(frozen=True)
class TreeBranch:
    """One probability-weighted branch leaving a chance node."""

    label: str
    probability_parameter_id: str
    child_node_id: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Branch label is mandatory.")
        if not self.probability_parameter_id.strip():
            raise ValueError("Branch probability parameter id is mandatory.")
        if not self.child_node_id.strip():
            raise ValueError("Branch child node id is mandatory.")


@dataclass(frozen=True)
class ChanceNode:
    """A chance node with mutually exclusive, collectively exhaustive branches."""

    id: str
    label: str
    branches: tuple[TreeBranch, ...]
    cost_parameter_ids: tuple[str, ...] = ()
    outcome_parameter_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.label.strip():
            raise ValueError("Chance-node id and label are mandatory.")
        if len(self.branches) < 2:
            raise ValueError("A chance node requires at least two branches.")


@dataclass(frozen=True)
class TerminalNode:
    """A terminal node where the decision-tree pathway ends."""

    id: str
    label: str
    cost_parameter_ids: tuple[str, ...] = ()
    outcome_parameter_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.label.strip():
            raise ValueError("Terminal-node id and label are mandatory.")


@dataclass(frozen=True)
class StrategyRoot:
    """Links one mutually exclusive strategy to its tree root."""

    strategy_id: str
    root_node_id: str

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.root_node_id.strip():
            raise ValueError("Strategy id and root node id are mandatory.")


@dataclass(frozen=True)
class DecisionTreeDefinition:
    """Complete structural definition of a decision-tree model."""

    strategy_roots: tuple[StrategyRoot, ...]
    chance_nodes: tuple[ChanceNode, ...]
    terminal_nodes: tuple[TerminalNode, ...]
    probability_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if len(self.strategy_roots) < 2:
            raise ValueError("A decision tree requires at least two strategies.")
        if not isfinite(self.probability_tolerance) or self.probability_tolerance <= 0:
            raise ValueError("Probability tolerance must be positive and finite.")


@dataclass(frozen=True)
class NodeExpectedValue:
    cost: float
    outcome: float


@dataclass(frozen=True)
class StrategyExpectedValue:
    strategy_id: str
    expected_cost: float
    expected_outcome: float


@dataclass(frozen=True)
class DecisionTreeRunResult:
    strategies: tuple[StrategyExpectedValue, ...]


class DecisionTreeValidationError(ValueError):
    """Raised when a decision-tree structure or resolved parameter set is invalid."""


class _ParameterResolver:
    def __init__(
        self,
        parameters: Sequence[Parameter],
        overrides: Mapping[str, float] | None = None,
        included_cost_bearers: Sequence[str] | None = None,
    ) -> None:
        self.parameters = {p.id: p for p in parameters}
        if len(self.parameters) != len(tuple(parameters)):
            raise DecisionTreeValidationError("Parameter ids must be unique.")
        self.overrides = dict(overrides or {})
        unknown_overrides = set(self.overrides) - set(self.parameters)
        if unknown_overrides:
            raise DecisionTreeValidationError(
                f"Unknown parameter override(s): {', '.join(sorted(unknown_overrides))}."
            )
        if any(not isfinite(value) for value in self.overrides.values()):
            raise DecisionTreeValidationError("Parameter overrides must be finite.")
        self.included_cost_bearers = (
            set(included_cost_bearers) if included_cost_bearers is not None else None
        )

    def parameter(self, parameter_id: str) -> Parameter:
        try:
            return self.parameters[parameter_id]
        except KeyError as exc:
            raise DecisionTreeValidationError(
                f"Tree references undefined parameter '{parameter_id}'."
            ) from exc

    def value(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        return self.overrides.get(parameter_id, parameter.value)

    def probability(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        value = self.value(parameter_id)
        if parameter.category == "cost":
            raise DecisionTreeValidationError(
                f"Probability parameter '{parameter_id}' cannot be a cost parameter."
            )
        if value < 0 or value > 1:
            raise DecisionTreeValidationError(
                f"Probability parameter '{parameter_id}' must lie between 0 and 1."
            )
        return value

    def cost(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        if parameter.category != "cost":
            raise DecisionTreeValidationError(
                f"Cost reward '{parameter_id}' must reference a cost parameter."
            )
        if self.included_cost_bearers is not None:
            if not set(parameter.cost_bearers).intersection(self.included_cost_bearers):
                return 0.0
        return self.value(parameter_id)

    def outcome(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        if parameter.category == "cost":
            raise DecisionTreeValidationError(
                f"Outcome reward '{parameter_id}' cannot reference a cost parameter."
            )
        return self.value(parameter_id)


def validate_decision_tree(
    tree: DecisionTreeDefinition,
    parameters: Sequence[Parameter],
    *,
    overrides: Mapping[str, float] | None = None,
) -> None:
    """Validate structure, parameter references, probabilities and acyclicity."""

    resolver = _ParameterResolver(parameters, overrides)
    chance = {node.id: node for node in tree.chance_nodes}
    terminal = {node.id: node for node in tree.terminal_nodes}

    all_ids = list(chance) + list(terminal)
    if len(all_ids) != len(set(all_ids)):
        raise DecisionTreeValidationError("Tree node ids must be globally unique.")

    root_strategy_ids = [root.strategy_id for root in tree.strategy_roots]
    if len(root_strategy_ids) != len(set(root_strategy_ids)):
        raise DecisionTreeValidationError("Each strategy may have only one tree root.")

    node_ids = set(all_ids)
    for root in tree.strategy_roots:
        if root.root_node_id not in node_ids:
            raise DecisionTreeValidationError(
                f"Strategy '{root.strategy_id}' references undefined root node "
                f"'{root.root_node_id}'."
            )

    for node in (*tree.chance_nodes, *tree.terminal_nodes):
        for parameter_id in node.cost_parameter_ids:
            resolver.cost(parameter_id)
        for parameter_id in node.outcome_parameter_ids:
            resolver.outcome(parameter_id)

    for node in tree.chance_nodes:
        labels = [branch.label for branch in node.branches]
        if len(labels) != len(set(labels)):
            raise DecisionTreeValidationError(
                f"Chance node '{node.id}' has duplicate branch labels."
            )

        total_probability = 0.0
        for branch in node.branches:
            if branch.child_node_id not in node_ids:
                raise DecisionTreeValidationError(
                    f"Branch '{branch.label}' from node '{node.id}' references "
                    f"undefined child node '{branch.child_node_id}'."
                )
            total_probability += resolver.probability(branch.probability_parameter_id)

        if not isclose(
            total_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=tree.probability_tolerance,
        ):
            raise DecisionTreeValidationError(
                f"Outgoing probabilities from chance node '{node.id}' sum to "
                f"{total_probability:.12g}, not 1."
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise DecisionTreeValidationError(
                f"Decision tree contains a cycle involving node '{node_id}'."
            )
        if node_id in visited:
            return
        visiting.add(node_id)
        node = chance.get(node_id)
        if node is not None:
            for branch in node.branches:
                visit(branch.child_node_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for root in tree.strategy_roots:
        visit(root.root_node_id)


def run_decision_tree(
    tree: DecisionTreeDefinition,
    parameters: Sequence[Parameter],
    *,
    overrides: Mapping[str, float] | None = None,
    included_cost_bearers: Sequence[str] | None = None,
) -> DecisionTreeRunResult:
    """Calculate expected cost and health outcome for every strategy.

    ``overrides`` enables deterministic, two-way and threshold sensitivity
    analyses without mutating the base parameter set.

    ``included_cost_bearers`` makes perspective computational. Cost parameters
    that do not share at least one included bearer are excluded. Parameters
    spanning more than one bearer should be split into separate components when
    different perspectives need to include only part of the value.
    """

    validate_decision_tree(tree, parameters, overrides=overrides)
    resolver = _ParameterResolver(
        parameters,
        overrides,
        included_cost_bearers=included_cost_bearers,
    )
    chance = {node.id: node for node in tree.chance_nodes}
    terminal = {node.id: node for node in tree.terminal_nodes}
    memo: dict[str, NodeExpectedValue] = {}

    def node_rewards(node: ChanceNode | TerminalNode) -> NodeExpectedValue:
        return NodeExpectedValue(
            cost=sum(resolver.cost(pid) for pid in node.cost_parameter_ids),
            outcome=sum(resolver.outcome(pid) for pid in node.outcome_parameter_ids),
        )

    def expected_value(node_id: str) -> NodeExpectedValue:
        if node_id in memo:
            return memo[node_id]

        if node_id in terminal:
            result = node_rewards(terminal[node_id])
            memo[node_id] = result
            return result

        node = chance[node_id]
        own = node_rewards(node)
        downstream_cost = 0.0
        downstream_outcome = 0.0
        for branch in node.branches:
            probability = resolver.probability(branch.probability_parameter_id)
            child = expected_value(branch.child_node_id)
            downstream_cost += probability * child.cost
            downstream_outcome += probability * child.outcome

        result = NodeExpectedValue(
            cost=own.cost + downstream_cost,
            outcome=own.outcome + downstream_outcome,
        )
        memo[node_id] = result
        return result

    strategy_results = []
    for root in tree.strategy_roots:
        value = expected_value(root.root_node_id)
        strategy_results.append(
            StrategyExpectedValue(
                strategy_id=root.strategy_id,
                expected_cost=value.cost,
                expected_outcome=value.outcome,
            )
        )

    return DecisionTreeRunResult(tuple(strategy_results))
