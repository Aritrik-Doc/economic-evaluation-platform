"""Deterministic decision-tree engine for health-economic evaluation.

Version 0.4 supports parameter-linked probabilities and rewards, explicit reward
timing with annual discounting, computational cost perspective, and parameter
overrides for sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Literal, Mapping, Sequence

from model.schema import Parameter


ProbabilityMode = Literal["direct", "complement"]


@dataclass(frozen=True)
class TimedReward:
    """A cost or outcome parameter accrued at an absolute time from model start."""

    parameter_id: str
    time_years: float = 0.0

    def __post_init__(self) -> None:
        if not self.parameter_id.strip():
            raise ValueError("Reward parameter id is mandatory.")
        if not isfinite(self.time_years) or self.time_years < 0:
            raise ValueError("Reward time must be a finite non-negative number of years.")


@dataclass(frozen=True)
class TreeBranch:
    """One probability-weighted branch leaving a chance node."""

    label: str
    probability_parameter_id: str
    child_node_id: str
    probability_mode: ProbabilityMode = "direct"

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Branch label is mandatory.")
        if not self.probability_parameter_id.strip():
            raise ValueError("Branch probability parameter id is mandatory.")
        if not self.child_node_id.strip():
            raise ValueError("Branch child node id is mandatory.")
        if self.probability_mode not in {"direct", "complement"}:
            raise ValueError("Branch probability mode must be 'direct' or 'complement'.")


@dataclass(frozen=True)
class ChanceNode:
    id: str
    label: str
    branches: tuple[TreeBranch, ...]
    cost_parameter_ids: tuple[str, ...] = ()
    outcome_parameter_ids: tuple[str, ...] = ()
    cost_rewards: tuple[TimedReward, ...] = ()
    outcome_rewards: tuple[TimedReward, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.label.strip():
            raise ValueError("Chance-node id and label are mandatory.")
        if len(self.branches) < 2:
            raise ValueError("A chance node requires at least two branches.")


@dataclass(frozen=True)
class TerminalNode:
    id: str
    label: str
    cost_parameter_ids: tuple[str, ...] = ()
    outcome_parameter_ids: tuple[str, ...] = ()
    cost_rewards: tuple[TimedReward, ...] = ()
    outcome_rewards: tuple[TimedReward, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.label.strip():
            raise ValueError("Terminal-node id and label are mandatory.")


@dataclass(frozen=True)
class StrategyRoot:
    strategy_id: str
    root_node_id: str

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.root_node_id.strip():
            raise ValueError("Strategy id and root node id are mandatory.")


@dataclass(frozen=True)
class DecisionTreeDefinition:
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
    pass


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
        unknown = set(self.overrides) - set(self.parameters)
        if unknown:
            raise DecisionTreeValidationError(
                f"Unknown parameter override(s): {', '.join(sorted(unknown))}."
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

    def probability(self, branch: TreeBranch) -> float:
        parameter = self.parameter(branch.probability_parameter_id)
        value = self.value(branch.probability_parameter_id)
        if parameter.category == "cost":
            raise DecisionTreeValidationError(
                f"Probability parameter '{parameter.id}' cannot be a cost parameter."
            )
        if value < 0 or value > 1:
            raise DecisionTreeValidationError(
                f"Probability parameter '{parameter.id}' must lie between 0 and 1."
            )
        return value if branch.probability_mode == "direct" else 1.0 - value

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


def _validate_discount_rate(rate: float, label: str) -> None:
    if not isfinite(rate) or rate < 0 or rate >= 1:
        raise DecisionTreeValidationError(
            f"{label} discount rate must be a finite proportion in [0, 1)."
        )


def discount_value(value: float, annual_rate: float, time_years: float) -> float:
    """Present value under annual discrete discounting."""
    _validate_discount_rate(annual_rate, "Annual")
    if not isfinite(time_years) or time_years < 0:
        raise DecisionTreeValidationError("Reward time must be finite and non-negative.")
    return value / ((1.0 + annual_rate) ** time_years)


def _cost_rewards(node: ChanceNode | TerminalNode) -> tuple[TimedReward, ...]:
    legacy = tuple(TimedReward(pid, 0.0) for pid in node.cost_parameter_ids)
    return legacy + node.cost_rewards


def _outcome_rewards(node: ChanceNode | TerminalNode) -> tuple[TimedReward, ...]:
    legacy = tuple(TimedReward(pid, 0.0) for pid in node.outcome_parameter_ids)
    return legacy + node.outcome_rewards


def validate_decision_tree(
    tree: DecisionTreeDefinition,
    parameters: Sequence[Parameter],
    *,
    overrides: Mapping[str, float] | None = None,
    cost_discount_rate: float = 0.0,
    outcome_discount_rate: float = 0.0,
) -> None:
    """Validate structure, references, probabilities, timing and acyclicity."""

    _validate_discount_rate(cost_discount_rate, "Cost")
    _validate_discount_rate(outcome_discount_rate, "Outcome")
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
                f"Strategy '{root.strategy_id}' references undefined root node '{root.root_node_id}'."
            )

    for node in (*tree.chance_nodes, *tree.terminal_nodes):
        for reward in _cost_rewards(node):
            resolver.cost(reward.parameter_id)
        for reward in _outcome_rewards(node):
            resolver.outcome(reward.parameter_id)

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
                    f"Branch '{branch.label}' from node '{node.id}' references undefined child node '{branch.child_node_id}'."
                )
            total_probability += resolver.probability(branch)
        if not isclose(total_probability, 1.0, rel_tol=0.0, abs_tol=tree.probability_tolerance):
            raise DecisionTreeValidationError(
                f"Outgoing probabilities from chance node '{node.id}' sum to {total_probability:.12g}, not 1."
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
    cost_discount_rate: float = 0.0,
    outcome_discount_rate: float = 0.0,
) -> DecisionTreeRunResult:
    """Calculate discounted expected cost and health outcome for each strategy.

    Reward times are absolute years from model start. Annual discrete discounting
    is applied separately to costs and health outcomes. Parameter overrides allow
    DSA/two-way/threshold workflows without mutating the base model.
    """

    validate_decision_tree(
        tree,
        parameters,
        overrides=overrides,
        cost_discount_rate=cost_discount_rate,
        outcome_discount_rate=outcome_discount_rate,
    )
    resolver = _ParameterResolver(parameters, overrides, included_cost_bearers)
    chance = {node.id: node for node in tree.chance_nodes}
    terminal = {node.id: node for node in tree.terminal_nodes}
    memo: dict[str, NodeExpectedValue] = {}

    def node_rewards(node: ChanceNode | TerminalNode) -> NodeExpectedValue:
        cost = sum(
            discount_value(resolver.cost(reward.parameter_id), cost_discount_rate, reward.time_years)
            for reward in _cost_rewards(node)
        )
        outcome = sum(
            discount_value(resolver.outcome(reward.parameter_id), outcome_discount_rate, reward.time_years)
            for reward in _outcome_rewards(node)
        )
        return NodeExpectedValue(cost=cost, outcome=outcome)

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
            probability = resolver.probability(branch)
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
            StrategyExpectedValue(root.strategy_id, value.cost, value.outcome)
        )
    return DecisionTreeRunResult(tuple(strategy_results))
