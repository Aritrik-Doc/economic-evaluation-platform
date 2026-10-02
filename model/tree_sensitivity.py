"""Decision-tree sensitivity helpers based on incremental net monetary benefit."""

from __future__ import annotations

from typing import Mapping, Sequence

from model.decision_tree import DecisionTreeDefinition, run_decision_tree
from model.schema import Parameter
from model.sensitivity import (
    OneWaySensitivitySpec,
    ThresholdAnalysisSpec,
    TwoWaySensitivitySpec,
    one_way_sensitivity,
    threshold_analysis,
    two_way_sensitivity,
)


def _inmb_evaluator(
    tree: DecisionTreeDefinition,
    parameters: Sequence[Parameter],
    *,
    intervention_id: str,
    comparator_id: str,
    willingness_to_pay: float,
    included_cost_bearers: Sequence[str] | None = None,
    cost_discount_rate: float = 0.0,
    outcome_discount_rate: float = 0.0,
):
    if intervention_id == comparator_id:
        raise ValueError("Intervention and comparator must be different strategies.")

    def evaluate(overrides: Mapping[str, float]) -> float:
        run = run_decision_tree(
            tree,
            parameters,
            overrides=overrides,
            included_cost_bearers=included_cost_bearers,
            cost_discount_rate=cost_discount_rate,
            outcome_discount_rate=outcome_discount_rate,
        )
        rows = {row.strategy_id: row for row in run.strategies}
        if intervention_id not in rows or comparator_id not in rows:
            raise ValueError("Sensitivity comparison references an unknown strategy.")
        intervention = rows[intervention_id]
        comparator = rows[comparator_id]
        delta_cost = intervention.expected_cost - comparator.expected_cost
        delta_effect = intervention.expected_outcome - comparator.expected_outcome
        return willingness_to_pay * delta_effect - delta_cost

    return evaluate


def one_way_tree_inmb(
    tree: DecisionTreeDefinition,
    parameters: Sequence[Parameter],
    spec: OneWaySensitivitySpec,
    **context,
):
    return one_way_sensitivity(spec, _inmb_evaluator(tree, parameters, **context))


def two_way_tree_inmb(
    tree: DecisionTreeDefinition,
    parameters: Sequence[Parameter],
    spec: TwoWaySensitivitySpec,
    **context,
):
    return two_way_sensitivity(spec, _inmb_evaluator(tree, parameters, **context))


def threshold_tree_inmb(
    tree: DecisionTreeDefinition,
    parameters: Sequence[Parameter],
    spec: ThresholdAnalysisSpec,
    **context,
):
    return threshold_analysis(spec, _inmb_evaluator(tree, parameters, **context))
