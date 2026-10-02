"""Cohort Markov sensitivity helpers based on incremental net monetary benefit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from model.markov import CohortMarkovDefinition, run_cohort_markov
from model.schema import Parameter
from model.sensitivity import (
    OneWaySensitivitySpec,
    ThresholdAnalysisSpec,
    TwoWaySensitivitySpec,
    one_way_sensitivity,
    threshold_analysis,
    two_way_sensitivity,
)


@dataclass(frozen=True)
class MarkovTornadoResult:
    parameter_id: str
    label: str
    low_value: float
    high_value: float
    low_inmb: float
    high_inmb: float
    base_inmb: float

    @property
    def impact(self) -> float:
        return max(abs(self.low_inmb - self.base_inmb), abs(self.high_inmb - self.base_inmb))


def _inmb_evaluator(
    model: CohortMarkovDefinition,
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
        run = run_cohort_markov(
            model,
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


def one_way_markov_inmb(
    model: CohortMarkovDefinition,
    parameters: Sequence[Parameter],
    spec: OneWaySensitivitySpec,
    **context,
):
    return one_way_sensitivity(spec, _inmb_evaluator(model, parameters, **context))


def two_way_markov_inmb(
    model: CohortMarkovDefinition,
    parameters: Sequence[Parameter],
    spec: TwoWaySensitivitySpec,
    **context,
):
    return two_way_sensitivity(spec, _inmb_evaluator(model, parameters, **context))


def threshold_markov_inmb(
    model: CohortMarkovDefinition,
    parameters: Sequence[Parameter],
    spec: ThresholdAnalysisSpec,
    **context,
):
    return threshold_analysis(spec, _inmb_evaluator(model, parameters, **context))


def tornado_markov_inmb(
    model: CohortMarkovDefinition,
    parameters: Sequence[Parameter],
    **context,
) -> tuple[MarkovTornadoResult, ...]:
    evaluate = _inmb_evaluator(model, parameters, **context)
    base_inmb = evaluate({})
    rows: list[MarkovTornadoResult] = []
    for parameter in parameters:
        dsa = parameter.dsa
        if dsa is None or not dsa.enabled or dsa.lower is None or dsa.upper is None:
            continue
        rows.append(
            MarkovTornadoResult(
                parameter_id=parameter.id,
                label=parameter.label,
                low_value=dsa.lower,
                high_value=dsa.upper,
                low_inmb=evaluate({parameter.id: dsa.lower}),
                high_inmb=evaluate({parameter.id: dsa.upper}),
                base_inmb=base_inmb,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.impact, reverse=True))
