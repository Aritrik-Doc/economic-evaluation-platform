"""Core two-strategy cost-effectiveness calculations for version 0.1.

The module is intentionally independent of Streamlit so that the economic
logic can be tested and reused by other interfaces later.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


ComparisonStatus = Literal[
    "dominant",
    "dominated",
    "more_costly_more_effective",
    "less_costly_less_effective",
    "equal_effect_more_costly",
    "equal_effect_less_costly",
    "equal_cost_more_effective",
    "equal_cost_less_effective",
    "no_difference",
]


@dataclass(frozen=True)
class Strategy:
    """A strategy with expected cost and expected health effect per patient."""

    name: str
    cost: float
    effect: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Strategy name cannot be empty.")
        if not isfinite(self.cost):
            raise ValueError("Cost must be a finite number.")
        if not isfinite(self.effect):
            raise ValueError("Effect must be a finite number.")


@dataclass(frozen=True)
class EconomicEvaluationResult:
    """Results comparing an intervention against a comparator."""

    comparator: Strategy
    intervention: Strategy
    willingness_to_pay: float
    incremental_cost: float
    incremental_effect: float
    icer: float | None
    comparator_nmb: float
    intervention_nmb: float
    incremental_nmb: float
    status: ComparisonStatus

    @property
    def preferred_by_nmb(self) -> str:
        """Return the strategy with the larger net monetary benefit."""
        if self.incremental_nmb > 0:
            return self.intervention.name
        if self.incremental_nmb < 0:
            return self.comparator.name
        return "Tie"


def _comparison_status(delta_cost: float, delta_effect: float) -> ComparisonStatus:
    if delta_cost == 0 and delta_effect == 0:
        return "no_difference"
    if delta_effect == 0:
        return "equal_effect_more_costly" if delta_cost > 0 else "equal_effect_less_costly"
    if delta_cost == 0:
        return "equal_cost_more_effective" if delta_effect > 0 else "equal_cost_less_effective"
    if delta_cost < 0 and delta_effect > 0:
        return "dominant"
    if delta_cost > 0 and delta_effect < 0:
        return "dominated"
    if delta_cost > 0 and delta_effect > 0:
        return "more_costly_more_effective"
    return "less_costly_less_effective"


def evaluate_two_strategies(
    comparator: Strategy,
    intervention: Strategy,
    willingness_to_pay: float,
) -> EconomicEvaluationResult:
    """Compare an intervention with a comparator.

    Definitions
    -----------
    Incremental cost = intervention cost - comparator cost
    Incremental effect = intervention effect - comparator effect
    ICER = incremental cost / incremental effect, when incremental effect != 0
    NMB = willingness_to_pay * effect - cost
    INMB = intervention NMB - comparator NMB

    The ICER is returned as ``None`` when incremental effect is exactly zero,
    because division by zero has no meaningful finite interpretation.
    Dominance is represented separately in ``status`` rather than by attempting
    to interpret the sign of the ICER.
    """

    if not isfinite(willingness_to_pay):
        raise ValueError("Willingness-to-pay threshold must be finite.")
    if willingness_to_pay < 0:
        raise ValueError("Willingness-to-pay threshold cannot be negative.")

    delta_cost = intervention.cost - comparator.cost
    delta_effect = intervention.effect - comparator.effect

    comparator_nmb = willingness_to_pay * comparator.effect - comparator.cost
    intervention_nmb = willingness_to_pay * intervention.effect - intervention.cost
    incremental_nmb = intervention_nmb - comparator_nmb

    icer = None if delta_effect == 0 else delta_cost / delta_effect
    status = _comparison_status(delta_cost, delta_effect)

    return EconomicEvaluationResult(
        comparator=comparator,
        intervention=intervention,
        willingness_to_pay=willingness_to_pay,
        incremental_cost=delta_cost,
        incremental_effect=delta_effect,
        icer=icer,
        comparator_nmb=comparator_nmb,
        intervention_nmb=intervention_nmb,
        incremental_nmb=incremental_nmb,
        status=status,
    )
