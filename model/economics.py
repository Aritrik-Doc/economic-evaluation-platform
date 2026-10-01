"""Core cost-effectiveness decision analysis.

The module is UI-independent. It supports:
- two-strategy pairwise analysis (kept for backwards compatibility)
- multi-strategy fully incremental analysis
- strong dominance
- extended dominance
- net monetary benefit at a selected threshold

All supported primary economic outcome measures are defined so that
"more is better" (QALYs, life-years gained, DALYs averted).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Literal, Sequence


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

FrontierStatus = Literal[
    "efficient",
    "strongly_dominated",
    "extendedly_dominated",
]


@dataclass(frozen=True)
class OutcomeMeasure:
    code: str
    label: str
    unit: str

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.label.strip() or not self.unit.strip():
            raise ValueError("Outcome measure code, label, and unit cannot be empty.")


OUTCOME_MEASURES: dict[str, OutcomeMeasure] = {
    "QALY": OutcomeMeasure("QALY", "QALYs gained", "QALY"),
    "LYG": OutcomeMeasure("LYG", "Life-years gained", "life-year"),
    "DALY_AVERTED": OutcomeMeasure(
        "DALY_AVERTED", "DALYs averted", "DALY averted"
    ),
}


@dataclass(frozen=True)
class Strategy:
    """A strategy with expected cost and expected economic outcome per patient."""

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
    """Pairwise results comparing an intervention against a comparator."""

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
        if self.incremental_nmb > 0:
            return self.intervention.name
        if self.incremental_nmb < 0:
            return self.comparator.name
        return "Tie"


@dataclass(frozen=True)
class IncrementalResult:
    """One strategy's position in a fully incremental analysis."""

    strategy: Strategy
    nmb: float
    status: FrontierStatus
    incremental_cost: float | None = None
    incremental_effect: float | None = None
    icer: float | None = None
    compared_with: str | None = None


@dataclass(frozen=True)
class MultiStrategyResult:
    """Fully incremental analysis for mutually exclusive strategies."""

    rows: tuple[IncrementalResult, ...]
    willingness_to_pay: float

    @property
    def efficient_frontier(self) -> tuple[IncrementalResult, ...]:
        return tuple(row for row in self.rows if row.status == "efficient")

    @property
    def preferred_by_nmb(self) -> tuple[str, ...]:
        best = max(row.nmb for row in self.rows)
        return tuple(row.strategy.name for row in self.rows if row.nmb == best)


def _validate_wtp(willingness_to_pay: float) -> None:
    if not isfinite(willingness_to_pay):
        raise ValueError("Willingness-to-pay threshold must be finite.")
    if willingness_to_pay < 0:
        raise ValueError("Willingness-to-pay threshold cannot be negative.")


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
    """Compare an intervention with a comparator."""

    _validate_wtp(willingness_to_pay)

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


def _is_strongly_dominated(candidate: Strategy, strategies: Sequence[Strategy]) -> bool:
    """Return True if another strategy is no more costly and no less effective."""
    for other in strategies:
        if other is candidate:
            continue
        weakly_better = other.cost <= candidate.cost and other.effect >= candidate.effect
        strictly_better = other.cost < candidate.cost or other.effect > candidate.effect
        if weakly_better and strictly_better:
            return True
    return False


def fully_incremental_analysis(
    strategies: Sequence[Strategy],
    willingness_to_pay: float,
) -> MultiStrategyResult:
    """Perform fully incremental cost-effectiveness analysis.

    Strategies are mutually exclusive and outcome effects must be oriented so
    that larger values are better.

    Steps:
    1. remove strongly dominated strategies;
    2. sort remaining strategies by increasing effect;
    3. remove extendedly dominated strategies using the lower convex frontier;
    4. calculate sequential ICERs along the efficient frontier.

    Exact duplicate strategies (same cost and effect) are rejected because they
    are economically indistinguishable and make a unique incremental frontier
    undefined.
    """
    _validate_wtp(willingness_to_pay)

    strategies = tuple(strategies)
    if len(strategies) < 2:
        raise ValueError("At least two strategies are required.")
    names = [s.name for s in strategies]
    if len(set(names)) != len(names):
        raise ValueError("Strategy names must be unique.")

    coordinates = [(s.cost, s.effect) for s in strategies]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(
            "Strategies with identical cost and effect should be represented once."
        )

    strongly_dominated = {
        s.name for s in strategies if _is_strongly_dominated(s, strategies)
    }

    candidates = sorted(
        (s for s in strategies if s.name not in strongly_dominated),
        key=lambda s: (s.effect, s.cost, s.name),
    )

    frontier: list[Strategy] = []
    extendedly_dominated: set[str] = set()

    for strategy in candidates:
        frontier.append(strategy)
        while len(frontier) >= 3:
            a, b, c = frontier[-3:]
            icer_ab = (b.cost - a.cost) / (b.effect - a.effect)
            icer_bc = (c.cost - b.cost) / (c.effect - b.effect)

            if icer_bc < icer_ab and not isclose(
                icer_bc, icer_ab, rel_tol=1e-12, abs_tol=1e-12
            ):
                extendedly_dominated.add(b.name)
                frontier.pop(-2)
            else:
                break

    rows: list[IncrementalResult] = []
    for strategy in sorted(strategies, key=lambda s: (s.effect, s.cost, s.name)):
        nmb = willingness_to_pay * strategy.effect - strategy.cost

        if strategy.name in strongly_dominated:
            rows.append(
                IncrementalResult(
                    strategy=strategy,
                    nmb=nmb,
                    status="strongly_dominated",
                )
            )
            continue

        if strategy.name in extendedly_dominated:
            rows.append(
                IncrementalResult(
                    strategy=strategy,
                    nmb=nmb,
                    status="extendedly_dominated",
                )
            )
            continue

        rows.append(
            IncrementalResult(
                strategy=strategy,
                nmb=nmb,
                status="efficient",
            )
        )

    efficient_map = {row.strategy.name: row for row in rows if row.status == "efficient"}
    efficient_rows: dict[str, IncrementalResult] = {}
    previous: Strategy | None = None

    for strategy in frontier:
        nmb = efficient_map[strategy.name].nmb
        if previous is None:
            efficient_rows[strategy.name] = IncrementalResult(
                strategy=strategy,
                nmb=nmb,
                status="efficient",
            )
        else:
            delta_cost = strategy.cost - previous.cost
            delta_effect = strategy.effect - previous.effect
            efficient_rows[strategy.name] = IncrementalResult(
                strategy=strategy,
                nmb=nmb,
                status="efficient",
                incremental_cost=delta_cost,
                incremental_effect=delta_effect,
                icer=delta_cost / delta_effect,
                compared_with=previous.name,
            )
        previous = strategy

    final_rows = tuple(efficient_rows.get(row.strategy.name, row) for row in rows)
    return MultiStrategyResult(final_rows, willingness_to_pay)
