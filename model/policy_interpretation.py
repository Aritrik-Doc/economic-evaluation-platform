"""Deterministic, auditable interpretation of economic and implementation results.

The interpretation layer translates calculated outputs into plain language. It
never changes the underlying results and does not issue adoption/rejection
recommendations. Every statement carries a ``basis`` tuple identifying the
analysis output that supports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Mapping, Sequence

from model.budget_impact import BudgetImpactRunResult
from model.economics import MultiStrategyResult
from model.resource_capacity import ResourceCapacityRunResult


InterpretationDomain = Literal["value", "affordability", "feasibility"]
InterpretationKind = Literal[
    "headline",
    "magnitude",
    "pattern",
    "uncertainty",
    "driver",
    "caveat",
]


@dataclass(frozen=True)
class InterpretationStatement:
    kind: InterpretationKind
    title: str
    text: str
    basis: tuple[str, ...]


@dataclass(frozen=True)
class PolicyInterpretation:
    domain: InterpretationDomain
    statements: tuple[InterpretationStatement, ...]
    caveats: tuple[str, ...] = ()


def _money(value: float, symbol: str) -> str:
    sign = "−" if value < 0 else ""
    return f"{sign}{symbol}{abs(value):,.0f}"


def _number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def _join_names(names: Sequence[str]) -> str:
    names = tuple(names)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def interpret_cost_effectiveness(
    result: MultiStrategyResult,
    *,
    outcome_label: str,
    currency_symbol: str,
    psa_probability_by_strategy: Mapping[str, float] | None = None,
    dsa_drivers: Sequence[tuple[str, float]] | None = None,
) -> PolicyInterpretation:
    """Translate a fully incremental CEA result into non-prescriptive statements.

    ``psa_probability_by_strategy`` should contain the probability that each
    strategy has the highest NMB at the same threshold used in ``result``.
    ``dsa_drivers`` may contain ``(parameter label, absolute INMB swing)`` pairs.
    """

    threshold = result.willingness_to_pay
    preferred = result.preferred_by_nmb
    statements: list[InterpretationStatement] = []

    if len(preferred) == 1:
        headline = (
            f"At the stated decision threshold of {_money(threshold, currency_symbol)} per {outcome_label}, "
            f"{preferred[0]} has the highest net monetary benefit in the base-case analysis."
        )
    else:
        headline = (
            f"At the stated decision threshold of {_money(threshold, currency_symbol)} per {outcome_label}, "
            f"{_join_names(preferred)} have equal highest net monetary benefit in the base-case analysis."
        )
    statements.append(
        InterpretationStatement("headline", "Value-for-money finding", headline, ("Base-case fully incremental CEA", "Net monetary benefit"))
    )

    frontier = result.efficient_frontier
    chosen = next((row for row in frontier if row.strategy.name in preferred), None)
    if chosen is not None and chosen.compared_with is not None:
        magnitude = (
            f"On the efficient frontier, {chosen.strategy.name} is compared with {chosen.compared_with}. "
            f"The incremental cost is {_money(chosen.incremental_cost or 0.0, currency_symbol)} and the incremental "
            f"effect is {_number(chosen.incremental_effect or 0.0, 4)} {outcome_label}."
        )
        if chosen.icer is not None and isfinite(chosen.icer):
            magnitude += f" The corresponding ICER is {_money(chosen.icer, currency_symbol)} per {outcome_label}."
        statements.append(
            InterpretationStatement("magnitude", "Incremental magnitude", magnitude, ("Efficient frontier", "Sequential ICER analysis"))
        )

    strong = [row.strategy.name for row in result.rows if row.status == "strongly_dominated"]
    extended = [row.strategy.name for row in result.rows if row.status == "extendedly_dominated"]
    if strong or extended:
        parts: list[str] = []
        if strong:
            parts.append(f"{_join_names(strong)} {'is' if len(strong) == 1 else 'are'} strongly dominated")
        if extended:
            parts.append(f"{_join_names(extended)} {'is' if len(extended) == 1 else 'are'} extendedly dominated")
        statements.append(
            InterpretationStatement(
                "pattern",
                "Position on the efficiency frontier",
                "; ".join(parts) + ". These classifications describe the incremental efficiency calculation and are not standalone policy recommendations.",
                ("Fully incremental CEA",),
            )
        )

    if psa_probability_by_strategy:
        missing = [name for name in preferred if name not in psa_probability_by_strategy]
        if not missing:
            probabilities = [float(psa_probability_by_strategy[name]) for name in preferred]
            if all(0.0 <= value <= 1.0 and isfinite(value) for value in probabilities):
                if len(preferred) == 1:
                    text = (
                        f"At the stated threshold, {preferred[0]} has the highest NMB in "
                        f"{probabilities[0] * 100:.1f}% of PSA simulations. The remaining simulations favour other strategies, "
                        "so this percentage describes decision uncertainty rather than certainty of benefit."
                    )
                else:
                    text = (
                        "The base case contains an NMB tie. In the PSA, the corresponding probabilities of having the highest NMB are "
                        + ", ".join(f"{name}: {psa_probability_by_strategy[name] * 100:.1f}%" for name in preferred)
                        + "."
                    )
                statements.append(
                    InterpretationStatement("uncertainty", "Probabilistic decision uncertainty", text, ("PSA", "CEAC / highest NMB at selected threshold"))
                )

    if dsa_drivers:
        valid = [(str(label), abs(float(swing))) for label, swing in dsa_drivers if isfinite(float(swing))]
        valid.sort(key=lambda item: item[1], reverse=True)
        if valid:
            top = valid[:3]
            text = "The largest configured deterministic changes in decision value are associated with " + ", ".join(
                f"{label} (INMB swing {_money(swing, currency_symbol)})" for label, swing in top
            ) + ". This ranking reflects the ranges entered by the modeller, not the intrinsic scientific importance of a parameter."
            statements.append(
                InterpretationStatement("driver", "Deterministic drivers", text, ("One-way DSA / tornado analysis",))
            )

    caveats = (
        "The result is conditional on the selected threshold, perspective, time horizon, model structure and evidence inputs.",
        "Cost-effectiveness does not establish affordability or implementation feasibility; those questions require the BIA and capacity analyses.",
    )
    return PolicyInterpretation("value", tuple(statements), caveats)


def interpret_budget_impact(
    result: BudgetImpactRunResult,
    *,
    currency_symbol: str,
    category_labels: Mapping[str, str] | None = None,
) -> PolicyInterpretation:
    """Translate annual BIA results into factual affordability statements."""

    if not result.years:
        raise ValueError("Budget-impact interpretation requires at least one model year.")
    labels = dict(category_labels or {})
    statements: list[InterpretationStatement] = []
    years = result.years
    cumulative = result.cumulative_budget_impact
    horizon = len(years)

    if cumulative > 1e-9:
        direction = f"an additional cumulative budget requirement of {_money(cumulative, currency_symbol)}"
    elif cumulative < -1e-9:
        direction = f"cumulative budget savings of {_money(abs(cumulative), currency_symbol)}"
    else:
        direction = "approximately no cumulative net budget change"
    statements.append(
        InterpretationStatement(
            "headline",
            "Affordability finding",
            f"Across the {horizon}-year budget horizon, the future treatment mix produces {direction} compared with the current mix.",
            ("Base-case Budget Impact Analysis", "Cumulative budget impact"),
        )
    )

    positive = [row for row in years if row.net_budget_impact > 1e-9]
    negative = [row for row in years if row.net_budget_impact < -1e-9]
    if positive and not negative:
        pattern = "The future scenario increases expenditure in every modelled year."
    elif negative and not positive:
        pattern = "The future scenario reduces expenditure in every modelled year."
    elif positive and negative:
        pattern = "The direction of budget impact changes across the modelled years, with both additional-spend and saving periods."
    else:
        pattern = "Annual expenditure is approximately unchanged in every modelled year."

    if positive:
        peak = max(positive, key=lambda row: row.net_budget_impact)
        pattern += f" The largest annual additional requirement is {_money(peak.net_budget_impact, currency_symbol)} in Year {peak.year}."
    if negative:
        saving = min(negative, key=lambda row: row.net_budget_impact)
        pattern += f" The largest annual saving is {_money(abs(saving.net_budget_impact), currency_symbol)} in Year {saving.year}."
    statements.append(
        InterpretationStatement("pattern", "Annual budget pattern", pattern, ("Annual net budget impact",))
    )

    pmpm_rows = [row for row in years if row.pmpm_budget_impact is not None]
    if pmpm_rows:
        first = pmpm_rows[0]
        last = pmpm_rows[-1]
        text = f"PMPM budget impact is {_money(first.pmpm_budget_impact or 0.0, currency_symbol)} in Year {first.year}"
        if last.year != first.year:
            text += f" and {_money(last.pmpm_budget_impact or 0.0, currency_symbol)} in Year {last.year}"
        text += "."
        statements.append(
            InterpretationStatement("magnitude", "Per-member budget effect", text, ("Covered lives", "PMPM budget impact"))
        )

    by_category: dict[str, dict[str, float]] = {}
    for row in result.category_rows:
        entry = by_category.setdefault(row.category, {"current": 0.0, "future": 0.0})
        entry[row.scenario] += row.total_cost
    category_deltas = [
        (category, values["future"] - values["current"])
        for category, values in by_category.items()
    ]
    category_deltas = [(category, delta) for category, delta in category_deltas if abs(delta) > 1e-9]
    if category_deltas:
        category, delta = max(category_deltas, key=lambda item: abs(item[1]))
        label = labels.get(category, category.replace("_", " ").title())
        text = (
            f"The largest absolute cumulative cost-category change is {label}: "
            f"{_money(delta, currency_symbol)} versus the current mix."
        )
        statements.append(
            InterpretationStatement("driver", "Largest cost-category change", text, ("BIA cost-category decomposition",))
        )

    caveats = (
        "Budget impact describes financial consequences for the specified payer and horizon; it does not by itself determine whether the change is affordable.",
        "Affordability depends on the decision maker's available budget, competing commitments, population and uptake assumptions, and any implementation constraints.",
    )
    return PolicyInterpretation("affordability", tuple(statements), caveats)


def interpret_capacity(
    result: ResourceCapacityRunResult,
) -> PolicyInterpretation:
    """Translate resource/capacity outputs into non-prescriptive feasibility statements."""

    if not result.comparison_rows:
        raise ValueError("Capacity interpretation requires at least one resource comparison row.")
    statements: list[InterpretationStatement] = []
    future_shortfalls = list(result.future_shortfall_rows)

    if not future_shortfalls:
        headline = (
            "No modelled resource exceeds the capacity available to this population under the future treatment mix over the planning horizon."
        )
    else:
        resources = sorted({row.resource_name for row in future_shortfalls})
        earliest = min(row.year for row in future_shortfalls)
        headline = (
            f"The future treatment mix exceeds available capacity for {len(resources)} modelled resource"
            f"{'s' if len(resources) != 1 else ''}; the first modelled shortfall occurs in Year {earliest}."
        )
    statements.append(
        InterpretationStatement("headline", "Implementation feasibility", headline, ("Future resource demand", "Available capacity"))
    )

    if future_shortfalls:
        largest = max(future_shortfalls, key=lambda row: row.future_shortfall)
        statements.append(
            InterpretationStatement(
                "magnitude",
                "Largest modelled shortfall",
                f"The largest absolute shortfall is {_number(largest.future_shortfall)} {largest.unit} of {largest.resource_name} in Year {largest.year}. "
                f"Future demand is {_number(largest.future_required_units)} {largest.unit} against {_number(largest.available_capacity)} {largest.unit} available capacity.",
                ("Resource demand", "Residual available capacity", "Shortfall"),
            )
        )

    finite_utilisation = [
        row for row in result.comparison_rows
        if row.future_utilization_ratio is not None and isfinite(row.future_utilization_ratio)
    ]
    if finite_utilisation:
        peak = max(finite_utilisation, key=lambda row: row.future_utilization_ratio or 0.0)
        statements.append(
            InterpretationStatement(
                "pattern",
                "Peak capacity utilisation",
                f"The highest modelled future utilisation is {(peak.future_utilization_ratio or 0.0) * 100:.1f}% for {peak.resource_name} in Year {peak.year}.",
                ("Future utilisation ratio",),
            )
        )

    changed = [row for row in result.comparison_rows if abs(row.net_change_units) > 1e-9]
    if changed:
        largest_change = max(changed, key=lambda row: abs(row.net_change_units))
        direction = "increase" if largest_change.net_change_units > 0 else "decrease"
        statements.append(
            InterpretationStatement(
                "driver",
                "Largest change in resource demand",
                f"The largest absolute change between current and future treatment mixes is a {direction} of {_number(abs(largest_change.net_change_units))} {largest_change.unit} of {largest_change.resource_name} in Year {largest_change.year}.",
                ("Current-versus-future resource comparison",),
            )
        )

    zero_capacity = [
        row for row in result.comparison_rows
        if row.available_capacity <= 1e-12 and row.future_required_units > 1e-12
    ]
    if zero_capacity:
        labels = sorted({f"{row.resource_name} (Year {row.year})" for row in zero_capacity})
        statements.append(
            InterpretationStatement(
                "caveat",
                "No residual capacity entered",
                "Future demand is positive while residual capacity is zero for " + ", ".join(labels) + ". Utilisation percentages are not meaningful for these rows; the shortfall is reported directly in natural units.",
                ("Capacity inputs",),
            )
        )

    caveats = (
        "Capacity findings are conditional on the resource definitions, natural-unit requirements, capacity already committed elsewhere, and the selected population-demand basis.",
        "A modelled shortfall identifies an implementation constraint; it does not prescribe whether capacity should be expanded, demand reduced, treatment phased, or resources reallocated.",
    )
    return PolicyInterpretation("feasibility", tuple(statements), caveats)


def combined_policy_summary(*interpretations: PolicyInterpretation) -> tuple[InterpretationStatement, ...]:
    """Return headline statements in value → affordability → feasibility order."""
    order = {"value": 0, "affordability": 1, "feasibility": 2}
    output: list[InterpretationStatement] = []
    for interpretation in sorted(interpretations, key=lambda item: order[item.domain]):
        output.extend(statement for statement in interpretation.statements if statement.kind == "headline")
    return tuple(output)
