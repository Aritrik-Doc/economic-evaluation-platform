"""Validated cross-page Budget Impact Analysis session contract.

The BIA page publishes canonical, validated population and treatment-mix rows.
Downstream pages consume those rows instead of reconstructing the analysis from
individual Streamlit widget keys. This keeps clinical linkage, capacity planning
and policy interpretation aligned with the BIA result currently shown to users.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model.budget_impact import (
    COST_CATEGORIES,
    AnnualCostInput,
    BudgetImpactDefinition,
    BudgetImpactValidationError,
    BudgetIntervention,
    PopulationYear,
    TreatmentMixShare,
    compound_series,
)
from model.budget_impact_profiles import BUDGET_IMPACT_PROFILES


class BIAContextError(ValueError):
    """Raised when no current validated BIA context is available."""


def _profile_horizon_currency(session: Mapping[str, Any]) -> tuple[int, str]:
    profile_code = session.get("bia_profile")
    if not profile_code:
        raise BIAContextError("Configure Budget Impact Analysis first.")
    if profile_code == "CUSTOM":
        return (
            int(session.get("bia_horizon_custom", 3)),
            str(session.get("bia_currency_custom", "GBP")),
        )
    if profile_code not in BUDGET_IMPACT_PROFILES:
        raise BIAContextError("The active BIA methods profile is not recognised.")
    profile = BUDGET_IMPACT_PROFILES[profile_code]
    return (
        int(session.get(f"bia_horizon_{profile.code}", profile.default_horizon_years)),
        str(session.get("bia_currency_profile", profile.default_currency)),
    )


def bia_definition_from_session(
    session: Mapping[str, Any],
) -> tuple[BudgetImpactDefinition, str, int]:
    """Rebuild the current validated BIA definition from the canonical session export."""

    if session.get("bia_context_valid") is not True:
        raise BIAContextError(
            "Open Budget Impact Analysis and complete a valid analysis first. "
            "Downstream pages do not reuse an earlier BIA state after the current inputs become invalid."
        )

    horizon, currency = _profile_horizon_currency(session)
    raw_interventions = session.get("bia_interventions")
    raw_population = session.get("bia_population_rows")
    raw_mix = session.get("bia_treatment_mix_rows")
    raw_costs = session.get("bia_cost_inputs")
    if not raw_interventions or not raw_population or not raw_mix or not raw_costs:
        raise BIAContextError(
            "The validated BIA handoff is incomplete. Re-open Budget Impact Analysis so the current inputs can be revalidated."
        )

    interventions = tuple(
        BudgetIntervention(str(row["id"]), str(row["name"]))
        for row in raw_interventions
    )
    population = tuple(
        PopulationYear(
            int(row["year"]),
            float(row["eligible_population"]),
            None
            if row.get("covered_lives") in (None, "", 0, 0.0)
            else float(row["covered_lives"]),
        )
        for row in raw_population
    )
    mix = tuple(
        TreatmentMixShare(
            str(row["scenario"]),
            int(row["year"]),
            str(row["intervention_id"]),
            float(row["share"]),
        )
        for row in raw_mix
    )

    if len(population) != horizon:
        raise BIAContextError(
            "The validated BIA population horizon does not match the active methods horizon. Re-open BIA and revalidate the analysis."
        )

    costs: list[AnnualCostInput] = []
    for intervention in interventions:
        data = raw_costs.get(intervention.id, {})
        growth = float(data.get("annual_change", 0.0))
        for category in COST_CATEGORIES:
            series = compound_series(float(data.get(category, 0.0)), growth, horizon)
            for year, value in enumerate(series, start=1):
                costs.append(AnnualCostInput(intervention.id, year, category, value))

    included = tuple(session.get("bia_included_categories", list(COST_CATEGORIES)))
    try:
        definition = BudgetImpactDefinition(
            interventions=interventions,
            population=population,
            treatment_mix=mix,
            costs=tuple(costs),
            included_cost_categories=included,
        )
    except (ValueError, BudgetImpactValidationError) as exc:
        raise BIAContextError(str(exc)) from exc
    return definition, currency, horizon
