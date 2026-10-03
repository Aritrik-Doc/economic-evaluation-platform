"""Budget impact analysis engine for payer and policy affordability assessment.

The engine deliberately uses a transparent annual cost-calculator framework:
eligible population × treatment share × per-treated-person cost.  Annual inputs
can vary over time, so open-population growth, uptake, price changes and resource
cost changes are represented explicitly rather than hidden in discounting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal, Mapping, Sequence


ScenarioName = Literal["current", "future"]
COST_CATEGORIES = (
    "acquisition",
    "administration",
    "monitoring",
    "adverse_events",
    "disease_management",
    "other",
)


class BudgetImpactValidationError(ValueError):
    """Raised when a budget-impact model is internally inconsistent."""


@dataclass(frozen=True)
class BudgetIntervention:
    id: str
    name: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("Intervention id and name are required.")


@dataclass(frozen=True)
class PopulationYear:
    year: int
    eligible_population: float
    covered_lives: float | None = None

    def __post_init__(self) -> None:
        if self.year < 1:
            raise ValueError("Budget-impact year must be a positive integer.")
        if not isfinite(self.eligible_population) or self.eligible_population < 0:
            raise ValueError("Eligible population must be finite and non-negative.")
        if self.covered_lives is not None:
            if not isfinite(self.covered_lives) or self.covered_lives <= 0:
                raise ValueError("Covered lives must be positive when supplied.")
            if self.eligible_population > self.covered_lives + 1e-9:
                raise ValueError("Eligible population cannot exceed covered lives.")


@dataclass(frozen=True)
class TreatmentMixShare:
    scenario: ScenarioName
    year: int
    intervention_id: str
    share: float

    def __post_init__(self) -> None:
        if self.scenario not in {"current", "future"}:
            raise ValueError("Treatment mix scenario must be current or future.")
        if self.year < 1:
            raise ValueError("Treatment mix year must be positive.")
        if not self.intervention_id.strip():
            raise ValueError("Treatment mix intervention id is required.")
        if not isfinite(self.share) or self.share < 0 or self.share > 1:
            raise ValueError("Treatment shares must lie between 0 and 1.")


@dataclass(frozen=True)
class AnnualCostInput:
    intervention_id: str
    year: int
    category: str
    cost_per_treated_person: float

    def __post_init__(self) -> None:
        if not self.intervention_id.strip():
            raise ValueError("Cost intervention id is required.")
        if self.year < 1:
            raise ValueError("Cost year must be positive.")
        if self.category not in COST_CATEGORIES:
            raise ValueError(
                f"Unknown budget-impact cost category '{self.category}'."
            )
        if not isfinite(self.cost_per_treated_person):
            raise ValueError("Cost input must be finite.")


@dataclass(frozen=True)
class BudgetImpactDefinition:
    interventions: tuple[BudgetIntervention, ...]
    population: tuple[PopulationYear, ...]
    treatment_mix: tuple[TreatmentMixShare, ...]
    costs: tuple[AnnualCostInput, ...]
    included_cost_categories: tuple[str, ...] = COST_CATEGORIES
    share_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if len(self.interventions) < 2:
            raise ValueError("Budget impact analysis requires at least two interventions/options.")
        if not self.population:
            raise ValueError("At least one budget period is required.")
        if not isfinite(self.share_tolerance) or self.share_tolerance <= 0:
            raise ValueError("Share tolerance must be positive and finite.")


@dataclass(frozen=True)
class InterventionBudgetRow:
    scenario: ScenarioName
    year: int
    intervention_id: str
    intervention_name: str
    share: float
    treated_people: float
    cost_per_treated_person: float
    total_cost: float


@dataclass(frozen=True)
class CategoryBudgetRow:
    scenario: ScenarioName
    year: int
    category: str
    total_cost: float


@dataclass(frozen=True)
class BudgetImpactYearResult:
    year: int
    eligible_population: float
    covered_lives: float | None
    current_cost: float
    future_cost: float
    net_budget_impact: float
    cumulative_budget_impact: float
    pmpm_budget_impact: float | None


@dataclass(frozen=True)
class BudgetImpactRunResult:
    years: tuple[BudgetImpactYearResult, ...]
    intervention_rows: tuple[InterventionBudgetRow, ...]
    category_rows: tuple[CategoryBudgetRow, ...]

    @property
    def cumulative_budget_impact(self) -> float:
        return self.years[-1].cumulative_budget_impact if self.years else 0.0


def _validate_definition(definition: BudgetImpactDefinition) -> None:
    intervention_ids = [item.id for item in definition.interventions]
    if len(intervention_ids) != len(set(intervention_ids)):
        raise BudgetImpactValidationError("Intervention ids must be unique.")
    names = [item.name for item in definition.interventions]
    if len(names) != len(set(names)):
        raise BudgetImpactValidationError("Intervention names must be unique.")

    years = [row.year for row in definition.population]
    if len(years) != len(set(years)):
        raise BudgetImpactValidationError("Population years must be unique.")
    if years != sorted(years):
        raise BudgetImpactValidationError("Population years must be supplied in ascending order.")
    valid_years = set(years)
    valid_interventions = set(intervention_ids)

    included = set(definition.included_cost_categories)
    unknown_categories = included - set(COST_CATEGORIES)
    if unknown_categories:
        raise BudgetImpactValidationError(
            "Unknown included cost categories: " + ", ".join(sorted(unknown_categories))
        )

    mix_keys: set[tuple[str, int, str]] = set()
    mix_by_scenario_year: dict[tuple[str, int], dict[str, float]] = {}
    for row in definition.treatment_mix:
        if row.year not in valid_years:
            raise BudgetImpactValidationError(
                f"Treatment mix references year {row.year}, which is outside the model horizon."
            )
        if row.intervention_id not in valid_interventions:
            raise BudgetImpactValidationError(
                f"Treatment mix references unknown intervention '{row.intervention_id}'."
            )
        key = (row.scenario, row.year, row.intervention_id)
        if key in mix_keys:
            raise BudgetImpactValidationError(
                f"Duplicate treatment-mix entry for {row.scenario}, year {row.year}, intervention '{row.intervention_id}'."
            )
        mix_keys.add(key)
        mix_by_scenario_year.setdefault((row.scenario, row.year), {})[
            row.intervention_id
        ] = row.share

    for scenario in ("current", "future"):
        for year in years:
            shares = mix_by_scenario_year.get((scenario, year), {})
            missing = valid_interventions - set(shares)
            if missing:
                raise BudgetImpactValidationError(
                    f"{scenario.title()} treatment mix for year {year} is missing: "
                    + ", ".join(sorted(missing))
                    + "."
                )
            total = sum(shares.values())
            if abs(total - 1.0) > definition.share_tolerance:
                raise BudgetImpactValidationError(
                    f"{scenario.title()} treatment shares for year {year} sum to {total:.6f}; they must sum to 1."
                )

    cost_keys: set[tuple[str, int, str]] = set()
    for row in definition.costs:
        if row.year not in valid_years:
            raise BudgetImpactValidationError(
                f"Cost input references year {row.year}, which is outside the model horizon."
            )
        if row.intervention_id not in valid_interventions:
            raise BudgetImpactValidationError(
                f"Cost input references unknown intervention '{row.intervention_id}'."
            )
        key = (row.intervention_id, row.year, row.category)
        if key in cost_keys:
            raise BudgetImpactValidationError(
                f"Duplicate cost input for intervention '{row.intervention_id}', year {row.year}, category '{row.category}'."
            )
        cost_keys.add(key)


def top_down_eligible_population(
    covered_population: float,
    prevalence: float,
    diagnosed_or_identified: float = 1.0,
    clinically_eligible: float = 1.0,
    access_or_coverage: float = 1.0,
) -> float:
    """Calculate an eligible population using a transparent top-down funnel."""
    values = {
        "covered_population": covered_population,
        "prevalence": prevalence,
        "diagnosed_or_identified": diagnosed_or_identified,
        "clinically_eligible": clinically_eligible,
        "access_or_coverage": access_or_coverage,
    }
    for name, value in values.items():
        if not isfinite(value):
            raise BudgetImpactValidationError(f"{name} must be finite.")
    if covered_population < 0:
        raise BudgetImpactValidationError("Covered population cannot be negative.")
    for name in (
        "prevalence",
        "diagnosed_or_identified",
        "clinically_eligible",
        "access_or_coverage",
    ):
        value = values[name]
        if value < 0 or value > 1:
            raise BudgetImpactValidationError(f"{name} must lie between 0 and 1.")
    return (
        covered_population
        * prevalence
        * diagnosed_or_identified
        * clinically_eligible
        * access_or_coverage
    )


def compound_series(start: float, annual_growth_rate: float, periods: int) -> tuple[float, ...]:
    if periods < 1:
        raise BudgetImpactValidationError("At least one period is required.")
    if not isfinite(start) or start < 0:
        raise BudgetImpactValidationError("Series starting value must be finite and non-negative.")
    if not isfinite(annual_growth_rate) or annual_growth_rate <= -1:
        raise BudgetImpactValidationError("Annual growth rate must be finite and greater than -100%.")
    return tuple(start * ((1.0 + annual_growth_rate) ** index) for index in range(periods))


def reallocate_target_share(
    shares: Mapping[str, float], target_intervention_id: str, target_share: float
) -> dict[str, float]:
    """Set one intervention share and redistribute the remainder proportionally.

    This is intended for transparent uptake scenarios.  It does not silently infer
    substitution patterns: the proportional redistribution rule should be shown to
    the modeller whenever it is used.
    """
    if target_intervention_id not in shares:
        raise BudgetImpactValidationError("Target intervention is not present in the treatment mix.")
    if not isfinite(target_share) or target_share < 0 or target_share > 1:
        raise BudgetImpactValidationError("Target treatment share must lie between 0 and 1.")
    total = sum(float(value) for value in shares.values())
    if abs(total - 1.0) > 1e-8:
        raise BudgetImpactValidationError("Input treatment shares must sum to 1 before reallocation.")

    other_ids = [key for key in shares if key != target_intervention_id]
    if not other_ids:
        if abs(target_share - 1.0) > 1e-8:
            raise BudgetImpactValidationError("A single-option mix must retain a share of 1.")
        return {target_intervention_id: 1.0}

    old_other_total = sum(float(shares[key]) for key in other_ids)
    remaining = 1.0 - target_share
    if old_other_total <= 1e-15:
        equal = remaining / len(other_ids)
        output = {key: equal for key in other_ids}
    else:
        output = {
            key: float(shares[key]) / old_other_total * remaining for key in other_ids
        }
    output[target_intervention_id] = target_share
    return output


def run_budget_impact(definition: BudgetImpactDefinition) -> BudgetImpactRunResult:
    _validate_definition(definition)

    names = {item.id: item.name for item in definition.interventions}
    population = {row.year: row for row in definition.population}
    mix = {
        (row.scenario, row.year, row.intervention_id): row.share
        for row in definition.treatment_mix
    }
    included = set(definition.included_cost_categories)
    cost_values: dict[tuple[str, int, str], float] = {
        (row.intervention_id, row.year, row.category): row.cost_per_treated_person
        for row in definition.costs
        if row.category in included
    }

    intervention_rows: list[InterventionBudgetRow] = []
    category_rows: list[CategoryBudgetRow] = []
    year_results: list[BudgetImpactYearResult] = []
    cumulative = 0.0

    for year in sorted(population):
        pop_row = population[year]
        scenario_totals: dict[str, float] = {}

        for scenario in ("current", "future"):
            total_cost = 0.0
            category_totals = {category: 0.0 for category in definition.included_cost_categories}
            for intervention in definition.interventions:
                share = mix[(scenario, year, intervention.id)]
                treated = pop_row.eligible_population * share
                annual_cost = 0.0
                for category in definition.included_cost_categories:
                    unit_cost = cost_values.get((intervention.id, year, category), 0.0)
                    component_total = treated * unit_cost
                    annual_cost += unit_cost
                    category_totals[category] += component_total
                intervention_total = treated * annual_cost
                total_cost += intervention_total
                intervention_rows.append(
                    InterventionBudgetRow(
                        scenario=scenario,
                        year=year,
                        intervention_id=intervention.id,
                        intervention_name=intervention.name,
                        share=share,
                        treated_people=treated,
                        cost_per_treated_person=annual_cost,
                        total_cost=intervention_total,
                    )
                )

            scenario_totals[scenario] = total_cost
            for category, value in category_totals.items():
                category_rows.append(
                    CategoryBudgetRow(
                        scenario=scenario,
                        year=year,
                        category=category,
                        total_cost=value,
                    )
                )

        net = scenario_totals["future"] - scenario_totals["current"]
        cumulative += net
        pmpm = (
            net / (pop_row.covered_lives * 12.0)
            if pop_row.covered_lives is not None
            else None
        )
        year_results.append(
            BudgetImpactYearResult(
                year=year,
                eligible_population=pop_row.eligible_population,
                covered_lives=pop_row.covered_lives,
                current_cost=scenario_totals["current"],
                future_cost=scenario_totals["future"],
                net_budget_impact=net,
                cumulative_budget_impact=cumulative,
                pmpm_budget_impact=pmpm,
            )
        )

    return BudgetImpactRunResult(
        years=tuple(year_results),
        intervention_rows=tuple(intervention_rows),
        category_rows=tuple(category_rows),
    )


def apply_simple_scenario(
    definition: BudgetImpactDefinition,
    *,
    population_multiplier: float = 1.0,
    target_intervention_id: str | None = None,
    uptake_multiplier: float = 1.0,
    target_cost_multiplier: float = 1.0,
) -> BudgetImpactDefinition:
    """Return a transparent one-way/multi-way BIA scenario definition.

    Population and target-intervention cost multipliers apply to all model years.
    If a target intervention is supplied, its future share is multiplied and capped
    at 1, with the remaining future mix redistributed proportionally across the
    other interventions.
    """
    for label, value in (
        ("population multiplier", population_multiplier),
        ("uptake multiplier", uptake_multiplier),
        ("target cost multiplier", target_cost_multiplier),
    ):
        if not isfinite(value) or value < 0:
            raise BudgetImpactValidationError(f"{label} must be finite and non-negative.")

    population = tuple(
        replace(row, eligible_population=row.eligible_population * population_multiplier)
        for row in definition.population
    )

    costs = definition.costs
    treatment_mix = definition.treatment_mix
    if target_intervention_id is not None:
        ids = {item.id for item in definition.interventions}
        if target_intervention_id not in ids:
            raise BudgetImpactValidationError("Scenario target intervention is unknown.")
        costs = tuple(
            replace(
                row,
                cost_per_treated_person=(
                    row.cost_per_treated_person * target_cost_multiplier
                    if row.intervention_id == target_intervention_id
                    else row.cost_per_treated_person
                ),
            )
            for row in definition.costs
        )

        current_rows = [row for row in definition.treatment_mix if row.scenario == "current"]
        future_by_year: dict[int, dict[str, float]] = {}
        for row in definition.treatment_mix:
            if row.scenario == "future":
                future_by_year.setdefault(row.year, {})[row.intervention_id] = row.share

        future_rows: list[TreatmentMixShare] = []
        for year, shares in sorted(future_by_year.items()):
            target = min(1.0, shares[target_intervention_id] * uptake_multiplier)
            adjusted = reallocate_target_share(shares, target_intervention_id, target)
            for intervention in definition.interventions:
                future_rows.append(
                    TreatmentMixShare(
                        "future", year, intervention.id, adjusted[intervention.id]
                    )
                )
        treatment_mix = tuple(current_rows + future_rows)

    return BudgetImpactDefinition(
        interventions=definition.interventions,
        population=population,
        treatment_mix=treatment_mix,
        costs=costs,
        included_cost_categories=definition.included_cost_categories,
        share_tolerance=definition.share_tolerance,
    )
