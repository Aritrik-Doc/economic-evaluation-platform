"""Shared population and uptake assumptions for affordability and implementation models.

This module deliberately sits outside Budget Impact Analysis.  It represents the
population exposed to a policy decision and the current/future allocation of that
population across interventions/options.  BIA and resource/capacity planning can
therefore consume the same assumptions without either module owning them.

The annual population can represent either a cross-sectional eligible/treated
population or annual new treatment starts.  That meaning is declared explicitly
through ``population_basis`` and downstream modules remain responsible for using
the basis coherently.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Mapping


ScenarioName = Literal["current", "future"]
PopulationBasis = Literal["annual_eligible_population", "new_treatment_starts"]


class PopulationUptakeValidationError(ValueError):
    """Raised when a shared population/uptake scenario is internally inconsistent."""


@dataclass(frozen=True)
class PopulationOption:
    id: str
    name: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("Population option id and name are required.")


@dataclass(frozen=True)
class PopulationYear:
    year: int
    eligible_population: float
    covered_lives: float | None = None

    def __post_init__(self) -> None:
        if self.year < 1:
            raise ValueError("Population year must be a positive integer.")
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
            raise ValueError("Treatment mix option id is required.")
        if not isfinite(self.share) or self.share < 0 or self.share > 1:
            raise ValueError("Treatment shares must lie between 0 and 1.")


@dataclass(frozen=True)
class PopulationUptakeDefinition:
    options: tuple[PopulationOption, ...]
    population: tuple[PopulationYear, ...]
    treatment_mix: tuple[TreatmentMixShare, ...]
    population_basis: PopulationBasis = "annual_eligible_population"
    share_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("Population and uptake planning requires at least two options.")
        if not self.population:
            raise ValueError("At least one population year is required.")
        if self.population_basis not in {"annual_eligible_population", "new_treatment_starts"}:
            raise ValueError("Unknown population basis.")
        if not isfinite(self.share_tolerance) or self.share_tolerance <= 0:
            raise ValueError("Share tolerance must be positive and finite.")


@dataclass(frozen=True)
class TreatedPopulationRow:
    scenario: ScenarioName
    year: int
    intervention_id: str
    intervention_name: str
    eligible_population: float
    share: float
    treated_people: float


@dataclass(frozen=True)
class PopulationUptakeRunResult:
    rows: tuple[TreatedPopulationRow, ...]
    population_basis: PopulationBasis

    def treated_people(self, scenario: ScenarioName, year: int, intervention_id: str) -> float:
        for row in self.rows:
            if row.scenario == scenario and row.year == year and row.intervention_id == intervention_id:
                return row.treated_people
        raise KeyError((scenario, year, intervention_id))


def validate_population_uptake(definition: PopulationUptakeDefinition) -> None:
    option_ids = [item.id for item in definition.options]
    if len(option_ids) != len(set(option_ids)):
        raise PopulationUptakeValidationError("Option ids must be unique.")
    option_names = [item.name for item in definition.options]
    if len(option_names) != len(set(option_names)):
        raise PopulationUptakeValidationError("Option names must be unique.")

    years = [row.year for row in definition.population]
    if years != sorted(years) or len(years) != len(set(years)):
        raise PopulationUptakeValidationError("Population years must be unique and in ascending order.")
    valid_years = set(years)
    valid_options = set(option_ids)

    seen: set[tuple[str, int, str]] = set()
    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for row in definition.treatment_mix:
        if row.year not in valid_years:
            raise PopulationUptakeValidationError(
                f"Treatment mix references year {row.year}, which is outside the population horizon."
            )
        if row.intervention_id not in valid_options:
            raise PopulationUptakeValidationError(
                f"Treatment mix references unknown option '{row.intervention_id}'."
            )
        key = (row.scenario, row.year, row.intervention_id)
        if key in seen:
            raise PopulationUptakeValidationError("Duplicate treatment-mix entry.")
        seen.add(key)
        grouped.setdefault((row.scenario, row.year), {})[row.intervention_id] = row.share

    for scenario in ("current", "future"):
        for year in years:
            shares = grouped.get((scenario, year), {})
            missing = valid_options - set(shares)
            if missing:
                raise PopulationUptakeValidationError(
                    f"{scenario.title()} mix for year {year} is missing: " + ", ".join(sorted(missing))
                )
            total = sum(shares.values())
            if abs(total - 1.0) > definition.share_tolerance:
                raise PopulationUptakeValidationError(
                    f"{scenario.title()} treatment shares for year {year} sum to {total:.6f}; they must sum to 1."
                )


def run_population_uptake(definition: PopulationUptakeDefinition) -> PopulationUptakeRunResult:
    validate_population_uptake(definition)
    population = {row.year: row for row in definition.population}
    names = {item.id: item.name for item in definition.options}
    shares = {
        (row.scenario, row.year, row.intervention_id): row.share
        for row in definition.treatment_mix
    }
    rows: list[TreatedPopulationRow] = []
    for scenario in ("current", "future"):
        for year in sorted(population):
            eligible = population[year].eligible_population
            for option in definition.options:
                share = shares[(scenario, year, option.id)]
                rows.append(
                    TreatedPopulationRow(
                        scenario=scenario,
                        year=year,
                        intervention_id=option.id,
                        intervention_name=names[option.id],
                        eligible_population=eligible,
                        share=share,
                        treated_people=eligible * share,
                    )
                )
    return PopulationUptakeRunResult(tuple(rows), definition.population_basis)


def top_down_eligible_population(
    covered_population: float,
    prevalence: float,
    diagnosed_or_identified: float = 1.0,
    clinically_eligible: float = 1.0,
    access_or_coverage: float = 1.0,
) -> float:
    values = {
        "covered_population": covered_population,
        "prevalence": prevalence,
        "diagnosed_or_identified": diagnosed_or_identified,
        "clinically_eligible": clinically_eligible,
        "access_or_coverage": access_or_coverage,
    }
    if any(not isfinite(value) for value in values.values()):
        raise PopulationUptakeValidationError("Population-funnel inputs must be finite.")
    if covered_population < 0:
        raise PopulationUptakeValidationError("Covered population cannot be negative.")
    for name in ("prevalence", "diagnosed_or_identified", "clinically_eligible", "access_or_coverage"):
        if values[name] < 0 or values[name] > 1:
            raise PopulationUptakeValidationError(f"{name} must lie between 0 and 1.")
    return covered_population * prevalence * diagnosed_or_identified * clinically_eligible * access_or_coverage


def compound_series(start: float, annual_growth_rate: float, periods: int) -> tuple[float, ...]:
    if periods < 1:
        raise PopulationUptakeValidationError("At least one period is required.")
    if not isfinite(start) or start < 0:
        raise PopulationUptakeValidationError("Series starting value must be finite and non-negative.")
    if not isfinite(annual_growth_rate) or annual_growth_rate <= -1:
        raise PopulationUptakeValidationError("Annual growth rate must be finite and greater than -100%.")
    return tuple(start * ((1.0 + annual_growth_rate) ** index) for index in range(periods))


def reallocate_target_share(
    shares: Mapping[str, float], target_intervention_id: str, target_share: float
) -> dict[str, float]:
    if target_intervention_id not in shares:
        raise PopulationUptakeValidationError("Target option is not present in the treatment mix.")
    if not isfinite(target_share) or target_share < 0 or target_share > 1:
        raise PopulationUptakeValidationError("Target treatment share must lie between 0 and 1.")
    total = sum(float(value) for value in shares.values())
    if abs(total - 1.0) > 1e-8:
        raise PopulationUptakeValidationError("Input treatment shares must sum to 1 before reallocation.")
    other_ids = [key for key in shares if key != target_intervention_id]
    if not other_ids:
        if abs(target_share - 1.0) > 1e-8:
            raise PopulationUptakeValidationError("A single-option mix must retain a share of 1.")
        return {target_intervention_id: 1.0}
    old_other_total = sum(float(shares[key]) for key in other_ids)
    remaining = 1.0 - target_share
    if old_other_total <= 1e-15:
        output = {key: remaining / len(other_ids) for key in other_ids}
    else:
        output = {key: float(shares[key]) / old_other_total * remaining for key in other_ids}
    output[target_intervention_id] = target_share
    return output
