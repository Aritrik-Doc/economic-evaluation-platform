"""Natural-unit resource and capacity planning for Budget Impact Analysis.

The module separates physical resource demand from monetary budget impact. It
uses the BIA population and treatment mix, explicit per-person resource
requirements, and annual capacity available to the modelled population.

Two population bases are supported:

* ``annual_treated_population`` applies a budget-year-specific resource input to
  the treated population in that same year;
* ``new_treatment_starts`` treats each annual population as a cohort of new
  starts and stacks year-since-initiation resource profiles across budget years.

Capacity can be entered as total system capacity less demand already committed
to services outside the modelled BIA population. This makes the residual
capacity available to the model explicit instead of assuming that nominal
capacity is fully free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal, Sequence

from model.budget_impact import BudgetIntervention, PopulationYear, TreatmentMixShare


DemandBasis = Literal["annual_treated_population", "new_treatment_starts"]
RESOURCE_CATEGORIES = (
    "workforce",
    "facility",
    "equipment",
    "diagnostic",
    "inpatient",
    "pharmacy",
    "consumable",
    "other",
)


class ResourceCapacityValidationError(ValueError):
    """Raised when a resource/capacity model is internally inconsistent."""


@dataclass(frozen=True)
class ResourceDefinition:
    id: str
    name: str
    unit: str
    category: str = "other"

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip() or not self.unit.strip():
            raise ValueError("Resource id, name and natural unit are required.")
        if self.category not in RESOURCE_CATEGORIES:
            raise ValueError(f"Unknown resource category '{self.category}'.")


@dataclass(frozen=True)
class ResourceRequirement:
    intervention_id: str
    resource_id: str
    period: int
    units_per_person: float

    def __post_init__(self) -> None:
        if not self.intervention_id.strip() or not self.resource_id.strip():
            raise ValueError("Resource requirement must reference an intervention and resource.")
        if self.period < 1:
            raise ValueError("Resource requirement period must be a positive integer.")
        if not isfinite(self.units_per_person) or self.units_per_person < 0:
            raise ValueError("Resource units per person must be finite and non-negative.")


@dataclass(frozen=True)
class AnnualResourceCapacity:
    resource_id: str
    year: int
    total_capacity: float
    committed_other_demand: float = 0.0

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ValueError("Annual capacity must reference a resource.")
        if self.year < 1:
            raise ValueError("Capacity year must be a positive integer.")
        if not isfinite(self.total_capacity) or self.total_capacity < 0:
            raise ValueError("Total capacity must be finite and non-negative.")
        if not isfinite(self.committed_other_demand) or self.committed_other_demand < 0:
            raise ValueError("Committed other demand must be finite and non-negative.")
        if self.committed_other_demand > self.total_capacity + 1e-9:
            raise ValueError("Committed other demand cannot exceed total capacity.")

    @property
    def available_capacity(self) -> float:
        return max(0.0, self.total_capacity - self.committed_other_demand)


@dataclass(frozen=True)
class ResourceCapacityDefinition:
    interventions: tuple[BudgetIntervention, ...]
    population: tuple[PopulationYear, ...]
    treatment_mix: tuple[TreatmentMixShare, ...]
    resources: tuple[ResourceDefinition, ...]
    requirements: tuple[ResourceRequirement, ...]
    capacities: tuple[AnnualResourceCapacity, ...]
    demand_basis: DemandBasis = "annual_treated_population"
    share_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if len(self.interventions) < 2:
            raise ValueError("Resource planning requires at least two interventions/options.")
        if not self.population:
            raise ValueError("At least one planning year is required.")
        if not self.resources:
            raise ValueError("At least one constrained or monitored resource is required.")
        if self.demand_basis not in {"annual_treated_population", "new_treatment_starts"}:
            raise ValueError("Unknown resource-demand basis.")
        if not isfinite(self.share_tolerance) or self.share_tolerance <= 0:
            raise ValueError("Share tolerance must be positive and finite.")


@dataclass(frozen=True)
class InterventionResourceRow:
    scenario: str
    budget_year: int
    intervention_id: str
    intervention_name: str
    resource_id: str
    resource_name: str
    unit: str
    source_period: int
    treated_people: float
    units_per_person: float
    required_units: float
    initiation_year: int | None = None


@dataclass(frozen=True)
class ResourceScenarioRow:
    scenario: str
    year: int
    resource_id: str
    resource_name: str
    unit: str
    category: str
    required_units: float
    total_capacity: float
    committed_other_demand: float
    available_capacity: float
    utilization_ratio: float | None
    headroom: float
    shortfall: float

    @property
    def status(self) -> str:
        if self.shortfall > 1e-9:
            return "shortfall"
        if self.available_capacity > 0 and abs(self.required_units - self.available_capacity) <= 1e-9:
            return "at_capacity"
        return "within_capacity"


@dataclass(frozen=True)
class ResourceComparisonRow:
    year: int
    resource_id: str
    resource_name: str
    unit: str
    category: str
    current_required_units: float
    future_required_units: float
    net_change_units: float
    available_capacity: float
    current_utilization_ratio: float | None
    future_utilization_ratio: float | None
    current_shortfall: float
    future_shortfall: float
    future_headroom: float


@dataclass(frozen=True)
class ResourceCapacityRunResult:
    scenario_rows: tuple[ResourceScenarioRow, ...]
    comparison_rows: tuple[ResourceComparisonRow, ...]
    intervention_rows: tuple[InterventionResourceRow, ...]

    @property
    def future_shortfall_rows(self) -> tuple[ResourceComparisonRow, ...]:
        return tuple(row for row in self.comparison_rows if row.future_shortfall > 1e-9)


def _validate_definition(definition: ResourceCapacityDefinition) -> None:
    intervention_ids = [item.id for item in definition.interventions]
    if len(intervention_ids) != len(set(intervention_ids)):
        raise ResourceCapacityValidationError("Intervention ids must be unique.")
    valid_interventions = set(intervention_ids)

    years = [row.year for row in definition.population]
    if years != sorted(years) or len(years) != len(set(years)):
        raise ResourceCapacityValidationError("Population years must be unique and in ascending order.")
    valid_years = set(years)
    horizon = max(years)

    resource_ids = [resource.id for resource in definition.resources]
    if len(resource_ids) != len(set(resource_ids)):
        raise ResourceCapacityValidationError("Resource ids must be unique.")
    resource_names = [resource.name for resource in definition.resources]
    if len(resource_names) != len(set(resource_names)):
        raise ResourceCapacityValidationError("Resource names must be unique.")
    valid_resources = set(resource_ids)

    mix_keys: set[tuple[str, int, str]] = set()
    mix_groups: dict[tuple[str, int], dict[str, float]] = {}
    for row in definition.treatment_mix:
        if row.year not in valid_years:
            raise ResourceCapacityValidationError("Treatment mix references a year outside the planning horizon.")
        if row.intervention_id not in valid_interventions:
            raise ResourceCapacityValidationError(
                f"Treatment mix references unknown intervention '{row.intervention_id}'."
            )
        key = (row.scenario, row.year, row.intervention_id)
        if key in mix_keys:
            raise ResourceCapacityValidationError("Duplicate treatment-mix entry in resource plan.")
        mix_keys.add(key)
        mix_groups.setdefault((row.scenario, row.year), {})[row.intervention_id] = row.share

    for scenario in ("current", "future"):
        for year in years:
            shares = mix_groups.get((scenario, year), {})
            missing = valid_interventions - set(shares)
            if missing:
                raise ResourceCapacityValidationError(
                    f"{scenario.title()} mix for year {year} is missing: " + ", ".join(sorted(missing))
                )
            total = sum(shares.values())
            if abs(total - 1.0) > definition.share_tolerance:
                raise ResourceCapacityValidationError(
                    f"{scenario.title()} mix for year {year} sums to {total:.6f}; it must sum to 1."
                )

    requirement_keys: set[tuple[str, str, int]] = set()
    for row in definition.requirements:
        if row.intervention_id not in valid_interventions:
            raise ResourceCapacityValidationError(
                f"Resource requirement references unknown intervention '{row.intervention_id}'."
            )
        if row.resource_id not in valid_resources:
            raise ResourceCapacityValidationError(
                f"Resource requirement references unknown resource '{row.resource_id}'."
            )
        if row.period > horizon:
            raise ResourceCapacityValidationError(
                f"Resource requirement period {row.period} exceeds the planning horizon of {horizon} years."
            )
        key = (row.intervention_id, row.resource_id, row.period)
        if key in requirement_keys:
            raise ResourceCapacityValidationError(
                f"Duplicate resource requirement for intervention '{row.intervention_id}', resource '{row.resource_id}', period {row.period}."
            )
        requirement_keys.add(key)

    capacity_keys: set[tuple[str, int]] = set()
    for row in definition.capacities:
        if row.resource_id not in valid_resources:
            raise ResourceCapacityValidationError(
                f"Capacity references unknown resource '{row.resource_id}'."
            )
        if row.year not in valid_years:
            raise ResourceCapacityValidationError("Capacity references a year outside the planning horizon.")
        key = (row.resource_id, row.year)
        if key in capacity_keys:
            raise ResourceCapacityValidationError(
                f"Duplicate capacity entry for resource '{row.resource_id}', year {row.year}."
            )
        capacity_keys.add(key)

    required_capacity_keys = {(resource_id, year) for resource_id in valid_resources for year in years}
    missing_capacity = required_capacity_keys - capacity_keys
    if missing_capacity:
        resource_id, year = sorted(missing_capacity)[0]
        raise ResourceCapacityValidationError(
            f"Capacity is missing for resource '{resource_id}' in year {year}."
        )


def _utilization(required: float, available: float) -> float | None:
    if available <= 0:
        return 0.0 if required <= 1e-12 else None
    return required / available


def run_resource_capacity_plan(definition: ResourceCapacityDefinition) -> ResourceCapacityRunResult:
    """Calculate natural-unit demand and compare it with available annual capacity."""
    _validate_definition(definition)

    population = {row.year: row for row in definition.population}
    shares = {
        (row.scenario, row.year, row.intervention_id): row.share
        for row in definition.treatment_mix
    }
    requirements = {
        (row.intervention_id, row.resource_id, row.period): row.units_per_person
        for row in definition.requirements
    }
    capacities = {(row.resource_id, row.year): row for row in definition.capacities}
    resources = {row.id: row for row in definition.resources}
    intervention_names = {row.id: row.name for row in definition.interventions}

    totals: dict[tuple[str, int, str], float] = {
        (scenario, year, resource_id): 0.0
        for scenario in ("current", "future")
        for year in population
        for resource_id in resources
    }
    intervention_rows: list[InterventionResourceRow] = []

    if definition.demand_basis == "annual_treated_population":
        for scenario in ("current", "future"):
            for year, pop_row in population.items():
                for intervention in definition.interventions:
                    treated = pop_row.eligible_population * shares[(scenario, year, intervention.id)]
                    for resource_id, resource in resources.items():
                        units_pp = requirements.get((intervention.id, resource_id, year), 0.0)
                        required = treated * units_pp
                        totals[(scenario, year, resource_id)] += required
                        if required or units_pp:
                            intervention_rows.append(
                                InterventionResourceRow(
                                    scenario=scenario,
                                    budget_year=year,
                                    intervention_id=intervention.id,
                                    intervention_name=intervention.name,
                                    resource_id=resource_id,
                                    resource_name=resource.name,
                                    unit=resource.unit,
                                    source_period=year,
                                    treated_people=treated,
                                    units_per_person=units_pp,
                                    required_units=required,
                                    initiation_year=None,
                                )
                            )
    else:
        years = sorted(population)
        for scenario in ("current", "future"):
            for initiation_year in years:
                starts_population = population[initiation_year].eligible_population
                for intervention in definition.interventions:
                    starts = starts_population * shares[(scenario, initiation_year, intervention.id)]
                    for budget_year in years:
                        if budget_year < initiation_year:
                            continue
                        period = budget_year - initiation_year + 1
                        for resource_id, resource in resources.items():
                            units_pp = requirements.get((intervention.id, resource_id, period), 0.0)
                            required = starts * units_pp
                            totals[(scenario, budget_year, resource_id)] += required
                            if required or units_pp:
                                intervention_rows.append(
                                    InterventionResourceRow(
                                        scenario=scenario,
                                        budget_year=budget_year,
                                        intervention_id=intervention.id,
                                        intervention_name=intervention.name,
                                        resource_id=resource_id,
                                        resource_name=resource.name,
                                        unit=resource.unit,
                                        source_period=period,
                                        treated_people=starts,
                                        units_per_person=units_pp,
                                        required_units=required,
                                        initiation_year=initiation_year,
                                    )
                                )

    scenario_rows: list[ResourceScenarioRow] = []
    scenario_lookup: dict[tuple[str, int, str], ResourceScenarioRow] = {}
    for scenario in ("current", "future"):
        for year in sorted(population):
            for resource_id, resource in resources.items():
                required = totals[(scenario, year, resource_id)]
                capacity = capacities[(resource_id, year)]
                available = capacity.available_capacity
                shortfall = max(required - available, 0.0)
                headroom = max(available - required, 0.0)
                row = ResourceScenarioRow(
                    scenario=scenario,
                    year=year,
                    resource_id=resource_id,
                    resource_name=resource.name,
                    unit=resource.unit,
                    category=resource.category,
                    required_units=required,
                    total_capacity=capacity.total_capacity,
                    committed_other_demand=capacity.committed_other_demand,
                    available_capacity=available,
                    utilization_ratio=_utilization(required, available),
                    headroom=headroom,
                    shortfall=shortfall,
                )
                scenario_rows.append(row)
                scenario_lookup[(scenario, year, resource_id)] = row

    comparisons: list[ResourceComparisonRow] = []
    for year in sorted(population):
        for resource_id, resource in resources.items():
            current = scenario_lookup[("current", year, resource_id)]
            future = scenario_lookup[("future", year, resource_id)]
            comparisons.append(
                ResourceComparisonRow(
                    year=year,
                    resource_id=resource_id,
                    resource_name=resource.name,
                    unit=resource.unit,
                    category=resource.category,
                    current_required_units=current.required_units,
                    future_required_units=future.required_units,
                    net_change_units=future.required_units - current.required_units,
                    available_capacity=future.available_capacity,
                    current_utilization_ratio=current.utilization_ratio,
                    future_utilization_ratio=future.utilization_ratio,
                    current_shortfall=current.shortfall,
                    future_shortfall=future.shortfall,
                    future_headroom=future.headroom,
                )
            )

    return ResourceCapacityRunResult(
        scenario_rows=tuple(scenario_rows),
        comparison_rows=tuple(comparisons),
        intervention_rows=tuple(intervention_rows),
    )


def apply_capacity_expansion(
    definition: ResourceCapacityDefinition,
    *,
    resource_id: str,
    from_year: int = 1,
    capacity_multiplier: float = 1.0,
    additional_capacity: float = 0.0,
) -> ResourceCapacityDefinition:
    """Return an explicit service-capacity expansion scenario.

    The change applies to total capacity for the selected resource from
    ``from_year`` onward. Existing demand already committed to other services is
    left unchanged.
    """
    if resource_id not in {resource.id for resource in definition.resources}:
        raise ResourceCapacityValidationError("Capacity scenario references an unknown resource.")
    if from_year < 1:
        raise ResourceCapacityValidationError("Capacity scenario start year must be positive.")
    if not isfinite(capacity_multiplier) or capacity_multiplier < 0:
        raise ResourceCapacityValidationError("Capacity multiplier must be finite and non-negative.")
    if not isfinite(additional_capacity):
        raise ResourceCapacityValidationError("Additional capacity must be finite.")

    updated: list[AnnualResourceCapacity] = []
    for row in definition.capacities:
        if row.resource_id == resource_id and row.year >= from_year:
            new_total = row.total_capacity * capacity_multiplier + additional_capacity
            if new_total < row.committed_other_demand - 1e-9:
                raise ResourceCapacityValidationError(
                    "Scenario total capacity cannot fall below capacity already committed to other services."
                )
            updated.append(replace(row, total_capacity=new_total))
        else:
            updated.append(row)
    return replace(definition, capacities=tuple(updated))
