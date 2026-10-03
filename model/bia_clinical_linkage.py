"""Clinical-model linkage for Budget Impact Analysis.

The linkage keeps two layers separate:

* the clinical model projects per-patient condition-related costs over time;
* the BIA applies those trajectories to annual treatment-start cohorts and the
  current/future utilisation mix for the budget holder.

Only explicitly selected cost-reward parameters are imported. Other clinical
cost parameters are set to zero for the linkage run, which is the main
protection against double counting acquisition/administration costs already
entered directly in the BIA.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isclose
from typing import Mapping, Sequence

from model.budget_impact import (
    BudgetImpactDefinition,
    BudgetImpactRunResult,
    BudgetImpactValidationError,
    run_budget_impact,
)
from model.markov import CohortMarkovDefinition, MarkovStrategyDefinition, run_cohort_markov
from model.schema import Parameter
from model.semi_markov import SemiMarkovDefinition, SemiMarkovStrategyDefinition, run_semi_markov


class ClinicalLinkageError(ValueError):
    """Raised when a clinical model cannot be linked coherently to a BIA."""


@dataclass(frozen=True)
class ClinicalCostProfile:
    strategy_id: str
    strategy_name: str
    annual_cost_per_patient: tuple[float, ...]
    source_model_type: str
    included_parameter_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClinicalCohortBudgetRow:
    scenario: str
    budget_year: int
    intervention_id: str
    strategy_id: str
    initiation_year: int
    year_since_initiation: int
    treatment_starts: float
    cost_per_patient: float
    total_clinical_cost: float


@dataclass(frozen=True)
class LinkedBudgetImpactYearResult:
    year: int
    eligible_population: float
    covered_lives: float | None
    direct_current_cost: float
    direct_future_cost: float
    clinical_current_cost: float
    clinical_future_cost: float
    current_cost: float
    future_cost: float
    net_budget_impact: float
    cumulative_budget_impact: float
    pmpm_budget_impact: float | None


@dataclass(frozen=True)
class LinkedBudgetImpactRunResult:
    years: tuple[LinkedBudgetImpactYearResult, ...]
    clinical_rows: tuple[ClinicalCohortBudgetRow, ...]
    direct_run: BudgetImpactRunResult
    linked_category: str

    @property
    def cumulative_budget_impact(self) -> float:
        return self.years[-1].cumulative_budget_impact if self.years else 0.0


def _cycles_per_year(cycle_length_years: float) -> int:
    if cycle_length_years <= 0 or cycle_length_years > 1:
        raise ClinicalLinkageError(
            "Clinical-model linkage currently requires a cycle length greater than 0 and no longer than 1 year."
        )
    reciprocal = 1.0 / cycle_length_years
    rounded = int(round(reciprocal))
    if rounded < 1 or not isclose(reciprocal, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ClinicalLinkageError(
            "For annual BIA linkage, the clinical-model cycle length must divide one year exactly "
            "(for example 1, 1/2, 1/4 or 1/12 year)."
        )
    return rounded


def _markov_cost_reward_ids(strategy: MarkovStrategyDefinition) -> set[str]:
    return {
        reward.parameter_id
        for reward in (*strategy.state_rewards, *strategy.transition_rewards)
        if reward.reward_type == "cost"
    }


def _semi_markov_cost_reward_ids(strategy: SemiMarkovStrategyDefinition) -> set[str]:
    return {
        reward.parameter_id
        for reward in (*strategy.state_rewards, *strategy.transition_rewards)
        if reward.reward_type == "cost"
    }


def available_markov_cost_parameters(model: CohortMarkovDefinition) -> tuple[str, ...]:
    return tuple(sorted(set().union(*(_markov_cost_reward_ids(strategy) for strategy in model.strategies))))


def available_semi_markov_cost_parameters(model: SemiMarkovDefinition) -> tuple[str, ...]:
    return tuple(sorted(set().union(*(_semi_markov_cost_reward_ids(strategy) for strategy in model.strategies))))


def _validate_selected_parameters(
    parameters: Sequence[Parameter],
    referenced_cost_ids: set[str],
    selected_cost_parameter_ids: Sequence[str],
) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(item).strip() for item in selected_cost_parameter_ids if str(item).strip()))
    if not selected:
        raise ClinicalLinkageError(
            "Select at least one clinical cost parameter to import. This explicit selection prevents double counting."
        )
    unknown = set(selected) - referenced_cost_ids
    if unknown:
        raise ClinicalLinkageError(
            "Selected clinical cost parameter(s) are not used as cost rewards in the model: "
            + ", ".join(sorted(unknown))
        )
    parameter_map = {parameter.id: parameter for parameter in parameters}
    missing = set(selected) - set(parameter_map)
    if missing:
        raise ClinicalLinkageError(
            "Selected clinical cost parameter(s) are undefined: " + ", ".join(sorted(missing))
        )
    non_cost = [pid for pid in selected if parameter_map[pid].category != "cost"]
    if non_cost:
        raise ClinicalLinkageError(
            "Only cost parameters can be linked to BIA: " + ", ".join(sorted(non_cost))
        )
    return selected


def _projection_overrides(
    referenced_cost_ids: set[str],
    selected_cost_parameter_ids: Sequence[str],
    base_overrides: Mapping[str, float] | None,
) -> dict[str, float]:
    selected = set(selected_cost_parameter_ids)
    overrides = dict(base_overrides or {})
    for parameter_id in referenced_cost_ids:
        if parameter_id not in selected:
            overrides[parameter_id] = 0.0
    return overrides


def _annualise_cycle_costs(
    cycle_costs: Sequence[float],
    *,
    cycle_length_years: float,
    horizon_years: int,
) -> tuple[float, ...]:
    if horizon_years < 1:
        raise ClinicalLinkageError("BIA linkage requires at least one budget year.")
    cycles_per_year = _cycles_per_year(cycle_length_years)
    required_cycles = horizon_years * cycles_per_year
    padded = [float(value) for value in cycle_costs[:required_cycles]]
    if len(padded) < required_cycles:
        padded.extend([0.0] * (required_cycles - len(padded)))
    return tuple(
        sum(padded[index * cycles_per_year : (index + 1) * cycles_per_year])
        for index in range(horizon_years)
    )


def project_markov_cost_profiles(
    model: CohortMarkovDefinition,
    parameters: Sequence[Parameter],
    *,
    selected_cost_parameter_ids: Sequence[str],
    horizon_years: int,
    included_cost_bearers: Sequence[str] | None = None,
    overrides: Mapping[str, float] | None = None,
) -> tuple[ClinicalCostProfile, ...]:
    cycles_per_year = _cycles_per_year(model.cycle_length_years)
    if model.max_cycles < horizon_years * cycles_per_year:
        raise ClinicalLinkageError(
            "The clinical model horizon is shorter than the requested BIA clinical-cost horizon. "
            "Extend the clinical model rather than silently assuming zero cost beyond its horizon."
        )
    referenced = set(available_markov_cost_parameters(model))
    selected = _validate_selected_parameters(parameters, referenced, selected_cost_parameter_ids)
    run = run_cohort_markov(
        model,
        parameters,
        overrides=_projection_overrides(referenced, selected, overrides),
        included_cost_bearers=included_cost_bearers,
        cost_discount_rate=0.0,
        outcome_discount_rate=0.0,
    )
    return tuple(
        ClinicalCostProfile(
            strategy_id=result.strategy_id,
            strategy_name=result.label,
            annual_cost_per_patient=_annualise_cycle_costs(
                result.cycle_costs,
                cycle_length_years=model.cycle_length_years,
                horizon_years=horizon_years,
            ),
            source_model_type="cohort_markov",
            included_parameter_ids=selected,
        )
        for result in run.strategies
    )


def project_semi_markov_cost_profiles(
    model: SemiMarkovDefinition,
    parameters: Sequence[Parameter],
    *,
    selected_cost_parameter_ids: Sequence[str],
    horizon_years: int,
    included_cost_bearers: Sequence[str] | None = None,
    overrides: Mapping[str, float] | None = None,
) -> tuple[ClinicalCostProfile, ...]:
    cycles_per_year = _cycles_per_year(model.cycle_length_years)
    if model.max_cycles < horizon_years * cycles_per_year:
        raise ClinicalLinkageError(
            "The clinical model horizon is shorter than the requested BIA clinical-cost horizon. "
            "Extend the clinical model rather than silently assuming zero cost beyond its horizon."
        )
    referenced = set(available_semi_markov_cost_parameters(model))
    selected = _validate_selected_parameters(parameters, referenced, selected_cost_parameter_ids)
    run = run_semi_markov(
        model,
        parameters,
        overrides=_projection_overrides(referenced, selected, overrides),
        included_cost_bearers=included_cost_bearers,
        cost_discount_rate=0.0,
        outcome_discount_rate=0.0,
    )
    return tuple(
        ClinicalCostProfile(
            strategy_id=result.strategy_id,
            strategy_name=result.label,
            annual_cost_per_patient=_annualise_cycle_costs(
                result.cycle_costs,
                cycle_length_years=model.cycle_length_years,
                horizon_years=horizon_years,
            ),
            source_model_type="semi_markov",
            included_parameter_ids=selected,
        )
        for result in run.strategies
    )


def run_linked_budget_impact(
    definition: BudgetImpactDefinition,
    clinical_profiles: Sequence[ClinicalCostProfile],
    *,
    intervention_to_strategy: Mapping[str, str],
    linked_category: str = "disease_management",
    population_basis: str = "new_treatment_starts",
) -> LinkedBudgetImpactRunResult:
    """Run BIA with condition-related costs supplied by clinical trajectories.

    The annual BIA population is interpreted as annual *new treatment starts*.
    Each initiation cohort retains its assigned intervention/clinical strategy and
    contributes Year-1, Year-2, ... clinical costs in subsequent budget years.

    Direct costs in ``linked_category`` are removed from the BIA before the
    clinical cost contribution is added. This replacement rule is deliberate and
    is the primary double-counting safeguard.
    """
    if population_basis != "new_treatment_starts":
        raise ClinicalLinkageError(
            "Clinical trajectory linkage currently supports only an explicit annual new-treatment-start cohort basis."
        )
    if linked_category not in definition.included_cost_categories:
        raise ClinicalLinkageError(
            f"Linked category '{linked_category}' must be included in the BIA so it can be replaced transparently."
        )

    profile_map = {profile.strategy_id: profile for profile in clinical_profiles}
    if len(profile_map) != len(tuple(clinical_profiles)):
        raise ClinicalLinkageError("Clinical strategy ids must be unique.")

    intervention_ids = {item.id for item in definition.interventions}
    if set(intervention_to_strategy) != intervention_ids:
        missing = intervention_ids - set(intervention_to_strategy)
        extra = set(intervention_to_strategy) - intervention_ids
        detail = []
        if missing:
            detail.append("missing mappings: " + ", ".join(sorted(missing)))
        if extra:
            detail.append("unknown mappings: " + ", ".join(sorted(extra)))
        raise ClinicalLinkageError("Every BIA intervention must map to one clinical strategy (" + "; ".join(detail) + ").")
    missing_profiles = set(intervention_to_strategy.values()) - set(profile_map)
    if missing_profiles:
        raise ClinicalLinkageError(
            "Clinical strategy mapping references unavailable profile(s): "
            + ", ".join(sorted(missing_profiles))
        )

    horizon = max(row.year for row in definition.population)
    for intervention_id, strategy_id in intervention_to_strategy.items():
        if len(profile_map[strategy_id].annual_cost_per_patient) < horizon:
            raise ClinicalLinkageError(
                f"Clinical profile '{strategy_id}' is shorter than the BIA horizon for intervention '{intervention_id}'."
            )

    direct_definition = replace(
        definition,
        included_cost_categories=tuple(
            category for category in definition.included_cost_categories if category != linked_category
        ),
    )
    try:
        direct_run = run_budget_impact(direct_definition)
    except BudgetImpactValidationError as exc:
        raise ClinicalLinkageError(str(exc)) from exc

    population = {row.year: row for row in definition.population}
    shares = {
        (row.scenario, row.year, row.intervention_id): row.share
        for row in definition.treatment_mix
    }
    clinical_rows: list[ClinicalCohortBudgetRow] = []
    clinical_totals: dict[tuple[str, int], float] = {
        (scenario, year): 0.0
        for scenario in ("current", "future")
        for year in population
    }

    for scenario in ("current", "future"):
        for initiation_year in sorted(population):
            starts_base = population[initiation_year].eligible_population
            for intervention in definition.interventions:
                treatment_starts = starts_base * shares[(scenario, initiation_year, intervention.id)]
                strategy_id = intervention_to_strategy[intervention.id]
                profile = profile_map[strategy_id]
                for budget_year in sorted(population):
                    if budget_year < initiation_year:
                        continue
                    year_since = budget_year - initiation_year + 1
                    cost_per_patient = profile.annual_cost_per_patient[year_since - 1]
                    total = treatment_starts * cost_per_patient
                    clinical_totals[(scenario, budget_year)] += total
                    clinical_rows.append(
                        ClinicalCohortBudgetRow(
                            scenario=scenario,
                            budget_year=budget_year,
                            intervention_id=intervention.id,
                            strategy_id=strategy_id,
                            initiation_year=initiation_year,
                            year_since_initiation=year_since,
                            treatment_starts=treatment_starts,
                            cost_per_patient=cost_per_patient,
                            total_clinical_cost=total,
                        )
                    )

    direct_by_year = {row.year: row for row in direct_run.years}
    results: list[LinkedBudgetImpactYearResult] = []
    cumulative = 0.0
    for year in sorted(population):
        direct = direct_by_year[year]
        clinical_current = clinical_totals[("current", year)]
        clinical_future = clinical_totals[("future", year)]
        current_cost = direct.current_cost + clinical_current
        future_cost = direct.future_cost + clinical_future
        net = future_cost - current_cost
        cumulative += net
        covered = population[year].covered_lives
        pmpm = None if covered is None else net / (covered * 12.0)
        results.append(
            LinkedBudgetImpactYearResult(
                year=year,
                eligible_population=population[year].eligible_population,
                covered_lives=covered,
                direct_current_cost=direct.current_cost,
                direct_future_cost=direct.future_cost,
                clinical_current_cost=clinical_current,
                clinical_future_cost=clinical_future,
                current_cost=current_cost,
                future_cost=future_cost,
                net_budget_impact=net,
                cumulative_budget_impact=cumulative,
                pmpm_budget_impact=pmpm,
            )
        )

    return LinkedBudgetImpactRunResult(
        years=tuple(results),
        clinical_rows=tuple(clinical_rows),
        direct_run=direct_run,
        linked_category=linked_category,
    )
