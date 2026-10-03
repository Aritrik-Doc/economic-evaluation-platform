import pytest

from model.budget_impact import BudgetIntervention, PopulationYear, TreatmentMixShare
from model.resource_capacity import (
    AnnualResourceCapacity,
    ResourceCapacityDefinition,
    ResourceCapacityValidationError,
    ResourceDefinition,
    ResourceRequirement,
    apply_capacity_expansion,
    run_resource_capacity_plan,
)


def _mix(years=(1, 2)):
    rows = []
    for year in years:
        rows.extend(
            [
                TreatmentMixShare("current", year, "standard", 1.0),
                TreatmentMixShare("current", year, "new", 0.0),
                TreatmentMixShare("future", year, "standard", 0.5),
                TreatmentMixShare("future", year, "new", 0.5),
            ]
        )
    return tuple(rows)


def _base_definition(*, demand_basis="annual_treated_population"):
    interventions = (
        BudgetIntervention("standard", "Standard care"),
        BudgetIntervention("new", "New treatment"),
    )
    population = (PopulationYear(1, 100.0), PopulationYear(2, 100.0))
    resources = (
        ResourceDefinition("chair", "Infusion-chair time", "chair-hours", "facility"),
    )
    requirements = (
        ResourceRequirement("standard", "chair", 1, 1.0),
        ResourceRequirement("standard", "chair", 2, 1.0),
        ResourceRequirement("new", "chair", 1, 3.0),
        ResourceRequirement("new", "chair", 2, 3.0),
    )
    capacities = (
        AnnualResourceCapacity("chair", 1, 250.0, 50.0),
        AnnualResourceCapacity("chair", 2, 250.0, 50.0),
    )
    return ResourceCapacityDefinition(
        interventions=interventions,
        population=population,
        treatment_mix=_mix(),
        resources=resources,
        requirements=requirements,
        capacities=capacities,
        demand_basis=demand_basis,
    )


def test_annual_treated_population_calculates_current_future_demand_and_capacity():
    result = run_resource_capacity_plan(_base_definition())
    year1 = next(row for row in result.comparison_rows if row.year == 1 and row.resource_id == "chair")
    assert year1.current_required_units == pytest.approx(100.0)
    assert year1.future_required_units == pytest.approx(200.0)
    assert year1.net_change_units == pytest.approx(100.0)
    assert year1.available_capacity == pytest.approx(200.0)
    assert year1.future_utilization_ratio == pytest.approx(1.0)
    assert year1.future_shortfall == pytest.approx(0.0)
    assert year1.future_headroom == pytest.approx(0.0)


def test_committed_other_demand_reduces_capacity_available_to_modelled_population():
    definition = _base_definition()
    capacities = (
        AnnualResourceCapacity("chair", 1, 250.0, 100.0),
        AnnualResourceCapacity("chair", 2, 250.0, 100.0),
    )
    definition = ResourceCapacityDefinition(
        interventions=definition.interventions,
        population=definition.population,
        treatment_mix=definition.treatment_mix,
        resources=definition.resources,
        requirements=definition.requirements,
        capacities=capacities,
    )
    result = run_resource_capacity_plan(definition)
    year1 = next(row for row in result.comparison_rows if row.year == 1)
    assert year1.available_capacity == pytest.approx(150.0)
    assert year1.future_shortfall == pytest.approx(50.0)
    assert year1.future_utilization_ratio == pytest.approx(200.0 / 150.0)


def test_new_treatment_start_basis_stacks_longitudinal_resource_profiles():
    interventions = (
        BudgetIntervention("standard", "Standard care"),
        BudgetIntervention("new", "New treatment"),
    )
    population = (PopulationYear(1, 100.0), PopulationYear(2, 100.0), PopulationYear(3, 100.0))
    mix = []
    for year in (1, 2, 3):
        mix.extend(
            [
                TreatmentMixShare("current", year, "standard", 1.0),
                TreatmentMixShare("current", year, "new", 0.0),
                TreatmentMixShare("future", year, "standard", 0.0),
                TreatmentMixShare("future", year, "new", 1.0),
            ]
        )
    resources = (ResourceDefinition("nurse", "Specialist nurse time", "hours", "workforce"),)
    requirements = (
        ResourceRequirement("standard", "nurse", 1, 1.0),
        ResourceRequirement("new", "nurse", 1, 2.0),
        ResourceRequirement("new", "nurse", 2, 0.5),
        ResourceRequirement("new", "nurse", 3, 0.25),
    )
    capacities = tuple(AnnualResourceCapacity("nurse", year, 1000.0) for year in (1, 2, 3))
    definition = ResourceCapacityDefinition(
        interventions=interventions,
        population=population,
        treatment_mix=tuple(mix),
        resources=resources,
        requirements=requirements,
        capacities=capacities,
        demand_basis="new_treatment_starts",
    )
    result = run_resource_capacity_plan(definition)
    future = {row.year: row.future_required_units for row in result.comparison_rows}
    assert future[1] == pytest.approx(200.0)
    assert future[2] == pytest.approx(250.0)  # 100 new starts * 2 + prior cohort * 0.5
    assert future[3] == pytest.approx(275.0)  # 200 + 50 + 25


def test_zero_available_capacity_reports_shortfall_without_division_error():
    definition = _base_definition()
    capacities = (
        AnnualResourceCapacity("chair", 1, 50.0, 50.0),
        AnnualResourceCapacity("chair", 2, 50.0, 50.0),
    )
    definition = ResourceCapacityDefinition(
        interventions=definition.interventions,
        population=definition.population,
        treatment_mix=definition.treatment_mix,
        resources=definition.resources,
        requirements=definition.requirements,
        capacities=capacities,
    )
    result = run_resource_capacity_plan(definition)
    row = next(item for item in result.comparison_rows if item.year == 1)
    assert row.future_utilization_ratio is None
    assert row.future_shortfall == pytest.approx(200.0)


def test_capacity_must_be_supplied_for_every_resource_year():
    definition = _base_definition()
    broken = ResourceCapacityDefinition(
        interventions=definition.interventions,
        population=definition.population,
        treatment_mix=definition.treatment_mix,
        resources=definition.resources,
        requirements=definition.requirements,
        capacities=(AnnualResourceCapacity("chair", 1, 250.0),),
    )
    with pytest.raises(ResourceCapacityValidationError, match="missing"):
        run_resource_capacity_plan(broken)


def test_treatment_mix_is_not_silently_normalised():
    definition = _base_definition()
    bad_mix = list(definition.treatment_mix)
    bad_mix[2] = TreatmentMixShare("future", 1, "standard", 0.6)
    broken = ResourceCapacityDefinition(
        interventions=definition.interventions,
        population=definition.population,
        treatment_mix=tuple(bad_mix),
        resources=definition.resources,
        requirements=definition.requirements,
        capacities=definition.capacities,
    )
    with pytest.raises(ResourceCapacityValidationError, match="sum"):
        run_resource_capacity_plan(broken)


def test_capacity_expansion_applies_from_selected_year_and_preserves_other_demand():
    definition = _base_definition()
    scenario = apply_capacity_expansion(
        definition,
        resource_id="chair",
        from_year=2,
        capacity_multiplier=1.2,
        additional_capacity=10.0,
    )
    capacities = {(row.resource_id, row.year): row for row in scenario.capacities}
    assert capacities[("chair", 1)].total_capacity == pytest.approx(250.0)
    assert capacities[("chair", 2)].total_capacity == pytest.approx(310.0)
    assert capacities[("chair", 2)].committed_other_demand == pytest.approx(50.0)


def test_resource_requirements_cannot_be_negative():
    with pytest.raises(ValueError, match="non-negative"):
        ResourceRequirement("new", "chair", 1, -1.0)
