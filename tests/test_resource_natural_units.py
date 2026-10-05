from model.budget_impact import BudgetIntervention, PopulationYear, TreatmentMixShare
from model.resource_capacity import (
    AnnualResourceCapacity,
    ResourceCapacityDefinition,
    ResourceDefinition,
    ResourceRequirement,
    run_resource_capacity_plan,
)


def test_capacity_engine_is_unit_agnostic_for_physical_resources():
    interventions = (
        BudgetIntervention("current", "Current"),
        BudgetIntervention("new", "New"),
    )
    population = (PopulationYear(1, 100.0),)
    mix = (
        TreatmentMixShare("current", 1, "current", 1.0),
        TreatmentMixShare("current", 1, "new", 0.0),
        TreatmentMixShare("future", 1, "current", 0.0),
        TreatmentMixShare("future", 1, "new", 1.0),
    )
    resources = (
        ResourceDefinition("beds", "Inpatient bed-days", "bed-days", "inpatient"),
        ResourceDefinition("syringes", "Syringes", "syringes", "consumable"),
        ResourceDefinition("vials", "Medicine vials", "vials", "pharmacy"),
        ResourceDefinition("ambulance", "Ambulance trips", "trips", "other"),
        ResourceDefinition("oxygen", "Oxygen supply", "litres", "consumable"),
        ResourceDefinition("blood", "Blood products", "units", "consumable"),
    )
    requirements = tuple(
        ResourceRequirement("new", resource.id, 1, 2.0)
        for resource in resources
    )
    capacities = tuple(
        AnnualResourceCapacity(resource.id, 1, 500.0)
        for resource in resources
    )
    result = run_resource_capacity_plan(
        ResourceCapacityDefinition(
            interventions=interventions,
            population=population,
            treatment_mix=mix,
            resources=resources,
            requirements=requirements,
            capacities=capacities,
        )
    )
    future = {row.resource_id: row for row in result.scenario_rows if row.scenario == "future"}
    assert set(future) == {resource.id for resource in resources}
    for resource in resources:
        assert future[resource.id].required_units == 200.0
        assert future[resource.id].unit == resource.unit
