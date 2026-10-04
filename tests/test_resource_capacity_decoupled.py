import pytest

from model.population_uptake import PopulationOption, PopulationYear, TreatmentMixShare
from model.resource_capacity import (
    AnnualResourceCapacity,
    ResourceCapacityDefinition,
    ResourceDefinition,
    ResourceRequirement,
    run_resource_capacity_plan,
)


def test_capacity_engine_accepts_shared_population_context_without_bia_definition():
    options = (PopulationOption("old", "Old care"), PopulationOption("new", "New care"))
    population = (PopulationYear(1, 100), PopulationYear(2, 120))
    mix = (
        TreatmentMixShare("current", 1, "old", 1.0),
        TreatmentMixShare("current", 1, "new", 0.0),
        TreatmentMixShare("future", 1, "old", 0.5),
        TreatmentMixShare("future", 1, "new", 0.5),
        TreatmentMixShare("current", 2, "old", 1.0),
        TreatmentMixShare("current", 2, "new", 0.0),
        TreatmentMixShare("future", 2, "old", 0.25),
        TreatmentMixShare("future", 2, "new", 0.75),
    )
    definition = ResourceCapacityDefinition(
        interventions=options,
        population=population,
        treatment_mix=mix,
        resources=(ResourceDefinition("chair", "Infusion chair", "hours", "facility"),),
        requirements=(
            ResourceRequirement("old", "chair", 1, 1.0),
            ResourceRequirement("old", "chair", 2, 1.0),
            ResourceRequirement("new", "chair", 1, 3.0),
            ResourceRequirement("new", "chair", 2, 3.0),
        ),
        capacities=(
            AnnualResourceCapacity("chair", 1, 200),
            AnnualResourceCapacity("chair", 2, 200),
        ),
        demand_basis="annual_treated_population",
    )
    result = run_resource_capacity_plan(definition)
    future = {(row.year, row.resource_id): row for row in result.scenario_rows if row.scenario == "future"}
    assert future[(1, "chair")].required_units == pytest.approx(200.0)
    assert future[(1, "chair")].shortfall == pytest.approx(0.0)
    assert future[(2, "chair")].required_units == pytest.approx(300.0)
    assert future[(2, "chair")].shortfall == pytest.approx(100.0)
