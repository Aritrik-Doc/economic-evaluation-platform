import pytest

from model.budget_impact import (
    AnnualCostInput,
    BudgetImpactDefinition,
    BudgetIntervention,
    PopulationYear as BIAPopulationYear,
    TreatmentMixShare as BIATreatmentMixShare,
    run_budget_impact,
)
from model.population_projection import project_direct_population
from model.population_uptake import (
    PopulationOption,
    PopulationUptakeDefinition,
    PopulationYear,
    TreatmentMixShare,
    run_population_uptake,
)
from model.resource_capacity import (
    AnnualResourceCapacity,
    ResourceCapacityDefinition,
    ResourceDefinition,
    ResourceRequirement,
    run_resource_capacity_plan,
)


def test_population_bia_capacity_pipeline_is_numerically_consistent():
    projection = project_direct_population(
        eligible_start=1000.0,
        eligible_growth_rate=0.10,
        horizon_years=2,
        covered_lives_start=10000.0,
        covered_lives_growth_rate=0.05,
    )
    assert [row.eligible_population for row in projection] == pytest.approx([1000.0, 1100.0])
    assert [row.covered_lives for row in projection] == pytest.approx([10000.0, 10500.0])

    options = (PopulationOption("a", "Current option"), PopulationOption("b", "New option"))
    population = tuple(
        PopulationYear(row.year, row.eligible_population, row.covered_lives)
        for row in projection
    )
    shared_mix = tuple(
        TreatmentMixShare(scenario, year, option, share)
        for year in (1, 2)
        for scenario, values in (
            ("current", {"a": 1.0, "b": 0.0}),
            ("future", {"a": 0.5, "b": 0.5}),
        )
        for option, share in values.items()
    )
    shared_result = run_population_uptake(
        PopulationUptakeDefinition(
            options=options,
            population=population,
            treatment_mix=shared_mix,
        )
    )
    assert shared_result.treated_people("future", 2, "b") == pytest.approx(550.0)

    interventions = (
        BudgetIntervention("a", "Current option"),
        BudgetIntervention("b", "New option"),
    )
    bia_population = tuple(
        BIAPopulationYear(row.year, row.eligible_population, row.covered_lives)
        for row in projection
    )
    bia_mix = tuple(
        BIATreatmentMixShare(row.scenario, row.year, row.intervention_id, row.share)
        for row in shared_mix
    )
    costs = tuple(
        AnnualCostInput(intervention_id, year, "acquisition", cost)
        for year in (1, 2)
        for intervention_id, cost in (("a", 100.0), ("b", 200.0))
    )
    bia_definition = BudgetImpactDefinition(
        interventions=interventions,
        population=bia_population,
        treatment_mix=bia_mix,
        costs=costs,
        included_cost_categories=("acquisition",),
    )
    bia_result = run_budget_impact(bia_definition)

    assert bia_result.years[0].current_cost == pytest.approx(100000.0)
    assert bia_result.years[0].future_cost == pytest.approx(150000.0)
    assert bia_result.years[0].net_budget_impact == pytest.approx(50000.0)
    assert bia_result.years[1].net_budget_impact == pytest.approx(55000.0)
    assert bia_result.cumulative_budget_impact == pytest.approx(105000.0)
    assert bia_result.years[0].pmpm_budget_impact == pytest.approx(50000.0 / (10000.0 * 12.0))

    resource = ResourceDefinition("beds", "Inpatient bed-days", "bed-days", "inpatient")
    requirements = tuple(
        ResourceRequirement(intervention_id, "beds", year, units)
        for year in (1, 2)
        for intervention_id, units in (("a", 1.0), ("b", 2.0))
    )
    capacities = (
        AnnualResourceCapacity("beds", 1, 1300.0),
        AnnualResourceCapacity("beds", 2, 1400.0),
    )
    capacity_result = run_resource_capacity_plan(
        ResourceCapacityDefinition(
            interventions=interventions,
            population=bia_population,
            treatment_mix=bia_mix,
            resources=(resource,),
            requirements=requirements,
            capacities=capacities,
            demand_basis="annual_treated_population",
        )
    )
    by_year = {row.year: row for row in capacity_result.comparison_rows}
    assert by_year[1].current_required_units == pytest.approx(1000.0)
    assert by_year[1].future_required_units == pytest.approx(1500.0)
    assert by_year[1].future_shortfall == pytest.approx(200.0)
    assert by_year[2].future_required_units == pytest.approx(1650.0)
    assert by_year[2].future_shortfall == pytest.approx(250.0)
