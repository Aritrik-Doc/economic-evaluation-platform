from model.budget_impact import (
    AnnualCostInput,
    BudgetImpactDefinition,
    BudgetIntervention,
    PopulationYear,
    TreatmentMixShare,
    run_budget_impact,
)
from model.economics import Strategy, fully_incremental_analysis
from model.policy_interpretation import (
    combined_policy_summary,
    interpret_budget_impact,
    interpret_capacity,
    interpret_cost_effectiveness,
)
from model.resource_capacity import (
    AnnualResourceCapacity,
    ResourceCapacityDefinition,
    ResourceDefinition,
    ResourceRequirement,
    run_resource_capacity_plan,
)


def test_cost_effectiveness_interpretation_is_threshold_explicit_and_non_prescriptive():
    result = fully_incremental_analysis(
        [
            Strategy("Standard", 10000, 5.0),
            Strategy("New", 12000, 5.2),
        ],
        30000,
    )
    interpretation = interpret_cost_effectiveness(
        result,
        outcome_label="QALY",
        currency_symbol="£",
        psa_probability_by_strategy={"Standard": 0.25, "New": 0.75},
    )
    text = " ".join(statement.text for statement in interpretation.statements).lower()
    assert "£30,000" in text
    assert "highest net monetary benefit" in text
    assert "75.0%" in text
    assert "adopt" not in text
    assert "recommend" not in text


def _bia_result():
    interventions = (
        BudgetIntervention("current", "Current"),
        BudgetIntervention("new", "New"),
    )
    population = (PopulationYear(1, 1000), PopulationYear(2, 1000))
    mix = (
        TreatmentMixShare("current", 1, "current", 1.0),
        TreatmentMixShare("current", 1, "new", 0.0),
        TreatmentMixShare("future", 1, "current", 0.5),
        TreatmentMixShare("future", 1, "new", 0.5),
        TreatmentMixShare("current", 2, "current", 1.0),
        TreatmentMixShare("current", 2, "new", 0.0),
        TreatmentMixShare("future", 2, "current", 0.25),
        TreatmentMixShare("future", 2, "new", 0.75),
    )
    costs = tuple(
        AnnualCostInput(iid, year, "acquisition", cost)
        for year in (1, 2)
        for iid, cost in (("current", 100.0), ("new", 200.0))
    )
    return run_budget_impact(
        BudgetImpactDefinition(
            interventions=interventions,
            population=population,
            treatment_mix=mix,
            costs=costs,
            included_cost_categories=("acquisition",),
        )
    )


def test_budget_interpretation_reports_cumulative_and_peak_impact():
    interpretation = interpret_budget_impact(
        _bia_result(), currency_symbol="£", category_labels={"acquisition": "Acquisition"}
    )
    text = " ".join(statement.text for statement in interpretation.statements)
    assert "additional cumulative budget requirement" in text
    assert "largest annual additional requirement" in text
    assert "Acquisition" in text


def _capacity_result():
    interventions = (
        BudgetIntervention("current", "Current"),
        BudgetIntervention("new", "New"),
    )
    population = (PopulationYear(1, 100), PopulationYear(2, 100))
    mix = (
        TreatmentMixShare("current", 1, "current", 1.0),
        TreatmentMixShare("current", 1, "new", 0.0),
        TreatmentMixShare("future", 1, "current", 0.0),
        TreatmentMixShare("future", 1, "new", 1.0),
        TreatmentMixShare("current", 2, "current", 1.0),
        TreatmentMixShare("current", 2, "new", 0.0),
        TreatmentMixShare("future", 2, "current", 0.0),
        TreatmentMixShare("future", 2, "new", 1.0),
    )
    resource = ResourceDefinition("chair", "Infusion chair time", "hours", "facility")
    requirements = (
        ResourceRequirement("current", "chair", 1, 1.0),
        ResourceRequirement("current", "chair", 2, 1.0),
        ResourceRequirement("new", "chair", 1, 3.0),
        ResourceRequirement("new", "chair", 2, 3.0),
    )
    capacities = (
        AnnualResourceCapacity("chair", 1, 250.0, 0.0),
        AnnualResourceCapacity("chair", 2, 250.0, 0.0),
    )
    return run_resource_capacity_plan(
        ResourceCapacityDefinition(
            interventions=interventions,
            population=population,
            treatment_mix=mix,
            resources=(resource,),
            requirements=requirements,
            capacities=capacities,
        )
    )


def test_capacity_interpretation_identifies_shortfall_without_prescribing_action():
    interpretation = interpret_capacity(_capacity_result())
    text = " ".join(statement.text for statement in interpretation.statements).lower()
    assert "shortfall" in text
    assert "infusion chair time" in text
    assert "adopt" not in text
    assert "expand capacity" not in text


def test_combined_summary_orders_value_affordability_feasibility():
    cea = interpret_cost_effectiveness(
        fully_incremental_analysis([Strategy("A", 100, 1.0), Strategy("B", 120, 1.1)], 500),
        outcome_label="QALY",
        currency_symbol="£",
    )
    bia = interpret_budget_impact(_bia_result(), currency_symbol="£")
    capacity = interpret_capacity(_capacity_result())
    headlines = combined_policy_summary(capacity, bia, cea)
    assert len(headlines) == 3
    assert headlines[0].title == "Value-for-money finding"
    assert headlines[1].title == "Affordability finding"
    assert headlines[2].title == "Implementation feasibility"
