"""Evidence-informed methodological profiles for Budget Impact Analysis (BIA)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetImpactProfile:
    code: str
    name: str
    perspective: str
    default_horizon_years: int
    minimum_horizon_years: int
    maximum_horizon_years: int
    default_currency: str
    discount_costs: bool
    population_guidance: str
    uncertainty_guidance: str
    reporting_guidance: str
    source_title: str
    source_url: str
    notes: str = ""


ISPOR_BIA = BudgetImpactProfile(
    code="ISPOR_BIA",
    name="ISPOR Budget Impact Good Practice",
    perspective="Specific healthcare budget holder / payer",
    default_horizon_years=3,
    minimum_horizon_years=1,
    maximum_horizon_years=5,
    default_currency="GBP",
    discount_costs=False,
    population_guidance=(
        "Use the population relevant to the specific decision maker; allow the eligible population, access, and treatment mix to change over time."
    ),
    uncertainty_guidance=(
        "Use scenario and sensitivity analyses that reflect the information needs and plausible assumptions of the decision maker."
    ),
    reporting_guidance=(
        "Report current and future treatment mixes, annual budget consequences, and sufficiently disaggregated inputs/calculations to support replication."
    ),
    source_title="ISPOR Principles of Good Practice for Budget Impact Analysis II (2014)",
    source_url="https://www.ispor.org/heor-resources/good-practices/article/principles-of-good-practice-for-budget-impact-analysis-ii",
    notes=(
        "ISPOR recommends choosing a horizon relevant to the budget holder; 1–5 years is common rather than a universal mandatory horizon."
    ),
)


INDIA_BIA_2021 = BudgetImpactProfile(
    code="INDIA_BIA_2021",
    name="India National BIA Guidelines (2021)",
    perspective="Healthcare payer / budget holder; multi-payer and single-payer scenarios",
    default_horizon_years=4,
    minimum_horizon_years=1,
    maximum_horizon_years=4,
    default_currency="INR",
    discount_costs=False,
    population_guidance=(
        "Use a top-down eligible-population approach reflecting epidemiology, indication, clinical guidance, access and care-seeking; the BIA population is open over time."
    ),
    uncertainty_guidance=(
        "Conduct deterministic sensitivity analysis for important parameters and scenario analysis for uncertain or structural assumptions."
    ),
    reporting_guidance=(
        "Present total and disaggregated budget impact year by year, by budget holder/organisational level and by resource type, including natural resource units where relevant."
    ),
    source_title="National Methodological Guidelines to Conduct Budget Impact Analysis for Health Technology Assessment in India (2021)",
    source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8238667/",
    notes=(
        "The guidance recommends no discounting for BIA. Prices should be expressed at the relevant current price level, with inflation/price assumptions documented explicitly."
    ),
)


NICE_RESOURCE_IMPACT_2026 = BudgetImpactProfile(
    code="NICE_RESOURCE_2026",
    name="NICE Resource Impact framing (2026)",
    perspective="NHS / relevant commissioning budget holder",
    default_horizon_years=3,
    minimum_horizon_years=1,
    maximum_horizon_years=3,
    default_currency="GBP",
    discount_costs=False,
    population_guidance=(
        "Estimate the population and uptake relevant to implementation of NICE guidance and make local variation in population/resource use explicit."
    ),
    uncertainty_guidance=(
        "Expose implementation, uptake and resource assumptions so commissioners can explore local variation."
    ),
    reporting_guidance=(
        "Present annual resource consequences and costs/savings in a form useful for implementation and service planning."
    ),
    source_title="NICE Assessing the resource impact of NICE guidance (updated 2026)",
    source_url="https://www.nice.org.uk/About/What-we-do/Into-practice/resource-impact-assessment/",
    notes=(
        "This is labelled as resource-impact framing rather than a complete standalone BIA reference case. NICE states that its resource-impact tools moved to 3 future years in January 2026."
    ),
)


BUDGET_IMPACT_PROFILES = {
    profile.code: profile
    for profile in (ISPOR_BIA, INDIA_BIA_2021, NICE_RESOURCE_IMPACT_2026)
}
