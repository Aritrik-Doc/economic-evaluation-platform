"""Guided Budget Impact Analysis workspace."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from model.budget_impact import (
    COST_CATEGORIES,
    AnnualCostInput,
    BudgetImpactDefinition,
    BudgetImpactValidationError,
    BudgetIntervention,
    PopulationYear,
    TreatmentMixShare,
    apply_simple_scenario,
    compound_series,
    run_budget_impact,
)
from model.budget_impact_profiles import BUDGET_IMPACT_PROFILES
from model.currency import CURRENCIES
from model.population_projection import project_direct_population, project_top_down_population
from ui.design_system import coloured_block, status_bar


st.set_page_config(page_title="Budget Impact Analysis", page_icon="💷", layout="wide")
st.title("Budget Impact Analysis")
st.caption("Version 0.15 — population-level affordability with validated cross-page handoffs")

coloured_block(
    "Affordability complements value for money",
    "Budget impact analysis estimates the financial consequences of changing the treatment mix for a defined budget holder and eligible population. It should be interpreted alongside—not as a substitute for—cost-effectiveness and implementation evidence.",
    tone="teal",
    kicker="Budget holder perspective",
)

c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("pages/10_Population_Uptake.py", label="Population & Uptake", icon="👥")
with c2:
    st.page_link("pages/7_BIA_Clinical_Linkage.py", label="Clinical model → BIA linkage", icon="🔗")
with c3:
    st.page_link("pages/8_Resource_Capacity_Planning.py", label="Resource & Capacity Planning", icon="🏥")


CATEGORY_LABELS = {
    "acquisition": "Acquisition / technology",
    "administration": "Administration / procedure",
    "monitoring": "Monitoring / follow-up",
    "adverse_events": "Adverse-event management",
    "disease_management": "Condition-related care",
    "other": "Other budgeted cost / credit",
}


def _slug(text: str) -> str:
    value = "".join(char.lower() if char.isalnum() else "_" for char in text.strip())
    value = "_".join(part for part in value.split("_") if part)
    return value or "intervention"


def _unique_id(label: str, existing: set[str]) -> str:
    base = _slug(label)
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _init_state() -> None:
    if "bia_interventions" not in st.session_state:
        st.session_state.bia_interventions = [
            {"id": "current_treatment", "name": "Current treatment"},
            {"id": "new_intervention", "name": "New intervention"},
        ]
    if "bia_cost_inputs" not in st.session_state:
        st.session_state.bia_cost_inputs = {
            "current_treatment": {
                "acquisition": 1000.0,
                "administration": 100.0,
                "monitoring": 200.0,
                "adverse_events": 50.0,
                "disease_management": 500.0,
                "other": 0.0,
                "annual_change": 0.0,
                "source": "Illustrative input — replace with local payer evidence",
                "rationale": "Replace with documented resource-use and price evidence before decision use.",
            },
            "new_intervention": {
                "acquisition": 1800.0,
                "administration": 80.0,
                "monitoring": 150.0,
                "adverse_events": 30.0,
                "disease_management": 350.0,
                "other": 0.0,
                "annual_change": 0.0,
                "source": "Illustrative input — replace with local payer evidence",
                "rationale": "Replace with documented resource-use and price evidence before decision use.",
            },
        }
    st.session_state.setdefault("bia_population_source", "Illustrative population — replace with local epidemiology / payer data")
    st.session_state.setdefault("bia_population_rationale", "Replace with an evidence-based eligible-population derivation.")
    st.session_state.setdefault("bia_current_mix_source", "Illustrative current mix — replace with local utilisation data")
    st.session_state.setdefault("bia_current_mix_rationale", "Current mix should reflect actual routine care for the budget holder.")
    st.session_state.setdefault("bia_future_mix_source", "Illustrative uptake assumption — replace with forecast evidence")
    st.session_state.setdefault("bia_future_mix_rationale", "Future uptake should reflect access restrictions, implementation and expected substitution.")
    st.session_state.setdefault("bia_population_entry_mode", "Use formula-derived annual values")
    st.session_state.setdefault("bia_context_valid", False)


_init_state()

methods_tab, population_tab, costs_tab, mix_tab, results_tab = st.tabs(
    [
        "1 · Methods",
        "2 · Eligible population",
        "3 · Interventions & costs",
        "4 · Current vs future mix",
        "5 · Results & scenarios",
    ]
)


with methods_tab:
    st.subheader("Budget holder and methodological framing")
    profile_options = [*BUDGET_IMPACT_PROFILES.keys(), "CUSTOM"]
    profile_code = st.selectbox(
        "Budget impact profile",
        profile_options,
        format_func=lambda code: BUDGET_IMPACT_PROFILES[code].name if code in BUDGET_IMPACT_PROFILES else "Custom budget-impact methods",
        key="bia_profile",
    )
    profile = BUDGET_IMPACT_PROFILES.get(profile_code)

    if profile is not None:
        c1, c2, c3 = st.columns(3)
        budget_holder = c1.text_input("Budget holder / payer", value=profile.perspective, key="bia_budget_holder")
        currency_code = c2.selectbox(
            "Analysis currency",
            list(CURRENCIES),
            index=list(CURRENCIES).index(profile.default_currency) if profile.default_currency in CURRENCIES else 0,
            key="bia_currency_profile",
        )
        horizon_years = int(
            c3.number_input(
                "Forecast horizon (years)",
                min_value=profile.minimum_horizon_years,
                max_value=profile.maximum_horizon_years,
                value=profile.default_horizon_years,
                step=1,
                key=f"bia_horizon_{profile.code}",
            )
        )
        st.caption(f"**Source:** {profile.source_title}")
        st.write(profile.population_guidance)
        st.write(profile.uncertainty_guidance)
        st.write(profile.reporting_guidance)
        if profile.notes:
            st.info(profile.notes)
    else:
        c1, c2, c3 = st.columns(3)
        budget_holder = c1.text_input("Budget holder / payer", value="Healthcare payer", key="bia_budget_holder_custom")
        currency_code = c2.selectbox("Analysis currency", list(CURRENCIES), key="bia_currency_custom")
        horizon_years = int(c3.number_input("Forecast horizon (years)", 1, 10, 3, 1, key="bia_horizon_custom"))
        st.warning(
            "Custom BIA methods should document the local payer perspective, budgeting period, eligible-population rules, treatment-mix assumptions, included cost categories and uncertainty approach."
        )

    st.info(
        "The BIA engine reports period-by-period financial consequences and does not discount annual budget flows. If a jurisdiction requires a discounted presentation, show that as an additional local-practice view rather than silently replacing annual cash-flow results."
    )
    included_categories = st.multiselect(
        "Cost categories included in this analysis",
        list(COST_CATEGORIES),
        default=list(COST_CATEGORIES),
        format_func=lambda code: CATEGORY_LABELS[code],
        key="bia_included_categories",
    )
    if not included_categories:
        st.error("Include at least one budgeted cost category.")


with population_tab:
    st.subheader("Eligible population over time")
    st.caption(
        "If the same population/uptake assumptions will also be used for capacity planning, you can define them in Population & Uptake and copy a validated snapshot into BIA."
    )
    population_mode = st.radio(
        "How would you like to estimate the population?",
        ["Top-down funnel", "Direct annual eligible population"],
        horizontal=True,
        key="bia_population_mode",
    )

    if population_mode == "Top-down funnel":
        with st.expander("What each population input changes", expanded=True):
            st.markdown(
                "**Eligible population = covered/catchment population × prevalence × diagnosed/identified × clinically eligible × access/coverage.**\n\n"
                "- **Covered / catchment population:** people served by the payer, programme or service.\n"
                "- **Prevalence / target-condition proportion:** share with the condition or policy-relevant characteristic.\n"
                "- **Diagnosed / identified:** share of affected people known to the system.\n"
                "- **Clinically eligible:** share meeting the intervention criteria.\n"
                "- **Access / coverage:** share expected to reach or be covered by the service.\n"
                "- **Annual covered/catchment growth:** changes the starting population in later years; the same funnel is then applied each year."
            )
        c1, c2, c3 = st.columns(3)
        covered_start = c1.number_input("Covered / catchment population — Year 1", min_value=0.0, value=1_000_000.0, step=10_000.0, key="bia_formula_covered_start")
        prevalence = c2.number_input("Prevalence / target-condition proportion", min_value=0.0, max_value=1.0, value=0.01, format="%.6f", key="bia_formula_prevalence")
        diagnosed = c3.number_input("Diagnosed / identified proportion", min_value=0.0, max_value=1.0, value=0.80, format="%.4f", key="bia_formula_diagnosed")
        c1, c2, c3 = st.columns(3)
        eligible = c1.number_input("Clinically eligible proportion", min_value=0.0, max_value=1.0, value=0.75, format="%.4f", key="bia_formula_eligible")
        access = c2.number_input("Access / covered proportion", min_value=0.0, max_value=1.0, value=0.90, format="%.4f", key="bia_formula_access")
        population_growth_pct = c3.number_input("Annual covered/catchment growth (%)", value=1.0, format="%.3f", key="bia_formula_covered_growth_pct")
        projection = project_top_down_population(
            covered_or_catchment_start=covered_start,
            covered_or_catchment_growth_rate=population_growth_pct / 100.0,
            prevalence=prevalence,
            diagnosed_or_identified=diagnosed,
            clinically_eligible=eligible,
            access_or_coverage=access,
            horizon_years=horizon_years,
        )
    else:
        with st.expander("What each population input changes", expanded=True):
            st.markdown(
                "- **Eligible population — Year 1:** number of people relevant to the budget decision in the first year.\n"
                "- **Eligible-population growth:** changes that eligible population in later years. A value of 2% gives Year 2 = Year 1 × 1.02.\n"
                "- **Covered lives:** optional denominator used for PMPM. It does **not** change eligible population in direct mode.\n"
                "- **Covered-lives growth:** changes only that denominator, allowing the payer population and eligible population to grow at different rates."
            )
        c1, c2 = st.columns(2)
        eligible_start = c1.number_input("Eligible population — Year 1", min_value=0.0, value=10_000.0, step=100.0, key="bia_formula_eligible_start")
        eligible_growth_pct = c2.number_input("Annual eligible-population growth (%)", value=2.0, format="%.3f", key="bia_formula_eligible_growth_pct")
        c1, c2 = st.columns(2)
        covered_start = c1.number_input("Covered lives — Year 1 (0 = not supplied)", min_value=0.0, value=0.0, step=1000.0, key="bia_formula_covered_direct")
        covered_growth_pct = c2.number_input("Annual covered-lives growth (%)", value=0.0, format="%.3f", key="bia_formula_covered_growth_pct_direct", disabled=covered_start <= 0)
        projection = project_direct_population(
            eligible_start=eligible_start,
            eligible_growth_rate=eligible_growth_pct / 100.0,
            horizon_years=horizon_years,
            covered_lives_start=covered_start if covered_start > 0 else None,
            covered_lives_growth_rate=covered_growth_pct / 100.0,
        )

    projection_df = pd.DataFrame(
        [
            {
                "Year": row.year,
                "Formula-derived eligible population": row.eligible_population,
                "Formula-derived covered lives": row.covered_lives,
            }
            for row in projection
        ]
    )
    population_entry_mode = st.radio(
        "How should the annual values be used?",
        ["Use formula-derived annual values", "Edit annual values manually"],
        horizontal=True,
        key="bia_population_entry_mode",
        help="Formula-derived mode always follows the current drivers above. Manual mode preserves year-specific overrides until you edit or reset them.",
    )

    population_rows: list[PopulationYear] = []
    if population_entry_mode == "Use formula-derived annual values":
        st.success("Live projection: changes to the population drivers above immediately update the annual values used in the BIA result and downstream validated handoffs.")
        st.dataframe(projection_df, use_container_width=True, hide_index=True)
        for row in projection:
            try:
                population_rows.append(PopulationYear(row.year, row.eligible_population, row.covered_lives))
            except ValueError as exc:
                st.error(f"Year {row.year}: {exc}")
    else:
        snapshot_basis = st.session_state.get("bia_population_uptake_snapshot_basis")
        if snapshot_basis:
            basis_text = "new treatment starts" if snapshot_basis == "new_treatment_starts" else "annual eligible / treated population"
            st.info(f"This BIA received a Population & Uptake snapshot whose declared population basis is **{basis_text}**. The annual values remain editable here as a separate BIA snapshot.")
        st.warning("Manual override is active. Formula changes update the preview but do not overwrite the year-specific BIA values unless you reset them.")
        st.dataframe(projection_df, use_container_width=True, hide_index=True)
        if st.button("Reset manual annual BIA values from the current formula", key="bia_reset_manual_population"):
            for index, row in enumerate(projection):
                st.session_state[f"bia_eligible_{horizon_years}_{index}"] = float(row.eligible_population)
                st.session_state[f"bia_covered_{horizon_years}_{index}"] = float(row.covered_lives or 0.0)
            st.rerun()
        for index, row in enumerate(projection):
            eligible_key = f"bia_eligible_{horizon_years}_{index}"
            covered_key = f"bia_covered_{horizon_years}_{index}"
            st.session_state.setdefault(eligible_key, float(row.eligible_population))
            st.session_state.setdefault(covered_key, float(row.covered_lives or 0.0))
            c1, c2 = st.columns(2)
            eligible_value = c1.number_input(
                f"Year {index + 1} eligible population",
                min_value=0.0,
                step=max(float(row.eligible_population) * 0.01, 1.0),
                key=eligible_key,
            )
            covered_value = c2.number_input(
                f"Year {index + 1} covered lives (0 = not supplied)",
                min_value=0.0,
                step=max(float(row.covered_lives or 0.0) * 0.01, 100.0) if row.covered_lives else 1000.0,
                key=covered_key,
            )
            try:
                population_rows.append(PopulationYear(index + 1, eligible_value, covered_value if covered_value > 0 else None))
            except ValueError as exc:
                st.error(f"Year {index + 1}: {exc}")

    st.markdown("#### Evidence and rationale")
    st.session_state.bia_population_source = st.text_area(
        "Population evidence source",
        st.session_state.bia_population_source,
        key="bia_population_source_widget",
    )
    st.session_state.bia_population_rationale = st.text_area(
        "Population derivation / assumption rationale",
        st.session_state.bia_population_rationale,
        key="bia_population_rationale_widget",
    )


with costs_tab:
    st.subheader("Interventions and annual cost components")
    interventions = st.session_state.bia_interventions
    intervention_ids = [item["id"] for item in interventions]
    selected_id = st.selectbox(
        "Intervention to edit",
        intervention_ids,
        format_func=lambda iid: next(item["name"] for item in interventions if item["id"] == iid),
        key="bia_selected_intervention",
    )
    selected_index = next(i for i, item in enumerate(interventions) if item["id"] == selected_id)
    selected = dict(interventions[selected_index])
    selected["name"] = st.text_input("Intervention / option name", selected["name"], key=f"bia_name_{selected_id}")
    interventions[selected_index] = selected
    st.session_state.bia_interventions = interventions
    st.caption(f"Stable intervention ID: `{selected_id}`")

    cost_data = dict(st.session_state.bia_cost_inputs.get(selected_id, {}))
    c1, c2 = st.columns(2)
    for idx, category in enumerate(COST_CATEGORIES):
        target = c1 if idx % 2 == 0 else c2
        cost_data[category] = target.number_input(
            f"{CATEGORY_LABELS[category]} — cost per treated person in Year 1",
            value=float(cost_data.get(category, 0.0)),
            format="%.2f",
            key=f"bia_cost_{selected_id}_{category}",
        )
    cost_data["annual_change"] = st.number_input(
        "Annual price/resource-cost change (proportion)",
        value=float(cost_data.get("annual_change", 0.0)),
        format="%.4f",
        key=f"bia_cost_growth_{selected_id}",
        help="Applied to all cost components for this intervention. Enter 0.02 for +2% per year, -0.02 for -2%, or 0 when costs are held constant.",
    )
    cost_data["source"] = st.text_area(
        "Cost/resource evidence source",
        str(cost_data.get("source", "")),
        key=f"bia_cost_source_{selected_id}",
    )
    cost_data["rationale"] = st.text_area(
        "Costing assumptions / rationale",
        str(cost_data.get("rationale", "")),
        key=f"bia_cost_rationale_{selected_id}",
    )
    st.session_state.bia_cost_inputs[selected_id] = cost_data

    with st.expander("+ Add intervention / option"):
        with st.form("bia_add_intervention", clear_on_submit=True):
            new_name = st.text_input("Name")
            new_id = st.text_input("ID (optional)")
            add = st.form_submit_button("Add intervention")
            if add and new_name.strip():
                existing = {item["id"] for item in st.session_state.bia_interventions}
                chosen = new_id.strip() or _unique_id(new_name, existing)
                if chosen in existing:
                    st.error(f"Intervention ID '{chosen}' already exists.")
                else:
                    st.session_state.bia_interventions.append({"id": chosen, "name": new_name.strip()})
                    st.session_state.bia_cost_inputs[chosen] = {
                        **{category: 0.0 for category in COST_CATEGORIES},
                        "annual_change": 0.0,
                        "source": "",
                        "rationale": "",
                    }
                    st.rerun()

    if len(st.session_state.bia_interventions) > 2:
        if st.button("Delete selected intervention", type="secondary", key="bia_delete_intervention"):
            st.session_state.bia_interventions = [item for item in st.session_state.bia_interventions if item["id"] != selected_id]
            st.session_state.bia_cost_inputs.pop(selected_id, None)
            for key in list(st.session_state):
                if key.startswith("bia_share_") and key.endswith("_" + selected_id):
                    del st.session_state[key]
            st.rerun()


with mix_tab:
    st.subheader("Current and future treatment / utilisation mix")
    interventions = st.session_state.bia_interventions
    ids = [item["id"] for item in interventions]
    labels = {item["id"]: item["name"] for item in interventions}
    st.caption("Shares must sum to 100% separately for the current and future scenario in every year. The app does not silently normalise them.")

    mix_rows: list[TreatmentMixShare] = []
    mix_valid = True
    for year in range(1, horizon_years + 1):
        with st.expander(f"Year {year}", expanded=year == 1):
            st.markdown("**Current mix**")
            current_values: dict[str, float] = {}
            for idx, iid in enumerate(ids):
                default = 1.0 if idx == 0 else 0.0
                current_values[iid] = st.number_input(
                    f"{labels[iid]} — current share",
                    0.0,
                    1.0,
                    value=float(st.session_state.get(f"bia_share_current_{year}_{iid}", default)),
                    format="%.4f",
                    key=f"bia_share_current_{year}_{iid}",
                )
            current_total = sum(current_values.values())
            st.caption(f"Current total: {current_total * 100:.2f}%")
            if abs(current_total - 1.0) > 1e-8:
                st.error("Current shares must sum to 100%.")
                mix_valid = False

            st.markdown("**Future mix after introduction / implementation**")
            future_values: dict[str, float] = {}
            for idx, iid in enumerate(ids):
                if len(ids) >= 2:
                    default = 0.75 if idx == 0 else (0.25 if idx == 1 else 0.0)
                else:
                    default = 1.0
                future_values[iid] = st.number_input(
                    f"{labels[iid]} — future share",
                    0.0,
                    1.0,
                    value=float(st.session_state.get(f"bia_share_future_{year}_{iid}", default)),
                    format="%.4f",
                    key=f"bia_share_future_{year}_{iid}",
                )
            future_total = sum(future_values.values())
            st.caption(f"Future total: {future_total * 100:.2f}%")
            if abs(future_total - 1.0) > 1e-8:
                st.error("Future shares must sum to 100%.")
                mix_valid = False

            for iid in ids:
                mix_rows.append(TreatmentMixShare("current", year, iid, current_values[iid]))
                mix_rows.append(TreatmentMixShare("future", year, iid, future_values[iid]))

    st.markdown("#### Evidence and rationale")
    st.session_state.bia_current_mix_source = st.text_area("Current treatment-mix evidence", st.session_state.bia_current_mix_source)
    st.session_state.bia_current_mix_rationale = st.text_area("Current mix rationale", st.session_state.bia_current_mix_rationale)
    st.session_state.bia_future_mix_source = st.text_area("Future uptake / implementation evidence", st.session_state.bia_future_mix_source)
    st.session_state.bia_future_mix_rationale = st.text_area("Future mix / substitution rationale", st.session_state.bia_future_mix_rationale)


interventions = tuple(BudgetIntervention(item["id"], item["name"]) for item in st.session_state.bia_interventions)
cost_rows: list[AnnualCostInput] = []
for intervention in interventions:
    data = st.session_state.bia_cost_inputs.get(intervention.id, {})
    growth = float(data.get("annual_change", 0.0))
    for category in COST_CATEGORIES:
        series = compound_series(float(data.get(category, 0.0)), growth, horizon_years)
        for index, value in enumerate(series, start=1):
            cost_rows.append(AnnualCostInput(intervention.id, index, category, value))

compiled_definition = None
base_run = None
compile_error = None
try:
    if len(population_rows) != horizon_years:
        raise BudgetImpactValidationError("Complete the eligible-population inputs for every year.")
    if not mix_valid:
        raise BudgetImpactValidationError("Treatment-mix shares must sum to 100% before results can run.")
    if not included_categories:
        raise BudgetImpactValidationError("At least one cost category must be included.")
    compiled_definition = BudgetImpactDefinition(
        interventions=interventions,
        population=tuple(population_rows),
        treatment_mix=tuple(mix_rows),
        costs=tuple(cost_rows),
        included_cost_categories=tuple(included_categories),
    )
    base_run = run_budget_impact(compiled_definition)
    st.session_state.bia_population_rows = [
        {
            "year": row.year,
            "eligible_population": row.eligible_population,
            "covered_lives": row.covered_lives,
        }
        for row in compiled_definition.population
    ]
    st.session_state.bia_treatment_mix_rows = [
        {
            "scenario": row.scenario,
            "year": row.year,
            "intervention_id": row.intervention_id,
            "share": row.share,
        }
        for row in compiled_definition.treatment_mix
    ]
    st.session_state.bia_context_valid = True
except (BudgetImpactValidationError, ValueError) as exc:
    compile_error = str(exc)
    st.session_state.bia_context_valid = False
    st.session_state.pop("bia_population_rows", None)
    st.session_state.pop("bia_treatment_mix_rows", None)


with results_tab:
    st.subheader("Annual and cumulative budget impact")
    if compile_error:
        st.error(compile_error)
        st.warning("The current BIA configuration is invalid, so clinical linkage, capacity planning and policy interpretation will not reuse an earlier valid BIA state from this session.")
    else:
        assert compiled_definition is not None and base_run is not None
        symbol = CURRENCIES[currency_code].symbol
        first = base_run.years[0]
        final = base_run.years[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Year 1 net budget impact", f"{symbol}{first.net_budget_impact:,.0f}")
        c2.metric(f"Year {horizon_years} net budget impact", f"{symbol}{final.net_budget_impact:,.0f}")
        c3.metric("Cumulative budget impact", f"{symbol}{base_run.cumulative_budget_impact:,.0f}")
        c4.metric(
            "Year 1 PMPM",
            f"{symbol}{first.pmpm_budget_impact:,.4f}" if first.pmpm_budget_impact is not None else "Not available",
        )

        annual_df = pd.DataFrame(
            [
                {
                    "Year": row.year,
                    "Eligible population": row.eligible_population,
                    "Covered lives": row.covered_lives,
                    "Current scenario cost": row.current_cost,
                    "Future scenario cost": row.future_cost,
                    "Net budget impact": row.net_budget_impact,
                    "Cumulative budget impact": row.cumulative_budget_impact,
                    "PMPM": row.pmpm_budget_impact,
                }
                for row in base_run.years
            ]
        )
        st.dataframe(annual_df, use_container_width=True, hide_index=True)

        budget_long = annual_df.melt(
            id_vars="Year",
            value_vars=["Current scenario cost", "Future scenario cost"],
            var_name="Scenario",
            value_name="Annual budget",
        )
        st.plotly_chart(
            px.line(budget_long, x="Year", y="Annual budget", color="Scenario", markers=True, title="Annual budget with and without the new implementation scenario"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(annual_df, x="Year", y="Net budget impact", title="Net budget impact by year"),
            use_container_width=True,
        )

        with st.expander("Treated population by intervention"):
            treated_df = pd.DataFrame(
                [
                    {
                        "Scenario": row.scenario,
                        "Year": row.year,
                        "Intervention": row.intervention_name,
                        "Share": row.share,
                        "Treated people": row.treated_people,
                        "Cost per treated person": row.cost_per_treated_person,
                        "Total cost": row.total_cost,
                    }
                    for row in base_run.intervention_rows
                ]
            )
            st.dataframe(treated_df, use_container_width=True, hide_index=True)

        with st.expander("Budget by cost category"):
            category_df = pd.DataFrame(
                [
                    {
                        "Scenario": row.scenario,
                        "Year": row.year,
                        "Category": CATEGORY_LABELS[row.category],
                        "Total cost": row.total_cost,
                    }
                    for row in base_run.category_rows
                ]
            )
            st.dataframe(category_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download annual budget-impact results (CSV)",
            annual_df.to_csv(index=False),
            file_name="budget_impact_results.csv",
            mime="text/csv",
        )

        st.markdown("### Scenario explorer")
        st.caption(
            "This explorer changes the eligible population, one intervention's cost, and its future uptake. When uptake changes, the remaining future shares are redistributed proportionally across the other options; the rule is explicit rather than hidden."
        )
        target_id = st.selectbox(
            "Target intervention",
            [item.id for item in interventions],
            format_func=lambda iid: next(item.name for item in interventions if item.id == iid),
            key="bia_scenario_target",
        )
        c1, c2, c3 = st.columns(3)
        population_multiplier = c1.number_input("Eligible-population multiplier", min_value=0.0, value=1.0, step=0.05, format="%.3f")
        uptake_multiplier = c2.number_input("Future-uptake multiplier", min_value=0.0, value=1.0, step=0.05, format="%.3f")
        cost_multiplier = c3.number_input("Target cost multiplier", min_value=0.0, value=1.0, step=0.05, format="%.3f")
        scenario_definition = apply_simple_scenario(
            compiled_definition,
            population_multiplier=population_multiplier,
            target_intervention_id=target_id,
            uptake_multiplier=uptake_multiplier,
            target_cost_multiplier=cost_multiplier,
        )
        scenario_run = run_budget_impact(scenario_definition)
        delta_vs_base = scenario_run.cumulative_budget_impact - base_run.cumulative_budget_impact
        c1, c2 = st.columns(2)
        c1.metric("Scenario cumulative budget impact", f"{symbol}{scenario_run.cumulative_budget_impact:,.0f}")
        c2.metric("Change versus base scenario", f"{symbol}{delta_vs_base:,.0f}")

        st.markdown("### Transparency check")
        checks: list[tuple[str, bool]] = [
            ("Eligible-population evidence documented", bool(st.session_state.bia_population_source.strip())),
            ("Eligible-population rationale documented", bool(st.session_state.bia_population_rationale.strip())),
            ("Current treatment-mix evidence documented", bool(st.session_state.bia_current_mix_source.strip())),
            ("Current treatment-mix rationale documented", bool(st.session_state.bia_current_mix_rationale.strip())),
            ("Future uptake evidence documented", bool(st.session_state.bia_future_mix_source.strip())),
            ("Future uptake/substitution rationale documented", bool(st.session_state.bia_future_mix_rationale.strip())),
        ]
        for intervention in interventions:
            data = st.session_state.bia_cost_inputs.get(intervention.id, {})
            checks.extend(
                [
                    (f"{intervention.name}: cost/resource source documented", bool(str(data.get("source", "")).strip())),
                    (f"{intervention.name}: costing rationale documented", bool(str(data.get("rationale", "")).strip())),
                ]
            )
        complete = sum(1 for _, ok in checks if ok)
        status_bar(
            [
                (f"{complete}/{len(checks)} documentation checks complete", "green" if complete == len(checks) else "amber"),
                ("Not a scientific quality score", "neutral"),
            ]
        )
        for label, ok in checks:
            st.write(("✓ " if ok else "⚠ ") + label)
        st.caption(
            "The Transparency check only assesses whether documentation is present. It does not establish that the population forecast, uptake assumptions, prices, resource use or evidence sources are valid or unbiased."
        )

st.divider()
st.caption(
    "Budget impact analysis estimates affordability for a defined budget holder. Annual eligible population, treatment mix, uptake and cost inputs should use local evidence wherever possible and all assumptions should be reviewable."
)
