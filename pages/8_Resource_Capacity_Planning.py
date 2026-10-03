"""Guided natural-resource and service-capacity planning workspace."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from model.budget_impact import BudgetIntervention, PopulationYear, TreatmentMixShare
from model.budget_impact_profiles import BUDGET_IMPACT_PROFILES
from model.resource_capacity import (
    RESOURCE_CATEGORIES,
    AnnualResourceCapacity,
    ResourceCapacityDefinition,
    ResourceCapacityValidationError,
    ResourceDefinition,
    ResourceRequirement,
    apply_capacity_expansion,
    run_resource_capacity_plan,
)
from ui.design_system import coloured_block, status_bar


st.set_page_config(page_title="Resource & Capacity Planning", page_icon="🏥", layout="wide")
st.title("Resource & Capacity Planning")
st.caption("Version 0.11 — natural-unit demand, service capacity, utilisation and shortfall planning")

coloured_block(
    "Plan physical implementation as well as financial affordability",
    "Translate the active BIA population and treatment mix into natural resource requirements such as clinician hours, infusion-chair hours, bed-days, scans, laboratory tests, procedures or device quantities. Compare demand with the capacity actually available to the modelled population in each year.",
    tone="teal",
    kicker="Capacity and demand",
)
coloured_block(
    "Capacity is not the same as budget",
    "A technology may be financially affordable while still exceeding workforce, facility, diagnostic or equipment capacity. This workspace reports physical demand and shortfalls separately from monetary Budget Impact Analysis.",
    tone="amber",
    kicker="Implementation constraint",
)

CATEGORY_LABELS = {
    "workforce": "Workforce / staff time",
    "facility": "Facility / treatment space",
    "equipment": "Equipment / machine time",
    "diagnostic": "Diagnostics / tests / imaging",
    "inpatient": "Inpatient / bed capacity",
    "pharmacy": "Pharmacy / preparation",
    "consumable": "Consumables / devices",
    "other": "Other resource",
}
PROVISIONAL_MARKERS = ("illustrative", "replace with", "placeholder", "example input")


def _slug(text: str) -> str:
    value = "".join(char.lower() if char.isalnum() else "_" for char in text.strip())
    value = "_".join(part for part in value.split("_") if part)
    return value or "resource"


def _unique_id(label: str, existing: set[str]) -> str:
    base = _slug(label)
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _documented(text: object) -> bool:
    value = str(text or "").strip()
    return bool(value) and not any(marker in value.lower() for marker in PROVISIONAL_MARKERS)


def _bia_context() -> tuple[tuple[BudgetIntervention, ...], tuple[PopulationYear, ...], tuple[TreatmentMixShare, ...], int]:
    profile_code = st.session_state.get("bia_profile")
    raw_interventions = st.session_state.get("bia_interventions")
    if not profile_code or not raw_interventions:
        raise ResourceCapacityValidationError(
            "Configure Budget Impact Analysis first so the planner can reuse its population, interventions and current/future treatment mix."
        )

    if profile_code == "CUSTOM":
        horizon = int(st.session_state.get("bia_horizon_custom", 3))
    else:
        if profile_code not in BUDGET_IMPACT_PROFILES:
            raise ResourceCapacityValidationError("The active BIA methods profile is not recognised.")
        profile = BUDGET_IMPACT_PROFILES[profile_code]
        horizon = int(st.session_state.get(f"bia_horizon_{profile.code}", profile.default_horizon_years))

    interventions = tuple(BudgetIntervention(str(row["id"]), str(row["name"])) for row in raw_interventions)
    ids = [item.id for item in interventions]

    population: list[PopulationYear] = []
    for index in range(horizon):
        eligible_key = f"bia_eligible_{horizon}_{index}"
        covered_key = f"bia_covered_{horizon}_{index}"
        if eligible_key not in st.session_state:
            raise ResourceCapacityValidationError(
                "Complete all annual eligible-population inputs in Budget Impact Analysis before resource planning."
            )
        eligible = float(st.session_state[eligible_key])
        covered = float(st.session_state.get(covered_key, 0.0) or 0.0)
        population.append(PopulationYear(index + 1, eligible, covered if covered > 0 else None))

    mix: list[TreatmentMixShare] = []
    for year in range(1, horizon + 1):
        for scenario in ("current", "future"):
            for intervention_id in ids:
                key = f"bia_share_{scenario}_{year}_{intervention_id}"
                if key not in st.session_state:
                    raise ResourceCapacityValidationError(
                        "Complete the current and future treatment mix in Budget Impact Analysis before resource planning."
                    )
                mix.append(TreatmentMixShare(scenario, year, intervention_id, float(st.session_state[key])))
    return interventions, tuple(population), tuple(mix), horizon


def _ensure_resources() -> None:
    if "rc_resources" not in st.session_state:
        st.session_state.rc_resources = [
            {
                "id": "clinical_staff_time",
                "name": "Clinical staff time",
                "unit": "hours",
                "category": "workforce",
                "source": "Illustrative input — replace with local service/capacity evidence",
                "rationale": "Illustrative resource for testing the capacity-planning workflow.",
            }
        ]


def _resource_meta(resource_id: str) -> dict:
    return next(row for row in st.session_state.rc_resources if row["id"] == resource_id)


def _resource_definitions() -> tuple[ResourceDefinition, ...]:
    return tuple(
        ResourceDefinition(
            id=str(row["id"]),
            name=str(row["name"]),
            unit=str(row["unit"]),
            category=str(row["category"]),
        )
        for row in st.session_state.rc_resources
    )


def _requirements(interventions, resource_ids, horizon, demand_basis):
    rows: list[ResourceRequirement] = []
    for intervention in interventions:
        for resource_id in resource_ids:
            for period in range(1, horizon + 1):
                value = float(st.session_state.get(f"rc_req_{demand_basis}_{intervention.id}_{resource_id}_{period}", 0.0))
                if value > 0:
                    rows.append(ResourceRequirement(intervention.id, resource_id, period, value))
    return tuple(rows)


def _capacities(resource_ids, horizon):
    rows: list[AnnualResourceCapacity] = []
    for resource_id in resource_ids:
        for year in range(1, horizon + 1):
            total = float(st.session_state.get(f"rc_capacity_total_{resource_id}_{year}", 1000.0))
            committed = float(st.session_state.get(f"rc_capacity_committed_{resource_id}_{year}", 0.0))
            rows.append(AnnualResourceCapacity(resource_id, year, total, committed))
    return tuple(rows)


try:
    interventions, population, treatment_mix, horizon_years = _bia_context()
except (ResourceCapacityValidationError, ValueError) as exc:
    st.error(str(exc))
    st.page_link("pages/6_Budget_Impact_Analysis.py", label="Open Budget Impact Analysis →", icon="💷")
    st.stop()

_ensure_resources()

basis_label = st.sidebar.selectbox(
    "Resource-demand basis",
    ["Annual treated population", "New treatment starts / longitudinal profile"],
    key="rc_basis_label",
    help=(
        "Annual treated population applies each year's resource requirement to that year's treated population. "
        "New treatment starts follows initiation cohorts and stacks Year-1, Year-2 and later resource use."
    ),
)
demand_basis = "annual_treated_population" if basis_label == "Annual treated population" else "new_treatment_starts"

status_bar(
    [
        (f"BIA horizon: {horizon_years} years", "blue"),
        (f"{len(interventions)} interventions", "neutral"),
        (f"{len(st.session_state.rc_resources)} resources", "teal"),
        (basis_label, "neutral"),
    ]
)

if demand_basis == "annual_treated_population":
    st.info(
        "**Annual treated-population basis:** Period 1, 2, 3… means Budget Year 1, 2, 3…. Use this when the BIA population represents people receiving care in each year and resource inputs are annual cross-sectional requirements."
    )
else:
    st.warning(
        "**New-treatment-start basis:** Period 1, 2, 3… means Year 1, Year 2, Year 3 since treatment initiation. The BIA annual population must represent new starts; initiation cohorts are stacked into later budget years."
    )

resources_tab, requirements_tab, capacity_tab, results_tab, transparency_tab = st.tabs(
    ["1 · Resources", "2 · Requirements", "3 · Available capacity", "4 · Results", "5 · Transparency check"]
)

with resources_tab:
    st.subheader("Define the resources that may constrain implementation")
    st.caption("Use observable natural units such as hours, appointments, bed-days, scans, tests, procedures, treatment slots or device units.")
    resource_ids = [str(row["id"]) for row in st.session_state.rc_resources]
    selected_id = st.selectbox(
        "Resource to edit", resource_ids, format_func=lambda rid: _resource_meta(rid)["name"], key="rc_selected_resource"
    )
    selected_index = next(i for i, row in enumerate(st.session_state.rc_resources) if row["id"] == selected_id)
    selected = dict(st.session_state.rc_resources[selected_index])
    c1, c2, c3 = st.columns(3)
    selected["name"] = c1.text_input("Resource name", str(selected["name"]), key=f"rc_name_{selected_id}")
    selected["unit"] = c2.text_input("Natural unit", str(selected["unit"]), key=f"rc_unit_{selected_id}")
    selected["category"] = c3.selectbox(
        "Resource category",
        list(RESOURCE_CATEGORIES),
        index=list(RESOURCE_CATEGORIES).index(str(selected["category"])),
        format_func=lambda code: CATEGORY_LABELS[code],
        key=f"rc_category_{selected_id}",
    )
    st.caption(f"Stable resource ID: `{selected_id}`")
    selected["source"] = st.text_area("Resource evidence source", str(selected.get("source", "")), key=f"rc_source_{selected_id}")
    selected["rationale"] = st.text_area("Resource definition / inclusion rationale", str(selected.get("rationale", "")), key=f"rc_rationale_{selected_id}")
    st.session_state.rc_resources[selected_index] = selected

    with st.expander("+ Add another resource"):
        with st.form("rc_add_resource", clear_on_submit=True):
            new_name = st.text_input("Resource name")
            new_unit = st.text_input("Natural unit", placeholder="e.g. hours, visits, scans, bed-days")
            new_category = st.selectbox("Category", list(RESOURCE_CATEGORIES), format_func=lambda code: CATEGORY_LABELS[code])
            if st.form_submit_button("Add resource") and new_name.strip() and new_unit.strip():
                existing = {str(row["id"]) for row in st.session_state.rc_resources}
                st.session_state.rc_resources.append(
                    {
                        "id": _unique_id(new_name, existing),
                        "name": new_name.strip(),
                        "unit": new_unit.strip(),
                        "category": new_category,
                        "source": "",
                        "rationale": "",
                    }
                )
                st.rerun()

    if len(st.session_state.rc_resources) > 1 and st.button("Delete selected resource", type="secondary", key="rc_delete_resource"):
        st.session_state.rc_resources = [row for row in st.session_state.rc_resources if row["id"] != selected_id]
        st.rerun()

    st.markdown("#### Current resource library")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Resource": row["name"],
                    "Category": CATEGORY_LABELS[row["category"]],
                    "Natural unit": row["unit"],
                    "Evidence status": "Documented" if _documented(row.get("source")) else "Provisional / missing",
                }
                for row in st.session_state.rc_resources
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with requirements_tab:
    st.subheader("Resource requirements per person")
    resource_ids = [str(row["id"]) for row in st.session_state.rc_resources]
    resource_labels = {rid: _resource_meta(rid)["name"] for rid in resource_ids}
    intervention_labels = {item.id: item.name for item in interventions}
    c1, c2 = st.columns(2)
    req_intervention = c1.selectbox("Intervention / option", [item.id for item in interventions], format_func=lambda iid: intervention_labels[iid], key="rc_req_intervention")
    req_resource = c2.selectbox("Resource", resource_ids, format_func=lambda rid: resource_labels[rid], key="rc_req_resource")
    unit = _resource_meta(req_resource)["unit"]

    for period in range(1, horizon_years + 1):
        key = f"rc_req_{demand_basis}_{req_intervention}_{req_resource}_{period}"
        label = (
            f"Budget Year {period} — {unit} per treated person"
            if demand_basis == "annual_treated_population"
            else f"Year {period} since initiation — {unit} per new start"
        )
        st.number_input(label, min_value=0.0, value=float(st.session_state.get(key, 0.0)), format="%.6f", key=key)

    req_source_key = f"rc_req_source_{demand_basis}_{req_intervention}_{req_resource}"
    req_rationale_key = f"rc_req_rationale_{demand_basis}_{req_intervention}_{req_resource}"
    st.text_area("Evidence source for this resource-use profile", value=str(st.session_state.get(req_source_key, "")), key=req_source_key)
    st.text_area("Resource-use assumptions / rationale", value=str(st.session_state.get(req_rationale_key, "")), key=req_rationale_key)

    with st.expander("Review all non-zero requirements"):
        review_rows = _requirements(interventions, resource_ids, horizon_years, demand_basis)
        if review_rows:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Intervention": intervention_labels[row.intervention_id],
                            "Resource": resource_labels[row.resource_id],
                            "Period": row.period,
                            "Units/person": row.units_per_person,
                        }
                        for row in review_rows
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No non-zero resource requirements have been entered yet.")

with capacity_tab:
    st.subheader("Capacity available to the modelled population")
    st.write(
        "Enter total service capacity and capacity already committed to **other services or populations outside this BIA**. The difference is the capacity available to the modelled population."
    )
    resource_ids = [str(row["id"]) for row in st.session_state.rc_resources]
    cap_resource = st.selectbox("Resource to configure", resource_ids, format_func=lambda rid: _resource_meta(rid)["name"], key="rc_cap_resource")
    unit = _resource_meta(cap_resource)["unit"]
    for year in range(1, horizon_years + 1):
        total_key = f"rc_capacity_total_{cap_resource}_{year}"
        committed_key = f"rc_capacity_committed_{cap_resource}_{year}"
        with st.expander(f"Year {year}", expanded=year == 1):
            c1, c2 = st.columns(2)
            total = c1.number_input(
                f"Total capacity ({unit})", min_value=0.0, value=float(st.session_state.get(total_key, 1000.0)), format="%.4f", key=total_key
            )
            committed = c2.number_input(
                f"Committed to other uses ({unit})", min_value=0.0, value=float(st.session_state.get(committed_key, 0.0)), format="%.4f", key=committed_key
            )
            if committed > total:
                st.error("Committed other demand exceeds total capacity. Correct the inputs; the app will not silently truncate this value.")
            else:
                st.caption(f"Available to this modelled population: {total - committed:,.4f} {unit}")

    cap_source_key = f"rc_capacity_source_{cap_resource}"
    cap_rationale_key = f"rc_capacity_rationale_{cap_resource}"
    st.text_area("Capacity evidence source", value=str(st.session_state.get(cap_source_key, "")), key=cap_source_key)
    st.text_area("Capacity assumptions / rationale", value=str(st.session_state.get(cap_rationale_key, "")), key=cap_rationale_key)

compiled_definition = None
capacity_run = None
compile_error = None
resource_ids = [str(row["id"]) for row in st.session_state.rc_resources]
try:
    compiled_definition = ResourceCapacityDefinition(
        interventions=interventions,
        population=population,
        treatment_mix=treatment_mix,
        resources=_resource_definitions(),
        requirements=_requirements(interventions, resource_ids, horizon_years, demand_basis),
        capacities=_capacities(resource_ids, horizon_years),
        demand_basis=demand_basis,
    )
    capacity_run = run_resource_capacity_plan(compiled_definition)
except (ResourceCapacityValidationError, ValueError) as exc:
    compile_error = str(exc)

with results_tab:
    st.subheader("Resource demand and capacity by year")
    if compile_error:
        st.error(compile_error)
    else:
        assert compiled_definition is not None and capacity_run is not None
        rows = capacity_run.comparison_rows
        shortfalls = [row for row in rows if row.future_shortfall > 1e-9]
        utilization_values = [row.future_utilization_ratio for row in rows if row.future_utilization_ratio is not None]
        earliest_shortfall = min((row.year for row in shortfalls), default=None)
        max_util = max(utilization_values) if utilization_values else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resources modelled", len(compiled_definition.resources))
        c2.metric("Resource-years with future shortfall", len(shortfalls))
        c3.metric("Earliest shortfall", f"Year {earliest_shortfall}" if earliest_shortfall else "None")
        c4.metric("Maximum future utilisation", f"{max_util * 100:,.1f}%" if max_util is not None else "Undefined")

        comparison_df = pd.DataFrame(
            [
                {
                    "Year": row.year,
                    "Resource": row.resource_name,
                    "Unit": row.unit,
                    "Current demand": row.current_required_units,
                    "Future demand": row.future_required_units,
                    "Net change": row.net_change_units,
                    "Available capacity": row.available_capacity,
                    "Future utilisation": row.future_utilization_ratio,
                    "Future headroom": row.future_headroom,
                    "Future shortfall": row.future_shortfall,
                }
                for row in rows
            ]
        )
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={"Future utilisation": st.column_config.NumberColumn(format="%.1%")},
        )

        selected_result_resource = st.selectbox(
            "Resource to visualise",
            [resource.id for resource in compiled_definition.resources],
            format_func=lambda rid: next(resource.name for resource in compiled_definition.resources if resource.id == rid),
            key="rc_result_resource",
        )
        selected_name = next(resource.name for resource in compiled_definition.resources if resource.id == selected_result_resource)
        plot_df = comparison_df[comparison_df["Resource"] == selected_name]
        long_df = plot_df.melt(
            id_vars=["Year", "Unit"],
            value_vars=["Current demand", "Future demand", "Available capacity"],
            var_name="Series",
            value_name="Natural units",
        )
        st.plotly_chart(
            px.line(long_df, x="Year", y="Natural units", color="Series", markers=True, title="Current demand, future demand and available capacity"),
            use_container_width=True,
        )

        if shortfalls:
            st.warning(
                "At least one resource exceeds available capacity under the future treatment mix. The table reports the physical shortfall; it does not assume unmet demand is automatically resolved, rationed or funded."
            )
        else:
            st.success("No future-scenario capacity shortfall is identified under the current inputs.")

        with st.expander("Demand by intervention / initiation cohort"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Scenario": row.scenario,
                            "Budget year": row.budget_year,
                            "Intervention": row.intervention_name,
                            "Resource": row.resource_name,
                            "Initiation year": row.initiation_year,
                            "Source period": row.source_period,
                            "People": row.treated_people,
                            "Units/person": row.units_per_person,
                            "Required units": row.required_units,
                        }
                        for row in capacity_run.intervention_rows
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Capacity expansion planner")
        st.caption("Test a concrete service-expansion plan. Richer resource uncertainty/scenario management will be added later.")
        c1, c2, c3, c4 = st.columns(4)
        target_resource = c1.selectbox(
            "Resource",
            [resource.id for resource in compiled_definition.resources],
            format_func=lambda rid: next(resource.name for resource in compiled_definition.resources if resource.id == rid),
            key="rc_scenario_resource",
        )
        from_year = int(c2.number_input("Expansion starts in year", 1, horizon_years, 1, step=1))
        capacity_multiplier = c3.number_input("Capacity multiplier", min_value=0.0, value=1.0, step=0.05, format="%.3f")
        additional_capacity = c4.number_input("Additional units/year", value=0.0, step=10.0, format="%.3f")
        try:
            scenario_run = run_resource_capacity_plan(
                apply_capacity_expansion(
                    compiled_definition,
                    resource_id=target_resource,
                    from_year=from_year,
                    capacity_multiplier=capacity_multiplier,
                    additional_capacity=additional_capacity,
                )
            )
        except ResourceCapacityValidationError as exc:
            st.error(str(exc))
        else:
            base_shortfall = sum(row.future_shortfall for row in capacity_run.comparison_rows if row.resource_id == target_resource)
            scenario_shortfall = sum(row.future_shortfall for row in scenario_run.comparison_rows if row.resource_id == target_resource)
            unit = next(resource.unit for resource in compiled_definition.resources if resource.id == target_resource)
            c1, c2, c3 = st.columns(3)
            c1.metric("Base cumulative shortfall", f"{base_shortfall:,.2f} {unit}")
            c2.metric("Expansion-scenario shortfall", f"{scenario_shortfall:,.2f} {unit}")
            c3.metric("Shortfall reduced by", f"{base_shortfall - scenario_shortfall:,.2f} {unit}")

        st.download_button(
            "Download capacity results (CSV)", comparison_df.to_csv(index=False), file_name="resource_capacity_results.csv", mime="text/csv"
        )

with transparency_tab:
    st.subheader("Transparency check")
    st.caption(
        "This checks documentation completeness for resource definitions, resource-use profiles and capacity assumptions. Placeholder or illustrative evidence is treated as provisional. It is not a scientific or operational quality score."
    )
    checks: list[tuple[str, bool]] = []
    for resource in st.session_state.rc_resources:
        rid = str(resource["id"])
        checks.extend(
            [
                (f"{resource['name']}: resource evidence source documented", _documented(resource.get("source"))),
                (f"{resource['name']}: resource definition rationale documented", _documented(resource.get("rationale"))),
                (f"{resource['name']}: capacity evidence source documented", _documented(st.session_state.get(f"rc_capacity_source_{rid}", ""))),
                (f"{resource['name']}: capacity rationale documented", _documented(st.session_state.get(f"rc_capacity_rationale_{rid}", ""))),
            ]
        )
        for intervention in interventions:
            non_zero = any(
                float(st.session_state.get(f"rc_req_{demand_basis}_{intervention.id}_{rid}_{period}", 0.0)) > 0
                for period in range(1, horizon_years + 1)
            )
            if non_zero:
                checks.extend(
                    [
                        (f"{intervention.name} × {resource['name']}: resource-use source documented", _documented(st.session_state.get(f"rc_req_source_{demand_basis}_{intervention.id}_{rid}", ""))),
                        (f"{intervention.name} × {resource['name']}: resource-use rationale documented", _documented(st.session_state.get(f"rc_req_rationale_{demand_basis}_{intervention.id}_{rid}", ""))),
                    ]
                )

    complete = sum(1 for _, ok in checks if ok)
    status_bar(
        [
            (f"{complete}/{len(checks)} documentation checks complete", "green" if checks and complete == len(checks) else "amber"),
            ("Not a quality score", "neutral"),
        ]
    )
    for label, ok in checks:
        st.write(("✓ " if ok else "⚠ ") + label)

    st.markdown("#### Population-basis declaration")
    if demand_basis == "annual_treated_population":
        st.write("The calculation interprets the active BIA population as the treated population relevant to each budget year.")
    else:
        st.write("The calculation interprets the active BIA population as annual new treatment starts and stacks later resource use across initiation cohorts.")
    st.caption(
        "The modeller remains responsible for ensuring that the selected population basis matches the epidemiology, uptake assumptions and service-delivery pathway being evaluated."
    )

st.divider()
st.page_link("pages/6_Budget_Impact_Analysis.py", label="← Return to Budget Impact Analysis", icon="💷")
