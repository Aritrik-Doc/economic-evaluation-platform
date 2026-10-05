"""Guided natural-resource and service-capacity planning workspace."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from model.budget_impact import BudgetIntervention, PopulationYear as BIAPopulationYear, TreatmentMixShare as BIATreatmentMixShare
from model.budget_impact_profiles import BUDGET_IMPACT_PROFILES
from model.clinical_resource_linkage import (
    ClinicalResourceLinkageError,
    StateResourceMapping,
    TreeResourceMapping,
    available_resource_parameters,
    combine_resource_requirements,
    profiles_to_requirements,
    project_decision_tree_resource_profiles,
    project_markov_resource_profiles,
    project_semi_markov_resource_profiles,
)
from model.markov_builder import compile_markov_tables
from model.population_uptake import PopulationOption, PopulationYear, TreatmentMixShare
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
from model.semi_markov_builder import compile_semi_markov_tables
from model.tree_builder import compile_builder_tables
from ui.design_system import coloured_block, status_bar


st.set_page_config(page_title="Resource & Capacity Planning", page_icon="🏥", layout="wide")
st.title("Resource & Capacity Planning")
st.caption("Version 0.14.1 — reusable population context, clinical resource linkage, natural-unit demand and capacity planning")

coloured_block(
    "Plan physical implementation independently of Budget Impact Analysis",
    "Capacity planning can now use the shared Population & Uptake workspace, reuse a Budget Impact Analysis population, or define a population locally. Clinical models can optionally supply natural-unit resource trajectories while capacity remains a separate implementation layer.",
    tone="teal",
    kicker="Capacity and demand",
)
coloured_block(
    "Population, clinical consequences and capacity remain separate",
    "A per-patient clinical model does not determine how many people are eligible. Population and uptake determine how many people receive each option; the clinical model can contribute resource use per patient; this workspace compares resulting demand with available capacity.",
    tone="amber",
    kicker="Transparent linkage",
)

CATEGORY_LABELS = {
    "workforce": "Workforce / staff time",
    "facility": "Facility / treatment space / chairs",
    "equipment": "Equipment / machines / vehicles",
    "diagnostic": "Diagnostics / tests / imaging",
    "inpatient": "Inpatient / bed-days / admissions",
    "pharmacy": "Pharmacy / preparation / vials",
    "consumable": "Consumables / syringes / oxygen / blood products",
    "other": "Transport / ambulances / other resource",
}
RESOURCE_EXAMPLES = {
    "Clinical staff time": {
        "name": "Clinical staff time",
        "unit": "hours",
        "category": "workforce",
        "note": "Use when clinician, nurse, pharmacist or technician time is the constrained resource.",
    },
    "Treatment-chair time": {
        "name": "Treatment-chair time",
        "unit": "chair-hours",
        "category": "facility",
        "note": "Useful for infusions, dialysis or other services constrained by treatment spaces over time.",
    },
    "Inpatient bed-days": {
        "name": "Inpatient bed-days",
        "unit": "bed-days",
        "category": "inpatient",
        "note": "Model demand and available capacity in the same bed-day unit; do not compare bed-days used directly with a simple count of beds.",
    },
    "Syringes": {
        "name": "Syringes",
        "unit": "syringes",
        "category": "consumable",
        "note": "Use physical units consumed per patient, procedure or treatment course.",
    },
    "Medicine vials": {
        "name": "Medicine vials",
        "unit": "vials",
        "category": "pharmacy",
        "note": "Useful when vial availability, compounding or pharmacy throughput is operationally constrained.",
    },
    "Ambulance trips": {
        "name": "Ambulance trips",
        "unit": "trips",
        "category": "other",
        "note": "Use trips when transport demand is event-based. Vehicle-hours can be defined separately when time is the constraint.",
    },
    "Vehicle-hours": {
        "name": "Vehicle-hours",
        "unit": "vehicle-hours",
        "category": "equipment",
        "note": "Use for outreach, mobile clinics or transport fleets where available vehicle time is the relevant capacity.",
    },
    "Oxygen supply": {
        "name": "Oxygen supply",
        "unit": "litres",
        "category": "consumable",
        "note": "Choose litres, cubic metres or cylinders consistently for both demand and available supply.",
    },
    "Blood products": {
        "name": "Blood products",
        "unit": "units",
        "category": "consumable",
        "note": "Use units/bags of the relevant blood component and document whether wastage or reserve stock is included.",
    },
    "Diagnostic scans": {
        "name": "Diagnostic scans",
        "unit": "scans",
        "category": "diagnostic",
        "note": "Use scan counts, machine-hours or appointment slots depending on the actual service constraint.",
    },
}
PROVISIONAL_MARKERS = ("illustrative", "replace with", "placeholder", "example input")


def _records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return [dict(row) for row in value]


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
    st.session_state.setdefault("rc_standalone_options", [
        {"id": "current_treatment", "name": "Current treatment"},
        {"id": "new_intervention", "name": "New intervention"},
    ])
    st.session_state.setdefault("rc_clinical_mappings", [])


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


def _shared_context():
    raw_options = st.session_state.get("pu_options")
    raw_population = st.session_state.get("pu_population_rows")
    raw_mix = st.session_state.get("pu_uptake_rows")
    if not raw_options or not raw_population or not raw_mix:
        raise ResourceCapacityValidationError(
            "Complete the Population & Uptake workspace before selecting it as the capacity-planning population source."
        )
    options = tuple(PopulationOption(str(row["id"]), str(row["name"])) for row in raw_options)
    population = tuple(
        PopulationYear(int(row["year"]), float(row["eligible_population"]), None if row.get("covered_lives") in (None, "") else float(row["covered_lives"]))
        for row in raw_population
    )
    mix = tuple(
        TreatmentMixShare(str(row["scenario"]), int(row["year"]), str(row["intervention_id"]), float(row["share"]))
        for row in raw_mix
    )
    basis = str(st.session_state.get("pu_population_basis", "annual_eligible_population"))
    return options, population, mix, len(population), basis, "Shared Population & Uptake"


def _bia_context():
    profile_code = st.session_state.get("bia_profile")
    raw_interventions = st.session_state.get("bia_interventions")
    if not profile_code or not raw_interventions:
        raise ResourceCapacityValidationError("Configure Budget Impact Analysis before selecting it as the population source.")
    if profile_code == "CUSTOM":
        horizon = int(st.session_state.get("bia_horizon_custom", 3))
    else:
        if profile_code not in BUDGET_IMPACT_PROFILES:
            raise ResourceCapacityValidationError("The active BIA methods profile is not recognised.")
        profile = BUDGET_IMPACT_PROFILES[profile_code]
        horizon = int(st.session_state.get(f"bia_horizon_{profile.code}", profile.default_horizon_years))
    interventions = tuple(BudgetIntervention(str(row["id"]), str(row["name"])) for row in raw_interventions)
    population = []
    for index in range(horizon):
        eligible_key = f"bia_eligible_{horizon}_{index}"
        if eligible_key not in st.session_state:
            raise ResourceCapacityValidationError("Complete all annual BIA population inputs before resource planning.")
        covered = float(st.session_state.get(f"bia_covered_{horizon}_{index}", 0.0) or 0.0)
        population.append(BIAPopulationYear(index + 1, float(st.session_state[eligible_key]), covered if covered > 0 else None))
    mix = []
    for year in range(1, horizon + 1):
        for scenario in ("current", "future"):
            for intervention in interventions:
                key = f"bia_share_{scenario}_{year}_{intervention.id}"
                if key not in st.session_state:
                    raise ResourceCapacityValidationError("Complete the current/future BIA treatment mix before resource planning.")
                mix.append(BIATreatmentMixShare(scenario, year, intervention.id, float(st.session_state[key])))
    basis_label = st.selectbox(
        "Meaning of the annual BIA population for capacity planning",
        ["Annual treated / eligible population", "New treatment starts"],
        key="rc_bia_population_basis_label",
        help="BIA itself does not determine whether annual population values are a prevalence stock or incident/new-start cohorts. Declare that meaning explicitly for capacity planning.",
    )
    basis = "annual_eligible_population" if basis_label.startswith("Annual") else "new_treatment_starts"
    return interventions, tuple(population), tuple(mix), horizon, basis, "Budget Impact Analysis"


def _standalone_context():
    st.markdown("### Local population and uptake inputs")
    st.caption("For repeated use across analyses, prefer the dedicated Population & Uptake workspace. These inputs are kept local to this capacity plan.")
    c1, c2 = st.columns(2)
    horizon = int(c1.number_input("Planning horizon (years)", 1, 10, 3, 1, key="rc_standalone_horizon"))
    basis_label = c2.selectbox(
        "Annual population basis",
        ["Annual eligible / treated population", "New treatment starts"],
        key="rc_standalone_basis_label",
    )
    basis = "annual_eligible_population" if basis_label.startswith("Annual eligible") else "new_treatment_starts"

    raw_options = st.session_state.rc_standalone_options
    option_ids = [str(row["id"]) for row in raw_options]
    selected = st.selectbox("Option to edit", option_ids, format_func=lambda oid: next(row["name"] for row in raw_options if row["id"] == oid), key="rc_standalone_selected")
    selected_index = next(index for index, row in enumerate(raw_options) if row["id"] == selected)
    edited = dict(raw_options[selected_index])
    edited["name"] = st.text_input("Option name", str(edited["name"]), key=f"rc_standalone_name_{selected}")
    raw_options[selected_index] = edited
    with st.expander("+ Add another option"):
        with st.form("rc_add_standalone_option", clear_on_submit=True):
            name = st.text_input("New option name")
            if st.form_submit_button("Add option") and name.strip():
                existing = {str(row["id"]) for row in raw_options}
                raw_options.append({"id": _unique_id(name, existing), "name": name.strip()})
                st.rerun()
    st.session_state.rc_standalone_options = raw_options
    options = tuple(PopulationOption(str(row["id"]), str(row["name"])) for row in raw_options)

    population = []
    mix = []
    for year in range(1, horizon + 1):
        with st.expander(f"Population and mix — Year {year}", expanded=year == 1):
            pop = st.number_input(f"Year {year} population", min_value=0.0, value=float(st.session_state.get(f"rc_standalone_pop_{year}", 10000.0)), step=100.0, key=f"rc_standalone_pop_{year}")
            population.append(PopulationYear(year, pop))
            for scenario in ("current", "future"):
                st.markdown(f"**{scenario.title()} mix**")
                cols = st.columns(len(options))
                shares = []
                for index, option in enumerate(options):
                    default = (1.0 if index == 0 else 0.0) if scenario == "current" else (0.7 if index == 0 else (0.3 if index == 1 else 0.0))
                    value = cols[index].number_input(
                        option.name,
                        0.0,
                        1.0,
                        float(st.session_state.get(f"rc_standalone_share_{scenario}_{year}_{option.id}", default)),
                        format="%.4f",
                        key=f"rc_standalone_share_{scenario}_{year}_{option.id}",
                    )
                    shares.append(value)
                    mix.append(TreatmentMixShare(scenario, year, option.id, value))
                if abs(sum(shares) - 1.0) > 1e-8:
                    st.error(f"{scenario.title()} shares sum to {sum(shares):.4f}; they must sum to 1.")
    return options, tuple(population), tuple(mix), horizon, basis, "Defined in capacity workspace"


def _manual_requirements(interventions, resource_ids, horizon, demand_basis):
    rows = []
    for intervention in interventions:
        for resource_id in resource_ids:
            for period in range(1, horizon + 1):
                value = float(st.session_state.get(f"rc_req_{demand_basis}_{intervention.id}_{resource_id}_{period}", 0.0))
                if value > 0:
                    rows.append(ResourceRequirement(intervention.id, resource_id, period, value))
    return tuple(rows)


def _capacities(resource_ids, horizon):
    rows = []
    for resource_id in resource_ids:
        for year in range(1, horizon + 1):
            rows.append(
                AnnualResourceCapacity(
                    resource_id,
                    year,
                    float(st.session_state.get(f"rc_capacity_total_{resource_id}_{year}", 1000.0)),
                    float(st.session_state.get(f"rc_capacity_committed_{resource_id}_{year}", 0.0)),
                )
            )
    return tuple(rows)


def _available_clinical_models():
    models = []
    if all(key in st.session_state for key in ("dt_parameter_rows", "dt_strategy_rows", "dt_node_rows", "dt_branch_rows")):
        models.append("Decision Tree")
    if all(key in st.session_state for key in ("markov_parameters", "markov_states", "markov_strategies", "markov_initial", "markov_transitions")):
        models.append("Cohort Markov")
    if all(key in st.session_state for key in ("adv_parameters", "adv_states", "adv_strategies", "adv_initial", "adv_transitions")):
        models.append("Advanced Markov")
    return models


def _compile_clinical_model(model_type):
    if model_type == "Decision Tree":
        compiled = compile_builder_tables(
            _records(st.session_state.dt_parameter_rows),
            _records(st.session_state.dt_strategy_rows),
            _records(st.session_state.dt_node_rows),
            _records(st.session_state.dt_branch_rows),
        )
        locations = {node.id: node.label for node in (*compiled.tree.chance_nodes, *compiled.tree.terminal_nodes)}
        strategies = dict(compiled.strategy_names)
        return "decision_tree", compiled, strategies, locations
    if model_type == "Cohort Markov":
        cycle_length = int(st.session_state.get("mk_cycle_months", 12)) / 12.0
        horizon = int(st.session_state.get("mk_horizon_years", 20))
        termination = st.session_state.get("mk_termination", "Fixed horizon")
        compiled = compile_markov_tables(
            _records(st.session_state.markov_parameters),
            _records(st.session_state.markov_states),
            _records(st.session_state.markov_strategies),
            _records(st.session_state.markov_initial),
            _records(st.session_state.markov_transitions),
            _records(st.session_state.markov_state_rewards),
            _records(st.session_state.markov_transition_rewards),
            cycle_length_years=cycle_length,
            max_cycles=max(1, int(round(horizon / cycle_length))),
            state_accrual_timing=st.session_state.get("mk_state_accrual", "half_cycle"),
            transition_reward_timing=st.session_state.get("mk_transition_timing", "mid_cycle"),
            termination_mode="fixed_cycles" if termination == "Fixed horizon" else "cohort_depletion",
            depletion_threshold=float(st.session_state.get("mk_depletion", 0.0001)),
        )
        locations = {state.id: state.label for state in compiled.model.states}
        return "cohort_markov", compiled, dict(compiled.strategy_names), locations
    cycle_length = int(st.session_state.get("adv_cycle_months", 12)) / 12.0
    horizon = int(st.session_state.get("adv_horizon_years", 20))
    termination = st.session_state.get("adv_termination", "Fixed horizon")
    compiled = compile_semi_markov_tables(
        _records(st.session_state.adv_parameters),
        _records(st.session_state.adv_states),
        _records(st.session_state.adv_strategies),
        _records(st.session_state.adv_initial),
        _records(st.session_state.adv_transitions),
        _records(st.session_state.adv_state_rewards),
        _records(st.session_state.adv_transition_rewards),
        _records(st.session_state.adv_mortality_table),
        _records(st.session_state.adv_mortality_rules),
        cycle_length_years=cycle_length,
        max_cycles=max(1, int(round(horizon / cycle_length))),
        state_accrual_timing=st.session_state.get("adv_state_accrual", "half_cycle"),
        transition_reward_timing=st.session_state.get("adv_transition_timing", "mid_cycle"),
        termination_mode="fixed_cycles" if termination == "Fixed horizon" else "cohort_depletion",
        depletion_threshold=float(st.session_state.get("adv_depletion", 0.0001)),
    )
    strategies = {strategy.strategy_id: strategy.label for strategy in compiled.model.strategies}
    locations = {state.id: state.label for state in compiled.model.states}
    return "semi_markov", compiled, strategies, locations


def _linked_requirements(interventions, resource_ids, horizon_years, demand_basis):
    if demand_basis != "new_treatment_starts":
        raise ClinicalResourceLinkageError(
            "Longitudinal clinical resource profiles require annual new treatment starts. A cross-sectional annual population does not identify time since treatment; use manual annual requirements or change to an explicit new-start population basis."
        )
    available_models = _available_clinical_models()
    if not available_models:
        raise ClinicalResourceLinkageError("Build a Decision Tree, Cohort Markov or Advanced Markov model before using clinical resource linkage.")
    model_type = st.session_state.get("rc_clinical_model_type", available_models[0])
    if model_type not in available_models:
        model_type = available_models[0]
    kind, compiled, strategies, locations = _compile_clinical_model(model_type)
    parameters = compiled.parameters
    resource_parameters = available_resource_parameters(parameters)
    if not resource_parameters:
        raise ClinicalResourceLinkageError(
            "The selected clinical model has no parameters with category 'resource_use'. Add explicit natural-unit resource parameters to the clinical parameter library before linking them."
        )
    parameter_labels = {parameter.id: parameter.label for parameter in parameters}
    current = [row for row in st.session_state.rc_clinical_mappings if row.get("model_type") == model_type]
    mappings = []
    for row in current:
        if row.get("resource_id") not in resource_ids or row.get("strategy_id") not in strategies or row.get("location_id") not in locations or row.get("parameter_id") not in resource_parameters:
            continue
        if kind == "decision_tree":
            mappings.append(
                TreeResourceMapping(
                    str(row["strategy_id"]),
                    str(row["location_id"]),
                    str(row["resource_id"]),
                    str(row["parameter_id"]),
                    float(row.get("time_years", 0.0)),
                )
            )
        else:
            mappings.append(
                StateResourceMapping(
                    str(row["strategy_id"]),
                    str(row["location_id"]),
                    str(row["resource_id"]),
                    str(row["parameter_id"]),
                    str(row.get("accrual", "per_year")),
                )
            )
    if kind == "decision_tree":
        profiles = project_decision_tree_resource_profiles(
            compiled.tree,
            parameters,
            mappings=mappings,
            horizon_years=horizon_years,
            strategy_names=strategies,
        )
    elif kind == "cohort_markov":
        profiles = project_markov_resource_profiles(compiled.model, parameters, mappings=mappings, horizon_years=horizon_years)
    else:
        profiles = project_semi_markov_resource_profiles(compiled.model, parameters, mappings=mappings, horizon_years=horizon_years)
    mapping = {
        intervention.id: str(st.session_state.get(f"rc_strategy_map_{model_type}_{intervention.id}", next(iter(strategies))))
        for intervention in interventions
    }
    return profiles_to_requirements(profiles, intervention_to_strategy=mapping, horizon_years=horizon_years), profiles


_ensure_resources()

population_source = st.sidebar.selectbox(
    "Population & uptake source",
    ["Shared Population & Uptake", "Budget Impact Analysis", "Define in this workspace"],
    key="rc_population_source",
)
requirement_source = st.sidebar.selectbox(
    "Resource-requirement source",
    ["Manual", "Linked clinical model", "Hybrid — clinical + manual"],
    key="rc_requirement_source",
)

try:
    if population_source == "Shared Population & Uptake":
        interventions, population, treatment_mix, horizon_years, population_basis, source_label = _shared_context()
    elif population_source == "Budget Impact Analysis":
        interventions, population, treatment_mix, horizon_years, population_basis, source_label = _bia_context()
    else:
        interventions, population, treatment_mix, horizon_years, population_basis, source_label = _standalone_context()
except (ResourceCapacityValidationError, ValueError) as exc:
    st.error(str(exc))
    c1, c2 = st.columns(2)
    with c1:
        st.page_link("pages/10_Population_Uptake.py", label="Open Population & Uptake →", icon="👥")
    with c2:
        st.page_link("pages/6_Budget_Impact_Analysis.py", label="Open Budget Impact Analysis →", icon="💷")
    st.stop()

demand_basis = "new_treatment_starts" if population_basis == "new_treatment_starts" else "annual_treated_population"
status_bar([
    (source_label, "blue"),
    (f"{horizon_years} years", "neutral"),
    (f"{len(interventions)} options", "teal"),
    ("New treatment starts" if demand_basis == "new_treatment_starts" else "Annual treated population", "neutral"),
])

if demand_basis == "new_treatment_starts":
    st.info("Annual population values are interpreted as new treatment starts. Longitudinal manual or clinical resource profiles are stacked across initiation cohorts.")
else:
    st.info("Annual population values are interpreted cross-sectionally. Resource requirements apply to the population in each budget year; longitudinal clinical trajectories are not inferred from a prevalence stock.")

resources_tab, requirements_tab, clinical_tab, capacity_tab, results_tab, transparency_tab = st.tabs(
    ["1 · Resources", "2 · Manual requirements", "3 · Clinical linkage", "4 · Available capacity", "5 · Results", "6 · Transparency check"]
)

with resources_tab:
    st.subheader("Define physical resources")
    st.write("Resources are not limited to staff time. Define the operational quantity that can become constrained, then use the same natural unit for demand and available capacity.")
    st.caption("Examples include staff-hours, chair-hours, bed-days, syringes, vials, ambulance trips, vehicle-hours, oxygen litres/cylinders, blood units, scans, tests, procedures, treatment slots and device units.")

    with st.expander("Common resource examples and unit choices", expanded=True):
        st.markdown(
            "**Choose the unit that matches the actual constraint.** For example, if demand is measured in bed-days, annual capacity should also be entered in bed-days (for example staffed beds × usable days, adjusted as appropriate) rather than as a simple count of beds. The same principle applies to oxygen, vehicles, blood products and other supplies."
        )
        example_name = st.selectbox("Quick-add example", list(RESOURCE_EXAMPLES), key="rc_resource_example")
        example = RESOURCE_EXAMPLES[example_name]
        st.caption(f"Suggested unit: `{example['unit']}` · {example['note']}")
        if st.button("Add this resource example", key="rc_add_resource_example"):
            existing_names = {str(row["name"]) for row in st.session_state.rc_resources}
            if example["name"] in existing_names:
                st.warning("That resource already exists in this capacity plan. Edit the existing resource instead of adding a duplicate name.")
            else:
                existing_ids = {str(row["id"]) for row in st.session_state.rc_resources}
                st.session_state.rc_resources.append(
                    {
                        "id": _unique_id(example["name"], existing_ids),
                        "name": example["name"],
                        "unit": example["unit"],
                        "category": example["category"],
                        "source": "Illustrative resource definition — replace with local evidence",
                        "rationale": example["note"],
                    }
                )
                st.rerun()

    resource_ids = [str(row["id"]) for row in st.session_state.rc_resources]
    selected_id = st.selectbox("Resource to edit", resource_ids, format_func=lambda rid: _resource_meta(rid)["name"], key="rc_selected_resource")
    selected_index = next(i for i, row in enumerate(st.session_state.rc_resources) if row["id"] == selected_id)
    selected = dict(st.session_state.rc_resources[selected_index])
    c1, c2, c3 = st.columns(3)
    selected["name"] = c1.text_input("Resource name", str(selected["name"]), key=f"rc_name_{selected_id}")
    selected["unit"] = c2.text_input("Natural unit", str(selected["unit"]), key=f"rc_unit_{selected_id}", help="Examples: hours, bed-days, syringes, vials, trips, vehicle-hours, litres, cylinders, blood units, scans or device units.")
    selected["category"] = c3.selectbox("Resource category", list(RESOURCE_CATEGORIES), index=list(RESOURCE_CATEGORIES).index(str(selected["category"])), format_func=lambda code: CATEGORY_LABELS[code], key=f"rc_category_{selected_id}")
    selected["source"] = st.text_area("Resource evidence source", str(selected.get("source", "")), key=f"rc_source_{selected_id}")
    selected["rationale"] = st.text_area("Resource definition / inclusion rationale", str(selected.get("rationale", "")), key=f"rc_rationale_{selected_id}")
    st.session_state.rc_resources[selected_index] = selected
    with st.expander("+ Add a custom resource"):
        with st.form("rc_add_resource", clear_on_submit=True):
            new_name = st.text_input("Resource name")
            new_unit = st.text_input("Natural unit", placeholder="e.g. bed-days, syringes, vials, trips, litres, blood units")
            new_category = st.selectbox("Category", list(RESOURCE_CATEGORIES), format_func=lambda code: CATEGORY_LABELS[code])
            if st.form_submit_button("Add resource") and new_name.strip() and new_unit.strip():
                existing = {str(row["id"]) for row in st.session_state.rc_resources}
                st.session_state.rc_resources.append({"id": _unique_id(new_name, existing), "name": new_name.strip(), "unit": new_unit.strip(), "category": new_category, "source": "", "rationale": ""})
                st.rerun()
    if len(st.session_state.rc_resources) > 1 and st.button("Delete selected resource", key="rc_delete_resource"):
        st.session_state.rc_resources = [row for row in st.session_state.rc_resources if row["id"] != selected_id]
        st.rerun()

with requirements_tab:
    st.subheader("Manual resource requirements")
    if requirement_source == "Linked clinical model":
        st.info("Manual requirements are ignored in linked-only mode. Use Hybrid when some resources come from the clinical model and others require direct service-planning inputs.")
    resource_ids = [str(row["id"]) for row in st.session_state.rc_resources]
    labels = {item.id: item.name for item in interventions}
    c1, c2 = st.columns(2)
    req_intervention = c1.selectbox("Option", [item.id for item in interventions], format_func=lambda iid: labels[iid], key="rc_req_intervention")
    req_resource = c2.selectbox("Resource", resource_ids, format_func=lambda rid: _resource_meta(rid)["name"], key="rc_req_resource")
    unit = _resource_meta(req_resource)["unit"]
    for period in range(1, horizon_years + 1):
        label = f"Budget Year {period} — {unit} per person" if demand_basis == "annual_treated_population" else f"Year {period} since initiation — {unit} per new start"
        key = f"rc_req_{demand_basis}_{req_intervention}_{req_resource}_{period}"
        st.number_input(label, min_value=0.0, value=float(st.session_state.get(key, 0.0)), format="%.6f", key=key)
    source_key = f"rc_req_source_{demand_basis}_{req_intervention}_{req_resource}"
    rationale_key = f"rc_req_rationale_{demand_basis}_{req_intervention}_{req_resource}"
    st.text_area("Evidence source for this manual resource profile", value=str(st.session_state.get(source_key, "")), key=source_key)
    st.text_area("Manual resource-use assumptions / rationale", value=str(st.session_state.get(rationale_key, "")), key=rationale_key)

clinical_profiles = ()
linked_requirements = ()
clinical_error = None
with clinical_tab:
    st.subheader("Link natural-unit resource use to a clinical model")
    st.caption("Clinical linkage is optional. It maps explicit `resource_use` parameters to clinical states or decision-tree nodes; it never converts costs into physical resources automatically.")
    available_models = _available_clinical_models()
    if not available_models:
        st.info("No clinical modeller is currently available in this session. Build a Decision Tree, Cohort Markov or Advanced Markov model first if you want linked resource trajectories.")
    else:
        model_type = st.selectbox("Clinical model", available_models, key="rc_clinical_model_type")
        try:
            kind, compiled_clinical, clinical_strategies, clinical_locations = _compile_clinical_model(model_type)
            resource_parameters = available_resource_parameters(compiled_clinical.parameters)
        except Exception as exc:
            st.error(f"Clinical model could not be compiled: {exc}")
            resource_parameters = ()
            clinical_strategies = {}
            clinical_locations = {}
            kind = ""
        if not resource_parameters:
            st.warning("Add one or more parameters with category `resource_use` to the selected clinical model. Values may be hours/year, bed-days/year, syringes/event, vials/event, ambulance trips/event, oxygen litres/event, blood units/event, scans/event, or another documented natural unit.")
        else:
            parameter_labels = {parameter.id: parameter.label for parameter in compiled_clinical.parameters}
            resource_ids = [str(row["id"]) for row in st.session_state.rc_resources]
            with st.expander("+ Add clinical resource mapping", expanded=not any(row.get("model_type") == model_type for row in st.session_state.rc_clinical_mappings)):
                c1, c2 = st.columns(2)
                strategy_id = c1.selectbox("Clinical strategy", list(clinical_strategies), format_func=lambda sid: clinical_strategies[sid], key="rc_map_strategy")
                location_id = c2.selectbox("Clinical state / node", list(clinical_locations), format_func=lambda lid: clinical_locations[lid], key="rc_map_location")
                c1, c2 = st.columns(2)
                resource_id = c1.selectbox("Capacity resource", resource_ids, format_func=lambda rid: _resource_meta(rid)["name"], key="rc_map_resource")
                parameter_id = c2.selectbox("Resource-use parameter", list(resource_parameters), format_func=lambda pid: parameter_labels[pid], key="rc_map_parameter")
                mapping = {"model_type": model_type, "strategy_id": strategy_id, "location_id": location_id, "resource_id": resource_id, "parameter_id": parameter_id}
                if kind == "decision_tree":
                    mapping["time_years"] = st.number_input("Absolute model time for this resource event (years)", min_value=0.0, value=0.0, step=0.25, key="rc_map_time")
                else:
                    mapping["accrual"] = st.selectbox("Accrual", ["per_year", "per_cycle"], format_func=lambda value: "Per year in state" if value == "per_year" else "Per model cycle in state", key="rc_map_accrual")
                if st.button("Add mapping", key="rc_add_mapping"):
                    st.session_state.rc_clinical_mappings.append(mapping)
                    st.rerun()

            current = [row for row in st.session_state.rc_clinical_mappings if row.get("model_type") == model_type]
            if current:
                st.markdown("#### Current mappings")
                for index, row in enumerate(current):
                    text = f"**{clinical_strategies.get(row['strategy_id'], row['strategy_id'])}** · {clinical_locations.get(row['location_id'], row['location_id'])} → {_resource_meta(row['resource_id'])['name']} using `{row['parameter_id']}`"
                    c1, c2 = st.columns([6, 1])
                    c1.write(text)
                    if c2.button("Delete", key=f"rc_delete_mapping_{model_type}_{index}"):
                        target = row
                        removed = False
                        updated = []
                        for existing in st.session_state.rc_clinical_mappings:
                            if not removed and existing == target:
                                removed = True
                                continue
                            updated.append(existing)
                        st.session_state.rc_clinical_mappings = updated
                        st.rerun()
            st.markdown("#### Map policy options to clinical strategies")
            for intervention in interventions:
                st.selectbox(
                    intervention.name,
                    list(clinical_strategies),
                    format_func=lambda sid: clinical_strategies[sid],
                    key=f"rc_strategy_map_{model_type}_{intervention.id}",
                )

        if requirement_source != "Manual":
            try:
                linked_requirements, clinical_profiles = _linked_requirements(interventions, [str(row["id"]) for row in st.session_state.rc_resources], horizon_years, demand_basis)
            except (ClinicalResourceLinkageError, ValueError) as exc:
                clinical_error = str(exc)
                st.error(clinical_error)
            else:
                st.success("Clinical resource profiles compiled successfully.")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Clinical strategy": profile.strategy_name,
                            "Resource": _resource_meta(profile.resource_id)["name"],
                            "Source model": profile.source_model_type,
                            "Parameter ids": ", ".join(profile.included_parameter_ids),
                            **{f"Year {year}": value for year, value in enumerate(profile.annual_units_per_patient, start=1)},
                        }
                        for profile in clinical_profiles
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

with capacity_tab:
    st.subheader("Capacity available to the modelled population")
    st.write("Enter total service capacity and the amount already committed to other services/populations. The difference is residual capacity available to this modelled population.")
    resource_ids = [str(row["id"]) for row in st.session_state.rc_resources]
    cap_resource = st.selectbox("Resource to configure", resource_ids, format_func=lambda rid: _resource_meta(rid)["name"], key="rc_cap_resource")
    unit = _resource_meta(cap_resource)["unit"]
    st.caption(f"Capacity for this resource must use the same unit as demand: `{unit}`. If the operational stock is expressed differently (for example beds rather than bed-days), convert it explicitly and document that assumption rather than mixing units.")
    for year in range(1, horizon_years + 1):
        total_key = f"rc_capacity_total_{cap_resource}_{year}"
        committed_key = f"rc_capacity_committed_{cap_resource}_{year}"
        with st.expander(f"Year {year}", expanded=year == 1):
            c1, c2 = st.columns(2)
            total = c1.number_input(f"Total capacity ({unit})", min_value=0.0, value=float(st.session_state.get(total_key, 1000.0)), format="%.4f", key=total_key)
            committed = c2.number_input(f"Committed to other uses ({unit})", min_value=0.0, value=float(st.session_state.get(committed_key, 0.0)), format="%.4f", key=committed_key)
            if committed > total:
                st.error("Committed other demand exceeds total capacity. Correct the inputs; the app will not silently truncate this value.")
            else:
                st.caption(f"Available to this modelled population: {total - committed:,.4f} {unit}")
    st.text_area("Capacity evidence source", value=str(st.session_state.get(f"rc_capacity_source_{cap_resource}", "")), key=f"rc_capacity_source_{cap_resource}")
    st.text_area("Capacity assumptions / rationale", value=str(st.session_state.get(f"rc_capacity_rationale_{cap_resource}", "")), key=f"rc_capacity_rationale_{cap_resource}")

compiled_definition = None
capacity_run = None
compile_error = None
resource_ids = [str(row["id"]) for row in st.session_state.rc_resources]
manual_requirements = _manual_requirements(interventions, resource_ids, horizon_years, demand_basis)
if requirement_source == "Manual":
    effective_requirements = manual_requirements
elif requirement_source == "Linked clinical model":
    effective_requirements = linked_requirements
else:
    effective_requirements = combine_resource_requirements(manual_requirements, linked_requirements) if not clinical_error else manual_requirements

try:
    if requirement_source != "Manual" and clinical_error:
        raise ResourceCapacityValidationError("Clinical linkage is selected but is not valid: " + clinical_error)
    compiled_definition = ResourceCapacityDefinition(
        interventions=tuple(interventions),
        population=tuple(population),
        treatment_mix=tuple(treatment_mix),
        resources=_resource_definitions(),
        requirements=tuple(effective_requirements),
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
        comparison_df = pd.DataFrame([
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
        ])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True, column_config={"Future utilisation": st.column_config.NumberColumn(format="%.1%")})
        selected_result_resource = st.selectbox("Resource to visualise", [resource.id for resource in compiled_definition.resources], format_func=lambda rid: next(resource.name for resource in compiled_definition.resources if resource.id == rid), key="rc_result_resource")
        selected_name = next(resource.name for resource in compiled_definition.resources if resource.id == selected_result_resource)
        plot_df = comparison_df[comparison_df["Resource"] == selected_name]
        long_df = plot_df.melt(id_vars=["Year", "Unit"], value_vars=["Current demand", "Future demand", "Available capacity"], var_name="Series", value_name="Natural units")
        st.plotly_chart(px.line(long_df, x="Year", y="Natural units", color="Series", markers=True, title="Current demand, future demand and available capacity"), use_container_width=True)
        if shortfalls:
            st.warning("At least one resource exceeds residual available capacity under the future mix. The result reports the physical constraint; it does not assume rationing, funding or service expansion.")
        else:
            st.success("No future-scenario capacity shortfall is identified under the current inputs.")
        with st.expander("Demand by option / initiation cohort"):
            st.dataframe(pd.DataFrame([
                {
                    "Scenario": row.scenario,
                    "Budget year": row.budget_year,
                    "Option": row.intervention_name,
                    "Resource": row.resource_name,
                    "Initiation year": row.initiation_year,
                    "Source period": row.source_period,
                    "People": row.treated_people,
                    "Units/person": row.units_per_person,
                    "Required units": row.required_units,
                }
                for row in capacity_run.intervention_rows
            ]), use_container_width=True, hide_index=True)
        st.markdown("### Capacity expansion planner")
        c1, c2, c3, c4 = st.columns(4)
        target_resource = c1.selectbox("Resource", [resource.id for resource in compiled_definition.resources], format_func=lambda rid: next(resource.name for resource in compiled_definition.resources if resource.id == rid), key="rc_scenario_resource")
        from_year = int(c2.number_input("Expansion starts in year", 1, horizon_years, 1, step=1))
        capacity_multiplier = c3.number_input("Capacity multiplier", min_value=0.0, value=1.0, step=0.05, format="%.3f")
        additional_capacity = c4.number_input("Additional units/year", value=0.0, step=10.0, format="%.3f")
        try:
            scenario_run = run_resource_capacity_plan(apply_capacity_expansion(compiled_definition, resource_id=target_resource, from_year=from_year, capacity_multiplier=capacity_multiplier, additional_capacity=additional_capacity))
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
        st.download_button("Download capacity results (CSV)", comparison_df.to_csv(index=False), file_name="resource_capacity_results.csv", mime="text/csv")

with transparency_tab:
    st.subheader("Transparency check")
    st.caption("Checks documentation completeness for the population source, resource definitions, manual/clinical requirements and capacity assumptions. It is not a quality score.")
    st.write(f"**Population & uptake source:** {source_label}")
    st.write(f"**Population basis:** {'new treatment starts' if demand_basis == 'new_treatment_starts' else 'annual eligible/treated population'}")
    st.write(f"**Resource-requirement source:** {requirement_source}")
    checks = []
    for resource in st.session_state.rc_resources:
        rid = str(resource["id"])
        checks.extend([
            (f"{resource['name']}: resource evidence documented", _documented(resource.get("source"))),
            (f"{resource['name']}: inclusion rationale documented", _documented(resource.get("rationale"))),
            (f"{resource['name']}: capacity evidence documented", _documented(st.session_state.get(f"rc_capacity_source_{rid}", ""))),
            (f"{resource['name']}: capacity rationale documented", _documented(st.session_state.get(f"rc_capacity_rationale_{rid}", ""))),
        ])
        if requirement_source in {"Manual", "Hybrid — clinical + manual"}:
            for intervention in interventions:
                non_zero = any(float(st.session_state.get(f"rc_req_{demand_basis}_{intervention.id}_{rid}_{period}", 0.0)) > 0 for period in range(1, horizon_years + 1))
                if non_zero:
                    checks.extend([
                        (f"{intervention.name} × {resource['name']}: manual resource-use source documented", _documented(st.session_state.get(f"rc_req_source_{demand_basis}_{intervention.id}_{rid}", ""))),
                        (f"{intervention.name} × {resource['name']}: manual resource-use rationale documented", _documented(st.session_state.get(f"rc_req_rationale_{demand_basis}_{intervention.id}_{rid}", ""))),
                    ])
    if requirement_source != "Manual":
        checks.append(("Clinical resource linkage compiles", not bool(clinical_error) and bool(clinical_profiles)))
    complete = sum(1 for _, ok in checks if ok)
    status_bar([(f"{complete}/{len(checks)} documentation checks complete", "green" if checks and complete == len(checks) else "amber"), ("Not a quality score", "neutral")])
    for label, ok in checks:
        st.write(("✓ " if ok else "⚠ ") + label)
    if demand_basis == "annual_treated_population" and requirement_source != "Manual":
        st.warning("Longitudinal clinical resource linkage is intentionally unavailable for a cross-sectional population stock because time since treatment is not identified.")

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/10_Population_Uptake.py", label="Population & Uptake", icon="👥")
with c2:
    st.page_link("pages/6_Budget_Impact_Analysis.py", label="Budget Impact Analysis", icon="💷")
