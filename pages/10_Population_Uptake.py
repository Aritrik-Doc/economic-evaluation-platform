"""Shared population and uptake workspace for BIA and capacity planning."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model.population_uptake import (
    PopulationOption,
    PopulationUptakeDefinition,
    PopulationUptakeValidationError,
    PopulationYear,
    TreatmentMixShare,
    compound_series,
    run_population_uptake,
    top_down_eligible_population,
)
from ui.design_system import coloured_block, status_bar


st.set_page_config(page_title="Population & Uptake", page_icon="👥", layout="wide")
st.title("Population & Uptake")
st.caption("Version 0.14 — shared eligible-population and treatment-mix assumptions for affordability and implementation planning")

coloured_block(
    "Define the population once, then reuse it",
    "Population size and treatment uptake are policy-level assumptions, not properties of a per-patient clinical model. This workspace keeps them separate so the same documented population scenario can feed Budget Impact Analysis, Resource & Capacity Planning, or both.",
    tone="teal",
    kicker="Shared policy context",
)


def _slug(text: str) -> str:
    value = "".join(char.lower() if char.isalnum() else "_" for char in text.strip())
    value = "_".join(part for part in value.split("_") if part)
    return value or "option"


def _unique_id(label: str, existing: set[str]) -> str:
    base = _slug(label)
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _init_state() -> None:
    if "pu_options" not in st.session_state:
        st.session_state.pu_options = [
            {"id": "current_treatment", "name": "Current treatment"},
            {"id": "new_intervention", "name": "New intervention"},
        ]
    st.session_state.setdefault("pu_population_source", "Illustrative population — replace with local epidemiology or service data")
    st.session_state.setdefault("pu_population_rationale", "Replace with a documented derivation of the population relevant to the decision.")
    st.session_state.setdefault("pu_current_mix_source", "Illustrative current mix — replace with local utilisation data")
    st.session_state.setdefault("pu_current_mix_rationale", "Current allocation should reflect actual routine care in the target system.")
    st.session_state.setdefault("pu_future_mix_source", "Illustrative future uptake — replace with forecast evidence")
    st.session_state.setdefault("pu_future_mix_rationale", "Future uptake should reflect access, implementation and expected substitution.")


def _copy_to_bia(definition: PopulationUptakeDefinition) -> None:
    """Copy the current shared scenario into BIA as an explicit editable snapshot."""
    horizon = len(definition.population)
    st.session_state["bia_profile"] = "CUSTOM"
    st.session_state["bia_horizon_custom"] = horizon
    st.session_state["bia_population_mode"] = "Direct annual eligible population"
    st.session_state["bia_interventions"] = [
        {"id": option.id, "name": option.name} for option in definition.options
    ]

    existing_costs = dict(st.session_state.get("bia_cost_inputs", {}))
    copied_costs = {}
    for option in definition.options:
        copied_costs[option.id] = dict(
            existing_costs.get(
                option.id,
                {
                    "acquisition": 0.0,
                    "administration": 0.0,
                    "monitoring": 0.0,
                    "adverse_events": 0.0,
                    "disease_management": 0.0,
                    "other": 0.0,
                    "annual_change": 0.0,
                    "source": "",
                    "rationale": "",
                },
            )
        )
    st.session_state["bia_cost_inputs"] = copied_costs

    for index, row in enumerate(definition.population):
        st.session_state[f"bia_eligible_{horizon}_{index}"] = float(row.eligible_population)
        st.session_state[f"bia_covered_{horizon}_{index}"] = float(row.covered_lives or 0.0)
    for row in definition.treatment_mix:
        st.session_state[f"bia_share_{row.scenario}_{row.year}_{row.intervention_id}"] = float(row.share)

    st.session_state["bia_population_source"] = st.session_state.pu_population_source
    st.session_state["bia_population_rationale"] = st.session_state.pu_population_rationale
    st.session_state["bia_current_mix_source"] = st.session_state.pu_current_mix_source
    st.session_state["bia_current_mix_rationale"] = st.session_state.pu_current_mix_rationale
    st.session_state["bia_future_mix_source"] = st.session_state.pu_future_mix_source
    st.session_state["bia_future_mix_rationale"] = st.session_state.pu_future_mix_rationale
    st.session_state["bia_population_uptake_snapshot_basis"] = definition.population_basis


_init_state()

methods_tab, population_tab, uptake_tab, review_tab = st.tabs(
    ["1 · Population frame", "2 · Annual population", "3 · Current & future uptake", "4 · Review & reuse"]
)

with methods_tab:
    st.subheader("Population frame")
    c1, c2 = st.columns(2)
    horizon = int(c1.number_input("Planning horizon (years)", min_value=1, max_value=10, value=3, step=1, key="pu_horizon"))
    basis_label = c2.selectbox(
        "What does each annual population value represent?",
        ["Annual eligible / treated population", "New treatment starts"],
        key="pu_basis_label",
        help="Choose explicitly. A cross-sectional eligible population and annual new starts are not interchangeable in longitudinal resource or clinical-cost projections.",
    )
    population_basis = "annual_eligible_population" if basis_label.startswith("Annual eligible") else "new_treatment_starts"
    st.session_state.pu_population_basis = population_basis

    if population_basis == "annual_eligible_population":
        st.info("Each year is interpreted as the population eligible for or receiving the modelled options in that year. This is suitable for cross-sectional annual costing/resource use when the same individuals are not automatically treated as new cohorts.")
    else:
        st.warning("Each year is interpreted as a cohort of new treatment starts. Downstream longitudinal models may stack Year-1, Year-2 and later consequences across initiation cohorts.")

    st.markdown("#### Options / interventions")
    option_ids = [str(row["id"]) for row in st.session_state.pu_options]
    selected_id = st.selectbox(
        "Option to edit",
        option_ids,
        format_func=lambda oid: next(row["name"] for row in st.session_state.pu_options if row["id"] == oid),
        key="pu_selected_option",
    )
    index = next(i for i, row in enumerate(st.session_state.pu_options) if row["id"] == selected_id)
    updated = dict(st.session_state.pu_options[index])
    updated["name"] = st.text_input("Option name", str(updated["name"]), key=f"pu_option_name_{selected_id}")
    st.caption(f"Stable option ID: `{selected_id}`")
    st.session_state.pu_options[index] = updated

    with st.expander("+ Add another option"):
        with st.form("pu_add_option", clear_on_submit=True):
            new_name = st.text_input("Option name")
            if st.form_submit_button("Add option") and new_name.strip():
                existing = {str(row["id"]) for row in st.session_state.pu_options}
                st.session_state.pu_options.append({"id": _unique_id(new_name, existing), "name": new_name.strip()})
                st.rerun()
    if len(st.session_state.pu_options) > 2 and st.button("Delete selected option", key="pu_delete_option"):
        st.session_state.pu_options = [row for row in st.session_state.pu_options if row["id"] != selected_id]
        st.rerun()

with population_tab:
    st.subheader("Eligible population by year")
    mode = st.radio(
        "Population entry method",
        ["Direct annual population", "Top-down funnel"],
        horizontal=True,
        key="pu_population_mode",
    )
    covered_series = None
    if mode == "Top-down funnel":
        c1, c2, c3 = st.columns(3)
        covered_start = c1.number_input("Covered / catchment population — Year 1", min_value=0.0, value=1_000_000.0, step=10_000.0, key="pu_covered_start")
        prevalence = c2.number_input("Prevalence / target-condition proportion", 0.0, 1.0, 0.01, format="%.6f", key="pu_prevalence")
        diagnosed = c3.number_input("Diagnosed / identified proportion", 0.0, 1.0, 0.80, format="%.4f", key="pu_diagnosed")
        c1, c2, c3 = st.columns(3)
        eligible_prop = c1.number_input("Clinically eligible proportion", 0.0, 1.0, 0.75, format="%.4f", key="pu_eligible_prop")
        access = c2.number_input("Access / covered proportion", 0.0, 1.0, 0.90, format="%.4f", key="pu_access")
        growth = c3.number_input("Annual covered-population growth", value=0.01, format="%.4f", key="pu_population_growth")
        covered_series = compound_series(covered_start, growth, horizon)
        derived = tuple(top_down_eligible_population(value, prevalence, diagnosed, eligible_prop, access) for value in covered_series)
    else:
        c1, c2, c3 = st.columns(3)
        start = c1.number_input("Eligible population — Year 1", min_value=0.0, value=10_000.0, step=100.0, key="pu_eligible_start")
        growth = c2.number_input("Annual eligible-population growth", value=0.02, format="%.4f", key="pu_eligible_growth")
        covered_start = c3.number_input("Covered lives — Year 1 (0 = not supplied)", min_value=0.0, value=0.0, step=1000.0, key="pu_covered_direct")
        derived = compound_series(start, growth, horizon)
        covered_series = None if covered_start <= 0 else compound_series(covered_start, growth, horizon)

    population_rows = []
    for year in range(1, horizon + 1):
        c1, c2 = st.columns(2)
        eligible_key = f"pu_eligible_{year}"
        covered_key = f"pu_covered_{year}"
        eligible_value = c1.number_input(
            f"Year {year} population",
            min_value=0.0,
            value=float(st.session_state.get(eligible_key, derived[year - 1])),
            step=max(float(derived[year - 1]) * 0.01, 1.0),
            key=eligible_key,
        )
        covered_default = float(covered_series[year - 1]) if covered_series is not None else 0.0
        covered_value = c2.number_input(
            f"Year {year} covered lives (0 = not supplied)",
            min_value=0.0,
            value=float(st.session_state.get(covered_key, covered_default)),
            step=max(covered_default * 0.01, 100.0) if covered_default else 1000.0,
            key=covered_key,
        )
        try:
            population_rows.append(PopulationYear(year, eligible_value, covered_value if covered_value > 0 else None))
        except ValueError as exc:
            st.error(f"Year {year}: {exc}")

    st.markdown("#### Evidence and rationale")
    st.session_state.pu_population_source = st.text_area("Population evidence source", st.session_state.pu_population_source, key="pu_population_source_widget")
    st.session_state.pu_population_rationale = st.text_area("Population derivation / rationale", st.session_state.pu_population_rationale, key="pu_population_rationale_widget")

with uptake_tab:
    st.subheader("Current and future option mix")
    options = [PopulationOption(str(row["id"]), str(row["name"])) for row in st.session_state.pu_options]
    mix_rows = []
    for year in range(1, horizon + 1):
        with st.expander(f"Year {year}", expanded=year == 1):
            for scenario in ("current", "future"):
                st.markdown(f"**{scenario.title()} mix**")
                cols = st.columns(len(options))
                values = []
                for index, option in enumerate(options):
                    key = f"pu_share_{scenario}_{year}_{option.id}"
                    if scenario == "current":
                        default = 1.0 if index == 0 else 0.0
                    else:
                        default = 0.7 if index == 0 else (0.3 if index == 1 else 0.0)
                    value = cols[index].number_input(
                        option.name,
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get(key, default)),
                        format="%.4f",
                        key=key,
                    )
                    values.append(value)
                    mix_rows.append(TreatmentMixShare(scenario, year, option.id, value))
                total = sum(values)
                if abs(total - 1.0) <= 1e-8:
                    st.caption("✓ Shares sum to 1.0000")
                else:
                    st.error(f"Shares sum to {total:.4f}; they must sum to 1. The platform will not normalise them automatically.")

    st.markdown("#### Uptake evidence")
    st.session_state.pu_current_mix_source = st.text_area("Current-mix evidence source", st.session_state.pu_current_mix_source, key="pu_current_mix_source_widget")
    st.session_state.pu_current_mix_rationale = st.text_area("Current-mix rationale", st.session_state.pu_current_mix_rationale, key="pu_current_mix_rationale_widget")
    st.session_state.pu_future_mix_source = st.text_area("Future-uptake evidence source", st.session_state.pu_future_mix_source, key="pu_future_mix_source_widget")
    st.session_state.pu_future_mix_rationale = st.text_area("Future-uptake rationale", st.session_state.pu_future_mix_rationale, key="pu_future_mix_rationale_widget")

compiled = None
run = None
compile_error = None
try:
    options = tuple(PopulationOption(str(row["id"]), str(row["name"])) for row in st.session_state.pu_options)
    compiled = PopulationUptakeDefinition(
        options=options,
        population=tuple(population_rows),
        treatment_mix=tuple(mix_rows),
        population_basis=population_basis,
    )
    run = run_population_uptake(compiled)
    st.session_state.pu_population_rows = [
        {"year": row.year, "eligible_population": row.eligible_population, "covered_lives": row.covered_lives}
        for row in compiled.population
    ]
    st.session_state.pu_uptake_rows = [
        {"scenario": row.scenario, "year": row.year, "intervention_id": row.intervention_id, "share": row.share}
        for row in compiled.treatment_mix
    ]
except (PopulationUptakeValidationError, ValueError) as exc:
    compile_error = str(exc)

with review_tab:
    st.subheader("Review the shared population scenario")
    if compile_error:
        st.error(compile_error)
    else:
        assert compiled is not None and run is not None
        status_bar([
            (f"{len(compiled.population)} years", "blue"),
            (f"{len(compiled.options)} options", "teal"),
            ("New starts" if compiled.population_basis == "new_treatment_starts" else "Annual eligible population", "neutral"),
            ("Reusable in BIA and capacity", "green"),
        ])
        st.dataframe(
            pd.DataFrame([
                {
                    "Scenario": row.scenario.title(),
                    "Year": row.year,
                    "Option": row.intervention_name,
                    "Population": row.eligible_population,
                    "Share": row.share,
                    "People allocated": row.treated_people,
                }
                for row in run.rows
            ]),
            use_container_width=True,
            hide_index=True,
            column_config={"Share": st.column_config.NumberColumn(format="%.1%")},
        )
        st.info("This population scenario contains no costs and no clinical outcomes. Capacity can consume it directly. Budget Impact Analysis currently receives an explicit editable snapshot so later BIA edits cannot silently change the shared scenario.")
        if st.button("Copy this population & uptake snapshot into BIA", type="primary", key="pu_copy_to_bia"):
            _copy_to_bia(compiled)
            st.success("Copied to Budget Impact Analysis as a Custom-profile snapshot. Cost inputs remain separate and must be completed in the BIA workspace.")
        c1, c2 = st.columns(2)
        with c1:
            st.page_link("pages/6_Budget_Impact_Analysis.py", label="Open Budget Impact Analysis →", icon="💷")
        with c2:
            st.page_link("pages/8_Resource_Capacity_Planning.py", label="Use directly in Resource & Capacity Planning →", icon="🏥")
