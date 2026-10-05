"""Shared population and uptake workspace for BIA and capacity planning."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model.population_projection import (
    project_direct_population,
    project_top_down_population,
)
from model.population_uptake import (
    PopulationOption,
    PopulationUptakeDefinition,
    PopulationUptakeValidationError,
    PopulationYear,
    TreatmentMixShare,
    run_population_uptake,
)
from ui.design_system import coloured_block, status_bar


st.set_page_config(page_title="Population & Uptake", page_icon="👥", layout="wide")
st.title("Population & Uptake")
st.caption("Version 0.14.1 — shared eligible-population and treatment-mix assumptions for affordability and implementation planning")

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
        st.info("Each year is a cross-sectional population relevant to that year. Use this when annual service demand/costing applies to the population present in each year and people are not automatically treated as fresh cohorts.")
    else:
        st.warning("Each year is a new initiation cohort. Longitudinal clinical/resource models can stack Year-1, Year-2 and later consequences from earlier cohorts into later calendar years.")

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
    st.write("Choose how the Year-1 population is derived, then decide whether the annual values should continue to follow that formula or be manually overridden.")

    mode = st.radio(
        "Population derivation method",
        ["Direct eligible population", "Top-down funnel"],
        horizontal=True,
        key="pu_population_mode_v0141",
    )

    if mode == "Top-down funnel":
        with st.expander("What each funnel input means", expanded=True):
            st.markdown(
                "**Eligible population = covered/catchment population × prevalence × diagnosed/identified × clinically eligible × access/coverage.**\n\n"
                "- **Covered / catchment population:** the starting population served by the payer, programme or service.\n"
                "- **Prevalence / target-condition proportion:** the share of that population with the condition or policy-relevant characteristic.\n"
                "- **Diagnosed / identified:** the share of affected people who are known to the system.\n"
                "- **Clinically eligible:** the share of diagnosed/identified people meeting the intervention criteria.\n"
                "- **Access / coverage:** the share of eligible people expected to reach or be covered by the service.\n"
                "- **Annual population growth:** changes the covered/catchment population in later years; the eligible population then changes through the same funnel."
            )
        c1, c2, c3 = st.columns(3)
        covered_start = c1.number_input("Covered / catchment population — Year 1", min_value=0.0, value=1_000_000.0, step=10_000.0, key="pu_covered_start")
        prevalence = c2.number_input("Prevalence / target-condition proportion", 0.0, 1.0, 0.01, format="%.6f", key="pu_prevalence")
        diagnosed = c3.number_input("Diagnosed / identified proportion", 0.0, 1.0, 0.80, format="%.4f", key="pu_diagnosed")
        c1, c2, c3 = st.columns(3)
        eligible_prop = c1.number_input("Clinically eligible proportion", 0.0, 1.0, 0.75, format="%.4f", key="pu_eligible_prop")
        access = c2.number_input("Access / covered proportion", 0.0, 1.0, 0.90, format="%.4f", key="pu_access")
        old_growth = float(st.session_state.get("pu_population_growth", 0.01)) * 100.0
        growth_pct = c3.number_input("Annual covered/catchment growth (%)", value=old_growth, format="%.3f", key="pu_population_growth_pct")
        projection = project_top_down_population(
            covered_or_catchment_start=covered_start,
            covered_or_catchment_growth_rate=growth_pct / 100.0,
            prevalence=prevalence,
            diagnosed_or_identified=diagnosed,
            clinically_eligible=eligible_prop,
            access_or_coverage=access,
            horizon_years=horizon,
        )
        st.caption("In top-down mode the covered/catchment population is also carried as the covered-lives denominator. If your PMPM denominator differs from the catchment population, use manual annual override below.")
    else:
        with st.expander("What each direct-population input means", expanded=True):
            st.markdown(
                "- **Eligible population — Year 1:** the number of people relevant to the policy decision in the first year.\n"
                "- **Eligible-population growth:** changes that eligible population in later years. A value of 2% makes Year 2 equal to Year 1 × 1.02.\n"
                "- **Covered lives:** an optional payer/population denominator used for outputs such as per-member-per-month. It does **not** change the eligible population in direct mode.\n"
                "- **Covered-lives growth:** changes only the denominator in later years. Keep it separate when the covered population and eligible population grow at different rates."
            )
        c1, c2 = st.columns(2)
        start = c1.number_input("Eligible population — Year 1", min_value=0.0, value=10_000.0, step=100.0, key="pu_eligible_start")
        old_eligible_growth = float(st.session_state.get("pu_eligible_growth", 0.02)) * 100.0
        eligible_growth_pct = c2.number_input("Annual eligible-population growth (%)", value=old_eligible_growth, format="%.3f", key="pu_eligible_growth_pct")
        c1, c2 = st.columns(2)
        covered_start = c1.number_input("Covered lives — Year 1 (0 = not supplied)", min_value=0.0, value=0.0, step=1000.0, key="pu_covered_direct")
        covered_growth_pct = c2.number_input("Annual covered-lives growth (%)", value=0.0, format="%.3f", key="pu_covered_growth_pct", disabled=covered_start <= 0)
        projection = project_direct_population(
            eligible_start=start,
            eligible_growth_rate=eligible_growth_pct / 100.0,
            horizon_years=horizon,
            covered_lives_start=covered_start if covered_start > 0 else None,
            covered_lives_growth_rate=covered_growth_pct / 100.0,
        )

    projection_df = pd.DataFrame([
        {
            "Year": row.year,
            "Formula-derived eligible population": row.eligible_population,
            "Formula-derived covered lives": row.covered_lives,
        }
        for row in projection
    ])

    entry_mode = st.radio(
        "How should the annual values be used?",
        ["Use formula-derived annual values", "Edit annual values manually"],
        horizontal=True,
        key="pu_annual_entry_mode",
        help="Formula-derived mode always follows the current inputs above. Manual mode freezes separate year-specific values until you edit or reset them.",
    )

    population_rows = []
    if entry_mode == "Use formula-derived annual values":
        st.success("Live projection: changing any population driver above immediately changes the annual values used by BIA and capacity planning.")
        st.dataframe(projection_df, use_container_width=True, hide_index=True)
        for row in projection:
            try:
                population_rows.append(PopulationYear(row.year, row.eligible_population, row.covered_lives))
            except ValueError as exc:
                st.error(f"Year {row.year}: {exc}")
    else:
        st.warning("Manual override is active. Changes to the formula inputs above will update the preview, but will not overwrite your year-specific manual values unless you reset them.")
        st.dataframe(projection_df, use_container_width=True, hide_index=True)
        if st.button("Reset all manual annual values from the current formula", key="pu_reset_manual_projection"):
            for row in projection:
                st.session_state[f"pu_manual_eligible_{row.year}"] = float(row.eligible_population)
                st.session_state[f"pu_manual_covered_{row.year}"] = float(row.covered_lives or 0.0)
            st.rerun()
        for row in projection:
            eligible_key = f"pu_manual_eligible_{row.year}"
            covered_key = f"pu_manual_covered_{row.year}"
            st.session_state.setdefault(eligible_key, float(row.eligible_population))
            st.session_state.setdefault(covered_key, float(row.covered_lives or 0.0))
            c1, c2 = st.columns(2)
            eligible_value = c1.number_input(
                f"Year {row.year} eligible population",
                min_value=0.0,
                step=max(float(row.eligible_population) * 0.01, 1.0),
                key=eligible_key,
            )
            covered_value = c2.number_input(
                f"Year {row.year} covered lives (0 = not supplied)",
                min_value=0.0,
                step=max(float(row.covered_lives or 0.0) * 0.01, 100.0) if row.covered_lives else 1000.0,
                key=covered_key,
            )
            try:
                population_rows.append(PopulationYear(row.year, eligible_value, covered_value if covered_value > 0 else None))
            except ValueError as exc:
                st.error(f"Year {row.year}: {exc}")

    st.markdown("#### Evidence and rationale")
    population_source_text = st.text_area("Population evidence source", value=str(st.session_state.pu_population_source), key="pu_population_source_widget")
    population_rationale_text = st.text_area("Population derivation / rationale", value=str(st.session_state.pu_population_rationale), key="pu_population_rationale_widget")
    st.session_state.pu_population_source = population_source_text
    st.session_state.pu_population_rationale = population_rationale_text

with uptake_tab:
    st.subheader("Current and future option mix")
    st.caption("Current mix describes what happens without the proposed change. Future mix describes the expected allocation after implementation. These shares determine how the annual population is split across options; they do not change the total population.")
    options = [PopulationOption(str(row["id"]), str(row["name"])) for row in st.session_state.pu_options]
    mix_rows = []
    for year in range(1, horizon + 1):
        with st.expander(f"Year {year}", expanded=year == 1):
            for scenario in ("current", "future"):
                st.markdown(f"**{scenario.title()} mix**")
                cols = st.columns(len(options))
                values = []
                for option_index, option in enumerate(options):
                    key = f"pu_share_{scenario}_{year}_{option.id}"
                    if scenario == "current":
                        default = 1.0 if option_index == 0 else 0.0
                    else:
                        default = 0.7 if option_index == 0 else (0.3 if option_index == 1 else 0.0)
                    value = cols[option_index].number_input(
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
    current_source = st.text_area("Current-mix evidence source", value=str(st.session_state.pu_current_mix_source), key="pu_current_mix_source_widget")
    current_rationale = st.text_area("Current-mix rationale", value=str(st.session_state.pu_current_mix_rationale), key="pu_current_mix_rationale_widget")
    future_source = st.text_area("Future-uptake evidence source", value=str(st.session_state.pu_future_mix_source), key="pu_future_mix_source_widget")
    future_rationale = st.text_area("Future-uptake rationale", value=str(st.session_state.pu_future_mix_rationale), key="pu_future_mix_rationale_widget")
    st.session_state.pu_current_mix_source = current_source
    st.session_state.pu_current_mix_rationale = current_rationale
    st.session_state.pu_future_mix_source = future_source
    st.session_state.pu_future_mix_rationale = future_rationale

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
        st.markdown("#### Annual population used downstream")
        st.dataframe(
            pd.DataFrame([
                {
                    "Year": row.year,
                    "Eligible population": row.eligible_population,
                    "Covered lives": row.covered_lives,
                }
                for row in compiled.population
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Allocation by option")
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
