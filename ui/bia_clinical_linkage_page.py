"""UI implementation for clinical-model to Budget Impact linkage."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from model.bia_clinical_linkage import (
    ClinicalLinkageError,
    available_markov_cost_parameters,
    available_semi_markov_cost_parameters,
    project_markov_cost_profiles,
    project_semi_markov_cost_profiles,
    run_linked_budget_impact,
)
from model.bia_decision_tree_linkage import (
    available_tree_cost_parameters,
    project_decision_tree_cost_profiles,
)
from model.budget_impact import (
    COST_CATEGORIES,
    AnnualCostInput,
    BudgetImpactDefinition,
    BudgetImpactValidationError,
    BudgetIntervention,
    PopulationYear,
    TreatmentMixShare,
    compound_series,
)
from model.budget_impact_profiles import BUDGET_IMPACT_PROFILES
from model.currency import CURRENCIES
from model.markov_builder import compile_markov_tables
from model.reference_cases import REFERENCE_CASES
from model.semi_markov_builder import compile_semi_markov_tables
from model.tree_builder import BuilderValidationError, compile_builder_tables
from ui.design_system import coloured_block, status_bar


def _records(value):
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict("records")
        except TypeError:
            pass
    return [dict(row) for row in value]


def _bia_from_session() -> tuple[BudgetImpactDefinition, str, int]:
    profile_code = st.session_state.get("bia_profile")
    if not profile_code:
        raise ClinicalLinkageError(
            "Configure the Budget Impact Analysis workspace first so the eligible population, treatment mix and costs are available."
        )

    if profile_code == "CUSTOM":
        horizon = int(st.session_state.get("bia_horizon_custom", 3))
        currency = str(st.session_state.get("bia_currency_custom", "GBP"))
    else:
        if profile_code not in BUDGET_IMPACT_PROFILES:
            raise ClinicalLinkageError("The active BIA methods profile is not recognised.")
        profile = BUDGET_IMPACT_PROFILES[profile_code]
        horizon = int(st.session_state.get(f"bia_horizon_{profile.code}", profile.default_horizon_years))
        currency = str(st.session_state.get("bia_currency_profile", profile.default_currency))

    raw_interventions = st.session_state.get("bia_interventions")
    raw_costs = st.session_state.get("bia_cost_inputs")
    if not raw_interventions or not raw_costs:
        raise ClinicalLinkageError("Configure BIA interventions and costs before linking a clinical model.")

    interventions = tuple(
        BudgetIntervention(str(row["id"]), str(row["name"])) for row in raw_interventions
    )
    ids = [item.id for item in interventions]

    population = []
    for index in range(horizon):
        eligible_key = f"bia_eligible_{horizon}_{index}"
        covered_key = f"bia_covered_{horizon}_{index}"
        if eligible_key not in st.session_state:
            raise ClinicalLinkageError(
                "Complete the annual eligible-population inputs in Budget Impact Analysis before linking."
            )
        eligible = float(st.session_state[eligible_key])
        covered_value = float(st.session_state.get(covered_key, 0.0) or 0.0)
        population.append(PopulationYear(index + 1, eligible, covered_value if covered_value > 0 else None))

    mix = []
    for year in range(1, horizon + 1):
        for scenario in ("current", "future"):
            for iid in ids:
                key = f"bia_share_{scenario}_{year}_{iid}"
                if key not in st.session_state:
                    raise ClinicalLinkageError(
                        "Complete the current and future treatment-mix inputs in Budget Impact Analysis before linking."
                    )
                mix.append(TreatmentMixShare(scenario, year, iid, float(st.session_state[key])))

    cost_rows = []
    for intervention in interventions:
        data = raw_costs.get(intervention.id, {})
        growth = float(data.get("annual_change", 0.0))
        for category in COST_CATEGORIES:
            for year, value in enumerate(
                compound_series(float(data.get(category, 0.0)), growth, horizon), start=1
            ):
                cost_rows.append(AnnualCostInput(intervention.id, year, category, value))

    included = tuple(st.session_state.get("bia_included_categories", list(COST_CATEGORIES)))
    try:
        definition = BudgetImpactDefinition(
            interventions=interventions,
            population=tuple(population),
            treatment_mix=tuple(mix),
            costs=tuple(cost_rows),
            included_cost_categories=included,
        )
    except (ValueError, BudgetImpactValidationError) as exc:
        raise ClinicalLinkageError(str(exc)) from exc
    return definition, currency, horizon


def _compile_decision_tree():
    required = ("dt_parameter_rows", "dt_strategy_rows", "dt_node_rows", "dt_branch_rows")
    if not all(key in st.session_state for key in required):
        raise ClinicalLinkageError("No active Decision Tree model is available in this session.")
    compiled = compile_builder_tables(
        _records(st.session_state.dt_parameter_rows),
        _records(st.session_state.dt_strategy_rows),
        _records(st.session_state.dt_node_rows),
        _records(st.session_state.dt_branch_rows),
    )
    reference_case = st.session_state.get("dt_reference_case", "NICE_TA")
    if reference_case in REFERENCE_CASES:
        bearers = tuple(REFERENCE_CASES[reference_case].perspective.included_cost_bearers)
    else:
        raw = st.session_state.get("dt_custom_cost_bearers", ["health_system"])
        bearers = tuple(raw) if not isinstance(raw, str) else tuple(
            item.strip() for item in raw.split(",") if item.strip()
        )
    return compiled, bearers


def _compile_cohort_markov():
    required = (
        "markov_parameters", "markov_states", "markov_strategies", "markov_initial",
        "markov_transitions", "markov_state_rewards", "markov_transition_rewards",
    )
    if not all(key in st.session_state for key in required):
        raise ClinicalLinkageError("No active Cohort Markov model is available in this session.")
    cycle_months = int(st.session_state.get("mk_cycle_months", 12))
    cycle_length = cycle_months / 12.0
    horizon = int(st.session_state.get("mk_horizon_years", 20))
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
        state_accrual_timing=str(st.session_state.get("mk_state_accrual", "half_cycle")),
        transition_reward_timing=str(st.session_state.get("mk_transition_timing", "mid_cycle")),
        termination_mode="fixed_cycles" if st.session_state.get("mk_termination", "Fixed horizon") == "Fixed horizon" else "cohort_depletion",
        depletion_threshold=float(st.session_state.get("mk_depletion", 0.0001)),
    )
    bearers = tuple(
        item.strip() for item in str(st.session_state.get("mk_cost_bearers", "health_system")).split(",") if item.strip()
    )
    return compiled, bearers


def _compile_advanced_markov():
    required = (
        "adv_parameters", "adv_states", "adv_strategies", "adv_initial", "adv_transitions",
        "adv_state_rewards", "adv_transition_rewards", "adv_mortality_table", "adv_mortality_rules",
    )
    if not all(key in st.session_state for key in required):
        raise ClinicalLinkageError("No active Advanced Markov model is available in this session.")
    cycle_months = int(st.session_state.get("adv_cycle_months", 12))
    cycle_length = cycle_months / 12.0
    horizon = int(st.session_state.get("adv_horizon_years", 20))
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
        state_accrual_timing=str(st.session_state.get("adv_state_accrual", "half_cycle")),
        transition_reward_timing=str(st.session_state.get("adv_transition_timing", "mid_cycle")),
        termination_mode="fixed_cycles" if st.session_state.get("adv_termination", "Fixed horizon") == "Fixed horizon" else "cohort_depletion",
        depletion_threshold=float(st.session_state.get("adv_depletion", 0.0001)),
    )
    bearers = tuple(
        item.strip() for item in str(st.session_state.get("adv_cost_bearers", "health_system")).split(",") if item.strip()
    )
    return compiled, bearers


def render_page() -> None:
    st.title("Clinical model → Budget Impact linkage")
    st.caption("Version 0.10 — use clinical-model trajectories to project condition-related budget consequences")

    coloured_block(
        "Keep the clinical model and payer model separate — but consistent",
        "The clinical model projects per-patient condition-related costs over time. The BIA applies those trajectories to annual treatment-start cohorts, treatment uptake and the budget holder's direct costs. Only cost parameters you explicitly select are imported.",
        tone="teal",
        kicker="Clinical linkage",
    )
    coloured_block(
        "Why annual treatment starts matter",
        "A linked multi-year clinical trajectory cannot safely treat a cross-sectional prevalent population as a brand-new cohort every year. Linked BIA therefore requires the annual BIA population values to represent new treatment starts / newly initiating patients. Each annual cohort is followed through its later clinical costs.",
        tone="amber",
        kicker="Cohort basis",
    )

    try:
        bia_definition, bia_currency, bia_horizon = _bia_from_session()
    except ClinicalLinkageError as exc:
        st.error(str(exc))
        st.page_link("pages/6_Budget_Impact_Analysis.py", label="Open Budget Impact Analysis →", icon="💷")
        return

    source_options = []
    if all(key in st.session_state for key in ("dt_parameter_rows", "dt_strategy_rows", "dt_node_rows", "dt_branch_rows")):
        source_options.append("Decision Tree Modeller")
    if all(key in st.session_state for key in ("markov_parameters", "markov_states", "markov_strategies", "markov_transitions")):
        source_options.append("Cohort Markov Modeller")
    if all(key in st.session_state for key in ("adv_parameters", "adv_states", "adv_strategies", "adv_transitions")):
        source_options.append("Advanced Markov Dynamics")

    if not source_options:
        st.error("No active clinical model is available in this session.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.page_link("pages/1_Decision_Tree_Builder.py", label="Decision Tree →", icon="🌿")
        with c2:
            st.page_link("pages/2_Cohort_Markov_Builder.py", label="Cohort Markov →", icon="🔁")
        with c3:
            st.page_link("pages/3_Advanced_Markov_Dynamics.py", label="Advanced Markov →", icon="🧭")
        return

    source = st.selectbox("Clinical model source", source_options)
    try:
        if source == "Decision Tree Modeller":
            compiled, included_bearers = _compile_decision_tree()
            clinical_model = compiled.tree
            clinical_parameters = compiled.parameters
            strategy_labels = dict(compiled.strategy_names)
            available_cost_ids = available_tree_cost_parameters(clinical_model)
            source_model_type = "decision_tree"
        elif source == "Cohort Markov Modeller":
            compiled, included_bearers = _compile_cohort_markov()
            clinical_model = compiled.model
            clinical_parameters = compiled.parameters
            strategy_labels = {strategy.strategy_id: strategy.label for strategy in clinical_model.strategies}
            available_cost_ids = available_markov_cost_parameters(clinical_model)
            source_model_type = "cohort_markov"
        else:
            compiled, included_bearers = _compile_advanced_markov()
            clinical_model = compiled.model
            clinical_parameters = compiled.parameters
            strategy_labels = {strategy.strategy_id: strategy.label for strategy in clinical_model.strategies}
            available_cost_ids = available_semi_markov_cost_parameters(clinical_model)
            source_model_type = "semi_markov"
    except (ClinicalLinkageError, BuilderValidationError, ValueError) as exc:
        st.error(f"The selected clinical model cannot currently be compiled: {exc}")
        return

    parameter_map = {parameter.id: parameter for parameter in clinical_parameters}
    status_bar(
        [
            (f"BIA: {bia_horizon} years", "blue"),
            (f"Currency: {bia_currency}", "neutral"),
            (source, "teal"),
            (f"{len(strategy_labels)} clinical strategies", "neutral"),
        ]
    )

    st.markdown("## 1. Select the clinical costs to import")
    st.caption(
        "Only selected cost-reward parameters are projected. Unselected clinical cost rewards are excluded from the linkage run so acquisition or administration costs can remain direct BIA inputs without double counting."
    )
    selected_cost_ids = st.multiselect(
        "Condition-related clinical cost parameters",
        list(available_cost_ids),
        format_func=lambda pid: f"{parameter_map[pid].label} (`{pid}`)",
        default=[],
    )
    if not available_cost_ids:
        st.warning("The selected clinical model does not contain any cost rewards to link.")

    currency_mismatches = []
    for pid in selected_cost_ids:
        parameter_currency = (parameter_map[pid].currency or "").upper()
        if parameter_currency and parameter_currency != bia_currency.upper():
            currency_mismatches.append(f"{pid}: {parameter_currency}")
    if currency_mismatches:
        st.error(
            "Selected clinical costs do not match the BIA currency. Convert and document them before linking: "
            + ", ".join(currency_mismatches)
        )

    st.info(
        "Linked clinical costs replace the BIA **Condition-related care / disease-management** category. Acquisition, administration, monitoring, adverse-event and other BIA cost categories remain direct payer inputs."
    )

    st.markdown("## 2. Map BIA options to clinical strategies")
    strategy_ids = list(strategy_labels)
    mapping: dict[str, str] = {}
    for index, intervention in enumerate(bia_definition.interventions):
        mapping[intervention.id] = st.selectbox(
            f"{intervention.name} → clinical strategy",
            strategy_ids,
            index=min(index, len(strategy_ids) - 1),
            format_func=lambda sid: strategy_labels[sid],
            key=f"bia_link_map_{source_model_type}_{intervention.id}",
        )

    st.markdown("## 3. Confirm the population interpretation")
    cohort_confirmed = st.checkbox(
        "For this linked analysis, the annual BIA population values represent new treatment starts / newly initiating patients.",
        value=False,
        key=f"bia_link_cohort_confirmed_{source_model_type}",
    )
    st.caption(
        "Each start cohort keeps its assigned treatment and contributes its first-year, second-year and later clinical costs in subsequent budget years. If your BIA population is a prevalence stock, revise it before using trajectory linkage."
    )

    can_run = bool(selected_cost_ids) and cohort_confirmed and not currency_mismatches
    if not st.button("Run linked Budget Impact Analysis", type="primary", disabled=not can_run):
        return

    try:
        if source_model_type == "decision_tree":
            profiles = project_decision_tree_cost_profiles(
                clinical_model,
                clinical_parameters,
                selected_cost_parameter_ids=selected_cost_ids,
                horizon_years=bia_horizon,
                strategy_names=strategy_labels,
                included_cost_bearers=included_bearers,
            )
        elif source_model_type == "cohort_markov":
            profiles = project_markov_cost_profiles(
                clinical_model,
                clinical_parameters,
                selected_cost_parameter_ids=selected_cost_ids,
                horizon_years=bia_horizon,
                included_cost_bearers=included_bearers,
            )
        else:
            profiles = project_semi_markov_cost_profiles(
                clinical_model,
                clinical_parameters,
                selected_cost_parameter_ids=selected_cost_ids,
                horizon_years=bia_horizon,
                included_cost_bearers=included_bearers,
            )
        linked_run = run_linked_budget_impact(
            bia_definition,
            profiles,
            intervention_to_strategy=mapping,
            linked_category="disease_management",
            population_basis="new_treatment_starts",
        )
    except (ClinicalLinkageError, BudgetImpactValidationError, ValueError) as exc:
        st.error(str(exc))
        return

    symbol = CURRENCIES[bia_currency].symbol if bia_currency in CURRENCIES else bia_currency + " "
    st.markdown("## Linked results")
    first, final = linked_run.years[0], linked_run.years[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Year 1 net impact", f"{symbol}{first.net_budget_impact:,.0f}")
    c2.metric(f"Year {bia_horizon} net impact", f"{symbol}{final.net_budget_impact:,.0f}")
    c3.metric("Cumulative impact", f"{symbol}{linked_run.cumulative_budget_impact:,.0f}")
    c4.metric(
        "Year 1 PMPM",
        f"{symbol}{first.pmpm_budget_impact:,.4f}" if first.pmpm_budget_impact is not None else "Not available",
    )

    annual_df = pd.DataFrame(
        [
            {
                "Year": row.year,
                "Direct current cost": row.direct_current_cost,
                "Clinical current cost": row.clinical_current_cost,
                "Total current cost": row.current_cost,
                "Direct future cost": row.direct_future_cost,
                "Clinical future cost": row.clinical_future_cost,
                "Total future cost": row.future_cost,
                "Net budget impact": row.net_budget_impact,
                "Cumulative budget impact": row.cumulative_budget_impact,
                "PMPM": row.pmpm_budget_impact,
            }
            for row in linked_run.years
        ]
    )
    st.dataframe(annual_df, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(
            annual_df,
            x="Year",
            y=["Direct future cost", "Clinical future cost"],
            title="Future-scenario budget: direct payer inputs vs linked clinical costs",
            barmode="stack",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(annual_df, x="Year", y="Net budget impact", markers=True, title="Linked net budget impact by year"),
        use_container_width=True,
    )

    st.markdown("### Per-patient clinical cost trajectories")
    profile_df = pd.DataFrame(
        [
            {
                "Clinical strategy": profile.strategy_name,
                "Year since initiation": year,
                "Condition-related cost per patient": value,
            }
            for profile in profiles
            for year, value in enumerate(profile.annual_cost_per_patient, start=1)
        ]
    )
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    with st.expander("Inspect stacked initiation cohorts"):
        cohort_df = pd.DataFrame(
            [
                {
                    "Scenario": row.scenario,
                    "Budget year": row.budget_year,
                    "BIA intervention": row.intervention_id,
                    "Clinical strategy": strategy_labels[row.strategy_id],
                    "Initiation year": row.initiation_year,
                    "Year since initiation": row.year_since_initiation,
                    "Treatment starts": row.treatment_starts,
                    "Cost per patient": row.cost_per_patient,
                    "Total clinical cost": row.total_clinical_cost,
                }
                for row in linked_run.clinical_rows
            ]
        )
        st.dataframe(cohort_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download linked annual results (CSV)",
        annual_df.to_csv(index=False),
        file_name="linked_budget_impact_results.csv",
        mime="text/csv",
    )

    st.markdown("### Linkage transparency")
    status_bar(
        [
            ("Annual new-start cohorts confirmed", "green"),
            (f"{len(selected_cost_ids)} clinical cost parameter(s) imported", "green"),
            ("Disease-management direct inputs replaced", "blue"),
            ("No clinical-cost discounting", "neutral"),
        ]
    )
    st.write("**Imported cost parameters:** " + ", ".join(selected_cost_ids))
    st.write(
        "**BIA-to-clinical mapping:** "
        + "; ".join(
            f"{intervention.name} → {strategy_labels[mapping[intervention.id]]}"
            for intervention in bia_definition.interventions
        )
    )
    if source_model_type == "decision_tree":
        st.caption(
            "Decision-tree timed rewards are placed into annual budget periods using absolute model time: t=0 and t=1 are Budget Year 1; a reward just after year 1 is Budget Year 2. Branch probabilities weight each pathway's expected cost."
        )
    else:
        st.caption(
            "State-transition models are rerun with zero cost discounting and unselected cost rewards set to zero. Cycle costs are aggregated into annual per-patient trajectories before cohort stacking. The clinical cycle length must divide one year exactly."
        )
