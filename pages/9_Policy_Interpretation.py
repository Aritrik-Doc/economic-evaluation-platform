"""Consolidated policymaker-facing interpretation of active platform analyses."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model.budget_impact import run_budget_impact
from model.currency import CURRENCIES
from model.decision_tree import run_decision_tree
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.markov import run_cohort_markov
from model.markov_builder import compile_markov_tables
from model.policy_interpretation import (
    PolicyInterpretation,
    interpret_budget_impact,
    interpret_capacity,
    interpret_cost_effectiveness,
)
from model.reference_cases import REFERENCE_CASES
from model.semi_markov import run_semi_markov
from model.semi_markov_builder import compile_semi_markov_tables
from model.tree_builder import compile_builder_tables
from ui.bia_context import bia_definition_from_session
from ui.design_system import coloured_block, status_bar
from ui.policy_interpretation import render_combined_headlines, render_policy_interpretation


st.set_page_config(page_title="Policy Interpretation", page_icon="🧾", layout="wide")
st.title("Policy Interpretation")
st.caption("Version 0.15 — deterministic translation of value, affordability and implementation outputs")

coloured_block(
    "Translate results without replacing judgement",
    "This workspace converts calculated outputs into structured plain-language statements for policy audiences. It does not make an adoption, reimbursement or service-allocation recommendation. Every statement remains conditional on the model inputs, methods and evidence documented elsewhere in the platform.",
    tone="teal",
    kicker="Decision support",
)


def _records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return [dict(row) for row in value]


def _float_text(value) -> float:
    text = str(value or "").replace(",", "").strip()
    if not text:
        raise ValueError("A decision threshold is required before cost-effectiveness interpretation can be generated.")
    return float(text)


def _decision_tree_interpretation() -> PolicyInterpretation:
    compiled = compile_builder_tables(
        _records(st.session_state.dt_parameter_rows),
        _records(st.session_state.dt_strategy_rows),
        _records(st.session_state.dt_node_rows),
        _records(st.session_state.dt_branch_rows),
    )
    profile_code = st.session_state.get("dt_reference_case", "NICE_TA")
    if profile_code == "CUSTOM":
        outcome_code = st.session_state.get("dt_outcome_code", "QALY")
        currency_code = st.session_state.get("dt_custom_currency", "GBP")
        cost_bearers = tuple(st.session_state.get("dt_custom_cost_bearers", ["health_system"]))
        cost_discount = float(st.session_state.get("dt_custom_cost_discount", 3.5)) / 100.0
        outcome_discount = float(st.session_state.get("dt_custom_outcome_discount", 3.5)) / 100.0
    else:
        profile = REFERENCE_CASES[profile_code]
        outcome_code = st.session_state.get("dt_outcome_code", profile.preferred_outcome_code)
        currency_code = profile.analysis_currency
        cost_bearers = tuple(profile.perspective.included_cost_bearers)
        cost_discount = profile.cost_discount_rate
        outcome_discount = profile.outcome_discount_rate
    threshold = _float_text(st.session_state.get("dt_threshold_text", ""))
    run = run_decision_tree(
        compiled.tree,
        compiled.parameters,
        included_cost_bearers=cost_bearers or None,
        cost_discount_rate=cost_discount,
        outcome_discount_rate=outcome_discount,
    )
    strategies = [
        Strategy(compiled.strategy_names[row.strategy_id], row.expected_cost, row.expected_outcome)
        for row in run.strategies
    ]
    incremental = fully_incremental_analysis(strategies, threshold)
    return interpret_cost_effectiveness(
        incremental,
        outcome_label=OUTCOME_MEASURES[outcome_code].unit,
        currency_symbol=CURRENCIES[currency_code].symbol,
    )


def _markov_interpretation() -> PolicyInterpretation:
    cycle_months = int(st.session_state.get("mk_cycle_months", 12))
    cycle_length = cycle_months / 12.0
    horizon_years = int(st.session_state.get("mk_horizon_years", 20))
    max_cycles = max(1, int(round(horizon_years / cycle_length)))
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
        max_cycles=max_cycles,
        state_accrual_timing=st.session_state.get("mk_state_accrual", "half_cycle"),
        transition_reward_timing=st.session_state.get("mk_transition_timing", "mid_cycle"),
        termination_mode="fixed_cycles" if termination == "Fixed horizon" else "cohort_depletion",
        depletion_threshold=float(st.session_state.get("mk_depletion", 0.0001)),
    )
    bearers = tuple(
        item.strip()
        for item in str(st.session_state.get("mk_cost_bearers", "health_system")).split(",")
        if item.strip()
    )
    run = run_cohort_markov(
        compiled.model,
        compiled.parameters,
        included_cost_bearers=bearers or None,
        cost_discount_rate=float(st.session_state.get("mk_cost_discount", 0.035)),
        outcome_discount_rate=float(st.session_state.get("mk_outcome_discount", 0.035)),
    )
    threshold = float(st.session_state.get("mk_threshold", 0.0))
    incremental = fully_incremental_analysis(
        [Strategy(row.label, row.expected_cost, row.expected_outcome) for row in run.strategies],
        threshold,
    )
    outcome_code = st.session_state.get("mk_outcome", "QALY")
    currency_code = st.session_state.get("mk_currency", "GBP")
    return interpret_cost_effectiveness(
        incremental,
        outcome_label=OUTCOME_MEASURES[outcome_code].unit,
        currency_symbol=CURRENCIES[currency_code].symbol,
    )


def _advanced_interpretation() -> PolicyInterpretation:
    cycle_months = int(st.session_state.get("adv_cycle_months", 12))
    cycle_length = cycle_months / 12.0
    horizon_years = int(st.session_state.get("adv_horizon_years", 20))
    max_cycles = max(1, int(round(horizon_years / cycle_length)))
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
        max_cycles=max_cycles,
        state_accrual_timing=st.session_state.get("adv_state_accrual", "half_cycle"),
        transition_reward_timing=st.session_state.get("adv_transition_timing", "mid_cycle"),
        termination_mode="fixed_cycles" if termination == "Fixed horizon" else "cohort_depletion",
        depletion_threshold=float(st.session_state.get("adv_depletion", 0.0001)),
    )
    bearers = tuple(
        item.strip()
        for item in str(st.session_state.get("adv_cost_bearers", "health_system")).split(",")
        if item.strip()
    )
    run = run_semi_markov(
        compiled.model,
        compiled.parameters,
        included_cost_bearers=bearers or None,
        cost_discount_rate=float(st.session_state.get("adv_cost_discount", 0.035)),
        outcome_discount_rate=float(st.session_state.get("adv_outcome_discount", 0.035)),
    )
    threshold = float(st.session_state.get("adv_threshold", 0.0))
    incremental = fully_incremental_analysis(
        [Strategy(row.label, row.expected_cost, row.expected_outcome) for row in run.strategies],
        threshold,
    )
    outcome_code = st.session_state.get("adv_outcome", "QALY")
    currency_code = st.session_state.get("adv_currency", "GBP")
    return interpret_cost_effectiveness(
        incremental,
        outcome_label=OUTCOME_MEASURES[outcome_code].unit,
        currency_symbol=CURRENCIES[currency_code].symbol,
    )


def _bia_definition():
    return bia_definition_from_session(dict(st.session_state))


def _bia_interpretation() -> PolicyInterpretation:
    definition, currency_code, _ = _bia_definition()
    result = run_budget_impact(definition)
    labels = {
        "acquisition": "Acquisition / technology",
        "administration": "Administration / procedure",
        "monitoring": "Monitoring / follow-up",
        "adverse_events": "Adverse-event management",
        "disease_management": "Condition-related care",
        "other": "Other budgeted cost / credit",
    }
    return interpret_budget_impact(
        result,
        currency_symbol=CURRENCIES[currency_code].symbol,
        category_labels=labels,
    )


def _capacity_interpretation() -> PolicyInterpretation:
    result = st.session_state.get("rc_last_result")
    if result is None:
        raise ValueError(
            "Open Resource & Capacity Planning and complete a valid capacity analysis first. "
            "Feasibility interpretation uses the validated capacity result rather than reconstructing it from BIA."
        )
    return interpret_capacity(result)


cea_sources = []
if all(
    key in st.session_state
    for key in ("dt_parameter_rows", "dt_strategy_rows", "dt_node_rows", "dt_branch_rows")
):
    cea_sources.append("Decision Tree")
if all(
    key in st.session_state
    for key in ("markov_parameters", "markov_states", "markov_strategies")
):
    cea_sources.append("Cohort Markov")
if all(
    key in st.session_state
    for key in ("adv_parameters", "adv_states", "adv_strategies")
):
    cea_sources.append("Advanced Markov")

value_interpretation = None
affordability_interpretation = None
feasibility_interpretation = None
errors = {}

if cea_sources:
    selected_source = st.selectbox(
        "Clinical/economic model to interpret",
        cea_sources,
        key="policy_cea_source",
    )
    try:
        value_interpretation = {
            "Decision Tree": _decision_tree_interpretation,
            "Cohort Markov": _markov_interpretation,
            "Advanced Markov": _advanced_interpretation,
        }[selected_source]()
    except Exception as exc:
        errors["Value for money"] = str(exc)
else:
    errors["Value for money"] = (
        "No configured Decision Tree or Markov model is available in this session."
    )

try:
    affordability_interpretation = _bia_interpretation()
except Exception as exc:
    errors["Affordability"] = str(exc)

try:
    feasibility_interpretation = _capacity_interpretation()
except Exception as exc:
    errors["Implementation feasibility"] = str(exc)

available = tuple(
    item
    for item in (
        value_interpretation,
        affordability_interpretation,
        feasibility_interpretation,
    )
    if item is not None
)
status_bar(
    [
        (
            "Value: available" if value_interpretation else "Value: not available",
            "green" if value_interpretation else "neutral",
        ),
        (
            "Affordability: available"
            if affordability_interpretation
            else "Affordability: not available",
            "green" if affordability_interpretation else "neutral",
        ),
        (
            "Feasibility: available"
            if feasibility_interpretation
            else "Feasibility: not available",
            "green" if feasibility_interpretation else "neutral",
        ),
    ]
)

summary_tab, value_tab, affordability_tab, feasibility_tab, methods_tab = st.tabs(
    [
        "Executive summary",
        "Value",
        "Affordability",
        "Feasibility",
        "Interpretation rules",
    ]
)

with summary_tab:
    st.subheader("Cross-domain decision summary")
    if available:
        render_combined_headlines(available)
    else:
        st.info("Configure at least one analysis workspace before generating a policy interpretation.")
    if errors:
        with st.expander("Analyses not currently available", expanded=False):
            for domain, message in errors.items():
                st.write(f"**{domain}:** {message}")
    coloured_block(
        "Read the three domains together",
        "A technology can have favourable value-for-money results while increasing payer expenditure or exceeding service capacity. Conversely, a financially saving option may still have uncertain health-economic value. The platform therefore keeps value, affordability and feasibility distinct before presenting them together.",
        tone="blue",
        kicker="No single composite score",
    )

with value_tab:
    if value_interpretation:
        render_policy_interpretation(value_interpretation)
    else:
        st.info(errors.get("Value for money", "Value interpretation is not available."))
        st.page_link("pages/1_Decision_Tree_Builder.py", label="Open Decision Tree Modeller →")
        st.page_link("pages/2_Cohort_Markov_Builder.py", label="Open Cohort Markov Modeller →")

with affordability_tab:
    if affordability_interpretation:
        render_policy_interpretation(affordability_interpretation)
    else:
        st.info(errors.get("Affordability", "Affordability interpretation is not available."))
        st.page_link("pages/6_Budget_Impact_Analysis.py", label="Open Budget Impact Analysis →")

with feasibility_tab:
    if feasibility_interpretation:
        render_policy_interpretation(feasibility_interpretation)
        context = st.session_state.get("rc_last_context") or {}
        if context:
            st.caption(
                "Capacity result source: "
                + str(context.get("population_source_label") or context.get("population_source") or "capacity workspace")
                + " · Resource requirements: "
                + str(context.get("requirement_source") or "configured in capacity workspace")
            )
    else:
        st.info(
            errors.get(
                "Implementation feasibility",
                "Feasibility interpretation is not available.",
            )
        )
        st.page_link(
            "pages/8_Resource_Capacity_Planning.py",
            label="Open Resource & Capacity Planning →",
        )

with methods_tab:
    st.subheader("How the interpretation is generated")
    st.write(
        "The interpretation is rules-based and deterministic. It reads validated numerical outputs and converts them into standardised statements. It does not use a generative model to infer a recommendation or fill missing evidence."
    )
    st.markdown(
        "**Value:** threshold-specific NMB, efficient-frontier position, incremental cost/effect and ICER.  \n"
        "**Affordability:** the current validated BIA handoff, including annual and cumulative budget impact, PMPM where available, and the largest cost-category change.  \n"
        "**Feasibility:** the most recently validated resource/capacity result, including demand, residual capacity, utilisation, headroom and shortfall. Feasibility does not require a BIA to exist."
    )
    st.write(
        "Each detailed statement displays its analytical basis. The language deliberately distinguishes calculated findings from policy judgements, and the three domains are not collapsed into a single score."
    )
    st.warning(
        "Interpretation quality cannot exceed model quality. Missing, provisional or weak evidence should be addressed through the Transparency Check and critical appraisal of the underlying sources."
    )

st.divider()
st.caption(
    "Policy Interpretation is a communication layer over the platform's analytical outputs. It does not replace appraisal of clinical evidence, economic methods, equity, ethics, legal considerations, implementation context or stakeholder judgement."
)
