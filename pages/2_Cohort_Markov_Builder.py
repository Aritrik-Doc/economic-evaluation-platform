"""Guided cohort state-transition modeller."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model.analysis_consistency import (
    AnalysisConsistencyError,
    ceac_threshold_grid,
    pairwise_inmb_from_strategy_results,
    require_matching_base_inmb,
)
from model.currency import CURRENCIES
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.markov import MarkovValidationError, run_cohort_markov
from model.markov_builder import compile_markov_tables, markov_structure_to_dot
from model.markov_psa import markov_psa_configuration_warnings, run_markov_psa
from model.markov_reproducibility import (
    MarkovReproducibilityError,
    markov_run_fingerprint,
    validate_analysis_currency,
)
from model.markov_sensitivity import tornado_markov_inmb
from model.psa import ceac, incremental_plane, pairwise_probability_cost_effective
from model.reference_cases import REFERENCE_CASES
from model.tree_builder import BuilderValidationError
from ui.markov_structure_editor import (
    render_initial_distribution,
    render_raw_structure_tables,
    render_states,
    render_standard_transitions,
    render_strategies,
)
from ui.parameter_library import render_parameter_library


st.set_page_config(page_title="Cohort Markov Builder", page_icon="🔁", layout="wide")


def _parameter_row(
    pid,
    label,
    value,
    unit,
    category,
    *,
    currency="",
    price_year=None,
    cost_bearers="",
    dsa=False,
    low=None,
    high=None,
    psa=False,
    family="",
    distribution_parameters=None,
):
    return {
        "id": pid,
        "label": label,
        "value": value,
        "unit": unit,
        "category": category,
        "currency": currency,
        "price_year": price_year,
        "cost_bearers": cost_bearers,
        "source_citation": "Illustrative input — replace with model evidence",
        "source_type": "user_assumption",
        "publication_year": None,
        "source_url": "",
        "source_details": "",
        "assumption": "Illustrative cohort-model value.",
        "assumption_rationale": "Replace with a documented evidence-based assumption before decision use.",
        "dsa_enabled": dsa,
        "dsa_lower": low,
        "dsa_upper": high,
        "dsa_rationale": "Illustrative low/high range." if dsa else "Not represented in DSA.",
        "psa_enabled": psa,
        "psa_rationale": "Illustrative sampling uncertainty." if psa else "Not represented in PSA.",
        "distribution_family": family,
        "distribution_parameterisation": "Alpha + Beta" if family == "beta" else "",
        "distribution_parameters": distribution_parameters or {},
        "correlation_group": "",
        "notes": "",
    }


def _defaults():
    parameters = [
        _parameter_row(
            "p_prog_standard", "Progression probability — standard care", 0.12,
            "probability/cycle", "clinical", dsa=True, low=0.08, high=0.16,
            psa=True, family="beta", distribution_parameters={"alpha": 12.0, "beta": 88.0},
        ),
        _parameter_row(
            "p_prog_new", "Progression probability — new treatment", 0.08,
            "probability/cycle", "clinical", dsa=True, low=0.05, high=0.12,
            psa=True, family="beta", distribution_parameters={"alpha": 8.0, "beta": 92.0},
        ),
        _parameter_row("p_death_stable", "Death probability — stable", 0.03, "probability/cycle", "clinical", dsa=True, low=0.02, high=0.05),
        _parameter_row("p_death_progressed", "Death probability — progressed", 0.15, "probability/cycle", "clinical", dsa=True, low=0.10, high=0.20),
        _parameter_row("cost_stable_standard", "Stable-state cost — standard care", 1000.0, "currency/year", "cost", currency="GBP", price_year=2026, cost_bearers="health_system", dsa=True, low=800, high=1200),
        _parameter_row("cost_stable_new", "Stable-state cost — new treatment", 3000.0, "currency/year", "cost", currency="GBP", price_year=2026, cost_bearers="health_system", dsa=True, low=2400, high=3600),
        _parameter_row("cost_progressed", "Progressed-state cost", 6000.0, "currency/year", "cost", currency="GBP", price_year=2026, cost_bearers="health_system", dsa=True, low=4800, high=7200),
        _parameter_row("utility_stable", "Stable-state utility", 0.82, "utility", "utility", dsa=True, low=0.75, high=0.88),
        _parameter_row("utility_progressed", "Progressed-state utility", 0.55, "utility", "utility", dsa=True, low=0.45, high=0.65),
    ]
    states = [
        {"state_id": "stable", "state_name": "Stable", "absorbing": False},
        {"state_id": "progressed", "state_name": "Progressed", "absorbing": False},
        {"state_id": "dead", "state_name": "Dead", "absorbing": True},
    ]
    strategies = [
        {"strategy_id": "standard", "strategy_name": "Standard care"},
        {"strategy_id": "new", "strategy_name": "New treatment"},
    ]
    initial = [
        {"strategy_id": "standard", "state_id": "stable", "proportion": 1.0, "proportion_mode": "fixed", "proportion_parameter_id": ""},
        {"strategy_id": "new", "state_id": "stable", "proportion": 1.0, "proportion_mode": "fixed", "proportion_parameter_id": ""},
    ]
    transitions = []
    for sid, progression_parameter in (("standard", "p_prog_standard"), ("new", "p_prog_new")):
        transitions.extend(
            [
                {"strategy_id": sid, "origin_state": "stable", "destination_state": "progressed", "probability_parameter_id": progression_parameter, "probability_mode": "direct"},
                {"strategy_id": sid, "origin_state": "stable", "destination_state": "dead", "probability_parameter_id": "p_death_stable", "probability_mode": "direct"},
                {"strategy_id": sid, "origin_state": "stable", "destination_state": "stable", "probability_parameter_id": "", "probability_mode": "residual"},
                {"strategy_id": sid, "origin_state": "progressed", "destination_state": "dead", "probability_parameter_id": "p_death_progressed", "probability_mode": "direct"},
                {"strategy_id": sid, "origin_state": "progressed", "destination_state": "progressed", "probability_parameter_id": "", "probability_mode": "residual"},
            ]
        )
    state_rewards = [
        {"strategy_id": "standard", "state_id": "stable", "parameter_id": "cost_stable_standard", "reward_type": "cost", "accrual": "per_year"},
        {"strategy_id": "new", "state_id": "stable", "parameter_id": "cost_stable_new", "reward_type": "cost", "accrual": "per_year"},
    ]
    for sid in ("standard", "new"):
        state_rewards.extend(
            [
                {"strategy_id": sid, "state_id": "progressed", "parameter_id": "cost_progressed", "reward_type": "cost", "accrual": "per_year"},
                {"strategy_id": sid, "state_id": "stable", "parameter_id": "utility_stable", "reward_type": "outcome", "accrual": "per_year"},
                {"strategy_id": sid, "state_id": "progressed", "parameter_id": "utility_progressed", "reward_type": "outcome", "accrual": "per_year"},
            ]
        )
    return parameters, states, strategies, initial, transitions, state_rewards, []


def _ensure_state():
    keys = [
        "markov_parameters", "markov_states", "markov_strategies", "markov_initial",
        "markov_transitions", "markov_state_rewards", "markov_transition_rewards",
    ]
    if not all(key in st.session_state for key in keys):
        for key, value in zip(keys, _defaults()):
            st.session_state[key] = value


def _records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return [dict(row) for row in value]


def _apply_loaded_settings():
    loaded = st.session_state.get("loaded_cohort_markov_settings")
    if not loaded:
        return
    token = loaded.get("restore_token") or json.dumps(loaded, sort_keys=True, default=str)
    if st.session_state.get("markov_loaded_settings_applied_token") == token:
        return
    methods = loaded.get("methods", {})
    engine = loaded.get("engine", {})
    if methods.get("reference_case_code") in REFERENCE_CASES:
        st.session_state["mk_reference_case"] = methods["reference_case_code"]
    if methods.get("outcome_code") in OUTCOME_MEASURES:
        st.session_state["mk_outcome"] = methods["outcome_code"]
    if methods.get("currency_code") in CURRENCIES:
        st.session_state["mk_currency"] = methods["currency_code"]
    if methods.get("threshold") is not None:
        st.session_state["mk_threshold"] = float(methods["threshold"])
    if methods.get("cost_discount_rate") is not None:
        st.session_state["mk_cost_discount"] = float(methods["cost_discount_rate"])
    if methods.get("outcome_discount_rate") is not None:
        st.session_state["mk_outcome_discount"] = float(methods["outcome_discount_rate"])
    if methods.get("included_cost_bearers"):
        st.session_state["mk_cost_bearers"] = ", ".join(methods["included_cost_bearers"])
    cycle_years = float(engine.get("cycle_length_years", 1.0))
    cycle_months = max(1, int(round(cycle_years * 12)))
    if cycle_months in {1, 3, 6, 12}:
        st.session_state["mk_cycle_months"] = cycle_months
    max_cycles = int(engine.get("max_cycles", 20))
    st.session_state["mk_horizon_years"] = max(1, int(round(max_cycles * cycle_years)))
    st.session_state["mk_state_accrual"] = engine.get("state_accrual_timing", "half_cycle")
    st.session_state["mk_transition_timing"] = engine.get("transition_reward_timing", "mid_cycle")
    st.session_state["mk_termination"] = "Fixed horizon" if engine.get("termination_mode", "fixed_cycles") == "fixed_cycles" else "Cohort depletion"
    st.session_state["mk_depletion"] = float(engine.get("depletion_threshold", 0.0001))
    st.session_state["markov_loaded_settings_applied_token"] = token
    st.session_state["markov_loaded_settings_applied"] = True


def _fingerprint(settings):
    return markov_run_fingerprint(
        parameter_rows=st.session_state.markov_parameters,
        state_rows=st.session_state.markov_states,
        strategy_rows=st.session_state.markov_strategies,
        initial_rows=st.session_state.markov_initial,
        transition_rows=st.session_state.markov_transitions,
        state_reward_rows=st.session_state.markov_state_rewards,
        transition_reward_rows=st.session_state.markov_transition_rewards,
        settings=settings,
    )


_ensure_state()
_apply_loaded_settings()

st.title("🔁 Cohort Markov / State-Transition Builder")
st.caption("Version 0.13 — guided cohort modelling with settings-consistent deterministic and probabilistic analysis")
st.info(
    "Starting cohorts may now be linked directly to probability parameters, so cure/response parameters can propagate correctly into DSA and PSA. "
    "The Analyse tab displays the exact cycle/horizon settings used in all analyses."
)

reference_case_code = st.sidebar.selectbox("Methods profile", list(REFERENCE_CASES), format_func=lambda code: REFERENCE_CASES[code].name, key="mk_reference_case")
profile = REFERENCE_CASES[reference_case_code]
outcome_code = st.sidebar.selectbox(
    "Economic outcome", list(OUTCOME_MEASURES),
    index=list(OUTCOME_MEASURES).index(profile.preferred_outcome_code) if profile.preferred_outcome_code in OUTCOME_MEASURES else 0,
    key="mk_outcome",
)
currency_code = st.sidebar.selectbox(
    "Analysis currency", list(CURRENCIES),
    index=list(CURRENCIES).index(profile.analysis_currency) if profile.analysis_currency in CURRENCIES else 0,
    key="mk_currency",
)
threshold_default = 0.0 if profile.threshold_range is None else float((profile.threshold_range.lower + profile.threshold_range.upper) / 2)
threshold = st.sidebar.number_input("Decision threshold", min_value=0.0, value=threshold_default, step=1000.0, key="mk_threshold")
cost_discount_rate = st.sidebar.number_input("Annual cost discount rate", min_value=0.0, max_value=0.99, value=float(profile.cost_discount_rate), format="%.4f", key="mk_cost_discount")
outcome_discount_rate = st.sidebar.number_input("Annual outcome discount rate", min_value=0.0, max_value=0.99, value=float(profile.outcome_discount_rate), format="%.4f", key="mk_outcome_discount")
included_cost_bearers_text = st.sidebar.text_input("Included cost bearers", value=", ".join(profile.perspective.included_cost_bearers), key="mk_cost_bearers")
included_cost_bearers = tuple(item.strip() for item in included_cost_bearers_text.split(",") if item.strip())
if profile.threshold_range is None:
    st.sidebar.caption("This reference case does not prescribe a single monetary threshold; the value above is used only for NMB/INMB analysis.")

methods_tab, parameters_tab, structure_tab, rewards_tab, analyse_tab = st.tabs(
    ["1 · Methods", "2 · Parameters", "3 · States & transitions", "4 · Rewards", "5 · Analyse"]
)

with methods_tab:
    st.subheader("Cycle structure and time horizon")
    c1, c2, c3 = st.columns(3)
    cycle_months = c1.selectbox("Cycle length", [1, 3, 6, 12], index=3, format_func=lambda value: f"{value} month" if value == 1 else f"{value} months", key="mk_cycle_months")
    horizon_years = int(c2.number_input("Maximum horizon (years)", min_value=1, max_value=200, value=20, step=1, key="mk_horizon_years"))
    termination_label = c3.selectbox("Termination", ["Fixed horizon", "Cohort depletion"], key="mk_termination")
    cycle_length_years = cycle_months / 12.0
    max_cycles = int(round(horizon_years / cycle_length_years))
    termination_mode = "fixed_cycles" if termination_label == "Fixed horizon" else "cohort_depletion"
    depletion_threshold = st.number_input(
        "Non-absorbing cohort threshold for depletion stopping", min_value=0.0, max_value=0.5,
        value=0.0001, format="%.6f", disabled=termination_mode != "cohort_depletion", key="mk_depletion"
    )
    st.subheader("Within-cycle accrual")
    c1, c2 = st.columns(2)
    state_accrual_timing = c1.selectbox(
        "State reward accrual", ["half_cycle", "start", "end"],
        format_func=lambda value: {"half_cycle": "Half-cycle / trapezoidal", "start": "Start of cycle", "end": "End of cycle"}[value],
        key="mk_state_accrual",
    )
    transition_reward_timing = c2.selectbox(
        "Transition-event reward timing", ["mid_cycle", "start", "end"],
        format_func=lambda value: {"mid_cycle": "Mid-cycle", "start": "Start of cycle", "end": "End of cycle"}[value],
        key="mk_transition_timing",
    )
    st.metric("Maximum cycles", max_cycles)
    if cycle_months != 12:
        st.warning("Transition probabilities are probabilities for the selected cycle. Changing cycle length does not automatically rescale existing probability parameters.")

with parameters_tab:
    render_parameter_library(session_key="markov_parameters", currency_code=currency_code, key_prefix="mk_param")

with structure_tab:
    st.subheader("Guided model structure")
    st.write("Define states and strategies first, then the starting cohort, then connect states with transitions.")
    render_states(states_key="markov_states", initial_key="markov_initial", transitions_key="markov_transitions", state_rewards_key="markov_state_rewards", transition_rewards_key="markov_transition_rewards", key_prefix="mk")
    st.divider()
    render_strategies(strategies_key="markov_strategies", initial_key="markov_initial", transitions_key="markov_transitions", state_rewards_key="markov_state_rewards", transition_rewards_key="markov_transition_rewards", key_prefix="mk")
    st.divider()
    render_initial_distribution(states_key="markov_states", strategies_key="markov_strategies", initial_key="markov_initial", parameters_key="markov_parameters", key_prefix="mk")
    st.divider()
    render_standard_transitions(states_key="markov_states", strategies_key="markov_strategies", parameters_key="markov_parameters", transitions_key="markov_transitions", key_prefix="mk")
    strategy_labels = {str(row.get("strategy_id")): str(row.get("strategy_name") or row.get("strategy_id")) for row in st.session_state.markov_strategies if row.get("strategy_id")}
    if strategy_labels:
        st.markdown("#### Model diagram")
        diagram_strategy = st.selectbox("Diagram strategy", list(strategy_labels), format_func=lambda sid: strategy_labels[sid], key="mk_diagram_strategy")
        st.graphviz_chart(markov_structure_to_dot(st.session_state.markov_states, st.session_state.markov_transitions, strategy_id=diagram_strategy), use_container_width=True)
    render_raw_structure_tables(states_key="markov_states", strategies_key="markov_strategies", initial_key="markov_initial", transitions_key="markov_transitions", key_prefix="mk", dynamic=False)

with rewards_tab:
    st.subheader("State rewards")
    st.caption("Per-year rewards are multiplied by cycle length. Use state rewards for recurring costs and outcomes while patients occupy a state.")
    state_rewards_df = st.data_editor(
        pd.DataFrame(st.session_state.markov_state_rewards), num_rows="dynamic", use_container_width=True, key="markov_state_rewards_editor_v013",
        column_config={
            "strategy_id": st.column_config.SelectboxColumn("Strategy", options=[row.get("strategy_id") for row in st.session_state.markov_strategies]),
            "state_id": st.column_config.SelectboxColumn("State", options=[row.get("state_id") for row in st.session_state.markov_states]),
            "parameter_id": st.column_config.SelectboxColumn("Parameter", options=[row.get("id") for row in st.session_state.markov_parameters]),
            "reward_type": st.column_config.SelectboxColumn("Reward type", options=["cost", "outcome"]),
            "accrual": st.column_config.SelectboxColumn("Accrual", options=["per_year", "per_cycle"]),
        },
    )
    st.session_state.markov_state_rewards = _records(state_rewards_df)
    st.subheader("Transition-event rewards")
    transition_rewards_df = st.data_editor(
        pd.DataFrame(st.session_state.markov_transition_rewards, columns=["strategy_id", "origin_state", "destination_state", "parameter_id", "reward_type"]),
        num_rows="dynamic", use_container_width=True, key="markov_transition_rewards_editor_v013",
        column_config={
            "strategy_id": st.column_config.SelectboxColumn("Strategy", options=[row.get("strategy_id") for row in st.session_state.markov_strategies]),
            "origin_state": st.column_config.SelectboxColumn("From", options=[row.get("state_id") for row in st.session_state.markov_states]),
            "destination_state": st.column_config.SelectboxColumn("To", options=[row.get("state_id") for row in st.session_state.markov_states]),
            "parameter_id": st.column_config.SelectboxColumn("Parameter", options=[row.get("id") for row in st.session_state.markov_parameters]),
            "reward_type": st.column_config.SelectboxColumn("Reward type", options=["cost", "outcome"]),
        },
    )
    st.session_state.markov_transition_rewards = _records(transition_rewards_df)

run_settings = {
    "reference_case_code": reference_case_code,
    "outcome_code": outcome_code,
    "analysis_currency": currency_code,
    "decision_threshold": threshold,
    "cost_discount_rate": cost_discount_rate,
    "outcome_discount_rate": outcome_discount_rate,
    "included_cost_bearers": included_cost_bearers,
    "cycle_length_years": cycle_length_years,
    "max_cycles": max_cycles,
    "state_accrual_timing": state_accrual_timing,
    "transition_reward_timing": transition_reward_timing,
    "termination_mode": termination_mode,
    "depletion_threshold": depletion_threshold,
}
current_fingerprint = _fingerprint(run_settings)

compiled = None
base_result = None
compile_error = None
try:
    validate_analysis_currency(st.session_state.markov_parameters, currency_code)
    compiled = compile_markov_tables(
        st.session_state.markov_parameters, st.session_state.markov_states, st.session_state.markov_strategies,
        st.session_state.markov_initial, st.session_state.markov_transitions, st.session_state.markov_state_rewards,
        st.session_state.markov_transition_rewards, cycle_length_years=cycle_length_years, max_cycles=max_cycles,
        state_accrual_timing=state_accrual_timing, transition_reward_timing=transition_reward_timing,
        termination_mode=termination_mode, depletion_threshold=depletion_threshold,
    )
    base_result = run_cohort_markov(
        compiled.model, compiled.parameters, included_cost_bearers=included_cost_bearers,
        cost_discount_rate=cost_discount_rate, outcome_discount_rate=outcome_discount_rate,
    )
except (BuilderValidationError, MarkovValidationError, MarkovReproducibilityError, ValueError) as exc:
    compile_error = str(exc)

with analyse_tab:
    st.subheader("Analysis settings used in every result below")
    st.dataframe(
        pd.DataFrame([
            {"Setting": "Cycle length", "Value": f"{cycle_length_years:g} years ({cycle_months} months)"},
            {"Setting": "Maximum cycles / horizon", "Value": f"{max_cycles} cycles / {horizon_years} years"},
            {"Setting": "State accrual", "Value": state_accrual_timing},
            {"Setting": "Transition reward timing", "Value": transition_reward_timing},
            {"Setting": "Cost discount", "Value": f"{cost_discount_rate:.4%}"},
            {"Setting": "Outcome discount", "Value": f"{outcome_discount_rate:.4%}"},
            {"Setting": "Decision threshold", "Value": f"{currency_code} {threshold:,.2f} per {OUTCOME_MEASURES[outcome_code].unit}"},
        ]), use_container_width=True, hide_index=True,
    )
    st.caption("Base case, DSA and PSA all use this same compiled model and settings fingerprint.")

    if compile_error:
        st.error(compile_error)
        st.info("Complete or correct the guided structure above, then return to this tab.")
    else:
        assert compiled is not None and base_result is not None
        st.success("Model structure, currency consistency and transition matrices validated.")
        outcome = OUTCOME_MEASURES[outcome_code]
        economic_strategies = [Strategy(row.label, row.expected_cost, row.expected_outcome) for row in base_result.strategies]
        incremental = fully_incremental_analysis(economic_strategies, threshold)
        inc_by_name = {row.strategy.name: row for row in incremental.rows}
        result_rows = [
            {
                "Strategy": row.label,
                f"Expected cost ({currency_code})": row.expected_cost,
                f"Expected {outcome.unit}": row.expected_outcome,
                "NMB": threshold * row.expected_outcome - row.expected_cost,
                "Frontier status": inc_by_name[row.label].status,
                "Incremental cost": inc_by_name[row.label].incremental_cost,
                "Incremental outcome": inc_by_name[row.label].incremental_effect,
                "ICER": inc_by_name[row.label].icer,
                "Cycles run": row.cycles_run,
            }
            for row in base_result.strategies
        ]
        st.subheader("Base-case economic results")
        st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)

        st.subheader("Cohort trace")
        trace_strategy_ids = [row.strategy_id for row in base_result.strategies]
        trace_labels = {row.strategy_id: row.label for row in base_result.strategies}
        selected_trace_strategy = st.selectbox("Strategy trace", trace_strategy_ids, format_func=lambda sid: trace_labels[sid], key="mk_trace_strategy")
        trace_result = next(row for row in base_result.strategies if row.strategy_id == selected_trace_strategy)
        trace_df = pd.DataFrame(trace_result.trace, columns=trace_result.state_ids)
        trace_df.insert(0, "Time (years)", np.arange(len(trace_df), dtype=float) * cycle_length_years)
        trace_df.insert(0, "Cycle", np.arange(len(trace_df), dtype=int))
        trace_long = trace_df.melt(id_vars=["Cycle", "Time (years)"], value_vars=list(trace_result.state_ids), var_name="State", value_name="Cohort proportion")
        st.plotly_chart(px.line(trace_long, x="Time (years)", y="Cohort proportion", color="State", title=f"Cohort trace — {trace_labels[selected_trace_strategy]}"), use_container_width=True)
        with st.expander("Trace table"):
            st.dataframe(trace_df, use_container_width=True, hide_index=True)

        strategy_ids = list(compiled.strategy_names)
        labels = dict(compiled.strategy_names)
        if len(strategy_ids) >= 2:
            st.subheader("Sensitivity analysis")
            c1, c2 = st.columns(2)
            comparator_id = c1.selectbox("Comparator", strategy_ids, format_func=lambda sid: labels[sid], key="mk_sa_comparator")
            intervention_options = [sid for sid in strategy_ids if sid != comparator_id]
            intervention_id = c2.selectbox("Intervention", intervention_options, format_func=lambda sid: labels[sid], key="mk_sa_intervention")
            analysis_mode = st.radio("Uncertainty view", ["Deterministic (DSA)", "Probabilistic (PSA)"], horizontal=True, key="mk_uncertainty_view")
            context = dict(
                intervention_id=intervention_id,
                comparator_id=comparator_id,
                willingness_to_pay=threshold,
                included_cost_bearers=included_cost_bearers,
                cost_discount_rate=cost_discount_rate,
                outcome_discount_rate=outcome_discount_rate,
            )
            base_pair_inmb = pairwise_inmb_from_strategy_results(
                base_result.strategies,
                intervention_id=intervention_id,
                comparator_id=comparator_id,
                willingness_to_pay=threshold,
            )
            if analysis_mode.startswith("Deterministic"):
                try:
                    tornado = tornado_markov_inmb(compiled.model, compiled.parameters, **context)
                    if not tornado:
                        st.info("No parameters currently have DSA enabled with low/high bounds.")
                    else:
                        require_matching_base_inmb(base_case_inmb=base_pair_inmb, sensitivity_base_inmb=tornado[0].base_inmb)
                        st.success("DSA consistency check passed: sensitivity base INMB matches the displayed base case under the current settings.")
                        tornado_df = pd.DataFrame([
                            {"Parameter": row.label, "Low value": row.low_value, "High value": row.high_value, "INMB at low": row.low_inmb, "INMB at high": row.high_inmb, "Impact": row.impact}
                            for row in tornado
                        ])
                        st.plotly_chart(px.bar(tornado_df.sort_values("Impact"), x="Impact", y="Parameter", orientation="h", title="Tornado diagram — impact on incremental NMB"), use_container_width=True)
                        st.dataframe(tornado_df, use_container_width=True, hide_index=True)
                except (MarkovValidationError, AnalysisConsistencyError, ValueError) as exc:
                    st.error(str(exc))
                    st.warning("Sensitivity output is withheld if its base case cannot be verified against the displayed analysis.")
            else:
                c1, c2 = st.columns(2)
                iterations = int(c1.number_input("Iterations", min_value=100, max_value=100000, value=1000, step=100, key="mk_psa_iterations"))
                seed = int(c2.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=2026, step=1, key="mk_psa_seed"))
                try:
                    for warning in markov_psa_configuration_warnings(compiled.parameters):
                        st.warning(warning)
                except ValueError as exc:
                    st.error(str(exc))
                if st.button("Run Markov PSA", type="primary", key="mk_run_psa"):
                    try:
                        st.session_state.markov_psa_result = run_markov_psa(
                            compiled.model, compiled.parameters, iterations=iterations, seed=seed,
                            included_cost_bearers=included_cost_bearers, cost_discount_rate=cost_discount_rate,
                            outcome_discount_rate=outcome_discount_rate,
                        )
                        st.session_state.markov_psa_fingerprint = current_fingerprint
                    except ValueError as exc:
                        st.error(str(exc))
                psa_result = st.session_state.get("markov_psa_result")
                if psa_result is not None and st.session_state.get("markov_psa_fingerprint") != current_fingerprint:
                    st.info("The stored PSA result belongs to an earlier model/settings state. Run PSA again for the current model.")
                    psa_result = None
                if psa_result is not None:
                    de, dc = incremental_plane(psa_result, intervention_id=intervention_id, comparator_id=comparator_id)
                    st.plotly_chart(px.scatter(pd.DataFrame({"Incremental outcome": de, "Incremental cost": dc}), x="Incremental outcome", y="Incremental cost", opacity=0.45, title=f"Cost-effectiveness plane — {labels[intervention_id]} vs {labels[comparator_id]}"), use_container_width=True)
                    probability = pairwise_probability_cost_effective(psa_result, intervention_id=intervention_id, comparator_id=comparator_id, willingness_to_pay=threshold)
                    st.metric("Pairwise probability cost-effective at selected threshold", f"{probability:.1%}")
                    st.markdown("##### CEAC threshold range")
                    c1, c2, c3 = st.columns(3)
                    ceac_min = float(c1.number_input("Minimum threshold", min_value=0.0, value=0.0, step=max(threshold / 20, 1.0), key="mk_ceac_min"))
                    ceac_default_max = max(float(threshold) * 3.0, float(threshold) + 1.0, 1.0)
                    ceac_max = float(c2.number_input("Maximum threshold", min_value=0.0, value=ceac_default_max, step=max(threshold / 10, 1.0), key="mk_ceac_max"))
                    ceac_points = int(c3.number_input("Grid points", min_value=21, max_value=501, value=101, step=10, key="mk_ceac_points"))
                    try:
                        thresholds = ceac_threshold_grid(ceac_min, ceac_max, ceac_points)
                        curve = ceac(psa_result, thresholds)
                        ceac_rows = [
                            {"Threshold": threshold_value, "Probability cost-effective": probability_value, "Strategy": labels.get(sid, sid)}
                            for sid, values in curve.probabilities.items()
                            for threshold_value, probability_value in zip(curve.thresholds, values)
                        ]
                        st.plotly_chart(px.line(pd.DataFrame(ceac_rows), x="Threshold", y="Probability cost-effective", color="Strategy", title="Cost-effectiveness acceptability curve"), use_container_width=True)
                    except AnalysisConsistencyError as exc:
                        st.error(str(exc))

        st.download_button("Download deterministic results CSV", data=pd.DataFrame(result_rows).to_csv(index=False), file_name="markov_deterministic_results.csv", mime="text/csv")

st.divider()
st.caption("Save / Load / Audit now stores the structural model and the exact active analysis settings as one reproducible snapshot.")
