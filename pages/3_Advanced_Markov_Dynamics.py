"""Guided advanced semi-Markov dynamics workbench."""

from __future__ import annotations

import ast
import hashlib
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model.analysis_consistency import (
    AnalysisConsistencyError,
    ceac_threshold_grid,
    pairwise_inmb_from_strategy_results,
    require_matching_base_inmb,
)
from model.currency import CURRENCIES
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.markov_reproducibility import MarkovReproducibilityError, validate_analysis_currency
from model.psa import ceac, incremental_plane, pairwise_probability_cost_effective
from model.semi_markov import SemiMarkovValidationError, run_semi_markov
from model.semi_markov_builder import compile_semi_markov_tables
from model.semi_markov_psa import run_semi_markov_psa, semi_markov_psa_configuration_warnings
from model.semi_markov_sensitivity import (
    one_way_semi_markov_inmb,
    threshold_semi_markov_inmb,
    tornado_semi_markov_inmb,
    two_way_semi_markov_inmb,
)
from model.sensitivity import OneWaySensitivitySpec, ThresholdAnalysisSpec, TwoWaySensitivitySpec
from model.transition_dynamics import (
    TransitionConversionError,
    competing_rates_to_probabilities,
    generator_to_transition_matrix,
    probability_to_rate,
    rate_to_probability,
    rescale_probability,
)
from model.tree_builder import BuilderValidationError
from ui.markov_structure_editor import (
    render_dynamic_transitions,
    render_initial_distribution,
    render_raw_structure_tables,
    render_states,
    render_strategies,
)
from ui.parameter_library import render_parameter_library


st.set_page_config(page_title="Advanced Markov Dynamics", page_icon="⏱️", layout="wide")
st.title("⏱️ Advanced Markov Dynamics")
st.caption(
    "Version 0.13 — semi-Markov memory, explicit mortality conversion, settings-consistent DSA/PSA and expanded sensitivity analysis"
)
st.info(
    "Advanced analyses always rerun the currently compiled model. The Analyse tab shows the exact cycle, horizon, accrual and discount settings used and checks DSA base-case INMB before displaying sensitivity results."
)


def _parameter_row(pid, label, value, unit, category="clinical", *, currency="", price_year=None, cost_bearers=""):
    row = {
        "id": pid,
        "label": label,
        "value": value,
        "unit": unit,
        "category": category,
        "source_citation": "Illustrative advanced-dynamics input — replace with evidence",
        "source_type": "user_assumption",
        "publication_year": None,
        "source_url": "",
        "source_details": "",
        "assumption": "Illustrative advanced-dynamics value.",
        "assumption_rationale": "Replace with evidence and a documented modelling rationale before decision use.",
        "dsa_enabled": False,
        "dsa_lower": None,
        "dsa_upper": None,
        "dsa_rationale": "Not represented in DSA.",
        "psa_enabled": False,
        "psa_rationale": "Not represented in PSA.",
        "distribution_family": "",
        "distribution_parameterisation": "",
        "distribution_parameters": {},
        "correlation_group": "",
        "notes": "",
        "currency": "",
        "price_year": None,
        "cost_bearers": "",
    }
    if category == "cost":
        row.update(
            currency=currency or "GBP",
            price_year=price_year or 2026,
            cost_bearers=cost_bearers or "health_system",
        )
    return row


def _defaults():
    parameters = [
        _parameter_row("h_prog_std_early", "Progression hazard — standard, first 2 years in stable", 0.15, "rate/year"),
        _parameter_row("h_prog_std_late", "Progression hazard — standard, after 2 years in stable", 0.10, "rate/year"),
        _parameter_row("h_prog_new_early", "Progression hazard — new treatment, first 2 years in stable", 0.08, "rate/year"),
        _parameter_row("h_prog_new_late", "Progression hazard — new treatment, after 2 years in stable", 0.07, "rate/year"),
        _parameter_row("h_death_progressed", "All-cause death hazard in progressed state", 0.20, "rate/year"),
        _parameter_row("smr_stable", "Background mortality SMR in stable state", 1.0, "ratio"),
        _parameter_row("cost_stable_std", "Stable-state annual cost — standard", 1200.0, "GBP/year", "cost", currency="GBP", price_year=2026),
        _parameter_row("cost_stable_new", "Stable-state annual cost — new treatment", 4200.0, "GBP/year", "cost", currency="GBP", price_year=2026),
        _parameter_row("cost_progressed", "Progressed-state annual cost", 7000.0, "GBP/year", "cost", currency="GBP", price_year=2026),
        _parameter_row("u_stable", "Stable-state utility", 0.82, "utility", "utility"),
        _parameter_row("u_progressed", "Progressed-state utility", 0.55, "utility", "utility"),
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
    transitions = [
        {"strategy_id": "standard", "origin_state": "stable", "destination_state": "progressed", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": 2.0, "parameter_id": "h_prog_std_early", "source_interval_years": None},
        {"strategy_id": "standard", "origin_state": "stable", "destination_state": "progressed", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 2.0, "end_time": None, "parameter_id": "h_prog_std_late", "source_interval_years": None},
        {"strategy_id": "new", "origin_state": "stable", "destination_state": "progressed", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": 2.0, "parameter_id": "h_prog_new_early", "source_interval_years": None},
        {"strategy_id": "new", "origin_state": "stable", "destination_state": "progressed", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 2.0, "end_time": None, "parameter_id": "h_prog_new_late", "source_interval_years": None},
        {"strategy_id": "standard", "origin_state": "progressed", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": None, "parameter_id": "h_death_progressed", "source_interval_years": None},
        {"strategy_id": "new", "origin_state": "progressed", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": None, "parameter_id": "h_death_progressed", "source_interval_years": None},
    ]
    state_rewards = []
    for sid, stable_cost in (("standard", "cost_stable_std"), ("new", "cost_stable_new")):
        state_rewards.extend(
            [
                {"strategy_id": sid, "state_id": "stable", "parameter_id": stable_cost, "reward_type": "cost", "accrual": "per_year"},
                {"strategy_id": sid, "state_id": "progressed", "parameter_id": "cost_progressed", "reward_type": "cost", "accrual": "per_year"},
                {"strategy_id": sid, "state_id": "stable", "parameter_id": "u_stable", "reward_type": "outcome", "accrual": "per_year"},
                {"strategy_id": sid, "state_id": "progressed", "parameter_id": "u_progressed", "reward_type": "outcome", "accrual": "per_year"},
            ]
        )
    mortality_table = []
    for age in range(60, 101):
        qx = min(0.005 * (1.085 ** (age - 60)), 0.60)
        mortality_table.append({"age": age, "annual_probability": qx})
    mortality_rules = [
        {"strategy_id": "standard", "destination_state": "dead", "initial_age": 60.0, "applicable_states": "stable", "smr_parameter_id": "smr_stable"},
        {"strategy_id": "new", "destination_state": "dead", "initial_age": 60.0, "applicable_states": "stable", "smr_parameter_id": "smr_stable"},
    ]
    return parameters, states, strategies, initial, transitions, state_rewards, [], mortality_table, mortality_rules


def _ensure_state():
    keys = [
        "adv_parameters", "adv_states", "adv_strategies", "adv_initial", "adv_transitions",
        "adv_state_rewards", "adv_transition_rewards", "adv_mortality_table", "adv_mortality_rules",
    ]
    if not all(key in st.session_state for key in keys):
        for key, value in zip(keys, _defaults()):
            st.session_state[key] = value


def _records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return [dict(row) for row in value]


def _apply_loaded_settings():
    loaded = st.session_state.get("loaded_semi_markov_settings")
    if not loaded:
        return
    token = loaded.get("restore_token") or json.dumps(loaded, sort_keys=True, default=str)
    if st.session_state.get("adv_loaded_settings_applied_token") == token:
        return
    methods = loaded.get("methods", {})
    engine = loaded.get("engine", {})
    if methods.get("currency_code") in CURRENCIES:
        st.session_state["adv_currency"] = methods["currency_code"]
    if methods.get("outcome_code") in OUTCOME_MEASURES:
        st.session_state["adv_outcome"] = methods["outcome_code"]
    if methods.get("threshold") is not None:
        st.session_state["adv_threshold"] = float(methods["threshold"])
    if methods.get("cost_discount_rate") is not None:
        st.session_state["adv_cost_discount"] = float(methods["cost_discount_rate"])
    if methods.get("outcome_discount_rate") is not None:
        st.session_state["adv_outcome_discount"] = float(methods["outcome_discount_rate"])
    if methods.get("included_cost_bearers"):
        st.session_state["adv_cost_bearers"] = ", ".join(methods["included_cost_bearers"])
    cycle_years = float(engine.get("cycle_length_years", 1.0))
    cycle_months = max(1, int(round(cycle_years * 12)))
    if cycle_months in {1, 3, 6, 12}:
        st.session_state["adv_cycle_months"] = cycle_months
    max_cycles = int(engine.get("max_cycles", 20))
    st.session_state["adv_horizon_years"] = max(1, int(round(max_cycles * cycle_years)))
    st.session_state["adv_state_accrual"] = engine.get("state_accrual_timing", "half_cycle")
    st.session_state["adv_transition_timing"] = engine.get("transition_reward_timing", "mid_cycle")
    st.session_state["adv_termination"] = "Fixed horizon" if engine.get("termination_mode", "fixed_cycles") == "fixed_cycles" else "Cohort depletion"
    st.session_state["adv_depletion"] = float(engine.get("depletion_threshold", 0.0001))
    st.session_state["adv_loaded_settings_applied_token"] = token
    st.session_state["adv_loaded_settings_applied"] = True


def _fingerprint(settings):
    payload = {
        "parameters": st.session_state.adv_parameters,
        "states": st.session_state.adv_states,
        "strategies": st.session_state.adv_strategies,
        "initial": st.session_state.adv_initial,
        "transitions": st.session_state.adv_transitions,
        "state_rewards": st.session_state.adv_state_rewards,
        "transition_rewards": st.session_state.adv_transition_rewards,
        "mortality_table": st.session_state.adv_mortality_table,
        "mortality_rules": st.session_state.adv_mortality_rules,
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _dynamic_dot(strategy_id: str) -> str:
    state_rows = st.session_state.adv_states
    transition_rows = st.session_state.adv_transitions
    lines = ["digraph SemiMarkov {", 'rankdir="LR";', 'node [fontname="Arial"];']
    for row in state_rows:
        sid = str(row.get("state_id") or "").strip()
        if not sid:
            continue
        label = str(row.get("state_name") or sid)
        shape = "doublecircle" if bool(row.get("absorbing")) else "circle"
        lines.append(f"{json.dumps(sid)} [label={json.dumps(label)}, shape={json.dumps(shape)}];")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in transition_rows:
        if str(row.get("strategy_id")) == strategy_id:
            grouped.setdefault((str(row.get("origin_state")), str(row.get("destination_state"))), []).append(row)
    for (origin, destination), rows in grouped.items():
        if origin == destination and any(str(row.get("probability_mode")) == "residual" for row in rows):
            continue
        types = {str(row.get("input_type") or "probability") for row in rows}
        clocks = {str(row.get("time_basis") or "model_time") for row in rows}
        label = ", ".join(sorted(types))
        if "state_time" in clocks:
            label += " · time in state"
        elif "model_time" in clocks:
            label += " · model time"
        if len(rows) > 1:
            label += f" · {len(rows)} bands"
        lines.append(f"{json.dumps(origin)} -> {json.dumps(destination)} [label={json.dumps(label)}];")
    lines.append("}")
    return "\n".join(lines)


_ensure_state()
_apply_loaded_settings()

with st.sidebar:
    st.header("Economic settings")
    currency_code = st.selectbox("Analysis currency", list(CURRENCIES), index=list(CURRENCIES).index("GBP"), key="adv_currency")
    outcome_code = st.selectbox("Outcome", list(OUTCOME_MEASURES), index=list(OUTCOME_MEASURES).index("QALY"), key="adv_outcome")
    threshold = st.number_input("Decision threshold", min_value=0.0, value=30000.0, step=1000.0, key="adv_threshold")
    cost_discount = st.number_input("Annual cost discount rate", min_value=0.0, max_value=0.99, value=0.035, format="%.4f", key="adv_cost_discount")
    outcome_discount = st.number_input("Annual outcome discount rate", min_value=0.0, max_value=0.99, value=0.035, format="%.4f", key="adv_outcome_discount")
    cost_bearers_text = st.text_input("Included cost bearers", value="health_system", key="adv_cost_bearers")
    included_cost_bearers = tuple(item.strip() for item in cost_bearers_text.split(",") if item.strip())

methods_tab, parameters_tab, structure_tab, rewards_tab, mortality_tab, analyse_tab, conversion_tab = st.tabs(
    ["1 · Methods", "2 · Parameters", "3 · States & dynamic transitions", "4 · Rewards", "5 · Mortality", "6 · Analyse", "7 · Hazard conversion"]
)

with methods_tab:
    st.subheader("Simulation settings")
    c1, c2, c3 = st.columns(3)
    cycle_months = c1.selectbox("Cycle length", [1, 3, 6, 12], index=3, format_func=lambda x: f"{x} month" if x == 1 else f"{x} months", key="adv_cycle_months")
    horizon_years = int(c2.number_input("Maximum horizon (years)", min_value=1, max_value=200, value=20, step=1, key="adv_horizon_years"))
    state_accrual = c3.selectbox("State reward accrual", ["half_cycle", "start", "end"], format_func=lambda x: {"half_cycle": "Half-cycle / trapezoidal", "start": "Start of cycle", "end": "End of cycle"}[x], key="adv_state_accrual")
    cycle_length = cycle_months / 12.0
    max_cycles = int(round(horizon_years / cycle_length))
    c1, c2, c3 = st.columns(3)
    transition_timing = c1.selectbox("Transition reward timing", ["mid_cycle", "start", "end"], key="adv_transition_timing")
    termination_label = c2.selectbox("Termination", ["Fixed horizon", "Cohort depletion"], key="adv_termination")
    termination_mode = "fixed_cycles" if termination_label == "Fixed horizon" else "cohort_depletion"
    depletion_threshold = c3.number_input("Depletion threshold", min_value=0.0, max_value=0.5, value=0.0001, format="%.6f", disabled=termination_mode != "cohort_depletion", key="adv_depletion")
    st.metric("Maximum cycles", max_cycles)
    st.markdown("**Model time** is time since simulation start. **Time in state** is time since entering the current health state and supplies semi-Markov memory.")

with parameters_tab:
    render_parameter_library(session_key="adv_parameters", currency_code=currency_code, key_prefix="adv_param")

with structure_tab:
    st.subheader("Guided dynamic model structure")
    render_states(
        states_key="adv_states", initial_key="adv_initial", transitions_key="adv_transitions",
        state_rewards_key="adv_state_rewards", transition_rewards_key="adv_transition_rewards",
        mortality_rules_key="adv_mortality_rules", key_prefix="adv"
    )
    st.divider()
    render_strategies(
        strategies_key="adv_strategies", initial_key="adv_initial", transitions_key="adv_transitions",
        state_rewards_key="adv_state_rewards", transition_rewards_key="adv_transition_rewards",
        mortality_rules_key="adv_mortality_rules", key_prefix="adv"
    )
    st.divider()
    render_initial_distribution(states_key="adv_states", strategies_key="adv_strategies", initial_key="adv_initial", parameters_key="adv_parameters", key_prefix="adv")
    st.divider()
    render_dynamic_transitions(states_key="adv_states", strategies_key="adv_strategies", parameters_key="adv_parameters", transitions_key="adv_transitions", key_prefix="adv")
    strategy_labels = {str(row.get("strategy_id")): str(row.get("strategy_name") or row.get("strategy_id")) for row in st.session_state.adv_strategies if row.get("strategy_id")}
    if strategy_labels:
        st.markdown("#### Model diagram")
        diagram_strategy = st.selectbox("Diagram strategy", list(strategy_labels), format_func=lambda sid: strategy_labels[sid], key="adv_diagram_strategy")
        st.graphviz_chart(_dynamic_dot(diagram_strategy), use_container_width=True)
    render_raw_structure_tables(states_key="adv_states", strategies_key="adv_strategies", initial_key="adv_initial", transitions_key="adv_transitions", key_prefix="adv", dynamic=True)

with rewards_tab:
    st.subheader("State rewards")
    state_rewards_df = st.data_editor(
        pd.DataFrame(st.session_state.adv_state_rewards), num_rows="dynamic", use_container_width=True,
        key="adv_state_rewards_editor_v013",
        column_config={
            "strategy_id": st.column_config.SelectboxColumn("Strategy", options=[row.get("strategy_id") for row in st.session_state.adv_strategies]),
            "state_id": st.column_config.SelectboxColumn("State", options=[row.get("state_id") for row in st.session_state.adv_states]),
            "parameter_id": st.column_config.SelectboxColumn("Parameter", options=[row.get("id") for row in st.session_state.adv_parameters]),
            "reward_type": st.column_config.SelectboxColumn("Reward type", options=["cost", "outcome"]),
            "accrual": st.column_config.SelectboxColumn("Accrual", options=["per_year", "per_cycle"]),
        },
    )
    st.session_state.adv_state_rewards = _records(state_rewards_df)
    st.subheader("Transition-event rewards")
    transition_rewards_df = st.data_editor(
        pd.DataFrame(st.session_state.adv_transition_rewards, columns=["strategy_id", "origin_state", "destination_state", "parameter_id", "reward_type"]),
        num_rows="dynamic", use_container_width=True, key="adv_transition_rewards_editor_v013",
        column_config={
            "strategy_id": st.column_config.SelectboxColumn("Strategy", options=[row.get("strategy_id") for row in st.session_state.adv_strategies]),
            "origin_state": st.column_config.SelectboxColumn("From", options=[row.get("state_id") for row in st.session_state.adv_states]),
            "destination_state": st.column_config.SelectboxColumn("To", options=[row.get("state_id") for row in st.session_state.adv_states]),
            "parameter_id": st.column_config.SelectboxColumn("Parameter", options=[row.get("id") for row in st.session_state.adv_parameters]),
            "reward_type": st.column_config.SelectboxColumn("Reward type", options=["cost", "outcome"]),
        },
    )
    st.session_state.adv_transition_rewards = _records(transition_rewards_df)

with mortality_tab:
    st.subheader("Age-specific background mortality")
    st.caption("Annual mortality probabilities are converted to a force of mortality; cycles crossing birthdays are integrated piecewise and SMRs multiply the mortality rate.")
    mortality_df = st.data_editor(pd.DataFrame(st.session_state.adv_mortality_table), num_rows="dynamic", use_container_width=True, height=420, key="adv_mortality_editor_v013")
    st.session_state.adv_mortality_table = _records(mortality_df)
    st.subheader("Mortality application by strategy")
    state_labels = {str(row.get("state_id")): str(row.get("state_name") or row.get("state_id")) for row in st.session_state.adv_states if row.get("state_id")}
    strategy_labels = {str(row.get("strategy_id")): str(row.get("strategy_name") or row.get("strategy_id")) for row in st.session_state.adv_strategies if row.get("strategy_id")}
    parameter_labels = {str(row.get("id")): str(row.get("label") or row.get("id")) for row in st.session_state.adv_parameters if row.get("id")}
    rules = [dict(row) for row in st.session_state.adv_mortality_rules]
    for index, row in enumerate(rules):
        sid = str(row.get("strategy_id") or "")
        with st.expander(strategy_labels.get(sid, sid) or f"Mortality rule {index + 1}", expanded=False):
            strategy_id = st.selectbox("Strategy", list(strategy_labels), index=list(strategy_labels).index(sid) if sid in strategy_labels else 0, format_func=lambda value: strategy_labels[value], key=f"adv_mortality_strategy_{index}") if strategy_labels else sid
            destination_current = str(row.get("destination_state") or "")
            destination = st.selectbox("Mortality destination state", list(state_labels), index=list(state_labels).index(destination_current) if destination_current in state_labels else 0, format_func=lambda value: state_labels[value], key=f"adv_mortality_destination_{index}") if state_labels else destination_current
            initial_age = st.number_input("Age at model entry", min_value=0.0, max_value=130.0, value=float(row.get("initial_age") or 0.0), step=1.0, key=f"adv_mortality_age_{index}")
            current_states = [item.strip() for item in str(row.get("applicable_states") or "").split(",") if item.strip()]
            applicable = st.multiselect("States receiving background mortality", list(state_labels), default=[item for item in current_states if item in state_labels], format_func=lambda value: state_labels[value], key=f"adv_mortality_applicable_{index}")
            smr_options = [""] + list(parameter_labels)
            current_smr = str(row.get("smr_parameter_id") or "")
            smr = st.selectbox("SMR parameter (optional)", smr_options, index=smr_options.index(current_smr) if current_smr in smr_options else 0, format_func=lambda value: "None — general population mortality" if value == "" else f"{parameter_labels[value]} (`{value}`)", key=f"adv_mortality_smr_{index}")
            rules[index] = {"strategy_id": strategy_id, "destination_state": destination, "initial_age": initial_age, "applicable_states": ", ".join(applicable), "smr_parameter_id": smr}
    st.session_state.adv_mortality_rules = rules
    with st.expander("Advanced · edit mortality rules as a table"):
        rules_df = st.data_editor(pd.DataFrame(st.session_state.adv_mortality_rules), num_rows="dynamic", use_container_width=True, key="adv_mortality_rules_raw_v013")
        if st.button("Apply mortality-rule table edits", key="adv_apply_mortality_rules_raw"):
            st.session_state.adv_mortality_rules = _records(rules_df)
            st.rerun()
    st.warning(
        "Automatic background mortality requires a rate-based competing-risk row. If disease progression is supplied as an interval probability, choose ‘Interval probability → constant cause-specific rate’ in the transition editor and document the constant-hazard assumption and source interval. Plain probability rows are not converted silently."
    )

run_settings = {
    "currency_code": currency_code,
    "outcome_code": outcome_code,
    "threshold": threshold,
    "cost_discount": cost_discount,
    "outcome_discount": outcome_discount,
    "included_cost_bearers": included_cost_bearers,
    "cycle_length_years": cycle_length,
    "max_cycles": max_cycles,
    "state_accrual_timing": state_accrual,
    "transition_reward_timing": transition_timing,
    "termination_mode": termination_mode,
    "depletion_threshold": depletion_threshold,
}
current_fingerprint = _fingerprint(run_settings)
compiled = None
run = None
compile_error = None
try:
    validate_analysis_currency(st.session_state.adv_parameters, currency_code)
    compiled = compile_semi_markov_tables(
        st.session_state.adv_parameters, st.session_state.adv_states, st.session_state.adv_strategies,
        st.session_state.adv_initial, st.session_state.adv_transitions, st.session_state.adv_state_rewards,
        st.session_state.adv_transition_rewards, st.session_state.adv_mortality_table, st.session_state.adv_mortality_rules,
        cycle_length_years=cycle_length, max_cycles=max_cycles, state_accrual_timing=state_accrual,
        transition_reward_timing=transition_timing, termination_mode=termination_mode,
        depletion_threshold=depletion_threshold,
    )
    run = run_semi_markov(
        compiled.model, compiled.parameters, included_cost_bearers=included_cost_bearers,
        cost_discount_rate=cost_discount, outcome_discount_rate=outcome_discount,
    )
except (BuilderValidationError, SemiMarkovValidationError, MarkovReproducibilityError, ValueError) as exc:
    compile_error = str(exc)

with analyse_tab:
    st.subheader("Analysis settings used in every result below")
    st.dataframe(
        pd.DataFrame([
            {"Setting": "Cycle length", "Value": f"{cycle_length:g} years ({cycle_months} months)"},
            {"Setting": "Maximum cycles / horizon", "Value": f"{max_cycles} cycles / {horizon_years} years"},
            {"Setting": "State accrual", "Value": state_accrual},
            {"Setting": "Transition reward timing", "Value": transition_timing},
            {"Setting": "Cost discount", "Value": f"{cost_discount:.4%}"},
            {"Setting": "Outcome discount", "Value": f"{outcome_discount:.4%}"},
            {"Setting": "Decision threshold", "Value": f"{currency_code} {threshold:,.2f} per {OUTCOME_MEASURES[outcome_code].unit}"},
        ]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Base case, DSA and PSA all use this same compiled model and settings fingerprint.")

    if compile_error:
        st.error(compile_error)
        st.info("Complete or correct the model structure, then return to this tab.")
    else:
        assert compiled is not None and run is not None
        currency = CURRENCIES[currency_code]
        outcome = OUTCOME_MEASURES[outcome_code]
        st.success("Dynamic cohort model validated and run.")
        economic = [Strategy(row.label, row.expected_cost, row.expected_outcome) for row in run.strategies]
        incremental = fully_incremental_analysis(economic, threshold)
        inc_map = {row.strategy.name: row for row in incremental.rows}
        st.subheader("Base-case economic results")
        st.dataframe(
            pd.DataFrame([
                {
                    "Strategy": row.label,
                    f"Expected cost ({currency_code})": row.expected_cost,
                    f"Expected {outcome.unit}": row.expected_outcome,
                    "NMB": threshold * row.expected_outcome - row.expected_cost,
                    "Frontier status": inc_map[row.label].status,
                    "Incremental cost": inc_map[row.label].incremental_cost,
                    "Incremental outcome": inc_map[row.label].incremental_effect,
                    "ICER": inc_map[row.label].icer,
                }
                for row in run.strategies
            ]),
            use_container_width=True, hide_index=True,
        )

        strategy_map = {row.strategy_id: row for row in run.strategies}
        selected = st.selectbox("Trace strategy", list(strategy_map), format_func=lambda sid: strategy_map[sid].label, key="adv_trace_strategy")
        selected_run = strategy_map[selected]
        trace = pd.DataFrame(selected_run.trace, columns=selected_run.state_ids)
        trace.insert(0, "Time (years)", np.arange(len(trace)) * cycle_length)
        trace_long = trace.melt(id_vars="Time (years)", var_name="State", value_name="Proportion")
        st.plotly_chart(px.line(trace_long, x="Time (years)", y="Proportion", color="State", title=f"State occupancy — {selected_run.label}"), use_container_width=True)
        mean_time = pd.DataFrame(selected_run.mean_state_time_trace, columns=selected_run.state_ids)
        mean_time.insert(0, "Time (years)", np.arange(len(mean_time)) * cycle_length)
        mean_long = mean_time.melt(id_vars="Time (years)", var_name="State", value_name="Mean time in state among current occupants")
        st.plotly_chart(px.line(mean_long, x="Time (years)", y="Mean time in state among current occupants", color="State", title="State-time memory / tunnel-state diagnostic"), use_container_width=True)

        strategy_ids = list(strategy_map)
        if len(strategy_ids) >= 2:
            st.subheader("Sensitivity analysis")
            c1, c2 = st.columns(2)
            comparator_id = c1.selectbox("Comparator", strategy_ids, format_func=lambda sid: strategy_map[sid].label, key="adv_sa_comparator")
            intervention_options = [sid for sid in strategy_ids if sid != comparator_id]
            intervention_id = c2.selectbox("Intervention", intervention_options, format_func=lambda sid: strategy_map[sid].label, key="adv_sa_intervention")
            mode = st.radio("Uncertainty view", ["Deterministic (DSA)", "Probabilistic (PSA)"], horizontal=True, key="adv_uncertainty_view")
            context = dict(
                intervention_id=intervention_id, comparator_id=comparator_id,
                willingness_to_pay=threshold, included_cost_bearers=included_cost_bearers,
                cost_discount_rate=cost_discount, outcome_discount_rate=outcome_discount,
            )
            base_pair_inmb = pairwise_inmb_from_strategy_results(
                run.strategies, intervention_id=intervention_id, comparator_id=comparator_id,
                willingness_to_pay=threshold,
            )

            if mode.startswith("Deterministic"):
                eligible = [p for p in compiled.parameters if p.dsa is not None and p.dsa.enabled and p.dsa.lower is not None and p.dsa.upper is not None]
                if not eligible:
                    st.info("No parameters currently have DSA enabled with low/high bounds.")
                else:
                    try:
                        tornado_check = tornado_semi_markov_inmb(compiled.model, compiled.parameters, **context)
                        if tornado_check:
                            require_matching_base_inmb(base_case_inmb=base_pair_inmb, sensitivity_base_inmb=tornado_check[0].base_inmb)
                        st.success("DSA consistency check passed: sensitivity base INMB matches the displayed base case under the current settings.")
                        dsa_view = st.segmented_control("DSA view", ["Tornado", "One-way", "Two-way", "Threshold"], default="Tornado", key="adv_dsa_view")
                        labels = {p.id: p.label for p in eligible}
                        params = {p.id: p for p in eligible}
                        if dsa_view == "Tornado":
                            tornado_df = pd.DataFrame([
                                {"Parameter": row.label, "Low value": row.low_value, "High value": row.high_value, "INMB at low": row.low_inmb, "INMB at high": row.high_inmb, "Impact": row.impact}
                                for row in tornado_check
                            ])
                            st.plotly_chart(px.bar(tornado_df.sort_values("Impact"), x="Impact", y="Parameter", orientation="h", title="Tornado diagram — impact on incremental NMB"), use_container_width=True)
                            st.dataframe(tornado_df, use_container_width=True, hide_index=True)
                        elif dsa_view == "One-way":
                            pid = st.selectbox("Parameter", list(labels), format_func=lambda x: labels[x], key="adv_owa_param")
                            p = params[pid]
                            values = tuple(float(x) for x in np.linspace(p.dsa.lower, p.dsa.upper, 41))
                            rows = one_way_semi_markov_inmb(compiled.model, compiled.parameters, OneWaySensitivitySpec(pid, values), **context)
                            frame = pd.DataFrame(rows, columns=["Parameter value", "INMB"])
                            fig = px.line(frame, x="Parameter value", y="INMB", title=f"One-way INMB — {labels[pid]}")
                            fig.add_hline(y=0)
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(frame, use_container_width=True, hide_index=True)
                        elif dsa_view == "Two-way":
                            if len(eligible) < 2:
                                st.info("Two-way analysis requires at least two DSA-enabled parameters.")
                            else:
                                c1, c2 = st.columns(2)
                                x_id = c1.selectbox("X parameter", list(labels), format_func=lambda x: labels[x], key="adv_two_x")
                                y_options = [pid for pid in labels if pid != x_id]
                                y_id = c2.selectbox("Y parameter", y_options, format_func=lambda x: labels[x], key="adv_two_y")
                                px_param, py_param = params[x_id], params[y_id]
                                x_values = tuple(float(v) for v in np.linspace(px_param.dsa.lower, px_param.dsa.upper, 21))
                                y_values = tuple(float(v) for v in np.linspace(py_param.dsa.lower, py_param.dsa.upper, 21))
                                rows = two_way_semi_markov_inmb(compiled.model, compiled.parameters, TwoWaySensitivitySpec(x_id, x_values, y_id, y_values), **context)
                                frame = pd.DataFrame(rows, columns=[labels[x_id], labels[y_id], "INMB"])
                                pivot = frame.pivot(index=labels[y_id], columns=labels[x_id], values="INMB")
                                fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorbar={"title": "INMB"}))
                                fig.update_layout(title="Two-way decision map — incremental NMB", xaxis_title=labels[x_id], yaxis_title=labels[y_id])
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption("INMB > 0 favours the selected intervention at the stated threshold; INMB < 0 favours the comparator. The map does not change the decision threshold.")
                        else:
                            pid = st.selectbox("Threshold parameter", list(labels), format_func=lambda x: labels[x], key="adv_threshold_param")
                            p = params[pid]
                            try:
                                switching = threshold_semi_markov_inmb(
                                    compiled.model, compiled.parameters,
                                    ThresholdAnalysisSpec(pid, float(p.dsa.lower), float(p.dsa.upper)),
                                    **context,
                                )
                                st.metric("INMB switching value", f"{switching:,.6g}")
                                values = tuple(float(v) for v in np.linspace(p.dsa.lower, p.dsa.upper, 81))
                                rows = one_way_semi_markov_inmb(compiled.model, compiled.parameters, OneWaySensitivitySpec(pid, values), **context)
                                frame = pd.DataFrame(rows, columns=["Parameter value", "INMB"])
                                fig = px.line(frame, x="Parameter value", y="INMB", title=f"Threshold analysis — {labels[pid]}")
                                fig.add_hline(y=0)
                                fig.add_vline(x=switching)
                                st.plotly_chart(fig, use_container_width=True)
                            except ValueError as exc:
                                st.info(f"No switching value is bracketed by the configured DSA range: {exc}")
                    except (ValueError, AnalysisConsistencyError) as exc:
                        st.error(str(exc))
                        st.warning("Sensitivity output is withheld because its base case could not be verified against the displayed analysis.")
            else:
                c1, c2 = st.columns(2)
                iterations = int(c1.number_input("Iterations", min_value=100, max_value=100000, value=1000, step=100, key="adv_psa_iterations"))
                seed = int(c2.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=2026, step=1, key="adv_psa_seed"))
                try:
                    for warning in semi_markov_psa_configuration_warnings(compiled.parameters):
                        st.warning(warning)
                except ValueError as exc:
                    st.error(str(exc))
                if st.button("Run semi-Markov PSA", type="primary", key="adv_run_psa"):
                    try:
                        st.session_state.adv_psa_result = run_semi_markov_psa(
                            compiled.model, compiled.parameters, iterations=iterations, seed=seed,
                            included_cost_bearers=included_cost_bearers,
                            cost_discount_rate=cost_discount, outcome_discount_rate=outcome_discount,
                        )
                        st.session_state.adv_psa_fingerprint = current_fingerprint
                    except ValueError as exc:
                        st.error(str(exc))
                psa_result = st.session_state.get("adv_psa_result")
                if psa_result is not None and st.session_state.get("adv_psa_fingerprint") != current_fingerprint:
                    st.info("The stored PSA result belongs to an earlier model/settings state. Run PSA again for the current model.")
                    psa_result = None
                if psa_result is not None:
                    de, dc = incremental_plane(psa_result, intervention_id=intervention_id, comparator_id=comparator_id)
                    st.plotly_chart(px.scatter(pd.DataFrame({"Incremental outcome": de, "Incremental cost": dc}), x="Incremental outcome", y="Incremental cost", opacity=0.45, title=f"Cost-effectiveness plane — {strategy_map[intervention_id].label} vs {strategy_map[comparator_id].label}"), use_container_width=True)
                    probability = pairwise_probability_cost_effective(psa_result, intervention_id=intervention_id, comparator_id=comparator_id, willingness_to_pay=threshold)
                    st.metric("Pairwise probability cost-effective at selected threshold", f"{probability:.1%}")
                    st.markdown("##### CEAC threshold range")
                    c1, c2, c3 = st.columns(3)
                    ceac_min = float(c1.number_input("Minimum threshold", min_value=0.0, value=0.0, step=max(threshold / 20, 1.0), key="adv_ceac_min"))
                    ceac_default_max = max(float(threshold) * 3.0, float(threshold) + 1.0, 1.0)
                    ceac_max = float(c2.number_input("Maximum threshold", min_value=0.0, value=ceac_default_max, step=max(threshold / 10, 1.0), key="adv_ceac_max"))
                    ceac_points = int(c3.number_input("Grid points", min_value=21, max_value=501, value=101, step=10, key="adv_ceac_points"))
                    try:
                        thresholds = ceac_threshold_grid(ceac_min, ceac_max, ceac_points)
                        curve = ceac(psa_result, thresholds)
                        ceac_rows = [
                            {"Threshold": threshold_value, "Probability cost-effective": probability_value, "Strategy": strategy_map[sid].label}
                            for sid, values in curve.probabilities.items()
                            for threshold_value, probability_value in zip(curve.thresholds, values)
                        ]
                        st.plotly_chart(px.line(pd.DataFrame(ceac_rows), x="Threshold", y="Probability cost-effective", color="Strategy", title="Cost-effectiveness acceptability curve"), use_container_width=True)
                    except AnalysisConsistencyError as exc:
                        st.error(str(exc))

with conversion_tab:
    st.subheader("Hazard / probability conversion")
    conversion_mode = st.radio("Conversion mode", ["Single event", "Rescale probability", "Competing hazards", "Generator matrix"], horizontal=True, key="adv_conversion_mode")
    if conversion_mode == "Single event":
        c1, c2 = st.columns(2)
        duration = c1.number_input("Interval (years)", min_value=0.000001, value=1.0, format="%.6f")
        direction = c2.radio("Direction", ["Rate → probability", "Probability → rate"], horizontal=True)
        if direction.startswith("Rate"):
            rate = st.number_input("Constant rate / hazard", min_value=0.0, value=0.1, format="%.6f")
            st.metric("Interval probability", f"{rate_to_probability(rate, duration):.6f}")
        else:
            probability_input = st.number_input("Interval probability", min_value=0.0, max_value=0.999999, value=0.10, format="%.6f")
            st.metric("Constant rate / hazard", f"{probability_to_rate(probability_input, duration):.6f}")
        st.caption("The probability→rate direction assumes a constant underlying hazard over the stated source interval. It is not an automatic conversion of arbitrary competing cumulative incidences.")
    elif conversion_mode == "Rescale probability":
        c1, c2, c3 = st.columns(3)
        probability = c1.number_input("Known interval probability", min_value=0.0, max_value=0.999999, value=0.20, format="%.6f")
        from_duration = c2.number_input("Known interval length (years)", min_value=0.000001, value=1.0, format="%.6f")
        to_duration = c3.number_input("Target interval length (years)", min_value=0.000001, value=0.5, format="%.6f")
        try:
            st.metric("Rescaled probability", f"{rescale_probability(probability, from_duration=from_duration, to_duration=to_duration):.6f}")
            st.caption("This assumes a constant underlying hazard across both intervals.")
        except TransitionConversionError as exc:
            st.error(str(exc))
    elif conversion_mode == "Competing hazards":
        competing_df = st.data_editor(pd.DataFrame([{"destination": "Progression", "rate": 0.10}, {"destination": "Death", "rate": 0.05}]), num_rows="dynamic", use_container_width=True, key="competing_hazards_editor_v013")
        duration = st.number_input("Interval (years)", min_value=0.000001, value=1.0, key="competing_duration_v013")
        try:
            rates = {str(row["destination"]): float(row["rate"]) for row in _records(competing_df) if str(row.get("destination") or "").strip()}
            converted = competing_rates_to_probabilities(rates, duration)
            output = [{"Destination": key, "Probability": value} for key, value in converted.destination_probabilities.items()]
            output.append({"Destination": "Remain in origin state", "Probability": converted.stay_probability})
            st.dataframe(pd.DataFrame(output), use_container_width=True, hide_index=True)
        except (TransitionConversionError, ValueError) as exc:
            st.error(str(exc))
    else:
        generator_text = st.text_area("Generator Q", value="[[-0.15, 0.10, 0.05], [0.0, -0.20, 0.20], [0.0, 0.0, 0.0]]", height=120)
        duration = st.number_input("Interval (years)", min_value=0.000001, value=1.0, key="generator_duration_v013")
        try:
            generator = np.asarray(ast.literal_eval(generator_text), dtype=float)
            st.dataframe(pd.DataFrame(generator_to_transition_matrix(generator, duration)), use_container_width=True)
            st.caption("Computes P(t)=exp(Q×t), allowing multi-step movement within the interval.")
        except (TransitionConversionError, ValueError, SyntaxError) as exc:
            st.error(str(exc))

st.divider()
st.caption("Use Save / Load / Audit to persist this model. Structure and the exact active analysis settings are stored together and restored together.")
