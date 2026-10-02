"""Advanced semi-Markov dynamics workbench for Economic Evaluation Platform v0.6."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model.currency import CURRENCIES
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.semi_markov import SemiMarkovValidationError, run_semi_markov
from model.semi_markov_builder import compile_semi_markov_tables
from model.transition_dynamics import (
    TransitionConversionError,
    competing_rates_to_probabilities,
    generator_to_transition_matrix,
    rate_to_probability,
    rescale_probability,
)
from model.tree_builder import BuilderValidationError


st.set_page_config(page_title="Advanced Markov Dynamics", page_icon="⏱️", layout="wide")
st.title("⏱️ Advanced Markov Dynamics")
st.caption(
    "Version 0.6 workbench — tunnel/semi-Markov state-time memory, simulation-time transitions, attained-age mortality and explicit hazard conversion"
)

st.info(
    "This page is intentionally separate from the v0.5 builder while the advanced dynamics are being validated. "
    "It does not yet save/load model files. Transition schedules can depend on model time or time since entry to the current state."
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
        "assumption": "Illustrative v0.6 workbench value.",
        "assumption_rationale": "Replace with evidence and a documented modelling rationale before decision use.",
        "dsa_enabled": False,
        "dsa_rationale": "Not configured in this workbench view.",
        "psa_enabled": False,
        "psa_rationale": "Not configured in this workbench view.",
        "distribution_family": "",
        "distribution_parameters": {},
        "correlation_group": "",
        "notes": "",
    }
    if category == "cost":
        row.update(currency=currency or "GBP", price_year=price_year or 2026, cost_bearers=cost_bearers or "health_system")
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
        {"strategy_id": "standard", "state_id": "stable", "proportion": 1.0},
        {"strategy_id": "new", "state_id": "stable", "proportion": 1.0},
    ]
    transitions = [
        {"strategy_id": "standard", "origin_state": "stable", "destination_state": "progressed", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": 2.0, "parameter_id": "h_prog_std_early"},
        {"strategy_id": "standard", "origin_state": "stable", "destination_state": "progressed", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 2.0, "end_time": None, "parameter_id": "h_prog_std_late"},
        {"strategy_id": "new", "origin_state": "stable", "destination_state": "progressed", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": 2.0, "parameter_id": "h_prog_new_early"},
        {"strategy_id": "new", "origin_state": "stable", "destination_state": "progressed", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 2.0, "end_time": None, "parameter_id": "h_prog_new_late"},
        {"strategy_id": "standard", "origin_state": "progressed", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": None, "parameter_id": "h_death_progressed"},
        {"strategy_id": "new", "origin_state": "progressed", "destination_state": "dead", "input_type": "rate", "probability_mode": "direct", "time_basis": "state_time", "start_time": 0.0, "end_time": None, "parameter_id": "h_death_progressed"},
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
        # Smooth illustrative qx curve only; users should replace this with a
        # documented life table appropriate to the jurisdiction and population.
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
        values = _defaults()
        for key, value in zip(keys, values):
            st.session_state[key] = value


def _records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return [dict(row) for row in value]


_ensure_state()

with st.sidebar:
    st.header("Economic settings")
    currency_code = st.selectbox("Analysis currency", list(CURRENCIES), index=list(CURRENCIES).index("GBP"))
    outcome_code = st.selectbox("Outcome", list(OUTCOME_MEASURES), index=list(OUTCOME_MEASURES).index("QALY"))
    threshold = st.number_input("Decision threshold", min_value=0.0, value=30000.0, step=1000.0)
    cost_discount = st.number_input("Annual cost discount rate", min_value=0.0, max_value=0.99, value=0.035, format="%.4f")
    outcome_discount = st.number_input("Annual outcome discount rate", min_value=0.0, max_value=0.99, value=0.035, format="%.4f")

methods_tab, parameters_tab, structure_tab, mortality_tab, analyse_tab, conversion_tab = st.tabs(
    ["1 · Methods", "2 · Parameters", "3 · Dynamic transitions", "4 · Mortality", "5 · Analyse", "6 · Hazard conversion"]
)

with methods_tab:
    st.subheader("Simulation settings")
    c1, c2, c3 = st.columns(3)
    cycle_months = c1.selectbox("Cycle length", [1, 3, 6, 12], index=3, format_func=lambda x: f"{x} month" if x == 1 else f"{x} months")
    horizon_years = int(c2.number_input("Maximum horizon (years)", min_value=1, max_value=100, value=20, step=1))
    accrual = c3.selectbox("State reward accrual", ["half_cycle", "start", "end"], format_func=lambda x: {"half_cycle": "Half-cycle / trapezoidal", "start": "Start of cycle", "end": "End of cycle"}[x])
    cycle_length = cycle_months / 12.0
    max_cycles = int(round(horizon_years / cycle_length))
    st.write(f"**Maximum cycles:** {max_cycles}")
    st.markdown(
        "**Time bases** — `model_time` changes according to time since the simulation started. `state_time` changes according to how long cohort mass has occupied the current state. The latter is the semi-Markov/tunnel-state mechanism."
    )

with parameters_tab:
    st.subheader("Parameter values")
    editable = pd.DataFrame(st.session_state.adv_parameters)[["id", "label", "value", "unit", "category"]]
    edited = st.data_editor(editable, use_container_width=True, hide_index=True, key="adv_parameter_editor")
    edited_by_id = {row["id"]: row for row in _records(edited)}
    updated = []
    for original in st.session_state.adv_parameters:
        row = dict(original)
        edit = edited_by_id.get(row["id"])
        if edit:
            row.update({key: edit[key] for key in ("label", "value", "unit", "category")})
        if row.get("category") == "cost":
            row["currency"] = currency_code
        updated.append(row)
    st.session_state.adv_parameters = updated
    st.caption("The workbench keeps illustrative provenance internally; full evidence/uncertainty editing remains in the main modeller until this workflow is consolidated.")

with structure_tab:
    st.subheader("Health states")
    states_df = st.data_editor(pd.DataFrame(st.session_state.adv_states), num_rows="dynamic", use_container_width=True, key="adv_states_editor")
    st.session_state.adv_states = _records(states_df)

    st.subheader("Strategies")
    strategies_df = st.data_editor(pd.DataFrame(st.session_state.adv_strategies), num_rows="dynamic", use_container_width=True, key="adv_strategies_editor")
    st.session_state.adv_strategies = _records(strategies_df)

    st.subheader("Initial cohort distribution")
    initial_df = st.data_editor(pd.DataFrame(st.session_state.adv_initial), num_rows="dynamic", use_container_width=True, key="adv_initial_editor")
    st.session_state.adv_initial = _records(initial_df)

    st.subheader("Time-varying / state-time transition schedules")
    st.caption("Each row is one piecewise-constant band [start_time, end_time). Leave end_time blank for the final open-ended band. All exits from an origin state must use either probabilities or rates consistently.")
    transitions_df = st.data_editor(
        pd.DataFrame(st.session_state.adv_transitions),
        num_rows="dynamic",
        use_container_width=True,
        key="adv_transitions_editor",
        column_config={
            "input_type": st.column_config.SelectboxColumn("input_type", options=["probability", "rate"]),
            "probability_mode": st.column_config.SelectboxColumn("probability_mode", options=["direct", "complement"]),
            "time_basis": st.column_config.SelectboxColumn("time_basis", options=["model_time", "state_time"]),
        },
    )
    st.session_state.adv_transitions = _records(transitions_df)

    with st.expander("State rewards"):
        rewards_df = st.data_editor(pd.DataFrame(st.session_state.adv_state_rewards), num_rows="dynamic", use_container_width=True, key="adv_state_rewards_editor")
        st.session_state.adv_state_rewards = _records(rewards_df)

with mortality_tab:
    st.subheader("Age-specific background mortality")
    st.caption("Annual mortality probabilities are converted to a constant force within each age year. Cycles crossing birthdays are integrated piecewise. SMRs multiply the mortality rate, not the probability.")
    mortality_df = st.data_editor(pd.DataFrame(st.session_state.adv_mortality_table), num_rows="dynamic", use_container_width=True, height=420, key="adv_mortality_editor")
    st.session_state.adv_mortality_table = _records(mortality_df)
    st.subheader("Mortality application by strategy")
    mortality_rules_df = st.data_editor(pd.DataFrame(st.session_state.adv_mortality_rules), num_rows="dynamic", use_container_width=True, key="adv_mortality_rules_editor")
    st.session_state.adv_mortality_rules = _records(mortality_rules_df)
    st.warning("Automatic background mortality is only combined with rate-based exits. If an origin row is probability-based, death must be represented explicitly in that probability schedule rather than added heuristically.")

compiled = None
run = None
compile_error = None
try:
    compiled = compile_semi_markov_tables(
        st.session_state.adv_parameters,
        st.session_state.adv_states,
        st.session_state.adv_strategies,
        st.session_state.adv_initial,
        st.session_state.adv_transitions,
        st.session_state.adv_state_rewards,
        st.session_state.adv_transition_rewards,
        st.session_state.adv_mortality_table,
        st.session_state.adv_mortality_rules,
        cycle_length_years=cycle_length,
        max_cycles=max_cycles,
        state_accrual_timing=accrual,
        transition_reward_timing="mid_cycle",
    )
    run = run_semi_markov(
        compiled.model,
        compiled.parameters,
        included_cost_bearers=("health_system",),
        cost_discount_rate=cost_discount,
        outcome_discount_rate=outcome_discount,
    )
except (BuilderValidationError, SemiMarkovValidationError, ValueError) as exc:
    compile_error = str(exc)

with analyse_tab:
    if compile_error:
        st.error(compile_error)
    else:
        assert compiled is not None and run is not None
        currency = CURRENCIES[currency_code]
        outcome = OUTCOME_MEASURES[outcome_code]
        st.success("Dynamic cohort model validated and run.")
        economic = [Strategy(row.label, row.expected_cost, row.expected_outcome) for row in run.strategies]
        incremental = fully_incremental_analysis(economic, threshold)
        inc_map = {row.strategy.name: row for row in incremental.rows}
        table = []
        for row in run.strategies:
            inc = inc_map[row.label]
            table.append({
                "Strategy": row.label,
                f"Expected cost ({currency_code})": row.expected_cost,
                f"Expected {outcome.unit}": row.expected_outcome,
                "NMB": threshold * row.expected_outcome - row.expected_cost,
                "Frontier status": inc.status,
                "Incremental cost": inc.incremental_cost,
                "Incremental outcome": inc.incremental_effect,
                "ICER": inc.icer,
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        strategy_map = {row.strategy_id: row for row in run.strategies}
        selected = st.selectbox("Trace strategy", list(strategy_map), format_func=lambda sid: strategy_map[sid].label)
        selected_run = strategy_map[selected]
        trace = pd.DataFrame(selected_run.trace, columns=selected_run.state_ids)
        trace.insert(0, "Time (years)", np.arange(len(trace)) * cycle_length)
        trace_long = trace.melt(id_vars="Time (years)", var_name="State", value_name="Proportion")
        st.plotly_chart(px.line(trace_long, x="Time (years)", y="Proportion", color="State", title=f"State occupancy — {selected_run.label}"), use_container_width=True)

        mean_time = pd.DataFrame(selected_run.mean_state_time_trace, columns=selected_run.state_ids)
        mean_time.insert(0, "Time (years)", np.arange(len(mean_time)) * cycle_length)
        mean_long = mean_time.melt(id_vars="Time (years)", var_name="State", value_name="Mean time in state among current occupants")
        st.plotly_chart(px.line(mean_long, x="Time (years)", y="Mean time in state among current occupants", color="State", title="State-time memory / tunnel-state diagnostic"), use_container_width=True)
        st.caption("The second chart is a diagnostic of the implicit tunnel-state distribution: it shows mean time since entry among people currently occupying each state.")

with conversion_tab:
    st.subheader("Hazard / probability conversion")
    mode = st.radio("Conversion mode", ["Single event", "Rescale probability", "Competing hazards", "Generator matrix"], horizontal=True)

    if mode == "Single event":
        c1, c2 = st.columns(2)
        rate = c1.number_input("Constant rate / hazard", min_value=0.0, value=0.1, format="%.6f")
        duration = c2.number_input("Interval (years)", min_value=0.000001, value=1.0, format="%.6f")
        st.metric("Interval probability", f"{rate_to_probability(rate, duration):.6f}")
        st.caption("Uses p = 1 − exp(−r × t). This is appropriate for a single constant hazard over the interval.")

    elif mode == "Rescale probability":
        c1, c2, c3 = st.columns(3)
        probability = c1.number_input("Known interval probability", min_value=0.0, max_value=0.999999, value=0.20, format="%.6f")
        from_duration = c2.number_input("Known interval length (years)", min_value=0.000001, value=1.0, format="%.6f")
        to_duration = c3.number_input("Target interval length (years)", min_value=0.000001, value=0.5, format="%.6f")
        try:
            converted = rescale_probability(probability, from_duration=from_duration, to_duration=to_duration)
            st.metric("Rescaled probability", f"{converted:.6f}")
            st.caption("This assumes the underlying hazard is constant across both intervals; it is not a generic rescaling identity.")
        except TransitionConversionError as exc:
            st.error(str(exc))

    elif mode == "Competing hazards":
        competing_df = st.data_editor(
            pd.DataFrame([{"destination": "Progression", "rate": 0.10}, {"destination": "Death", "rate": 0.05}]),
            num_rows="dynamic",
            use_container_width=True,
            key="competing_hazards_editor",
        )
        duration = st.number_input("Interval (years)", min_value=0.000001, value=1.0, key="competing_duration")
        try:
            rates = {str(row["destination"]): float(row["rate"]) for row in _records(competing_df) if str(row.get("destination") or "").strip()}
            converted = competing_rates_to_probabilities(rates, duration)
            output = [{"Destination": key, "Probability": value} for key, value in converted.destination_probabilities.items()]
            output.append({"Destination": "Remain in origin state", "Probability": converted.stay_probability})
            st.dataframe(pd.DataFrame(output), use_container_width=True, hide_index=True)
            st.caption("Competing rates are converted jointly so the destination probabilities plus remaining in the origin state sum to one.")
        except (TransitionConversionError, ValueError) as exc:
            st.error(str(exc))

    else:
        st.caption("Enter a Python-style square generator matrix Q. Off-diagonal entries are non-negative transition intensities and each row must sum to zero.")
        generator_text = st.text_area("Generator Q", value="[[-0.15, 0.10, 0.05], [0.0, -0.20, 0.20], [0.0, 0.0, 0.0]]", height=120)
        duration = st.number_input("Interval (years)", min_value=0.000001, value=1.0, key="generator_duration")
        try:
            generator = np.asarray(ast.literal_eval(generator_text), dtype=float)
            p = generator_to_transition_matrix(generator, duration)
            st.dataframe(pd.DataFrame(p), use_container_width=True)
            st.caption("This computes the full CTMC transition matrix P(t)=exp(Q×t), including possible multi-step movement within the interval.")
        except (TransitionConversionError, ValueError, SyntaxError) as exc:
            st.error(str(exc))
