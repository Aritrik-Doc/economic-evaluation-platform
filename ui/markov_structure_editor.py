"""Guided Streamlit editors for cohort and semi-Markov model structure.

These components deliberately keep the row-oriented model tables as the source of
truth while presenting a friendlier choose/configure/add workflow. Raw tables
remain available under an Advanced expander for experienced modellers.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from model.guided_markov import (
    GuidedMarkovError,
    add_dynamic_transition,
    add_state,
    add_strategy,
    add_transition,
    delete_state,
    delete_strategy,
    delete_transition,
    slugify,
    update_state,
    update_strategy,
)


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return [dict(row) for row in value]


def _state_labels(states: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("state_id")): str(row.get("state_name") or row.get("state_id"))
        for row in states
        if row.get("state_id")
    }


def _strategy_labels(strategies: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("strategy_id")): str(row.get("strategy_name") or row.get("strategy_id"))
        for row in strategies
        if row.get("strategy_id")
    }


def _parameter_labels(parameters: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("id")): str(row.get("label") or row.get("id"))
        for row in parameters
        if row.get("id")
    }


def render_states(
    *,
    states_key: str,
    initial_key: str,
    transitions_key: str,
    state_rewards_key: str,
    transition_rewards_key: str,
    key_prefix: str,
    mortality_rules_key: str | None = None,
) -> None:
    states = [dict(row) for row in st.session_state[states_key]]
    st.markdown("#### 1. Health states")
    st.caption(
        "A health state represents a clinically meaningful condition a patient can occupy. "
        "Mark a state as absorbing when patients cannot leave it, such as Death."
    )

    delete_id: str | None = None
    for index, original in enumerate(states):
        sid = str(original.get("state_id") or "")
        name = str(original.get("state_name") or sid)
        absorbing = bool(original.get("absorbing"))
        badge = " · absorbing" if absorbing else ""
        with st.expander(f"{name}  (`{sid}`){badge}", expanded=False):
            c1, c2 = st.columns([3, 1])
            new_name = c1.text_input(
                "State name",
                value=name,
                key=f"{key_prefix}_state_name_{sid}_{index}",
            )
            new_absorbing = c2.toggle(
                "Absorbing",
                value=absorbing,
                key=f"{key_prefix}_state_abs_{sid}_{index}",
            )
            st.caption(f"Stable state ID: `{sid}`")
            if new_name != name or new_absorbing != absorbing:
                try:
                    states = update_state(
                        states,
                        sid,
                        name=new_name,
                        absorbing=new_absorbing,
                    )
                    st.session_state[states_key] = states
                except GuidedMarkovError as exc:
                    st.error(str(exc))
            if st.button(
                "Delete state",
                key=f"{key_prefix}_delete_state_{sid}_{index}",
                type="secondary",
            ):
                delete_id = sid

    if delete_id is not None:
        try:
            result = delete_state(
                delete_id,
                states=st.session_state[states_key],
                initial=st.session_state[initial_key],
                transitions=st.session_state[transitions_key],
                state_rewards=st.session_state[state_rewards_key],
                transition_rewards=st.session_state[transition_rewards_key],
                mortality_rules=(
                    st.session_state[mortality_rules_key]
                    if mortality_rules_key is not None
                    else ()
                ),
            )
            st.session_state[states_key] = result["states"]
            st.session_state[initial_key] = result["initial"]
            st.session_state[transitions_key] = result["transitions"]
            st.session_state[state_rewards_key] = result["state_rewards"]
            st.session_state[transition_rewards_key] = result["transition_rewards"]
            if mortality_rules_key is not None:
                st.session_state[mortality_rules_key] = result["mortality_rules"]
            st.rerun()
        except GuidedMarkovError as exc:
            st.error(str(exc))

    with st.form(f"{key_prefix}_add_state_form", clear_on_submit=True):
        st.markdown("**Add another state**")
        c1, c2 = st.columns([2, 1])
        new_name = c1.text_input("State name", placeholder="e.g. Progressed disease")
        new_absorbing = c2.checkbox("Absorbing state")
        new_id = st.text_input(
            "State ID (optional)",
            placeholder="Generated automatically from the name",
        )
        submitted = st.form_submit_button("+ Add state")
        if submitted:
            try:
                st.session_state[states_key] = add_state(
                    st.session_state[states_key],
                    name=new_name,
                    absorbing=new_absorbing,
                    state_id=new_id.strip() or None,
                )
                st.rerun()
            except GuidedMarkovError as exc:
                st.error(str(exc))


def render_strategies(
    *,
    strategies_key: str,
    initial_key: str,
    transitions_key: str,
    state_rewards_key: str,
    transition_rewards_key: str,
    key_prefix: str,
    mortality_rules_key: str | None = None,
) -> None:
    strategies = [dict(row) for row in st.session_state[strategies_key]]
    st.markdown("#### 2. Strategies")
    st.caption(
        "Strategies are the mutually exclusive alternatives being compared, such as standard care and a new treatment."
    )

    delete_id: str | None = None
    for index, original in enumerate(strategies):
        sid = str(original.get("strategy_id") or "")
        name = str(original.get("strategy_name") or sid)
        with st.expander(f"{name}  (`{sid}`)", expanded=False):
            new_name = st.text_input(
                "Strategy name",
                value=name,
                key=f"{key_prefix}_strategy_name_{sid}_{index}",
            )
            st.caption(f"Stable strategy ID: `{sid}`")
            if new_name != name:
                try:
                    strategies = update_strategy(strategies, sid, name=new_name)
                    st.session_state[strategies_key] = strategies
                except GuidedMarkovError as exc:
                    st.error(str(exc))
            if st.button(
                "Delete strategy",
                key=f"{key_prefix}_delete_strategy_{sid}_{index}",
                type="secondary",
            ):
                delete_id = sid

    if delete_id is not None:
        try:
            result = delete_strategy(
                delete_id,
                strategies=st.session_state[strategies_key],
                initial=st.session_state[initial_key],
                transitions=st.session_state[transitions_key],
                state_rewards=st.session_state[state_rewards_key],
                transition_rewards=st.session_state[transition_rewards_key],
                mortality_rules=(
                    st.session_state[mortality_rules_key]
                    if mortality_rules_key is not None
                    else ()
                ),
            )
            st.session_state[strategies_key] = result["strategies"]
            st.session_state[initial_key] = result["initial"]
            st.session_state[transitions_key] = result["transitions"]
            st.session_state[state_rewards_key] = result["state_rewards"]
            st.session_state[transition_rewards_key] = result["transition_rewards"]
            if mortality_rules_key is not None:
                st.session_state[mortality_rules_key] = result["mortality_rules"]
            st.rerun()
        except GuidedMarkovError as exc:
            st.error(str(exc))

    with st.form(f"{key_prefix}_add_strategy_form", clear_on_submit=True):
        st.markdown("**Add another strategy**")
        new_name = st.text_input("Strategy name", placeholder="e.g. New treatment")
        new_id = st.text_input(
            "Strategy ID (optional)",
            placeholder="Generated automatically from the name",
        )
        submitted = st.form_submit_button("+ Add strategy")
        if submitted:
            try:
                st.session_state[strategies_key] = add_strategy(
                    st.session_state[strategies_key],
                    name=new_name,
                    strategy_id=new_id.strip() or None,
                )
                st.rerun()
            except GuidedMarkovError as exc:
                st.error(str(exc))


def render_initial_distribution(
    *,
    states_key: str,
    strategies_key: str,
    initial_key: str,
    key_prefix: str,
) -> None:
    states = [dict(row) for row in st.session_state[states_key]]
    strategies = [dict(row) for row in st.session_state[strategies_key]]
    state_labels = _state_labels(states)
    strategy_labels = _strategy_labels(strategies)
    strategy_ids = list(strategy_labels)

    st.markdown("#### 3. Starting cohort")
    st.caption(
        "For each strategy, specify where the cohort is located at time zero. The proportions must sum to 1."
    )
    if not strategy_ids or not state_labels:
        st.info("Add at least one strategy and one health state first.")
        return

    selected_strategy = st.selectbox(
        "Strategy to configure",
        strategy_ids,
        format_func=lambda sid: strategy_labels[sid],
        key=f"{key_prefix}_initial_strategy",
    )
    existing = {
        str(row.get("state_id")): float(row.get("proportion") or 0.0)
        for row in st.session_state[initial_key]
        if str(row.get("strategy_id")) == selected_strategy
    }

    allocations: dict[str, float] = {}
    columns = st.columns(2)
    for index, (state_id, label) in enumerate(state_labels.items()):
        allocations[state_id] = columns[index % 2].number_input(
            label,
            min_value=0.0,
            max_value=1.0,
            value=float(existing.get(state_id, 0.0)),
            step=0.01,
            format="%.6f",
            key=f"{key_prefix}_initial_{selected_strategy}_{state_id}",
        )

    total = sum(allocations.values())
    if abs(total - 1.0) <= 1e-8:
        st.success(f"Starting distribution sums to 1.000000 for {strategy_labels[selected_strategy]}.")
    else:
        st.warning(f"Starting distribution currently sums to {total:.6f}; it must equal 1 before the model can run.")

    if st.button(
        "Apply starting distribution",
        key=f"{key_prefix}_apply_initial_{selected_strategy}",
        type="primary",
        disabled=abs(total - 1.0) > 1e-8,
    ):
        remaining = [
            dict(row)
            for row in st.session_state[initial_key]
            if str(row.get("strategy_id")) != selected_strategy
        ]
        remaining.extend(
            {
                "strategy_id": selected_strategy,
                "state_id": state_id,
                "proportion": value,
            }
            for state_id, value in allocations.items()
            if value != 0
        )
        st.session_state[initial_key] = remaining
        st.rerun()

    if len(strategy_ids) > 1 and st.button(
        "Copy this starting distribution to all strategies",
        key=f"{key_prefix}_copy_initial_all",
        disabled=abs(total - 1.0) > 1e-8,
    ):
        rows: list[dict[str, Any]] = []
        for sid in strategy_ids:
            rows.extend(
                {
                    "strategy_id": sid,
                    "state_id": state_id,
                    "proportion": value,
                }
                for state_id, value in allocations.items()
                if value != 0
            )
        st.session_state[initial_key] = rows
        st.rerun()


def _transition_summary(
    row: Mapping[str, Any],
    state_labels: Mapping[str, str],
    parameter_labels: Mapping[str, str],
) -> str:
    origin = state_labels.get(str(row.get("origin_state")), str(row.get("origin_state")))
    destination = state_labels.get(
        str(row.get("destination_state")), str(row.get("destination_state"))
    )
    mode = str(row.get("probability_mode") or "direct")
    parameter = str(row.get("probability_parameter_id") or "")
    if mode == "residual":
        definition = "remaining probability"
    elif mode == "complement":
        definition = f"1 − {parameter_labels.get(parameter, parameter)}"
    else:
        definition = parameter_labels.get(parameter, parameter)
    return f"{origin} → {destination} · {definition}"


def render_standard_transitions(
    *,
    states_key: str,
    strategies_key: str,
    parameters_key: str,
    transitions_key: str,
    key_prefix: str,
) -> None:
    states = [dict(row) for row in st.session_state[states_key]]
    strategies = [dict(row) for row in st.session_state[strategies_key]]
    parameters = [dict(row) for row in st.session_state[parameters_key]]
    transitions = [dict(row) for row in st.session_state[transitions_key]]
    state_labels = _state_labels(states)
    strategy_labels = _strategy_labels(strategies)
    parameter_labels = _parameter_labels(parameters)
    state_ids = list(state_labels)
    strategy_ids = list(strategy_labels)
    parameter_ids = list(parameter_labels)

    st.markdown("#### 4. Transitions")
    st.caption(
        "Choose a strategy, the state patients leave, and the state they enter. "
        "Use a residual transition for the probability left after the other exits from an origin state."
    )
    if not state_ids or not strategy_ids:
        st.info("Add states and strategies before defining transitions.")
        return

    with st.form(f"{key_prefix}_add_transition_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        strategy_id = c1.selectbox(
            "Strategy",
            strategy_ids,
            format_func=lambda sid: strategy_labels[sid],
        )
        origin = c2.selectbox(
            "From state",
            state_ids,
            format_func=lambda sid: state_labels[sid],
        )
        destination = c3.selectbox(
            "To state",
            state_ids,
            format_func=lambda sid: state_labels[sid],
        )
        mode = st.selectbox(
            "How is this probability defined?",
            ["direct", "complement", "residual"],
            format_func=lambda value: {
                "direct": "Use a probability parameter directly",
                "complement": "Use 1 minus a probability parameter",
                "residual": "Use the remaining probability after other exits",
            }[value],
        )
        if mode == "residual":
            parameter_id = ""
            st.info("Residual transition probability = 1 − the sum of the other outgoing transition probabilities from this state.")
        else:
            parameter_id = st.selectbox(
                "Probability parameter",
                parameter_ids,
                format_func=lambda pid: f"{parameter_labels[pid]}  (`{pid}`)",
            ) if parameter_ids else ""
        submitted = st.form_submit_button("+ Add transition")
        if submitted:
            try:
                st.session_state[transitions_key] = add_transition(
                    st.session_state[transitions_key],
                    strategy_id=strategy_id,
                    origin_state=origin,
                    destination_state=destination,
                    parameter_id=parameter_id,
                    probability_mode=mode,
                )
                st.rerun()
            except GuidedMarkovError as exc:
                st.error(str(exc))

    st.markdown("**Existing transitions**")
    selected_strategy = st.selectbox(
        "Show transitions for",
        strategy_ids,
        format_func=lambda sid: strategy_labels[sid],
        key=f"{key_prefix}_show_transition_strategy",
    )
    visible = [
        (index, row)
        for index, row in enumerate(transitions)
        if str(row.get("strategy_id")) == selected_strategy
    ]
    if not visible:
        st.info("No transitions have been defined for this strategy yet.")
    delete_index: int | None = None
    for index, row in visible:
        summary = _transition_summary(row, state_labels, parameter_labels)
        with st.expander(summary, expanded=False):
            c1, c2 = st.columns(2)
            c1.write(f"**From:** {state_labels.get(str(row.get('origin_state')), row.get('origin_state'))}")
            c2.write(f"**To:** {state_labels.get(str(row.get('destination_state')), row.get('destination_state'))}")
            mode = str(row.get("probability_mode") or "direct")
            if mode == "residual":
                st.write("**Probability:** Remaining probability")
            else:
                pid = str(row.get("probability_parameter_id") or "")
                text = parameter_labels.get(pid, pid)
                st.write(f"**Probability:** {'1 − ' if mode == 'complement' else ''}{text}")
            if st.button(
                "Delete transition",
                key=f"{key_prefix}_delete_transition_{index}",
                type="secondary",
            ):
                delete_index = index
    if delete_index is not None:
        st.session_state[transitions_key] = delete_transition(
            st.session_state[transitions_key], delete_index
        )
        st.rerun()


def _dynamic_summary(
    row: Mapping[str, Any],
    state_labels: Mapping[str, str],
    parameter_labels: Mapping[str, str],
) -> str:
    origin = state_labels.get(str(row.get("origin_state")), str(row.get("origin_state")))
    destination = state_labels.get(
        str(row.get("destination_state")), str(row.get("destination_state"))
    )
    start = float(row.get("start_time") or 0.0)
    raw_end = row.get("end_time")
    end = "∞" if raw_end in (None, "") else f"{float(raw_end):g}"
    basis = "time in state" if str(row.get("time_basis")) == "state_time" else "model time"
    pid = str(row.get("parameter_id") or "")
    representation = str(row.get("input_type") or "probability")
    return (
        f"{origin} → {destination} · {basis} {start:g}–{end} years · "
        f"{representation}: {parameter_labels.get(pid, pid)}"
    )


def render_dynamic_transitions(
    *,
    states_key: str,
    strategies_key: str,
    parameters_key: str,
    transitions_key: str,
    key_prefix: str,
) -> None:
    states = [dict(row) for row in st.session_state[states_key]]
    strategies = [dict(row) for row in st.session_state[strategies_key]]
    parameters = [dict(row) for row in st.session_state[parameters_key]]
    transitions = [dict(row) for row in st.session_state[transitions_key]]
    state_labels = _state_labels(states)
    strategy_labels = _strategy_labels(strategies)
    parameter_labels = _parameter_labels(parameters)
    state_ids = list(state_labels)
    strategy_ids = list(strategy_labels)
    parameter_ids = list(parameter_labels)

    st.markdown("#### 4. Time-varying / semi-Markov transitions")
    st.caption(
        "Add one time band at a time. Use model time when a transition changes with time since the analysis began, "
        "or time in state when it depends on how long patients have occupied the current state."
    )
    if not state_ids or not strategy_ids or not parameter_ids:
        st.info("Add states, strategies and transition parameters before defining dynamic transitions.")
        return

    with st.form(f"{key_prefix}_add_dynamic_transition_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        strategy_id = c1.selectbox(
            "Strategy",
            strategy_ids,
            format_func=lambda sid: strategy_labels[sid],
        )
        origin = c2.selectbox("From state", state_ids, format_func=lambda sid: state_labels[sid])
        destination = c3.selectbox("To state", state_ids, format_func=lambda sid: state_labels[sid])
        c1, c2 = st.columns(2)
        input_type = c1.selectbox(
            "Transition input",
            ["probability", "rate"],
            format_func=lambda value: "Probability for each model cycle" if value == "probability" else "Cause-specific rate / hazard per year",
        )
        time_basis = c2.selectbox(
            "What clock controls this band?",
            ["model_time", "state_time"],
            format_func=lambda value: "Time since model start" if value == "model_time" else "Time since entering the current state",
        )
        c1, c2 = st.columns(2)
        start_time = c1.number_input("Band starts at (years)", min_value=0.0, value=0.0, step=0.25)
        open_ended = c2.checkbox("Final open-ended band", value=True)
        if open_ended:
            end_time = None
            c2.caption("This band continues indefinitely.")
        else:
            end_time = c2.number_input("Band ends at (years)", min_value=float(start_time) + 0.000001, value=max(float(start_time) + 1.0, 1.0), step=0.25)
        parameter_id = st.selectbox(
            "Transition parameter",
            parameter_ids,
            format_func=lambda pid: f"{parameter_labels[pid]}  (`{pid}`)",
        )
        if input_type == "probability":
            probability_mode = st.selectbox(
                "Probability definition",
                ["direct", "complement"],
                format_func=lambda value: "Use parameter directly" if value == "direct" else "Use 1 minus parameter",
            )
        else:
            probability_mode = "direct"
            st.caption("Rates are combined jointly as competing rates by the advanced engine.")
        submitted = st.form_submit_button("+ Add transition band")
        if submitted:
            try:
                st.session_state[transitions_key] = add_dynamic_transition(
                    st.session_state[transitions_key],
                    strategy_id=strategy_id,
                    origin_state=origin,
                    destination_state=destination,
                    input_type=input_type,
                    parameter_id=parameter_id,
                    time_basis=time_basis,
                    start_time=start_time,
                    end_time=end_time,
                    probability_mode=probability_mode,
                )
                st.rerun()
            except GuidedMarkovError as exc:
                st.error(str(exc))

    st.markdown("**Existing schedule bands**")
    selected_strategy = st.selectbox(
        "Show schedule for",
        strategy_ids,
        format_func=lambda sid: strategy_labels[sid],
        key=f"{key_prefix}_show_dynamic_strategy",
    )
    visible = [
        (index, row)
        for index, row in enumerate(transitions)
        if str(row.get("strategy_id")) == selected_strategy
    ]
    if not visible:
        st.info("No time-varying transition bands have been defined for this strategy yet.")
    delete_index: int | None = None
    for index, row in visible:
        with st.expander(_dynamic_summary(row, state_labels, parameter_labels), expanded=False):
            st.write(
                f"**Input:** {row.get('input_type')} · **Clock:** {row.get('time_basis')} · "
                f"**Parameter:** `{row.get('parameter_id')}`"
            )
            st.write(
                f"**Band:** {row.get('start_time')} to {row.get('end_time') if row.get('end_time') not in (None, '') else 'open ended'} years"
            )
            if str(row.get("input_type")) == "probability":
                st.write(f"**Probability mode:** {row.get('probability_mode', 'direct')}")
            if st.button(
                "Delete this band",
                key=f"{key_prefix}_delete_dynamic_{index}",
                type="secondary",
            ):
                delete_index = index
    if delete_index is not None:
        st.session_state[transitions_key] = delete_transition(
            st.session_state[transitions_key], delete_index
        )
        st.rerun()


def render_raw_structure_tables(
    *,
    states_key: str,
    strategies_key: str,
    initial_key: str,
    transitions_key: str,
    key_prefix: str,
    dynamic: bool,
) -> None:
    with st.expander("Advanced · edit underlying structure tables"):
        st.warning(
            "This view is intended for experienced users and bulk edits. Invalid combinations will still be rejected by model validation."
        )
        states_df = st.data_editor(
            pd.DataFrame(st.session_state[states_key]),
            num_rows="dynamic",
            use_container_width=True,
            key=f"{key_prefix}_raw_states",
        )
        strategies_df = st.data_editor(
            pd.DataFrame(st.session_state[strategies_key]),
            num_rows="dynamic",
            use_container_width=True,
            key=f"{key_prefix}_raw_strategies",
        )
        initial_df = st.data_editor(
            pd.DataFrame(st.session_state[initial_key]),
            num_rows="dynamic",
            use_container_width=True,
            key=f"{key_prefix}_raw_initial",
        )
        if dynamic:
            transitions_df = st.data_editor(
                pd.DataFrame(st.session_state[transitions_key]),
                num_rows="dynamic",
                use_container_width=True,
                key=f"{key_prefix}_raw_transitions",
                column_config={
                    "input_type": st.column_config.SelectboxColumn(
                        "input_type", options=["probability", "rate"]
                    ),
                    "probability_mode": st.column_config.SelectboxColumn(
                        "probability_mode", options=["direct", "complement"]
                    ),
                    "time_basis": st.column_config.SelectboxColumn(
                        "time_basis", options=["model_time", "state_time"]
                    ),
                },
            )
        else:
            transitions_df = st.data_editor(
                pd.DataFrame(st.session_state[transitions_key]),
                num_rows="dynamic",
                use_container_width=True,
                key=f"{key_prefix}_raw_transitions",
                column_config={
                    "probability_mode": st.column_config.SelectboxColumn(
                        "probability_mode", options=["direct", "complement", "residual"]
                    ),
                },
            )
        if st.button("Apply advanced table edits", key=f"{key_prefix}_apply_raw"):
            st.session_state[states_key] = _records(states_df)
            st.session_state[strategies_key] = _records(strategies_df)
            st.session_state[initial_key] = _records(initial_df)
            st.session_state[transitions_key] = _records(transitions_df)
            st.rerun()
