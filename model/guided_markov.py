"""Pure helpers for guided Markov/state-transition structure editing."""

from __future__ import annotations

import re
from math import isfinite
from typing import Any, Mapping, Sequence


class GuidedMarkovError(ValueError):
    """Raised when a guided editing operation would create an invalid structure."""


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return value or "item"


def unique_id(base: str, existing: Sequence[str]) -> str:
    existing_set = {str(item) for item in existing}
    if base not in existing_set:
        return base
    counter = 2
    while f"{base}_{counter}" in existing_set:
        counter += 1
    return f"{base}_{counter}"


def add_state(
    states: Sequence[Mapping[str, Any]],
    *,
    name: str,
    absorbing: bool = False,
    state_id: str | None = None,
) -> list[dict[str, Any]]:
    if not name.strip():
        raise GuidedMarkovError("State name is required.")
    rows = [dict(row) for row in states]
    existing_ids = [str(row.get("state_id") or "") for row in rows]
    sid = (state_id or slugify(name)).strip()
    if not sid:
        raise GuidedMarkovError("State ID is required.")
    if sid in existing_ids:
        raise GuidedMarkovError(f"State ID '{sid}' already exists.")
    rows.append({"state_id": sid, "state_name": name.strip(), "absorbing": bool(absorbing)})
    return rows


def update_state(
    states: Sequence[Mapping[str, Any]],
    state_id: str,
    *,
    name: str,
    absorbing: bool,
) -> list[dict[str, Any]]:
    if not name.strip():
        raise GuidedMarkovError("State name is required.")
    found = False
    rows: list[dict[str, Any]] = []
    for original in states:
        row = dict(original)
        if str(row.get("state_id")) == state_id:
            row["state_name"] = name.strip()
            row["absorbing"] = bool(absorbing)
            found = True
        rows.append(row)
    if not found:
        raise GuidedMarkovError(f"Unknown state '{state_id}'.")
    return rows


def delete_state(
    state_id: str,
    *,
    states: Sequence[Mapping[str, Any]],
    initial: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    state_rewards: Sequence[Mapping[str, Any]],
    transition_rewards: Sequence[Mapping[str, Any]],
    mortality_rules: Sequence[Mapping[str, Any]] = (),
) -> dict[str, list[dict[str, Any]]]:
    if len(states) <= 2:
        raise GuidedMarkovError("A state-transition model must retain at least two states.")
    if not any(str(row.get("state_id")) == state_id for row in states):
        raise GuidedMarkovError(f"Unknown state '{state_id}'.")

    cleaned_rules: list[dict[str, Any]] = []
    for original in mortality_rules:
        row = dict(original)
        applicable = [
            item.strip()
            for item in str(row.get("applicable_states") or "").split(",")
            if item.strip() and item.strip() != state_id
        ]
        row["applicable_states"] = ", ".join(applicable)
        if str(row.get("destination_state")) != state_id:
            cleaned_rules.append(row)

    return {
        "states": [dict(row) for row in states if str(row.get("state_id")) != state_id],
        "initial": [dict(row) for row in initial if str(row.get("state_id")) != state_id],
        "transitions": [
            dict(row)
            for row in transitions
            if str(row.get("origin_state")) != state_id
            and str(row.get("destination_state")) != state_id
        ],
        "state_rewards": [
            dict(row) for row in state_rewards if str(row.get("state_id")) != state_id
        ],
        "transition_rewards": [
            dict(row)
            for row in transition_rewards
            if str(row.get("origin_state")) != state_id
            and str(row.get("destination_state")) != state_id
        ],
        "mortality_rules": cleaned_rules,
    }


def add_strategy(
    strategies: Sequence[Mapping[str, Any]],
    *,
    name: str,
    strategy_id: str | None = None,
) -> list[dict[str, Any]]:
    if not name.strip():
        raise GuidedMarkovError("Strategy name is required.")
    rows = [dict(row) for row in strategies]
    existing_ids = [str(row.get("strategy_id") or "") for row in rows]
    sid = (strategy_id or slugify(name)).strip()
    if not sid:
        raise GuidedMarkovError("Strategy ID is required.")
    if sid in existing_ids:
        raise GuidedMarkovError(f"Strategy ID '{sid}' already exists.")
    if any(str(row.get("strategy_name") or "").strip() == name.strip() for row in rows):
        raise GuidedMarkovError(f"Strategy name '{name.strip()}' already exists.")
    rows.append({"strategy_id": sid, "strategy_name": name.strip()})
    return rows


def update_strategy(
    strategies: Sequence[Mapping[str, Any]], strategy_id: str, *, name: str
) -> list[dict[str, Any]]:
    if not name.strip():
        raise GuidedMarkovError("Strategy name is required.")
    found = False
    rows: list[dict[str, Any]] = []
    for original in strategies:
        row = dict(original)
        if str(row.get("strategy_id")) == strategy_id:
            row["strategy_name"] = name.strip()
            found = True
        rows.append(row)
    if not found:
        raise GuidedMarkovError(f"Unknown strategy '{strategy_id}'.")
    return rows


def delete_strategy(
    strategy_id: str,
    *,
    strategies: Sequence[Mapping[str, Any]],
    initial: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    state_rewards: Sequence[Mapping[str, Any]],
    transition_rewards: Sequence[Mapping[str, Any]],
    mortality_rules: Sequence[Mapping[str, Any]] = (),
) -> dict[str, list[dict[str, Any]]]:
    if len(strategies) <= 2:
        raise GuidedMarkovError("Keep at least two strategies for cost-effectiveness comparison.")
    if not any(str(row.get("strategy_id")) == strategy_id for row in strategies):
        raise GuidedMarkovError(f"Unknown strategy '{strategy_id}'.")
    return {
        "strategies": [
            dict(row) for row in strategies if str(row.get("strategy_id")) != strategy_id
        ],
        "initial": [
            dict(row) for row in initial if str(row.get("strategy_id")) != strategy_id
        ],
        "transitions": [
            dict(row) for row in transitions if str(row.get("strategy_id")) != strategy_id
        ],
        "state_rewards": [
            dict(row) for row in state_rewards if str(row.get("strategy_id")) != strategy_id
        ],
        "transition_rewards": [
            dict(row)
            for row in transition_rewards
            if str(row.get("strategy_id")) != strategy_id
        ],
        "mortality_rules": [
            dict(row) for row in mortality_rules if str(row.get("strategy_id")) != strategy_id
        ],
    }


def set_initial_distribution(
    initial: Sequence[Mapping[str, Any]],
    *,
    strategy_id: str,
    allocations: Mapping[str, float],
    tolerance: float = 1e-8,
) -> list[dict[str, Any]]:
    values = {str(state): float(value) for state, value in allocations.items()}
    if not values:
        raise GuidedMarkovError("At least one state allocation is required.")
    if any(not isfinite(value) or value < 0 or value > 1 for value in values.values()):
        raise GuidedMarkovError("Initial state proportions must be finite values between 0 and 1.")
    total = sum(values.values())
    if abs(total - 1.0) > tolerance:
        raise GuidedMarkovError(f"Initial cohort proportions must sum to 1. Current total: {total:.6f}.")

    rows = [dict(row) for row in initial if str(row.get("strategy_id")) != strategy_id]
    for state_id, proportion in values.items():
        if proportion != 0:
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "state_id": state_id,
                    "proportion": proportion,
                    "proportion_mode": "fixed",
                    "proportion_parameter_id": "",
                }
            )
    return rows


def add_transition(
    transitions: Sequence[Mapping[str, Any]],
    *,
    strategy_id: str,
    origin_state: str,
    destination_state: str,
    parameter_id: str = "",
    probability_mode: str = "direct",
) -> list[dict[str, Any]]:
    mode = probability_mode.strip().lower()
    if mode not in {"direct", "complement", "residual"}:
        raise GuidedMarkovError("Probability mode must be direct, complement or residual.")
    if mode != "residual" and not parameter_id.strip():
        raise GuidedMarkovError("Select a probability parameter for a direct or complement transition.")
    if mode == "residual" and parameter_id.strip():
        raise GuidedMarkovError("Residual transitions do not use a probability parameter.")

    rows = [dict(row) for row in transitions]
    if any(
        str(row.get("strategy_id")) == strategy_id
        and str(row.get("origin_state")) == origin_state
        and str(row.get("destination_state")) == destination_state
        for row in rows
    ):
        raise GuidedMarkovError(
            "That strategy already has a transition between the selected origin and destination. Edit or delete it first."
        )
    if mode == "residual" and any(
        str(row.get("strategy_id")) == strategy_id
        and str(row.get("origin_state")) == origin_state
        and str(row.get("probability_mode")) == "residual"
        for row in rows
    ):
        raise GuidedMarkovError("Only one residual transition is allowed from an origin state.")

    rows.append(
        {
            "strategy_id": strategy_id,
            "origin_state": origin_state,
            "destination_state": destination_state,
            "probability_parameter_id": "" if mode == "residual" else parameter_id.strip(),
            "probability_mode": mode,
        }
    )
    return rows


def add_dynamic_transition(
    transitions: Sequence[Mapping[str, Any]],
    *,
    strategy_id: str,
    origin_state: str,
    destination_state: str,
    input_type: str,
    parameter_id: str,
    time_basis: str,
    start_time: float,
    end_time: float | None,
    probability_mode: str = "direct",
    source_interval_years: float | None = None,
) -> list[dict[str, Any]]:
    representation = input_type.strip().lower()
    if representation not in {"probability", "rate", "probability_to_rate"}:
        raise GuidedMarkovError(
            "Input type must be probability, rate or probability_to_rate."
        )
    basis = time_basis.strip().lower()
    if basis not in {"model_time", "state_time"}:
        raise GuidedMarkovError("Time basis must be model_time or state_time.")
    if not parameter_id.strip():
        raise GuidedMarkovError("Select a parameter for the transition band.")
    if not isfinite(float(start_time)) or float(start_time) < 0:
        raise GuidedMarkovError("Schedule start time must be finite and non-negative.")
    if end_time is not None:
        if not isfinite(float(end_time)) or float(end_time) <= float(start_time):
            raise GuidedMarkovError("Schedule end time must be greater than its start time.")
    mode = probability_mode.strip().lower()
    if representation in {"rate", "probability_to_rate"}:
        mode = "direct"
    elif mode not in {"direct", "complement"}:
        raise GuidedMarkovError("Probability schedules support direct or complement mode.")

    if representation == "probability_to_rate":
        if source_interval_years is None or not isfinite(float(source_interval_years)) or float(source_interval_years) <= 0:
            raise GuidedMarkovError(
                "Probability-to-rate conversion requires the interval over which the probability was measured."
            )
        interval = float(source_interval_years)
    else:
        interval = None

    rows = [dict(row) for row in transitions]
    rows.append(
        {
            "strategy_id": strategy_id,
            "origin_state": origin_state,
            "destination_state": destination_state,
            "input_type": representation,
            "probability_mode": mode,
            "time_basis": basis,
            "start_time": float(start_time),
            "end_time": None if end_time is None else float(end_time),
            "parameter_id": parameter_id.strip(),
            "source_interval_years": interval,
        }
    )
    return rows


def delete_transition(transitions: Sequence[Mapping[str, Any]], index: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in transitions]
    if index < 0 or index >= len(rows):
        raise GuidedMarkovError("Transition index is out of range.")
    return [row for pos, row in enumerate(rows) if pos != index]
