"""Pure compiler helpers for the cohort Markov builder."""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any, Mapping, Sequence

from model.markov import (
    CohortMarkovDefinition,
    InitialStateAllocation,
    MarkovState,
    MarkovStrategyDefinition,
    StateReward,
    TransitionProbability,
    TransitionReward,
)
from model.schema import Parameter
from model.tree_builder import BuilderValidationError, compile_parameter_rows


@dataclass(frozen=True)
class CompiledMarkovModel:
    model: CohortMarkovDefinition
    parameters: tuple[Parameter, ...]
    strategy_names: Mapping[str, str]


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and isnan(value):
        return True
    return not str(value).strip()


def _text(value: Any, field: str) -> str:
    if _blank(value):
        raise BuilderValidationError(f"{field} is required.")
    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    return None if _blank(value) else str(value).strip()


def _float(value: Any, field: str) -> float:
    if _blank(value):
        raise BuilderValidationError(f"{field} is required.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BuilderValidationError(f"{field} must be numeric.") from exc


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def compile_markov_tables(
    parameter_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    strategy_rows: Sequence[Mapping[str, Any]],
    initial_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    state_reward_rows: Sequence[Mapping[str, Any]],
    transition_reward_rows: Sequence[Mapping[str, Any]],
    *,
    cycle_length_years: float,
    max_cycles: int,
    state_accrual_timing: str = "half_cycle",
    transition_reward_timing: str = "mid_cycle",
    termination_mode: str = "fixed_cycles",
    depletion_threshold: float = 1e-6,
) -> CompiledMarkovModel:
    parameters = compile_parameter_rows(parameter_rows)

    states: list[MarkovState] = []
    for index, row in enumerate(state_rows, start=1):
        if _blank(row.get("state_id")):
            continue
        try:
            states.append(
                MarkovState(
                    id=_text(row.get("state_id"), f"State row {index}: state_id"),
                    label=_text(row.get("state_name"), f"State row {index}: state_name"),
                    absorbing=_bool(row.get("absorbing")),
                )
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc
    state_ids = [state.id for state in states]
    if len(states) < 2:
        raise BuilderValidationError("At least two health states are required.")
    if len(state_ids) != len(set(state_ids)):
        raise BuilderValidationError("Health-state ids must be unique.")

    strategy_names: dict[str, str] = {}
    for index, row in enumerate(strategy_rows, start=1):
        if _blank(row.get("strategy_id")):
            continue
        strategy_id = _text(row.get("strategy_id"), f"Strategy row {index}: strategy_id")
        strategy_name = _text(row.get("strategy_name"), f"Strategy '{strategy_id}': strategy_name")
        if strategy_id in strategy_names:
            raise BuilderValidationError("Strategy ids must be unique.")
        if strategy_name in strategy_names.values():
            raise BuilderValidationError("Strategy names must be unique.")
        strategy_names[strategy_id] = strategy_name
    if len(strategy_names) < 2:
        raise BuilderValidationError("At least two strategies are required.")

    initial_by_strategy: dict[str, list[InitialStateAllocation]] = {sid: [] for sid in strategy_names}
    for index, row in enumerate(initial_rows, start=1):
        if _blank(row.get("strategy_id")):
            continue
        sid = _text(row.get("strategy_id"), f"Initial row {index}: strategy_id")
        if sid not in strategy_names:
            raise BuilderValidationError(f"Initial row references unknown strategy '{sid}'.")
        mode = (_optional_text(row.get("proportion_mode")) or _optional_text(row.get("allocation_mode")) or "fixed").lower()
        parameter_id = _optional_text(row.get("proportion_parameter_id")) or _optional_text(row.get("parameter_id"))
        proportion = None if mode != "fixed" else _float(row.get("proportion"), f"Initial row {index}: proportion")
        try:
            initial_by_strategy[sid].append(
                InitialStateAllocation(
                    state_id=_text(row.get("state_id"), f"Initial row {index}: state_id"),
                    proportion=proportion,
                    proportion_parameter_id=parameter_id,
                    proportion_mode=mode,
                )
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc

    transitions_by_strategy: dict[str, list[TransitionProbability]] = {sid: [] for sid in strategy_names}
    for index, row in enumerate(transition_rows, start=1):
        if _blank(row.get("strategy_id")):
            continue
        sid = _text(row.get("strategy_id"), f"Transition row {index}: strategy_id")
        if sid not in strategy_names:
            raise BuilderValidationError(f"Transition row references unknown strategy '{sid}'.")
        mode = (_optional_text(row.get("probability_mode")) or "direct").lower()
        parameter_id = _optional_text(row.get("probability_parameter_id"))
        try:
            transitions_by_strategy[sid].append(
                TransitionProbability(
                    origin_state=_text(row.get("origin_state"), f"Transition row {index}: origin_state"),
                    destination_state=_text(row.get("destination_state"), f"Transition row {index}: destination_state"),
                    probability_parameter_id=parameter_id,
                    probability_mode=mode,
                )
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc

    state_rewards_by_strategy: dict[str, list[StateReward]] = {sid: [] for sid in strategy_names}
    for index, row in enumerate(state_reward_rows, start=1):
        if _blank(row.get("strategy_id")):
            continue
        sid = _text(row.get("strategy_id"), f"State reward row {index}: strategy_id")
        if sid not in strategy_names:
            raise BuilderValidationError(f"State reward references unknown strategy '{sid}'.")
        try:
            state_rewards_by_strategy[sid].append(
                StateReward(
                    state_id=_text(row.get("state_id"), f"State reward row {index}: state_id"),
                    parameter_id=_text(row.get("parameter_id"), f"State reward row {index}: parameter_id"),
                    reward_type=_text(row.get("reward_type"), f"State reward row {index}: reward_type").lower(),
                    accrual=(_optional_text(row.get("accrual")) or "per_cycle").lower(),
                )
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc

    transition_rewards_by_strategy: dict[str, list[TransitionReward]] = {sid: [] for sid in strategy_names}
    for index, row in enumerate(transition_reward_rows, start=1):
        if _blank(row.get("strategy_id")):
            continue
        sid = _text(row.get("strategy_id"), f"Transition reward row {index}: strategy_id")
        if sid not in strategy_names:
            raise BuilderValidationError(f"Transition reward references unknown strategy '{sid}'.")
        try:
            transition_rewards_by_strategy[sid].append(
                TransitionReward(
                    origin_state=_text(row.get("origin_state"), f"Transition reward row {index}: origin_state"),
                    destination_state=_text(row.get("destination_state"), f"Transition reward row {index}: destination_state"),
                    parameter_id=_text(row.get("parameter_id"), f"Transition reward row {index}: parameter_id"),
                    reward_type=_text(row.get("reward_type"), f"Transition reward row {index}: reward_type").lower(),
                )
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc

    strategies = tuple(
        MarkovStrategyDefinition(
            strategy_id=sid,
            label=name,
            initial_distribution=tuple(initial_by_strategy[sid]),
            transitions=tuple(transitions_by_strategy[sid]),
            state_rewards=tuple(state_rewards_by_strategy[sid]),
            transition_rewards=tuple(transition_rewards_by_strategy[sid]),
        )
        for sid, name in strategy_names.items()
    )

    try:
        model = CohortMarkovDefinition(
            states=tuple(states),
            strategies=strategies,
            cycle_length_years=float(cycle_length_years),
            max_cycles=int(max_cycles),
            state_accrual_timing=state_accrual_timing,
            transition_reward_timing=transition_reward_timing,
            termination_mode=termination_mode,
            depletion_threshold=float(depletion_threshold),
        )
    except (TypeError, ValueError) as exc:
        raise BuilderValidationError(str(exc)) from exc

    return CompiledMarkovModel(model=model, parameters=parameters, strategy_names=strategy_names)


def markov_structure_to_dot(
    state_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    *,
    strategy_id: str,
) -> str:
    """Create a strategy-specific Graphviz representation for the UI."""
    import json

    def q(value: Any) -> str:
        return json.dumps(str(value))

    lines = [
        "digraph MarkovModel {",
        'rankdir="LR";',
        'graph [pad="0.2", nodesep="0.45", ranksep="0.65"];',
        'node [fontname="Arial"];',
    ]
    for row in state_rows:
        if _blank(row.get("state_id")):
            continue
        state_id = str(row.get("state_id")).strip()
        label = str(row.get("state_name") or state_id).strip()
        shape = "doublecircle" if _bool(row.get("absorbing")) else "circle"
        lines.append(f"{q(state_id)} [label={q(label)}, shape={q(shape)}];")

    for row in transition_rows:
        if _blank(row.get("strategy_id")) or str(row.get("strategy_id")).strip() != strategy_id:
            continue
        if _blank(row.get("origin_state")) or _blank(row.get("destination_state")):
            continue
        origin = str(row.get("origin_state")).strip()
        destination = str(row.get("destination_state")).strip()
        mode = str(row.get("probability_mode") or "direct").strip().lower()
        parameter = str(row.get("probability_parameter_id") or "").strip()
        if mode == "residual":
            label = "residual"
        elif mode == "complement":
            label = f"1 - {parameter}"
        else:
            label = parameter
        lines.append(f"{q(origin)} -> {q(destination)} [label={q(label)}];")

    lines.append("}")
    return "\n".join(lines)
