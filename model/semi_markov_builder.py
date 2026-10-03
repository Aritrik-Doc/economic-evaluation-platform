"""Compiler from editable UI tables to semi-Markov model definitions.

The compiler is the compatibility boundary between editable/imported row data
and the stricter advanced state-transition engine. It accepts legacy aliases
where they are unambiguous, removes only residual self-stay rows (because the
semi-Markov engine derives staying internally), and never silently converts an
interval probability to a rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any, Mapping, Sequence

from model.markov import InitialStateAllocation, MarkovState, StateReward, TransitionReward
from model.schema import Parameter
from model.semi_markov import (
    BackgroundMortalityRule,
    DynamicTransition,
    SemiMarkovDefinition,
    SemiMarkovStrategyDefinition,
)
from model.transition_dynamics import (
    AgeSpecificMortalityTable,
    ParameterBand,
    PiecewiseParameterSchedule,
)
from model.tree_builder import BuilderValidationError, compile_parameter_rows, parse_id_list


@dataclass(frozen=True)
class CompiledSemiMarkovModel:
    model: SemiMarkovDefinition
    parameters: tuple[Parameter, ...]


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and isnan(value):
        return True
    return not str(value).strip()


def _text(value: Any, field: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise BuilderValidationError(f"{field} is required.")
    return text


def _optional_text(value: Any) -> str | None:
    return None if _blank(value) else str(value).strip()


def _float(value: Any, field: str) -> float:
    if _blank(value):
        raise BuilderValidationError(f"{field} is required.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BuilderValidationError(f"{field} must be numeric.") from exc


def _optional_float(value: Any) -> float | None:
    if _blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BuilderValidationError("Optional numeric value is invalid.") from exc


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def compile_states(rows: Sequence[Mapping[str, Any]]) -> tuple[MarkovState, ...]:
    states: list[MarkovState] = []
    for index, row in enumerate(rows, start=1):
        if not str(row.get("state_id") or "").strip():
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
    if len(states) < 2:
        raise BuilderValidationError("At least two health states are required.")
    ids = [state.id for state in states]
    if len(ids) != len(set(ids)):
        raise BuilderValidationError("Health-state ids must be unique.")
    return tuple(states)


def compile_mortality_table(rows: Sequence[Mapping[str, Any]]) -> AgeSpecificMortalityTable | None:
    values: list[tuple[int, float]] = []
    for index, row in enumerate(rows, start=1):
        if _blank(row.get("age")):
            continue
        age = int(_float(row.get("age"), f"Mortality row {index}: age"))
        probability = _float(
            row.get("annual_probability"),
            f"Mortality row {index}: annual_probability",
        )
        values.append((age, probability))
    if not values:
        return None
    try:
        return AgeSpecificMortalityTable(tuple(values))
    except ValueError as exc:
        raise BuilderValidationError(str(exc)) from exc


def normalize_semi_markov_transition_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize compatible standard/legacy transition rows for the advanced engine.

    Standard cohort models often store an explicit residual self-transition.
    Advanced semi-Markov models derive staying automatically, so that specific
    row is safely omitted. A residual transition to another state is ambiguous
    and is therefore rejected rather than reinterpreted.
    """

    normalized: list[dict[str, Any]] = []
    for index, original in enumerate(rows, start=1):
        row = dict(original)
        if _blank(row.get("strategy_id")):
            continue
        origin = _text(row.get("origin_state"), f"Transition row {index}: origin_state")
        destination = _text(
            row.get("destination_state"), f"Transition row {index}: destination_state"
        )
        probability_mode = (_optional_text(row.get("probability_mode")) or "direct").lower()
        if probability_mode == "residual":
            if origin == destination:
                continue
            raise BuilderValidationError(
                "Advanced Markov derives the probability of remaining in an origin state automatically. "
                "A residual transition to a different state cannot be translated safely; replace it with an explicit probability/rate definition."
            )

        input_type = (_optional_text(row.get("input_type")) or "probability").lower()
        parameter_id = (
            _optional_text(row.get("parameter_id"))
            or _optional_text(row.get("probability_parameter_id"))
        )
        time_basis = (_optional_text(row.get("time_basis")) or "model_time").lower()
        start_time = 0.0 if _blank(row.get("start_time")) else _float(
            row.get("start_time"), f"Transition row {index}: start_time"
        )
        source_interval = _optional_float(
            row.get("source_interval_years", row.get("probability_interval_years"))
        )
        normalized.append(
            {
                **row,
                "strategy_id": _text(row.get("strategy_id"), f"Transition row {index}: strategy_id"),
                "origin_state": origin,
                "destination_state": destination,
                "input_type": input_type,
                "probability_mode": probability_mode,
                "time_basis": time_basis,
                "start_time": start_time,
                "end_time": _optional_float(row.get("end_time")),
                "parameter_id": parameter_id,
                "source_interval_years": source_interval,
            }
        )
    return normalized


def _compile_transition_groups(rows: Sequence[Mapping[str, Any]]):
    groups: dict[
        tuple[str, str, str, str, str, str],
        dict[str, Any],
    ] = {}
    for index, row in enumerate(normalize_semi_markov_transition_rows(rows), start=1):
        strategy_id = _text(row.get("strategy_id"), f"Transition row {index}: strategy_id")
        origin = _text(row.get("origin_state"), f"Transition row {index}: origin_state")
        destination = _text(row.get("destination_state"), f"Transition row {index}: destination_state")
        input_type = _text(row.get("input_type"), f"Transition row {index}: input_type").lower()
        probability_mode = str(row.get("probability_mode") or "direct").strip().lower()
        time_basis = _text(row.get("time_basis"), f"Transition row {index}: time_basis").lower()
        parameter_id = _text(row.get("parameter_id"), f"Transition row {index}: parameter_id")
        start = _float(row.get("start_time"), f"Transition row {index}: start_time")
        end = _optional_float(row.get("end_time"))
        source_interval = _optional_float(row.get("source_interval_years"))
        key = (
            strategy_id,
            origin,
            destination,
            input_type,
            probability_mode,
            time_basis,
        )
        group = groups.setdefault(key, {"bands": [], "source_interval_years": source_interval})
        if group["source_interval_years"] != source_interval:
            raise BuilderValidationError(
                f"Transition {origin} -> {destination} uses different source probability intervals across time bands. "
                "Use one source interval for all bands of the same converted transition."
            )
        try:
            group["bands"].append(ParameterBand(start, parameter_id, end))
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc
    return groups


def compile_dynamic_transitions(rows: Sequence[Mapping[str, Any]]):
    by_strategy: dict[str, list[DynamicTransition]] = {}
    for key, group in _compile_transition_groups(rows).items():
        strategy_id, origin, destination, input_type, probability_mode, time_basis = key
        bands = sorted(group["bands"], key=lambda band: band.start)
        try:
            schedule = PiecewiseParameterSchedule(time_basis, tuple(bands))
            transition = DynamicTransition(
                origin_state=origin,
                destination_state=destination,
                schedule=schedule,
                input_type=input_type,
                probability_mode=probability_mode,
                source_interval_years=group["source_interval_years"],
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc
        by_strategy.setdefault(strategy_id, []).append(transition)
    return {key: tuple(values) for key, values in by_strategy.items()}


def compile_initial(rows: Sequence[Mapping[str, Any]]):
    by_strategy: dict[str, list[InitialStateAllocation]] = {}
    for index, row in enumerate(rows, start=1):
        if not str(row.get("strategy_id") or "").strip():
            continue
        strategy_id = _text(row.get("strategy_id"), f"Initial row {index}: strategy_id")
        state_id = _text(row.get("state_id"), f"Initial row {index}: state_id")
        mode = (
            _optional_text(row.get("proportion_mode"))
            or _optional_text(row.get("allocation_mode"))
            or "fixed"
        ).lower()
        parameter_id = (
            _optional_text(row.get("proportion_parameter_id"))
            or _optional_text(row.get("parameter_id"))
        )
        proportion = None if mode != "fixed" else _float(
            row.get("proportion"), f"Initial row {index}: proportion"
        )
        try:
            allocation = InitialStateAllocation(
                state_id=state_id,
                proportion=proportion,
                proportion_parameter_id=parameter_id,
                proportion_mode=mode,
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc
        by_strategy.setdefault(strategy_id, []).append(allocation)
    return {key: tuple(values) for key, values in by_strategy.items()}


def compile_state_rewards(rows: Sequence[Mapping[str, Any]]):
    by_strategy: dict[str, list[StateReward]] = {}
    for index, row in enumerate(rows, start=1):
        if not str(row.get("strategy_id") or "").strip():
            continue
        strategy_id = _text(row.get("strategy_id"), f"State reward row {index}: strategy_id")
        try:
            reward = StateReward(
                state_id=_text(row.get("state_id"), f"State reward row {index}: state_id"),
                parameter_id=_text(row.get("parameter_id"), f"State reward row {index}: parameter_id"),
                reward_type=_text(row.get("reward_type"), f"State reward row {index}: reward_type").lower(),
                accrual=str(row.get("accrual") or "per_cycle").strip().lower(),
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc
        by_strategy.setdefault(strategy_id, []).append(reward)
    return {key: tuple(values) for key, values in by_strategy.items()}


def compile_transition_rewards(rows: Sequence[Mapping[str, Any]]):
    by_strategy: dict[str, list[TransitionReward]] = {}
    for index, row in enumerate(rows, start=1):
        if not str(row.get("strategy_id") or "").strip():
            continue
        strategy_id = _text(row.get("strategy_id"), f"Transition reward row {index}: strategy_id")
        try:
            reward = TransitionReward(
                origin_state=_text(row.get("origin_state"), f"Transition reward row {index}: origin_state"),
                destination_state=_text(row.get("destination_state"), f"Transition reward row {index}: destination_state"),
                parameter_id=_text(row.get("parameter_id"), f"Transition reward row {index}: parameter_id"),
                reward_type=_text(row.get("reward_type"), f"Transition reward row {index}: reward_type").lower(),
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc
        by_strategy.setdefault(strategy_id, []).append(reward)
    return {key: tuple(values) for key, values in by_strategy.items()}


def compile_mortality_rules(
    rows: Sequence[Mapping[str, Any]],
    mortality_table: AgeSpecificMortalityTable | None,
):
    rules: dict[str, BackgroundMortalityRule] = {}
    for index, row in enumerate(rows, start=1):
        if not str(row.get("strategy_id") or "").strip():
            continue
        if mortality_table is None:
            raise BuilderValidationError(
                "Background mortality is configured but the mortality table is empty."
            )
        strategy_id = _text(row.get("strategy_id"), f"Mortality rule row {index}: strategy_id")
        if strategy_id in rules:
            raise BuilderValidationError("Each strategy may define at most one background mortality rule.")
        applicable = parse_id_list(row.get("applicable_states"))
        smr_parameter_id = str(row.get("smr_parameter_id") or "").strip() or None
        try:
            rules[strategy_id] = BackgroundMortalityRule(
                destination_state=_text(row.get("destination_state"), f"Mortality rule row {index}: destination_state"),
                mortality_table=mortality_table,
                initial_age=_float(row.get("initial_age"), f"Mortality rule row {index}: initial_age"),
                applicable_states=applicable,
                smr_parameter_id=smr_parameter_id,
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc
    return rules


def compile_semi_markov_tables(
    parameter_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    strategy_rows: Sequence[Mapping[str, Any]],
    initial_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    state_reward_rows: Sequence[Mapping[str, Any]],
    transition_reward_rows: Sequence[Mapping[str, Any]],
    mortality_table_rows: Sequence[Mapping[str, Any]],
    mortality_rule_rows: Sequence[Mapping[str, Any]],
    *,
    cycle_length_years: float,
    max_cycles: int,
    state_accrual_timing: str = "half_cycle",
    transition_reward_timing: str = "mid_cycle",
    termination_mode: str = "fixed_cycles",
    depletion_threshold: float = 1e-6,
) -> CompiledSemiMarkovModel:
    parameters = compile_parameter_rows(parameter_rows)
    states = compile_states(state_rows)
    transitions = compile_dynamic_transitions(transition_rows)
    initial = compile_initial(initial_rows)
    state_rewards = compile_state_rewards(state_reward_rows)
    transition_rewards = compile_transition_rewards(transition_reward_rows)
    mortality_table = compile_mortality_table(mortality_table_rows)
    mortality_rules = compile_mortality_rules(mortality_rule_rows, mortality_table)

    strategies: list[SemiMarkovStrategyDefinition] = []
    for index, row in enumerate(strategy_rows, start=1):
        if not str(row.get("strategy_id") or "").strip():
            continue
        strategy_id = _text(row.get("strategy_id"), f"Strategy row {index}: strategy_id")
        label = _text(row.get("strategy_name"), f"Strategy row {index}: strategy_name")
        try:
            strategies.append(
                SemiMarkovStrategyDefinition(
                    strategy_id=strategy_id,
                    label=label,
                    initial_distribution=initial.get(strategy_id, ()),
                    transitions=transitions.get(strategy_id, ()),
                    state_rewards=state_rewards.get(strategy_id, ()),
                    transition_rewards=transition_rewards.get(strategy_id, ()),
                    background_mortality=mortality_rules.get(strategy_id),
                )
            )
        except ValueError as exc:
            raise BuilderValidationError(str(exc)) from exc

    if len(strategies) < 2:
        raise BuilderValidationError("At least two strategies are required.")
    ids = [strategy.strategy_id for strategy in strategies]
    if len(ids) != len(set(ids)):
        raise BuilderValidationError("Strategy ids must be unique.")

    try:
        model = SemiMarkovDefinition(
            states=states,
            strategies=tuple(strategies),
            cycle_length_years=float(cycle_length_years),
            max_cycles=int(max_cycles),
            state_accrual_timing=state_accrual_timing,
            transition_reward_timing=transition_reward_timing,
            termination_mode=termination_mode,
            depletion_threshold=float(depletion_threshold),
        )
    except ValueError as exc:
        raise BuilderValidationError(str(exc)) from exc
    return CompiledSemiMarkovModel(model=model, parameters=parameters)
