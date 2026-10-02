"""Semi-Markov and time-varying cohort state-transition engine.

Unlike the v0.5 homogeneous cohort engine, this runner tracks how long cohort
mass has spent in each state. Transition schedules can depend on either model
(simulation) time or time since entry to the current state. This implements
state-time memory without forcing users to manually create dozens of tunnel
states.

Outgoing transitions from a state must use one coherent representation:
probabilities or cause-specific rates. Rate-based exits are converted jointly
as competing risks. Optional age-specific background mortality joins those
cause-specific rates; it is deliberately not combined heuristically with
probability-based exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Literal, Mapping, Sequence

import numpy as np

from model.markov import (
    InitialStateAllocation,
    MarkovState,
    StateAccrualTiming,
    StateReward,
    TerminationMode,
    TransitionReward,
    TransitionRewardTiming,
)
from model.schema import Parameter
from model.transition_dynamics import (
    AgeSpecificMortalityTable,
    PiecewiseParameterSchedule,
    competing_rates_to_probabilities,
    probability_to_rate,
)


TransitionInput = Literal["probability", "rate"]
ProbabilityMode = Literal["direct", "complement"]


class SemiMarkovValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DynamicTransition:
    origin_state: str
    destination_state: str
    schedule: PiecewiseParameterSchedule
    input_type: TransitionInput = "probability"
    probability_mode: ProbabilityMode = "direct"

    def __post_init__(self) -> None:
        if not self.origin_state.strip() or not self.destination_state.strip():
            raise ValueError("Dynamic transition origin and destination are mandatory.")
        if self.origin_state == self.destination_state:
            raise ValueError("Dynamic transitions describe exits only; staying is derived automatically.")
        if self.input_type not in {"probability", "rate"}:
            raise ValueError("Dynamic transition input_type must be probability or rate.")
        if self.probability_mode not in {"direct", "complement"}:
            raise ValueError("Probability mode must be direct or complement.")
        if self.input_type == "rate" and self.probability_mode != "direct":
            raise ValueError("Complement mode is not meaningful for rate inputs.")


@dataclass(frozen=True)
class BackgroundMortalityRule:
    destination_state: str
    mortality_table: AgeSpecificMortalityTable
    initial_age: float
    applicable_states: tuple[str, ...]
    smr_parameter_id: str | None = None

    def __post_init__(self) -> None:
        if not self.destination_state.strip():
            raise ValueError("Mortality destination state is mandatory.")
        if not isfinite(self.initial_age) or self.initial_age < 0:
            raise ValueError("Initial age must be finite and non-negative.")
        if not self.applicable_states:
            raise ValueError("Background mortality must identify at least one applicable state.")
        if len(set(self.applicable_states)) != len(self.applicable_states):
            raise ValueError("Background mortality applicable states must be unique.")


@dataclass(frozen=True)
class SemiMarkovStrategyDefinition:
    strategy_id: str
    label: str
    initial_distribution: tuple[InitialStateAllocation, ...]
    transitions: tuple[DynamicTransition, ...]
    state_rewards: tuple[StateReward, ...] = ()
    transition_rewards: tuple[TransitionReward, ...] = ()
    background_mortality: BackgroundMortalityRule | None = None

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.label.strip():
            raise ValueError("Semi-Markov strategy id and label are mandatory.")
        if not self.initial_distribution:
            raise ValueError("Each strategy requires an initial state distribution.")


@dataclass(frozen=True)
class SemiMarkovDefinition:
    states: tuple[MarkovState, ...]
    strategies: tuple[SemiMarkovStrategyDefinition, ...]
    cycle_length_years: float
    max_cycles: int
    state_accrual_timing: StateAccrualTiming = "half_cycle"
    transition_reward_timing: TransitionRewardTiming = "mid_cycle"
    termination_mode: TerminationMode = "fixed_cycles"
    depletion_threshold: float = 1e-6
    probability_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if len(self.states) < 2:
            raise ValueError("A semi-Markov model requires at least two states.")
        if len(self.strategies) < 2:
            raise ValueError("A semi-Markov model requires at least two strategies.")
        if not isfinite(self.cycle_length_years) or self.cycle_length_years <= 0:
            raise ValueError("Cycle length must be positive and finite.")
        if not isinstance(self.max_cycles, int) or self.max_cycles < 1:
            raise ValueError("Maximum cycles must be a positive integer.")
        if self.state_accrual_timing not in {"start", "end", "half_cycle"}:
            raise ValueError("State accrual timing must be start, end or half_cycle.")
        if self.transition_reward_timing not in {"start", "mid_cycle", "end"}:
            raise ValueError("Transition reward timing must be start, mid_cycle or end.")
        if self.termination_mode not in {"fixed_cycles", "cohort_depletion"}:
            raise ValueError("Termination mode must be fixed_cycles or cohort_depletion.")
        if not isfinite(self.depletion_threshold) or not 0 <= self.depletion_threshold < 1:
            raise ValueError("Depletion threshold must lie in [0, 1).")


@dataclass(frozen=True)
class SemiMarkovStrategyResult:
    strategy_id: str
    label: str
    expected_cost: float
    expected_outcome: float
    state_ids: tuple[str, ...]
    trace: tuple[tuple[float, ...], ...]
    cycle_costs: tuple[float, ...]
    cycle_outcomes: tuple[float, ...]
    mean_state_time_trace: tuple[tuple[float, ...], ...]
    cycles_run: int
    stopped_early: bool


@dataclass(frozen=True)
class SemiMarkovRunResult:
    strategies: tuple[SemiMarkovStrategyResult, ...]


class _Resolver:
    def __init__(
        self,
        parameters: Sequence[Parameter],
        overrides: Mapping[str, float] | None,
        included_cost_bearers: Sequence[str] | None,
    ) -> None:
        self.parameters = {parameter.id: parameter for parameter in parameters}
        if len(self.parameters) != len(tuple(parameters)):
            raise SemiMarkovValidationError("Parameter ids must be unique.")
        self.overrides = dict(overrides or {})
        unknown = set(self.overrides) - set(self.parameters)
        if unknown:
            raise SemiMarkovValidationError(
                "Unknown parameter override(s): " + ", ".join(sorted(unknown))
            )
        if any(not isfinite(value) for value in self.overrides.values()):
            raise SemiMarkovValidationError("Parameter overrides must be finite.")
        self.included_cost_bearers = (
            set(included_cost_bearers) if included_cost_bearers is not None else None
        )

    def parameter(self, parameter_id: str) -> Parameter:
        if parameter_id not in self.parameters:
            raise SemiMarkovValidationError(f"Undefined parameter '{parameter_id}'.")
        return self.parameters[parameter_id]

    def value(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        return float(self.overrides.get(parameter_id, parameter.value))

    def transition_value(self, parameter_id: str, input_type: TransitionInput) -> float:
        parameter = self.parameter(parameter_id)
        if parameter.category == "cost":
            raise SemiMarkovValidationError("Transition inputs cannot reference cost parameters.")
        value = self.value(parameter_id)
        if input_type == "probability" and not 0 <= value <= 1:
            raise SemiMarkovValidationError(
                f"Transition probability '{parameter_id}' must lie in [0, 1]."
            )
        if input_type == "rate" and value < 0:
            raise SemiMarkovValidationError(
                f"Transition rate '{parameter_id}' must be non-negative."
            )
        return value

    def cost(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        if parameter.category != "cost":
            raise SemiMarkovValidationError(
                f"Cost reward '{parameter_id}' must reference a cost parameter."
            )
        if self.included_cost_bearers is not None and not set(parameter.cost_bearers).intersection(
            self.included_cost_bearers
        ):
            return 0.0
        return self.value(parameter_id)

    def outcome(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        if parameter.category == "cost":
            raise SemiMarkovValidationError(
                f"Outcome reward '{parameter_id}' cannot reference a cost parameter."
            )
        return self.value(parameter_id)


def _discount(value: float, rate: float, time_years: float) -> float:
    if not isfinite(rate) or not 0 <= rate < 1:
        raise SemiMarkovValidationError("Discount rates must be finite proportions in [0, 1).")
    return value / ((1.0 + rate) ** time_years)


def _reward_time(model: SemiMarkovDefinition, cycle: int, *, transition: bool) -> float:
    start = cycle * model.cycle_length_years
    timing = model.transition_reward_timing if transition else model.state_accrual_timing
    if timing == "start":
        return start
    if timing == "end":
        return start + model.cycle_length_years
    return start + 0.5 * model.cycle_length_years


def _initial_tenure_matrix(
    strategy: SemiMarkovStrategyDefinition,
    state_index: Mapping[str, int],
    max_cycles: int,
    tolerance: float,
) -> np.ndarray:
    matrix = np.zeros((len(state_index), max_cycles + 1), dtype=float)
    seen: set[str] = set()
    for allocation in strategy.initial_distribution:
        if allocation.state_id not in state_index:
            raise SemiMarkovValidationError(
                f"Initial distribution references undefined state '{allocation.state_id}'."
            )
        if allocation.state_id in seen:
            raise SemiMarkovValidationError("Initial distribution contains duplicate state allocation.")
        seen.add(allocation.state_id)
        matrix[state_index[allocation.state_id], 0] = allocation.proportion
    total = float(matrix.sum())
    if not isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise SemiMarkovValidationError(
            f"Initial distribution for strategy '{strategy.strategy_id}' sums to {total:.12g}, not 1."
        )
    return matrix


def _transition_groups(strategy: SemiMarkovStrategyDefinition) -> dict[str, tuple[DynamicTransition, ...]]:
    grouped: dict[str, list[DynamicTransition]] = {}
    pairs: set[tuple[str, str]] = set()
    for transition in strategy.transitions:
        pair = (transition.origin_state, transition.destination_state)
        if pair in pairs:
            raise SemiMarkovValidationError(
                f"Duplicate dynamic transition {pair[0]} -> {pair[1]}."
            )
        pairs.add(pair)
        grouped.setdefault(transition.origin_state, []).append(transition)
    return {origin: tuple(values) for origin, values in grouped.items()}


def _validate_strategy(
    model: SemiMarkovDefinition,
    strategy: SemiMarkovStrategyDefinition,
    resolver: _Resolver,
) -> None:
    state_ids = {state.id for state in model.states}
    state_map = {state.id: state for state in model.states}
    groups = _transition_groups(strategy)

    for origin, transitions in groups.items():
        if origin not in state_ids:
            raise SemiMarkovValidationError(f"Undefined transition origin '{origin}'.")
        if state_map[origin].absorbing:
            raise SemiMarkovValidationError(f"Absorbing state '{origin}' cannot have exit transitions.")
        for transition in transitions:
            if transition.destination_state not in state_ids:
                raise SemiMarkovValidationError(
                    f"Undefined transition destination '{transition.destination_state}'."
                )
        input_types = {transition.input_type for transition in transitions}
        if len(input_types) != 1:
            raise SemiMarkovValidationError(
                f"All exits from state '{origin}' must use probabilities or rates consistently."
            )

    for state in model.states:
        if not state.absorbing and state.id not in groups:
            raise SemiMarkovValidationError(
                f"Non-absorbing state '{state.id}' requires at least one exit transition."
            )

    mortality = strategy.background_mortality
    if mortality is not None:
        if mortality.destination_state not in state_ids:
            raise SemiMarkovValidationError("Background mortality destination state is undefined.")
        if not state_map[mortality.destination_state].absorbing:
            raise SemiMarkovValidationError("Background mortality destination should be an absorbing state.")
        for state_id in mortality.applicable_states:
            if state_id not in state_ids:
                raise SemiMarkovValidationError(
                    f"Background mortality references undefined state '{state_id}'."
                )
            transitions = groups.get(state_id, ())
            if transitions and any(item.input_type != "rate" for item in transitions):
                raise SemiMarkovValidationError(
                    "Background mortality can only be combined automatically with rate-based exits. "
                    "For probability-based rows, represent mortality explicitly in the probability schedule."
                )
        if mortality.smr_parameter_id is not None:
            smr = resolver.value(mortality.smr_parameter_id)
            if smr < 0:
                raise SemiMarkovValidationError("Mortality SMR parameter must be non-negative.")

    for reward in strategy.state_rewards:
        if reward.state_id not in state_ids:
            raise SemiMarkovValidationError("State reward references an undefined state.")
        resolver.cost(reward.parameter_id) if reward.reward_type == "cost" else resolver.outcome(reward.parameter_id)

    structural_pairs = {
        (transition.origin_state, transition.destination_state)
        for transition in strategy.transitions
    }
    if mortality is not None:
        structural_pairs.update(
            (origin, mortality.destination_state) for origin in mortality.applicable_states
        )
    for reward in strategy.transition_rewards:
        if (reward.origin_state, reward.destination_state) not in structural_pairs:
            raise SemiMarkovValidationError(
                "Transition reward references a transition not represented in the dynamic model."
            )
        resolver.cost(reward.parameter_id) if reward.reward_type == "cost" else resolver.outcome(reward.parameter_id)


def validate_semi_markov(
    model: SemiMarkovDefinition,
    parameters: Sequence[Parameter],
    *,
    overrides: Mapping[str, float] | None = None,
    cost_discount_rate: float = 0.0,
    outcome_discount_rate: float = 0.0,
) -> None:
    _discount(0.0, cost_discount_rate, 0.0)
    _discount(0.0, outcome_discount_rate, 0.0)
    state_ids = [state.id for state in model.states]
    if len(state_ids) != len(set(state_ids)):
        raise SemiMarkovValidationError("State ids must be unique.")
    strategy_ids = [strategy.strategy_id for strategy in model.strategies]
    if len(strategy_ids) != len(set(strategy_ids)):
        raise SemiMarkovValidationError("Strategy ids must be unique.")
    if model.termination_mode == "cohort_depletion" and not any(state.absorbing for state in model.states):
        raise SemiMarkovValidationError("Cohort depletion requires at least one absorbing state.")

    resolver = _Resolver(parameters, overrides, None)
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    for strategy in model.strategies:
        _initial_tenure_matrix(
            strategy, state_index, model.max_cycles, model.probability_tolerance
        )
        _validate_strategy(model, strategy, resolver)


def _row_probabilities(
    *,
    model: SemiMarkovDefinition,
    strategy: SemiMarkovStrategyDefinition,
    transitions: tuple[DynamicTransition, ...],
    origin_state: str,
    model_time: float,
    state_time: float,
    resolver: _Resolver,
) -> dict[str, float]:
    input_type = transitions[0].input_type
    destinations: dict[str, float] = {}

    if input_type == "probability":
        total = 0.0
        for transition in transitions:
            parameter_id = transition.schedule.parameter_id(
                model_time=model_time,
                state_time=state_time,
            )
            base = resolver.transition_value(parameter_id, "probability")
            probability = base if transition.probability_mode == "direct" else 1.0 - base
            destinations[transition.destination_state] = probability
            total += probability
        if total > 1.0 + model.probability_tolerance:
            raise SemiMarkovValidationError(
                f"Exit probabilities from state '{origin_state}' sum to {total:.12g}, exceeding 1."
            )
        destinations[origin_state] = max(0.0, 1.0 - total)
        return destinations

    rates: dict[str, float] = {}
    for transition in transitions:
        parameter_id = transition.schedule.parameter_id(
            model_time=model_time,
            state_time=state_time,
        )
        rates[transition.destination_state] = resolver.transition_value(parameter_id, "rate")

    mortality = strategy.background_mortality
    if mortality is not None and origin_state in mortality.applicable_states:
        if mortality.destination_state in rates:
            raise SemiMarkovValidationError(
                "Background mortality destination duplicates an explicitly modelled rate transition."
            )
        smr = 1.0 if mortality.smr_parameter_id is None else resolver.value(mortality.smr_parameter_id)
        death_probability = mortality.mortality_table.probability(
            age_start=mortality.initial_age + model_time,
            duration_years=model.cycle_length_years,
            standardized_mortality_ratio=smr,
        )
        rates[mortality.destination_state] = probability_to_rate(
            death_probability,
            model.cycle_length_years,
        )

    converted = competing_rates_to_probabilities(rates, model.cycle_length_years)
    destinations.update(converted.destination_probabilities)
    destinations[origin_state] = converted.stay_probability
    return destinations


def _mean_state_time(tenure: np.ndarray, cycle_length: float) -> tuple[float, ...]:
    times = np.arange(tenure.shape[1], dtype=float) * cycle_length
    output: list[float] = []
    for row in tenure:
        mass = float(row.sum())
        output.append(0.0 if mass == 0 else float((row * times).sum() / mass))
    return tuple(output)


def run_semi_markov(
    model: SemiMarkovDefinition,
    parameters: Sequence[Parameter],
    *,
    overrides: Mapping[str, float] | None = None,
    included_cost_bearers: Sequence[str] | None = None,
    cost_discount_rate: float = 0.0,
    outcome_discount_rate: float = 0.0,
) -> SemiMarkovRunResult:
    validate_semi_markov(
        model,
        parameters,
        overrides=overrides,
        cost_discount_rate=cost_discount_rate,
        outcome_discount_rate=outcome_discount_rate,
    )
    resolver = _Resolver(parameters, overrides, included_cost_bearers)
    state_ids = tuple(state.id for state in model.states)
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    state_map = {state.id: state for state in model.states}
    non_absorbing = np.asarray([not state.absorbing for state in model.states], dtype=bool)
    results: list[SemiMarkovStrategyResult] = []

    for strategy in model.strategies:
        groups = _transition_groups(strategy)
        tenure = _initial_tenure_matrix(
            strategy, state_index, model.max_cycles, model.probability_tolerance
        )
        aggregated = tenure.sum(axis=1)
        trace: list[tuple[float, ...]] = [tuple(float(value) for value in aggregated)]
        mean_time_trace: list[tuple[float, ...]] = [
            _mean_state_time(tenure, model.cycle_length_years)
        ]
        cycle_costs: list[float] = []
        cycle_outcomes: list[float] = []
        total_cost = 0.0
        total_outcome = 0.0
        stopped_early = False

        for cycle in range(model.max_cycles):
            model_time = cycle * model.cycle_length_years
            next_tenure = np.zeros_like(tenure)
            flows: dict[tuple[str, str], float] = {}

            for origin_id in state_ids:
                origin_idx = state_index[origin_id]
                state = state_map[origin_id]
                for tenure_cycle in range(cycle + 1):
                    mass = float(tenure[origin_idx, tenure_cycle])
                    if mass == 0:
                        continue
                    if state.absorbing:
                        next_tenure[origin_idx, min(tenure_cycle + 1, model.max_cycles)] += mass
                        flows[(origin_id, origin_id)] = flows.get((origin_id, origin_id), 0.0) + mass
                        continue

                    transitions = groups[origin_id]
                    probabilities = _row_probabilities(
                        model=model,
                        strategy=strategy,
                        transitions=transitions,
                        origin_state=origin_id,
                        model_time=model_time,
                        state_time=tenure_cycle * model.cycle_length_years,
                        resolver=resolver,
                    )
                    row_total = sum(probabilities.values())
                    if not isclose(row_total, 1.0, rel_tol=0.0, abs_tol=model.probability_tolerance):
                        raise SemiMarkovValidationError(
                            f"Dynamic transition row for '{origin_id}' sums to {row_total:.12g}, not 1."
                        )
                    for destination_id, probability in probabilities.items():
                        flow = mass * probability
                        flows[(origin_id, destination_id)] = flows.get((origin_id, destination_id), 0.0) + flow
                        destination_idx = state_index[destination_id]
                        if destination_id == origin_id:
                            next_tenure[destination_idx, min(tenure_cycle + 1, model.max_cycles)] += flow
                        else:
                            next_tenure[destination_idx, 0] += flow

            start_occupancy = tenure.sum(axis=1)
            end_occupancy = next_tenure.sum(axis=1)
            if not isclose(float(end_occupancy.sum()), 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise SemiMarkovValidationError(
                    f"Cohort mass is not conserved in cycle {cycle + 1}."
                )
            if model.state_accrual_timing == "start":
                state_weights = start_occupancy
            elif model.state_accrual_timing == "end":
                state_weights = end_occupancy
            else:
                state_weights = 0.5 * (start_occupancy + end_occupancy)

            state_timepoint = _reward_time(model, cycle, transition=False)
            transition_timepoint = _reward_time(model, cycle, transition=True)
            cycle_cost = 0.0
            cycle_outcome = 0.0

            for reward in strategy.state_rewards:
                exposure = float(state_weights[state_index[reward.state_id]])
                if reward.accrual == "per_year":
                    exposure *= model.cycle_length_years
                if reward.reward_type == "cost":
                    cycle_cost += _discount(
                        exposure * resolver.cost(reward.parameter_id),
                        cost_discount_rate,
                        state_timepoint,
                    )
                else:
                    cycle_outcome += _discount(
                        exposure * resolver.outcome(reward.parameter_id),
                        outcome_discount_rate,
                        state_timepoint,
                    )

            for reward in strategy.transition_rewards:
                flow = flows.get((reward.origin_state, reward.destination_state), 0.0)
                if reward.reward_type == "cost":
                    cycle_cost += _discount(
                        flow * resolver.cost(reward.parameter_id),
                        cost_discount_rate,
                        transition_timepoint,
                    )
                else:
                    cycle_outcome += _discount(
                        flow * resolver.outcome(reward.parameter_id),
                        outcome_discount_rate,
                        transition_timepoint,
                    )

            total_cost += cycle_cost
            total_outcome += cycle_outcome
            cycle_costs.append(cycle_cost)
            cycle_outcomes.append(cycle_outcome)
            tenure = next_tenure
            trace.append(tuple(float(value) for value in end_occupancy))
            mean_time_trace.append(_mean_state_time(tenure, model.cycle_length_years))

            if model.termination_mode == "cohort_depletion":
                remaining = float(end_occupancy[non_absorbing].sum())
                if remaining <= model.depletion_threshold:
                    stopped_early = True
                    break

        results.append(
            SemiMarkovStrategyResult(
                strategy_id=strategy.strategy_id,
                label=strategy.label,
                expected_cost=total_cost,
                expected_outcome=total_outcome,
                state_ids=state_ids,
                trace=tuple(trace),
                cycle_costs=tuple(cycle_costs),
                cycle_outcomes=tuple(cycle_outcomes),
                mean_state_time_trace=tuple(mean_time_trace),
                cycles_run=len(cycle_costs),
                stopped_early=stopped_early,
            )
        )

    return SemiMarkovRunResult(tuple(results))
