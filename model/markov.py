"""Cohort state-transition (Markov) engine for health-economic evaluation.

Version 0.5 implements a closed-cohort, discrete-time, time-homogeneous model
with explicit health states, strategy-specific transition matrices, state and
transition rewards, discounting, optional trapezoidal (half-cycle) state
accrual, parameter overrides, computational cost perspective, and bounded
cohort-depletion termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Literal, Mapping, Sequence

import numpy as np

from model.schema import Parameter


ProbabilityMode = Literal["direct", "complement", "residual"]
RewardType = Literal["cost", "outcome"]
RewardAccrual = Literal["per_cycle", "per_year"]
StateAccrualTiming = Literal["start", "end", "half_cycle"]
TransitionRewardTiming = Literal["start", "mid_cycle", "end"]
TerminationMode = Literal["fixed_cycles", "cohort_depletion"]


class MarkovValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MarkovState:
    id: str
    label: str
    absorbing: bool = False

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.label.strip():
            raise ValueError("Health-state id and label are mandatory.")


@dataclass(frozen=True)
class InitialStateAllocation:
    state_id: str
    proportion: float

    def __post_init__(self) -> None:
        if not self.state_id.strip():
            raise ValueError("Initial-state allocation requires a state id.")
        if not isfinite(self.proportion) or self.proportion < 0 or self.proportion > 1:
            raise ValueError("Initial-state proportions must be finite values in [0, 1].")


@dataclass(frozen=True)
class TransitionProbability:
    origin_state: str
    destination_state: str
    probability_parameter_id: str | None = None
    probability_mode: ProbabilityMode = "direct"

    def __post_init__(self) -> None:
        if not self.origin_state.strip() or not self.destination_state.strip():
            raise ValueError("Transition origin and destination are mandatory.")
        if self.probability_mode not in {"direct", "complement", "residual"}:
            raise ValueError("Transition probability mode must be direct, complement or residual.")
        if self.probability_mode == "residual":
            if self.probability_parameter_id is not None:
                raise ValueError("Residual transitions must not reference a probability parameter.")
        elif not (self.probability_parameter_id or "").strip():
            raise ValueError("Direct/complement transitions require a probability parameter.")


@dataclass(frozen=True)
class StateReward:
    state_id: str
    parameter_id: str
    reward_type: RewardType
    accrual: RewardAccrual = "per_cycle"

    def __post_init__(self) -> None:
        if not self.state_id.strip() or not self.parameter_id.strip():
            raise ValueError("State reward requires state and parameter ids.")
        if self.reward_type not in {"cost", "outcome"}:
            raise ValueError("State reward type must be cost or outcome.")
        if self.accrual not in {"per_cycle", "per_year"}:
            raise ValueError("State reward accrual must be per_cycle or per_year.")


@dataclass(frozen=True)
class TransitionReward:
    origin_state: str
    destination_state: str
    parameter_id: str
    reward_type: RewardType

    def __post_init__(self) -> None:
        if not self.origin_state.strip() or not self.destination_state.strip() or not self.parameter_id.strip():
            raise ValueError("Transition reward requires origin, destination and parameter ids.")
        if self.reward_type not in {"cost", "outcome"}:
            raise ValueError("Transition reward type must be cost or outcome.")


@dataclass(frozen=True)
class MarkovStrategyDefinition:
    strategy_id: str
    label: str
    initial_distribution: tuple[InitialStateAllocation, ...]
    transitions: tuple[TransitionProbability, ...]
    state_rewards: tuple[StateReward, ...] = ()
    transition_rewards: tuple[TransitionReward, ...] = ()

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.label.strip():
            raise ValueError("Markov strategy id and label are mandatory.")
        if not self.initial_distribution:
            raise ValueError("Each Markov strategy requires an initial state distribution.")


@dataclass(frozen=True)
class CohortMarkovDefinition:
    states: tuple[MarkovState, ...]
    strategies: tuple[MarkovStrategyDefinition, ...]
    cycle_length_years: float
    max_cycles: int
    state_accrual_timing: StateAccrualTiming = "half_cycle"
    transition_reward_timing: TransitionRewardTiming = "mid_cycle"
    termination_mode: TerminationMode = "fixed_cycles"
    depletion_threshold: float = 1e-6
    probability_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if len(self.states) < 2:
            raise ValueError("A cohort state-transition model requires at least two health states.")
        if len(self.strategies) < 2:
            raise ValueError("A cohort state-transition model requires at least two strategies.")
        if not isfinite(self.cycle_length_years) or self.cycle_length_years <= 0:
            raise ValueError("Cycle length must be a positive finite number of years.")
        if not isinstance(self.max_cycles, int) or self.max_cycles < 1:
            raise ValueError("Maximum cycles must be a positive integer.")
        if self.state_accrual_timing not in {"start", "end", "half_cycle"}:
            raise ValueError("State accrual timing must be start, end or half_cycle.")
        if self.transition_reward_timing not in {"start", "mid_cycle", "end"}:
            raise ValueError("Transition reward timing must be start, mid_cycle or end.")
        if self.termination_mode not in {"fixed_cycles", "cohort_depletion"}:
            raise ValueError("Termination mode must be fixed_cycles or cohort_depletion.")
        if not isfinite(self.depletion_threshold) or self.depletion_threshold < 0 or self.depletion_threshold >= 1:
            raise ValueError("Cohort-depletion threshold must be a finite proportion in [0, 1).")
        if not isfinite(self.probability_tolerance) or self.probability_tolerance <= 0:
            raise ValueError("Probability tolerance must be positive and finite.")


@dataclass(frozen=True)
class MarkovStrategyResult:
    strategy_id: str
    label: str
    expected_cost: float
    expected_outcome: float
    state_ids: tuple[str, ...]
    trace: tuple[tuple[float, ...], ...]
    cycle_costs: tuple[float, ...]
    cycle_outcomes: tuple[float, ...]
    cycles_run: int
    stopped_early: bool


@dataclass(frozen=True)
class CohortMarkovRunResult:
    strategies: tuple[MarkovStrategyResult, ...]


class _ParameterResolver:
    def __init__(
        self,
        parameters: Sequence[Parameter],
        overrides: Mapping[str, float] | None = None,
        included_cost_bearers: Sequence[str] | None = None,
    ) -> None:
        self.parameters = {parameter.id: parameter for parameter in parameters}
        if len(self.parameters) != len(tuple(parameters)):
            raise MarkovValidationError("Parameter ids must be unique.")
        self.overrides = dict(overrides or {})
        unknown = set(self.overrides) - set(self.parameters)
        if unknown:
            raise MarkovValidationError(
                "Unknown parameter override(s): " + ", ".join(sorted(unknown)) + "."
            )
        if any(not isfinite(value) for value in self.overrides.values()):
            raise MarkovValidationError("Parameter overrides must be finite.")
        self.included_cost_bearers = (
            set(included_cost_bearers) if included_cost_bearers is not None else None
        )

    def parameter(self, parameter_id: str) -> Parameter:
        try:
            return self.parameters[parameter_id]
        except KeyError as exc:
            raise MarkovValidationError(
                f"State-transition model references undefined parameter '{parameter_id}'."
            ) from exc

    def value(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        return float(self.overrides.get(parameter_id, parameter.value))

    def probability_parameter(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        if parameter.category == "cost":
            raise MarkovValidationError(
                f"Transition probability '{parameter_id}' cannot reference a cost parameter."
            )
        value = self.value(parameter_id)
        if value < 0 or value > 1:
            raise MarkovValidationError(
                f"Transition probability parameter '{parameter_id}' must lie between 0 and 1."
            )
        return value

    def cost(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        if parameter.category != "cost":
            raise MarkovValidationError(
                f"Cost reward '{parameter_id}' must reference a cost parameter."
            )
        if self.included_cost_bearers is not None:
            if not set(parameter.cost_bearers).intersection(self.included_cost_bearers):
                return 0.0
        return self.value(parameter_id)

    def outcome(self, parameter_id: str) -> float:
        parameter = self.parameter(parameter_id)
        if parameter.category == "cost":
            raise MarkovValidationError(
                f"Outcome reward '{parameter_id}' cannot reference a cost parameter."
            )
        return self.value(parameter_id)


def _validate_discount_rate(rate: float, label: str) -> None:
    if not isfinite(rate) or rate < 0 or rate >= 1:
        raise MarkovValidationError(
            f"{label} discount rate must be a finite proportion in [0, 1)."
        )


def _discount(value: float, annual_rate: float, time_years: float) -> float:
    return value / ((1.0 + annual_rate) ** time_years)


def _state_reward_time(model: CohortMarkovDefinition, cycle: int) -> float:
    start = cycle * model.cycle_length_years
    if model.state_accrual_timing == "start":
        return start
    if model.state_accrual_timing == "end":
        return start + model.cycle_length_years
    return start + 0.5 * model.cycle_length_years


def _transition_reward_time(model: CohortMarkovDefinition, cycle: int) -> float:
    start = cycle * model.cycle_length_years
    if model.transition_reward_timing == "start":
        return start
    if model.transition_reward_timing == "end":
        return start + model.cycle_length_years
    return start + 0.5 * model.cycle_length_years


def _initial_vector(
    strategy: MarkovStrategyDefinition,
    state_index: Mapping[str, int],
    tolerance: float,
) -> np.ndarray:
    vector = np.zeros(len(state_index), dtype=float)
    seen: set[str] = set()
    for allocation in strategy.initial_distribution:
        if allocation.state_id not in state_index:
            raise MarkovValidationError(
                f"Strategy '{strategy.strategy_id}' allocates cohort to undefined state '{allocation.state_id}'."
            )
        if allocation.state_id in seen:
            raise MarkovValidationError(
                f"Strategy '{strategy.strategy_id}' has duplicate initial allocation for state '{allocation.state_id}'."
            )
        seen.add(allocation.state_id)
        vector[state_index[allocation.state_id]] = allocation.proportion
    total = float(vector.sum())
    if not isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise MarkovValidationError(
            f"Initial distribution for strategy '{strategy.strategy_id}' sums to {total:.12g}, not 1."
        )
    return vector


def _transition_matrix(
    model: CohortMarkovDefinition,
    strategy: MarkovStrategyDefinition,
    resolver: _ParameterResolver,
    state_index: Mapping[str, int],
) -> np.ndarray:
    state_ids = tuple(state_index)
    matrix = np.zeros((len(state_ids), len(state_ids)), dtype=float)
    by_origin: dict[str, list[TransitionProbability]] = {state_id: [] for state_id in state_ids}
    pairs: set[tuple[str, str]] = set()

    for transition in strategy.transitions:
        if transition.origin_state not in state_index or transition.destination_state not in state_index:
            raise MarkovValidationError(
                f"Strategy '{strategy.strategy_id}' has a transition referencing an undefined state."
            )
        pair = (transition.origin_state, transition.destination_state)
        if pair in pairs:
            raise MarkovValidationError(
                f"Strategy '{strategy.strategy_id}' has duplicate transition {pair[0]} -> {pair[1]}."
            )
        pairs.add(pair)
        by_origin[transition.origin_state].append(transition)

    state_map = {state.id: state for state in model.states}

    for origin in state_ids:
        row_specs = by_origin[origin]
        origin_index = state_index[origin]
        state = state_map[origin]

        if not row_specs:
            if state.absorbing:
                matrix[origin_index, origin_index] = 1.0
                continue
            raise MarkovValidationError(
                f"Non-absorbing state '{origin}' has no outgoing transitions for strategy '{strategy.strategy_id}'."
            )

        residual_specs = [item for item in row_specs if item.probability_mode == "residual"]
        if len(residual_specs) > 1:
            raise MarkovValidationError(
                f"State '{origin}' may have at most one residual transition per strategy."
            )

        assigned = 0.0
        for transition in row_specs:
            if transition.probability_mode == "residual":
                continue
            assert transition.probability_parameter_id is not None
            base = resolver.probability_parameter(transition.probability_parameter_id)
            probability = base if transition.probability_mode == "direct" else 1.0 - base
            matrix[origin_index, state_index[transition.destination_state]] = probability
            assigned += probability

        if residual_specs:
            residual = 1.0 - assigned
            if residual < -model.probability_tolerance:
                raise MarkovValidationError(
                    f"Outgoing non-residual probabilities from state '{origin}' exceed 1 for strategy '{strategy.strategy_id}'."
                )
            if abs(residual) <= model.probability_tolerance:
                residual = 0.0
            transition = residual_specs[0]
            matrix[origin_index, state_index[transition.destination_state]] = residual

        row_sum = float(matrix[origin_index].sum())
        if not isclose(row_sum, 1.0, rel_tol=0.0, abs_tol=model.probability_tolerance):
            raise MarkovValidationError(
                f"Outgoing probabilities from state '{origin}' for strategy '{strategy.strategy_id}' sum to {row_sum:.12g}, not 1."
            )
        if np.any(matrix[origin_index] < -model.probability_tolerance):
            raise MarkovValidationError("Transition probabilities cannot be negative.")

        if state.absorbing:
            self_probability = matrix[origin_index, origin_index]
            others = row_sum - self_probability
            if not isclose(self_probability, 1.0, rel_tol=0.0, abs_tol=model.probability_tolerance) or not isclose(
                others, 0.0, rel_tol=0.0, abs_tol=model.probability_tolerance
            ):
                raise MarkovValidationError(
                    f"Absorbing state '{origin}' must transition to itself with probability 1."
                )

    return matrix


def validate_cohort_markov(
    model: CohortMarkovDefinition,
    parameters: Sequence[Parameter],
    *,
    overrides: Mapping[str, float] | None = None,
    cost_discount_rate: float = 0.0,
    outcome_discount_rate: float = 0.0,
) -> None:
    """Validate structure, parameter references and stochastic matrices."""

    _validate_discount_rate(cost_discount_rate, "Cost")
    _validate_discount_rate(outcome_discount_rate, "Outcome")
    resolver = _ParameterResolver(parameters, overrides)

    state_ids = [state.id for state in model.states]
    if len(state_ids) != len(set(state_ids)):
        raise MarkovValidationError("Health-state ids must be unique.")
    strategy_ids = [strategy.strategy_id for strategy in model.strategies]
    if len(strategy_ids) != len(set(strategy_ids)):
        raise MarkovValidationError("Strategy ids must be unique.")

    if model.termination_mode == "cohort_depletion" and not any(state.absorbing for state in model.states):
        raise MarkovValidationError(
            "Cohort-depletion termination requires at least one absorbing state."
        )

    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    for strategy in model.strategies:
        _initial_vector(strategy, state_index, model.probability_tolerance)
        matrix = _transition_matrix(model, strategy, resolver, state_index)
        structural_pairs = {
            (item.origin_state, item.destination_state) for item in strategy.transitions
        }
        for state in model.states:
            if state.absorbing and not any(item.origin_state == state.id for item in strategy.transitions):
                structural_pairs.add((state.id, state.id))

        for reward in strategy.state_rewards:
            if reward.state_id not in state_index:
                raise MarkovValidationError(
                    f"State reward references undefined state '{reward.state_id}'."
                )
            if reward.reward_type == "cost":
                resolver.cost(reward.parameter_id)
            else:
                resolver.outcome(reward.parameter_id)

        for reward in strategy.transition_rewards:
            pair = (reward.origin_state, reward.destination_state)
            if reward.origin_state not in state_index or reward.destination_state not in state_index:
                raise MarkovValidationError("Transition reward references an undefined state.")
            if pair not in structural_pairs:
                raise MarkovValidationError(
                    f"Transition reward references transition {pair[0]} -> {pair[1]} that is not defined for strategy '{strategy.strategy_id}'."
                )
            if reward.reward_type == "cost":
                resolver.cost(reward.parameter_id)
            else:
                resolver.outcome(reward.parameter_id)

        if np.any(matrix > 1.0 + model.probability_tolerance):
            raise MarkovValidationError("Transition probabilities cannot exceed 1.")


def _state_weights(
    model: CohortMarkovDefinition,
    start_occupancy: np.ndarray,
    end_occupancy: np.ndarray,
) -> np.ndarray:
    if model.state_accrual_timing == "start":
        return start_occupancy
    if model.state_accrual_timing == "end":
        return end_occupancy
    return 0.5 * (start_occupancy + end_occupancy)


def run_cohort_markov(
    model: CohortMarkovDefinition,
    parameters: Sequence[Parameter],
    *,
    overrides: Mapping[str, float] | None = None,
    included_cost_bearers: Sequence[str] | None = None,
    cost_discount_rate: float = 0.0,
    outcome_discount_rate: float = 0.0,
) -> CohortMarkovRunResult:
    """Run the closed cohort and return discounted per-person costs/outcomes."""

    validate_cohort_markov(
        model,
        parameters,
        overrides=overrides,
        cost_discount_rate=cost_discount_rate,
        outcome_discount_rate=outcome_discount_rate,
    )
    resolver = _ParameterResolver(parameters, overrides, included_cost_bearers)
    state_ids = tuple(state.id for state in model.states)
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    non_absorbing = np.asarray([not state.absorbing for state in model.states], dtype=bool)

    results: list[MarkovStrategyResult] = []
    for strategy in model.strategies:
        matrix = _transition_matrix(model, strategy, resolver, state_index)
        occupancy = _initial_vector(strategy, state_index, model.probability_tolerance)
        trace: list[tuple[float, ...]] = [tuple(float(value) for value in occupancy)]
        cycle_costs: list[float] = []
        cycle_outcomes: list[float] = []
        total_cost = 0.0
        total_outcome = 0.0
        stopped_early = False

        for cycle in range(model.max_cycles):
            end_occupancy = occupancy @ matrix
            if not isclose(float(end_occupancy.sum()), 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise MarkovValidationError(
                    f"Cohort mass is not conserved in cycle {cycle + 1} for strategy '{strategy.strategy_id}'."
                )

            weights = _state_weights(model, occupancy, end_occupancy)
            state_time = _state_reward_time(model, cycle)
            transition_time = _transition_reward_time(model, cycle)
            cycle_cost = 0.0
            cycle_outcome = 0.0

            for reward in strategy.state_rewards:
                scale = model.cycle_length_years if reward.accrual == "per_year" else 1.0
                exposure = float(weights[state_index[reward.state_id]]) * scale
                if reward.reward_type == "cost":
                    value = exposure * resolver.cost(reward.parameter_id)
                    cycle_cost += _discount(value, cost_discount_rate, state_time)
                else:
                    value = exposure * resolver.outcome(reward.parameter_id)
                    cycle_outcome += _discount(value, outcome_discount_rate, state_time)

            flows = occupancy[:, None] * matrix
            for reward in strategy.transition_rewards:
                flow = float(
                    flows[
                        state_index[reward.origin_state],
                        state_index[reward.destination_state],
                    ]
                )
                if reward.reward_type == "cost":
                    value = flow * resolver.cost(reward.parameter_id)
                    cycle_cost += _discount(value, cost_discount_rate, transition_time)
                else:
                    value = flow * resolver.outcome(reward.parameter_id)
                    cycle_outcome += _discount(value, outcome_discount_rate, transition_time)

            total_cost += cycle_cost
            total_outcome += cycle_outcome
            cycle_costs.append(cycle_cost)
            cycle_outcomes.append(cycle_outcome)
            occupancy = end_occupancy
            trace.append(tuple(float(value) for value in occupancy))

            if model.termination_mode == "cohort_depletion":
                remaining = float(occupancy[non_absorbing].sum())
                if remaining <= model.depletion_threshold:
                    stopped_early = True
                    break

        results.append(
            MarkovStrategyResult(
                strategy_id=strategy.strategy_id,
                label=strategy.label,
                expected_cost=total_cost,
                expected_outcome=total_outcome,
                state_ids=state_ids,
                trace=tuple(trace),
                cycle_costs=tuple(cycle_costs),
                cycle_outcomes=tuple(cycle_outcomes),
                cycles_run=len(cycle_costs),
                stopped_early=stopped_early,
            )
        )

    return CohortMarkovRunResult(tuple(results))
