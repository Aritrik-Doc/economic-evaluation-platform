"""Time-dependent transition and mortality utilities for cohort models.

The module keeps three concepts distinct:

* probabilities over an explicit interval;
* continuous-time transition rates / hazards;
* piecewise schedules indexed by model time or time since state entry.

For multiple competing exits from one state, rates are converted jointly rather
than converting each cause independently. For a complete continuous-time Markov
chain, ``generator_to_transition_matrix`` converts an infinitesimal generator Q
into an interval transition matrix using uniformization, avoiding a SciPy
runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, floor, isfinite, log
from typing import Literal, Mapping

import numpy as np


TimeBasis = Literal["model_time", "state_time"]


class TransitionConversionError(ValueError):
    pass


def _positive_duration(duration: float) -> float:
    if not isfinite(duration) or duration <= 0:
        raise TransitionConversionError("Time interval must be positive and finite.")
    return float(duration)


def rate_to_probability(rate: float, duration: float = 1.0) -> float:
    """Convert a constant event rate/hazard to an interval probability."""

    duration = _positive_duration(duration)
    if not isfinite(rate) or rate < 0:
        raise TransitionConversionError("Rate must be non-negative and finite.")
    return 1.0 - exp(-float(rate) * duration)


def probability_to_rate(probability: float, duration: float = 1.0) -> float:
    """Convert an interval probability to a constant rate/hazard assumption."""

    duration = _positive_duration(duration)
    if not isfinite(probability) or probability < 0 or probability >= 1:
        raise TransitionConversionError(
            "Probability must be finite and in [0, 1) to convert to a finite rate."
        )
    if probability == 0:
        return 0.0
    return -log(1.0 - float(probability)) / duration


def rescale_probability(
    probability: float,
    *,
    from_duration: float,
    to_duration: float,
) -> float:
    """Rescale a probability assuming a constant underlying hazard."""

    return rate_to_probability(
        probability_to_rate(probability, from_duration),
        to_duration,
    )


@dataclass(frozen=True)
class CompetingRiskProbabilities:
    destination_probabilities: dict[str, float]
    stay_probability: float

    @property
    def total_probability(self) -> float:
        return self.stay_probability + sum(self.destination_probabilities.values())


def competing_rates_to_probabilities(
    rates: Mapping[str, float],
    duration: float = 1.0,
) -> CompetingRiskProbabilities:
    """Jointly convert constant competing cause-specific rates over one interval.

    If H is the sum of cause-specific rates, the probability of any event is
    ``1-exp(-H*t)`` and cause j receives the share ``h_j/H``. This guarantees
    that competing exits plus remaining in the origin state sum to one.
    """

    duration = _positive_duration(duration)
    if not rates:
        raise TransitionConversionError("At least one competing rate is required.")
    checked: dict[str, float] = {}
    for destination, value in rates.items():
        if not str(destination).strip():
            raise TransitionConversionError("Competing-risk destination cannot be blank.")
        if not isfinite(value) or value < 0:
            raise TransitionConversionError("Competing rates must be non-negative and finite.")
        checked[str(destination)] = float(value)

    total_rate = sum(checked.values())
    if total_rate == 0:
        return CompetingRiskProbabilities(
            destination_probabilities={key: 0.0 for key in checked},
            stay_probability=1.0,
        )

    stay = exp(-total_rate * duration)
    event_probability = 1.0 - stay
    destination_probabilities = {
        destination: (rate / total_rate) * event_probability
        for destination, rate in checked.items()
    }
    return CompetingRiskProbabilities(destination_probabilities, stay)


def validate_generator_matrix(generator, tolerance: float = 1e-10) -> np.ndarray:
    """Validate and return a continuous-time Markov generator matrix Q."""

    matrix = np.asarray(generator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise TransitionConversionError("Generator matrix must be square with at least two states.")
    if not np.all(np.isfinite(matrix)):
        raise TransitionConversionError("Generator matrix entries must be finite.")
    if not isfinite(tolerance) or tolerance <= 0:
        raise TransitionConversionError("Tolerance must be positive and finite.")

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if np.any(off_diagonal < -tolerance):
        raise TransitionConversionError("Generator off-diagonal rates cannot be negative.")
    if np.any(np.diag(matrix) > tolerance):
        raise TransitionConversionError("Generator diagonal entries cannot be positive.")
    if not np.allclose(matrix.sum(axis=1), 0.0, rtol=0.0, atol=tolerance):
        raise TransitionConversionError("Every generator row must sum to zero.")
    return matrix


def _uniformization_step(generator: np.ndarray, duration: float, tolerance: float) -> np.ndarray:
    """Compute exp(Q*t) for a moderate Poisson mean by uniformization."""

    size = generator.shape[0]
    identity = np.eye(size)
    uniform_rate = float(np.max(-np.diag(generator)))
    if uniform_rate == 0:
        return identity

    embedded = identity + generator / uniform_rate
    embedded[np.abs(embedded) < tolerance] = 0.0
    if np.any(embedded < -tolerance) or not np.allclose(
        embedded.sum(axis=1), 1.0, rtol=0.0, atol=1e-8
    ):
        raise TransitionConversionError("Generator could not be uniformized into a stochastic matrix.")

    mean = uniform_rate * duration
    poisson_weight = exp(-mean)
    power = identity.copy()
    result = poisson_weight * power

    # mean is deliberately bounded by the caller; 10,000 terms is therefore
    # an extremely conservative safety limit.
    for k in range(1, 10_001):
        power = power @ embedded
        poisson_weight *= mean / k
        addition = poisson_weight * power
        result += addition
        if poisson_weight < tolerance and np.max(np.abs(addition)) < tolerance:
            break
    else:
        raise TransitionConversionError("Generator exponential did not converge.")

    return result


def generator_to_transition_matrix(
    generator,
    duration: float = 1.0,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Convert a CTMC generator Q to P(t)=exp(Q*t).

    Uniformization is numerically stable for stochastic generators. Long
    intervals are split so the Poisson mean in each sub-step remains moderate,
    then the sub-step transition matrices are multiplied.
    """

    duration = _positive_duration(duration)
    matrix = validate_generator_matrix(generator)
    uniform_rate = float(np.max(-np.diag(matrix)))
    if uniform_rate == 0:
        return np.eye(matrix.shape[0])

    subdivisions = max(1, int(ceil(uniform_rate * duration / 50.0)))
    step_duration = duration / subdivisions
    step = _uniformization_step(matrix, step_duration, tolerance)
    result = np.eye(matrix.shape[0])
    for _ in range(subdivisions):
        result = result @ step

    # Remove floating-point noise while preserving stochastic rows.
    result[np.abs(result) < 1e-14] = 0.0
    if np.any(result < -1e-10):
        raise TransitionConversionError("Converted transition matrix contains negative probabilities.")
    result = np.maximum(result, 0.0)
    row_sums = result.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise TransitionConversionError("Converted transition matrix has an invalid row sum.")
    result = result / row_sums
    return result


@dataclass(frozen=True)
class ParameterBand:
    """Piecewise-constant parameter selection on an interval [start, end)."""

    start: float
    parameter_id: str
    end: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.start) or self.start < 0:
            raise ValueError("Schedule band start must be finite and non-negative.")
        if self.end is not None and (not isfinite(self.end) or self.end <= self.start):
            raise ValueError("Schedule band end must be greater than start.")
        if not self.parameter_id.strip():
            raise ValueError("Schedule band parameter id cannot be blank.")

    def contains(self, time_value: float) -> bool:
        return time_value >= self.start and (self.end is None or time_value < self.end)


@dataclass(frozen=True)
class PiecewiseParameterSchedule:
    """Select a parameter by model time or time since entry to a state."""

    basis: TimeBasis
    bands: tuple[ParameterBand, ...]

    def __post_init__(self) -> None:
        if self.basis not in {"model_time", "state_time"}:
            raise ValueError("Schedule basis must be model_time or state_time.")
        if not self.bands:
            raise ValueError("A time-varying schedule requires at least one band.")
        ordered = sorted(self.bands, key=lambda band: band.start)
        if tuple(ordered) != self.bands:
            raise ValueError("Schedule bands must be ordered by start time.")
        for previous, current in zip(self.bands, self.bands[1:]):
            if previous.end is None or previous.end > current.start:
                raise ValueError("Schedule bands cannot overlap or follow an open-ended band.")

    def parameter_id(self, *, model_time: float, state_time: float) -> str:
        value = model_time if self.basis == "model_time" else state_time
        if not isfinite(value) or value < 0:
            raise TransitionConversionError("Schedule lookup time must be finite and non-negative.")
        for band in self.bands:
            if band.contains(value):
                return band.parameter_id
        raise TransitionConversionError(
            f"No schedule band covers {self.basis.replace('_', ' ')} {value:g}."
        )


@dataclass(frozen=True)
class AgeSpecificMortalityTable:
    """Annual mortality probabilities by integer attained age.

    Within each age year, mortality is represented by a constant force. A cycle
    that crosses birthdays is integrated piecewise. An optional SMR multiplies
    the mortality *rate*, not the probability.
    """

    annual_probabilities: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not self.annual_probabilities:
            raise ValueError("Mortality table cannot be empty.")
        ages = [age for age, _ in self.annual_probabilities]
        if ages != sorted(ages) or len(ages) != len(set(ages)):
            raise ValueError("Mortality ages must be unique and increasing.")
        if any(next_age != age + 1 for age, next_age in zip(ages, ages[1:])):
            raise ValueError("Mortality table ages must be consecutive integers.")
        for age, probability in self.annual_probabilities:
            if age < 0:
                raise ValueError("Mortality ages cannot be negative.")
            if not isfinite(probability) or probability < 0 or probability >= 1:
                raise ValueError("Annual mortality probabilities must lie in [0, 1).")

    @property
    def minimum_age(self) -> int:
        return self.annual_probabilities[0][0]

    @property
    def maximum_age(self) -> int:
        return self.annual_probabilities[-1][0]

    def _annual_probability(self, integer_age: int) -> float:
        if integer_age < self.minimum_age or integer_age > self.maximum_age:
            raise TransitionConversionError(
                f"Mortality table does not cover attained age {integer_age}."
            )
        return self.annual_probabilities[integer_age - self.minimum_age][1]

    def probability(
        self,
        *,
        age_start: float,
        duration_years: float,
        standardized_mortality_ratio: float = 1.0,
    ) -> float:
        duration_years = _positive_duration(duration_years)
        if not isfinite(age_start) or age_start < 0:
            raise TransitionConversionError("Starting age must be finite and non-negative.")
        if not isfinite(standardized_mortality_ratio) or standardized_mortality_ratio < 0:
            raise TransitionConversionError("Standardized mortality ratio must be non-negative and finite.")

        remaining = duration_years
        age = float(age_start)
        integrated_hazard = 0.0
        while remaining > 1e-15:
            integer_age = int(floor(age))
            annual_probability = self._annual_probability(integer_age)
            annual_rate = probability_to_rate(annual_probability, 1.0)
            time_to_birthday = (integer_age + 1.0) - age
            exposure = min(remaining, time_to_birthday)
            integrated_hazard += annual_rate * standardized_mortality_ratio * exposure
            age += exposure
            remaining -= exposure
        return 1.0 - exp(-integrated_hazard)
