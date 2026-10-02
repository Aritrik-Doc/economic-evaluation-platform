"""Sensitivity-analysis specifications and model-agnostic helpers.

The helpers operate on callbacks so they can later drive decision-tree, Markov,
partitioned-survival or other model engines without duplicating methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping


MetricEvaluator = Callable[[Mapping[str, float]], float]


@dataclass(frozen=True)
class OneWaySensitivitySpec:
    parameter_id: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.parameter_id.strip() or not self.values:
            raise ValueError("One-way sensitivity analysis requires a parameter and values.")
        if any(not isfinite(v) for v in self.values):
            raise ValueError("Sensitivity-analysis values must be finite.")


@dataclass(frozen=True)
class TwoWaySensitivitySpec:
    parameter_x: str
    x_values: tuple[float, ...]
    parameter_y: str
    y_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.parameter_x.strip() or not self.parameter_y.strip():
            raise ValueError("Two-way sensitivity parameters are mandatory.")
        if self.parameter_x == self.parameter_y:
            raise ValueError("Two-way sensitivity analysis requires two different parameters.")
        if not self.x_values or not self.y_values:
            raise ValueError("Two-way sensitivity analysis requires values for both parameters.")
        if any(not isfinite(v) for v in self.x_values + self.y_values):
            raise ValueError("Sensitivity-analysis values must be finite.")


@dataclass(frozen=True)
class ThresholdAnalysisSpec:
    parameter_id: str
    lower: float
    upper: float
    target_metric: float = 0.0
    tolerance: float = 1e-6
    max_iterations: int = 100

    def __post_init__(self) -> None:
        if not self.parameter_id.strip():
            raise ValueError("Threshold analysis requires a parameter id.")
        if not all(isfinite(v) for v in (self.lower, self.upper, self.target_metric, self.tolerance)):
            raise ValueError("Threshold-analysis values must be finite.")
        if self.lower >= self.upper:
            raise ValueError("Threshold-analysis lower bound must be below upper bound.")
        if self.tolerance <= 0 or self.max_iterations < 1:
            raise ValueError("Threshold-analysis tolerance and iterations must be positive.")


@dataclass(frozen=True)
class ScenarioAnalysisSpec:
    name: str
    description: str
    parameter_overrides: tuple[tuple[str, float], ...] = ()
    structural_changes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("Scenario name and description are mandatory.")


@dataclass(frozen=True)
class ProbabilisticSensitivitySpec:
    iterations: int
    seed: int

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("PSA iterations must be positive.")


def one_way_sensitivity(spec: OneWaySensitivitySpec, evaluate: MetricEvaluator) -> tuple[tuple[float, float], ...]:
    return tuple((value, evaluate({spec.parameter_id: value})) for value in spec.values)


def two_way_sensitivity(
    spec: TwoWaySensitivitySpec,
    evaluate: MetricEvaluator,
) -> tuple[tuple[float, float, float], ...]:
    """Evaluate a decision metric across the full Cartesian grid of two parameters."""
    return tuple(
        (x, y, evaluate({spec.parameter_x: x, spec.parameter_y: y}))
        for x in spec.x_values
        for y in spec.y_values
    )


def threshold_analysis(spec: ThresholdAnalysisSpec, evaluate: MetricEvaluator) -> float:
    """Find a parameter switching value using bisection.

    ``evaluate`` should return a continuous decision metric such as incremental
    net monetary benefit. The switching value is where the metric reaches
    ``target_metric``. The supplied bounds must bracket the target.
    """
    low, high = spec.lower, spec.upper

    def adjusted(value: float) -> float:
        metric = evaluate({spec.parameter_id: value})
        if not isfinite(metric):
            raise ValueError("Threshold evaluator returned a non-finite value.")
        return metric - spec.target_metric

    f_low = adjusted(low)
    f_high = adjusted(high)
    if abs(f_low) <= spec.tolerance:
        return low
    if abs(f_high) <= spec.tolerance:
        return high
    if f_low * f_high > 0:
        raise ValueError("Threshold-analysis bounds do not bracket a switching value.")

    for _ in range(spec.max_iterations):
        mid = (low + high) / 2
        f_mid = adjusted(mid)
        if abs(f_mid) <= spec.tolerance or (high - low) / 2 <= spec.tolerance:
            return mid
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

    return (low + high) / 2
