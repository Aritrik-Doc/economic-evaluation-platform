"""Small reproducibility guards shared by sensitivity and PSA result pages."""

from __future__ import annotations

from math import isclose, isfinite
from typing import Sequence

import numpy as np


class AnalysisConsistencyError(ValueError):
    pass


def pairwise_inmb_from_strategy_results(
    strategies: Sequence[object],
    *,
    intervention_id: str,
    comparator_id: str,
    willingness_to_pay: float,
) -> float:
    if not isfinite(willingness_to_pay) or willingness_to_pay < 0:
        raise AnalysisConsistencyError("Decision threshold must be finite and non-negative.")
    rows = {str(getattr(row, "strategy_id")): row for row in strategies}
    if intervention_id == comparator_id:
        raise AnalysisConsistencyError("Intervention and comparator must differ.")
    if intervention_id not in rows or comparator_id not in rows:
        raise AnalysisConsistencyError("Pairwise comparison references an unknown strategy.")
    intervention = rows[intervention_id]
    comparator = rows[comparator_id]
    delta_cost = float(getattr(intervention, "expected_cost")) - float(getattr(comparator, "expected_cost"))
    delta_outcome = float(getattr(intervention, "expected_outcome")) - float(getattr(comparator, "expected_outcome"))
    return willingness_to_pay * delta_outcome - delta_cost


def require_matching_base_inmb(
    *,
    base_case_inmb: float,
    sensitivity_base_inmb: float,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-8,
) -> None:
    if not all(isfinite(value) for value in (base_case_inmb, sensitivity_base_inmb)):
        raise AnalysisConsistencyError("Base-case and sensitivity INMB must be finite.")
    if not isclose(
        base_case_inmb,
        sensitivity_base_inmb,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise AnalysisConsistencyError(
            "Sensitivity analysis does not reproduce the base-case INMB under the current settings. "
            "Do not interpret the sensitivity output until cycle length, horizon, discounting, perspective and model state are aligned."
        )


def ceac_threshold_grid(lower: float, upper: float, points: int) -> np.ndarray:
    if not isfinite(lower) or not isfinite(upper) or lower < 0 or upper <= lower:
        raise AnalysisConsistencyError("CEAC threshold range must be finite, non-negative and increasing.")
    if not isinstance(points, int) or points < 2 or points > 5001:
        raise AnalysisConsistencyError("CEAC grid requires between 2 and 5,001 points.")
    return np.linspace(float(lower), float(upper), points)
