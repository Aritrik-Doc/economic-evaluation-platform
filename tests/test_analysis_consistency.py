import numpy as np
import pytest

from model.analysis_consistency import (
    AnalysisConsistencyError,
    ceac_threshold_grid,
    pairwise_inmb_from_strategy_results,
    require_matching_base_inmb,
)


class ResultRow:
    def __init__(self, strategy_id, cost, outcome):
        self.strategy_id = strategy_id
        self.expected_cost = cost
        self.expected_outcome = outcome


def test_pairwise_inmb_is_derived_from_displayed_strategy_results():
    rows = [ResultRow("comparator", 1000.0, 1.0), ResultRow("intervention", 1400.0, 1.2)]
    assert pairwise_inmb_from_strategy_results(
        rows,
        intervention_id="intervention",
        comparator_id="comparator",
        willingness_to_pay=3000.0,
    ) == pytest.approx(200.0)


def test_matching_sensitivity_base_inmb_passes_and_mismatch_is_blocked():
    require_matching_base_inmb(base_case_inmb=123.456, sensitivity_base_inmb=123.45600000001)
    with pytest.raises(AnalysisConsistencyError, match="does not reproduce the base-case INMB"):
        require_matching_base_inmb(base_case_inmb=123.456, sensitivity_base_inmb=120.0)


def test_ceac_grid_uses_user_selected_limits_instead_of_threshold_multiple():
    grid = ceac_threshold_grid(100.0, 4200.0, 83)
    assert len(grid) == 83
    assert grid[0] == pytest.approx(100.0)
    assert grid[-1] == pytest.approx(4200.0)
    assert np.all(np.diff(grid) > 0)


def test_invalid_ceac_range_is_rejected():
    with pytest.raises(AnalysisConsistencyError):
        ceac_threshold_grid(5000.0, 4200.0, 51)
