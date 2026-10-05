import pytest

from model.population_projection import (
    project_direct_population,
    project_top_down_population,
)


def test_direct_projection_updates_eligible_and_covered_series_independently():
    rows = project_direct_population(
        eligible_start=1000.0,
        eligible_growth_rate=0.10,
        horizon_years=3,
        covered_lives_start=10000.0,
        covered_lives_growth_rate=0.05,
    )
    assert [row.eligible_population for row in rows] == pytest.approx([1000.0, 1100.0, 1210.0])
    assert [row.covered_lives for row in rows] == pytest.approx([10000.0, 10500.0, 11025.0])


def test_direct_projection_without_covered_lives_keeps_denominator_missing():
    rows = project_direct_population(
        eligible_start=500.0,
        eligible_growth_rate=0.0,
        horizon_years=2,
        covered_lives_start=0.0,
    )
    assert [row.eligible_population for row in rows] == pytest.approx([500.0, 500.0])
    assert [row.covered_lives for row in rows] == [None, None]


def test_top_down_projection_recomputes_from_growth_and_funnel_inputs():
    rows = project_top_down_population(
        covered_or_catchment_start=100000.0,
        covered_or_catchment_growth_rate=0.10,
        prevalence=0.10,
        diagnosed_or_identified=0.50,
        clinically_eligible=0.80,
        access_or_coverage=0.50,
        horizon_years=2,
    )
    assert [row.covered_lives for row in rows] == pytest.approx([100000.0, 110000.0])
    assert [row.eligible_population for row in rows] == pytest.approx([2000.0, 2200.0])
