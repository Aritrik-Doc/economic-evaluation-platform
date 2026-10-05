"""Pure annual population projections used by the shared Population & Uptake UI.

Keeping the projection outside Streamlit prevents widget/session-state behaviour
from changing the mathematical result. The UI may display or manually override a
projection, but these functions always recompute from the current drivers.
"""

from __future__ import annotations

from dataclasses import dataclass

from model.population_uptake import compound_series, top_down_eligible_population


@dataclass(frozen=True)
class PopulationProjectionRow:
    year: int
    eligible_population: float
    covered_lives: float | None


def project_direct_population(
    *,
    eligible_start: float,
    eligible_growth_rate: float,
    horizon_years: int,
    covered_lives_start: float | None = None,
    covered_lives_growth_rate: float = 0.0,
) -> tuple[PopulationProjectionRow, ...]:
    """Project an explicitly entered eligible population and optional covered lives.

    Eligible-population growth and covered-lives growth are intentionally separate
    assumptions. A zero/None covered-lives value means that no PMPM denominator is
    supplied; it does not change the eligible population.
    """

    eligible = compound_series(eligible_start, eligible_growth_rate, horizon_years)
    covered = None
    if covered_lives_start is not None and covered_lives_start > 0:
        covered = compound_series(
            covered_lives_start,
            covered_lives_growth_rate,
            horizon_years,
        )

    rows = []
    for index, eligible_value in enumerate(eligible):
        covered_value = covered[index] if covered is not None else None
        rows.append(
            PopulationProjectionRow(
                year=index + 1,
                eligible_population=float(eligible_value),
                covered_lives=None if covered_value is None else float(covered_value),
            )
        )
    return tuple(rows)


def project_top_down_population(
    *,
    covered_or_catchment_start: float,
    covered_or_catchment_growth_rate: float,
    prevalence: float,
    diagnosed_or_identified: float,
    clinically_eligible: float,
    access_or_coverage: float,
    horizon_years: int,
) -> tuple[PopulationProjectionRow, ...]:
    """Project eligible population from a payer/catchment population funnel.

    The starting population is also retained as the covered-lives denominator.
    This is appropriate when the base population represents the payer's covered
    population. If the user's denominator differs, they should use direct mode or
    manually override the annual covered-lives values in the UI.
    """

    covered = compound_series(
        covered_or_catchment_start,
        covered_or_catchment_growth_rate,
        horizon_years,
    )
    return tuple(
        PopulationProjectionRow(
            year=index + 1,
            eligible_population=float(
                top_down_eligible_population(
                    covered_value,
                    prevalence,
                    diagnosed_or_identified,
                    clinically_eligible,
                    access_or_coverage,
                )
            ),
            covered_lives=float(covered_value),
        )
        for index, covered_value in enumerate(covered)
    )
