"""Probabilistic sensitivity analysis for semi-Markov cohort models."""

from __future__ import annotations

from math import isfinite
from typing import Sequence

import numpy as np

from model.markov_psa import _dirichlet_alpha, _sampling_plan
from model.psa import PSAConfigurationError, PSAResult, sample_distribution
from model.schema import Parameter
from model.semi_markov import (
    SemiMarkovDefinition,
    SemiMarkovValidationError,
    run_semi_markov,
)


def semi_markov_psa_configuration_warnings(
    parameters: Sequence[Parameter],
) -> tuple[str, ...]:
    return _sampling_plan(parameters)[3]


def run_semi_markov_psa(
    model: SemiMarkovDefinition,
    parameters: Sequence[Parameter],
    *,
    iterations: int,
    seed: int,
    included_cost_bearers: Sequence[str] | None = None,
    cost_discount_rate: float = 0.0,
    outcome_discount_rate: float = 0.0,
) -> PSAResult:
    if iterations < 1:
        raise PSAConfigurationError("PSA iterations must be positive.")
    if iterations > 1_000_000:
        raise PSAConfigurationError("PSA iterations are capped at 1,000,000 per run.")

    sampled, independent, dirichlet_groups, warnings = _sampling_plan(parameters)
    rng = np.random.default_rng(seed)
    parameter_draws = {
        parameter.id: np.empty(iterations, dtype=float) for parameter in sampled
    }

    base = run_semi_markov(
        model,
        parameters,
        included_cost_bearers=included_cost_bearers,
        cost_discount_rate=cost_discount_rate,
        outcome_discount_rate=outcome_discount_rate,
    )
    strategy_ids = tuple(row.strategy_id for row in base.strategies)
    costs = {strategy_id: np.empty(iterations, dtype=float) for strategy_id in strategy_ids}
    outcomes = {strategy_id: np.empty(iterations, dtype=float) for strategy_id in strategy_ids}

    dirichlet_alpha = {
        group: np.asarray([_dirichlet_alpha(parameter) for parameter in members], dtype=float)
        for group, members in dirichlet_groups.items()
    }

    for iteration in range(iterations):
        overrides: dict[str, float] = {}

        for parameter in independent:
            psa = parameter.psa
            assert psa is not None and psa.distribution is not None
            draw = sample_distribution(psa.distribution, rng)
            if not isfinite(draw):
                raise PSAConfigurationError(
                    f"Distribution for '{parameter.id}' produced a non-finite draw."
                )
            overrides[parameter.id] = draw
            parameter_draws[parameter.id][iteration] = draw

        for group, members in dirichlet_groups.items():
            vector = rng.dirichlet(dirichlet_alpha[group])
            for parameter, draw in zip(members, vector):
                value = float(draw)
                overrides[parameter.id] = value
                parameter_draws[parameter.id][iteration] = value

        try:
            run = run_semi_markov(
                model,
                parameters,
                overrides=overrides,
                included_cost_bearers=included_cost_bearers,
                cost_discount_rate=cost_discount_rate,
                outcome_discount_rate=outcome_discount_rate,
            )
        except SemiMarkovValidationError as exc:
            raise PSAConfigurationError(
                f"PSA draw {iteration + 1} produced an invalid semi-Markov model: {exc}. "
                "Time-varying probability exits that must remain mutually exclusive need a coherent joint parameterisation; rate-based competing exits are converted jointly."
            ) from exc

        for row in run.strategies:
            costs[row.strategy_id][iteration] = row.expected_cost
            outcomes[row.strategy_id][iteration] = row.expected_outcome

    return PSAResult(
        strategy_ids=strategy_ids,
        costs=costs,
        outcomes=outcomes,
        parameter_draws=parameter_draws,
        iterations=iterations,
        seed=seed,
        warnings=warnings,
    )
