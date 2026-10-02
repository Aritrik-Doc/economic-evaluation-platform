"""Probabilistic sensitivity analysis for cohort Markov models."""

from __future__ import annotations

from math import isfinite
from typing import Sequence

import numpy as np

from model.markov import CohortMarkovDefinition, MarkovValidationError, run_cohort_markov
from model.psa import PSAConfigurationError, PSAResult, sample_distribution
from model.schema import Parameter


def _family(parameter: Parameter) -> str:
    psa = parameter.psa
    if psa is None or not psa.enabled or psa.distribution is None:
        raise PSAConfigurationError(
            f"Parameter '{parameter.id}' is not configured with a PSA distribution."
        )
    return psa.distribution.family.strip().lower()


def _dirichlet_alpha(parameter: Parameter) -> float:
    psa = parameter.psa
    assert psa is not None and psa.distribution is not None
    values = {name.lower(): float(value) for name, value in psa.distribution.parameters}
    alpha = values.get("alpha", values.get("concentration"))
    if alpha is None or alpha <= 0:
        raise PSAConfigurationError(
            f"Dirichlet component '{parameter.id}' requires alpha>0."
        )
    return alpha


def _sampling_plan(parameters: Sequence[Parameter]):
    sampled = tuple(
        parameter
        for parameter in parameters
        if parameter.psa is not None and parameter.psa.enabled
    )
    if not sampled:
        raise PSAConfigurationError("PSA requires at least one parameter explicitly enabled for PSA.")

    groups: dict[str, list[Parameter]] = {}
    for parameter in sampled:
        psa = parameter.psa
        assert psa is not None
        if psa.correlation_group:
            groups.setdefault(psa.correlation_group, []).append(parameter)

    dirichlet_groups: dict[str, tuple[Parameter, ...]] = {}
    dirichlet_ids: set[str] = set()
    warnings: list[str] = []

    for parameter in sampled:
        if _family(parameter) == "dirichlet":
            psa = parameter.psa
            assert psa is not None
            if not psa.correlation_group:
                raise PSAConfigurationError(
                    f"Dirichlet parameter '{parameter.id}' requires a correlation_group."
                )

    for group, members in groups.items():
        families = {_family(parameter) for parameter in members}
        if "dirichlet" in families:
            if families != {"dirichlet"}:
                raise PSAConfigurationError(
                    f"Correlation group '{group}' mixes Dirichlet and non-Dirichlet distributions."
                )
            if len(members) < 2:
                raise PSAConfigurationError(
                    f"Dirichlet correlation group '{group}' requires at least two component parameters."
                )
            for parameter in members:
                _dirichlet_alpha(parameter)
            dirichlet_groups[group] = tuple(members)
            dirichlet_ids.update(parameter.id for parameter in members)
        elif len(members) > 1:
            warnings.append(
                "Correlation group '"
                + group
                + "' is declared for "
                + ", ".join(parameter.id for parameter in members)
                + ". No joint covariance/sampling structure is configured, so these parameters are sampled independently."
            )

    independent = tuple(parameter for parameter in sampled if parameter.id not in dirichlet_ids)
    return sampled, independent, dirichlet_groups, tuple(warnings)


def markov_psa_configuration_warnings(parameters: Sequence[Parameter]) -> tuple[str, ...]:
    return _sampling_plan(parameters)[3]


def run_markov_psa(
    model: CohortMarkovDefinition,
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

    base = run_cohort_markov(
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
            run = run_cohort_markov(
                model,
                parameters,
                overrides=overrides,
                included_cost_bearers=included_cost_bearers,
                cost_discount_rate=cost_discount_rate,
                outcome_discount_rate=outcome_discount_rate,
            )
        except MarkovValidationError as exc:
            raise PSAConfigurationError(
                f"PSA draw {iteration + 1} produced an invalid transition model: {exc}. "
                "Competing probabilities that must sum to one should use a coherent joint parameterisation such as grouped Dirichlet components or a residual transition."
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
