"""Probabilistic sensitivity analysis for decision-tree models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from model.decision_tree import DecisionTreeDefinition, DecisionTreeValidationError, run_decision_tree
from model.schema import DistributionSpec, Parameter


class PSAConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class PSAResult:
    strategy_ids: tuple[str, ...]
    costs: Mapping[str, np.ndarray]
    outcomes: Mapping[str, np.ndarray]
    parameter_draws: Mapping[str, np.ndarray]
    iterations: int
    seed: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CEACResult:
    thresholds: np.ndarray
    probabilities: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _PSAPlan:
    sampled: tuple[Parameter, ...]
    independently_sampled: tuple[Parameter, ...]
    dirichlet_groups: Mapping[str, tuple[Parameter, ...]]
    warnings: tuple[str, ...]


def _distribution_parameters(distribution: DistributionSpec) -> dict[str, float]:
    values = {name.lower(): float(value) for name, value in distribution.parameters}
    if len(values) != len(distribution.parameters):
        raise PSAConfigurationError("Distribution parameter names must be unique.")
    return values


def _distribution(parameter: Parameter) -> DistributionSpec:
    psa = parameter.psa
    if psa is None or not psa.enabled or psa.distribution is None:
        raise PSAConfigurationError(
            f"Parameter '{parameter.id}' is not configured with an enabled PSA distribution."
        )
    return psa.distribution


def _family(parameter: Parameter) -> str:
    return _distribution(parameter).family.strip().lower()


def _dirichlet_alpha(parameter: Parameter) -> float:
    params = _distribution_parameters(_distribution(parameter))
    alpha = params.get("alpha", params.get("concentration"))
    if alpha is None or alpha <= 0:
        raise PSAConfigurationError(
            f"Dirichlet component '{parameter.id}' requires alpha>0 (or concentration>0)."
        )
    return alpha


def sample_distribution(distribution: DistributionSpec, rng: np.random.Generator) -> float:
    """Sample one scalar distribution; Dirichlet is sampled jointly elsewhere."""
    family = distribution.family.strip().lower()
    params = _distribution_parameters(distribution)

    if family == "beta":
        alpha = params.get("alpha")
        beta = params.get("beta")
        if alpha is None or beta is None or alpha <= 0 or beta <= 0:
            raise PSAConfigurationError("Beta distributions require alpha>0 and beta>0.")
        return float(rng.beta(alpha, beta))
    if family == "gamma":
        shape = params.get("shape", params.get("alpha"))
        scale = params.get("scale")
        if scale is None:
            rate = params.get("rate", params.get("lambda"))
            if rate is not None and rate > 0:
                scale = 1.0 / rate
        if shape is None or scale is None or shape <= 0 or scale <= 0:
            raise PSAConfigurationError(
                "Gamma distributions require shape and scale, or alpha with a positive rate/lambda."
            )
        return float(rng.gamma(shape, scale))
    if family == "lognormal":
        meanlog = params.get("meanlog")
        sdlog = params.get("sdlog")
        if meanlog is None or sdlog is None or sdlog <= 0:
            raise PSAConfigurationError("Lognormal distributions require meanlog and sdlog>0.")
        return float(rng.lognormal(meanlog, sdlog))
    if family == "normal":
        mean = params.get("mean")
        sd = params.get("sd")
        if mean is None or sd is None or sd <= 0:
            raise PSAConfigurationError("Normal distributions require mean and sd>0.")
        return float(rng.normal(mean, sd))
    if family == "uniform":
        low = params.get("low")
        high = params.get("high")
        if low is None or high is None or high <= low:
            raise PSAConfigurationError("Uniform distributions require high>low.")
        return float(rng.uniform(low, high))
    if family == "dirichlet":
        raise PSAConfigurationError(
            "Dirichlet distributions must be sampled jointly through a shared correlation_group."
        )
    raise PSAConfigurationError(
        f"Unsupported PSA distribution family '{distribution.family}'. Supported families are beta, gamma, lognormal, normal, uniform and grouped dirichlet."
    )


def _build_psa_plan(parameters: Sequence[Parameter]) -> _PSAPlan:
    sampled = tuple(p for p in parameters if p.psa is not None and p.psa.enabled)
    if not sampled:
        raise PSAConfigurationError("PSA requires at least one parameter explicitly enabled for PSA.")

    grouped: dict[str, list[Parameter]] = {}
    ungrouped_dirichlet: list[str] = []
    for parameter in sampled:
        family = _family(parameter)
        assert parameter.psa is not None
        group = parameter.psa.correlation_group
        if family == "dirichlet" and not group:
            ungrouped_dirichlet.append(parameter.id)
        if group:
            grouped.setdefault(group, []).append(parameter)

    if ungrouped_dirichlet:
        raise PSAConfigurationError(
            "Dirichlet parameters require a shared correlation_group. Missing for: "
            + ", ".join(sorted(ungrouped_dirichlet)) + "."
        )

    dirichlet_groups: dict[str, tuple[Parameter, ...]] = {}
    warnings: list[str] = []
    dirichlet_ids: set[str] = set()
    for group, members in grouped.items():
        families = {_family(parameter) for parameter in members}
        if "dirichlet" in families:
            if families != {"dirichlet"}:
                raise PSAConfigurationError(
                    f"Correlation group '{group}' mixes Dirichlet and non-Dirichlet distributions. A joint group must use one coherent sampling structure."
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
                "Correlation group '" + group + "' is declared for "
                + ", ".join(parameter.id for parameter in members)
                + ". No joint distribution/covariance structure is configured, so these parameters will be sampled independently. This can misrepresent decision uncertainty."
            )

    independently_sampled = tuple(
        parameter for parameter in sampled if parameter.id not in dirichlet_ids
    )
    return _PSAPlan(sampled, independently_sampled, dirichlet_groups, tuple(warnings))


def psa_configuration_warnings(parameters: Sequence[Parameter]) -> tuple[str, ...]:
    return _build_psa_plan(parameters).warnings


def run_tree_psa(
    tree: DecisionTreeDefinition,
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

    plan = _build_psa_plan(parameters)
    rng = np.random.default_rng(seed)
    parameter_draws = {parameter.id: np.empty(iterations, dtype=float) for parameter in plan.sampled}

    base = run_decision_tree(
        tree,
        parameters,
        included_cost_bearers=included_cost_bearers,
        cost_discount_rate=cost_discount_rate,
        outcome_discount_rate=outcome_discount_rate,
    )
    strategy_ids = tuple(row.strategy_id for row in base.strategies)
    costs = {sid: np.empty(iterations, dtype=float) for sid in strategy_ids}
    outcomes = {sid: np.empty(iterations, dtype=float) for sid in strategy_ids}
    dirichlet_alpha = {
        group: np.asarray([_dirichlet_alpha(parameter) for parameter in members], dtype=float)
        for group, members in plan.dirichlet_groups.items()
    }

    for iteration in range(iterations):
        overrides: dict[str, float] = {}
        for parameter in plan.independently_sampled:
            draw = sample_distribution(_distribution(parameter), rng)
            if not isfinite(draw):
                raise PSAConfigurationError(f"Distribution for '{parameter.id}' produced a non-finite draw.")
            overrides[parameter.id] = draw
            parameter_draws[parameter.id][iteration] = draw

        for group, members in plan.dirichlet_groups.items():
            vector = rng.dirichlet(dirichlet_alpha[group])
            for parameter, draw in zip(members, vector):
                value = float(draw)
                overrides[parameter.id] = value
                parameter_draws[parameter.id][iteration] = value

        try:
            run = run_decision_tree(
                tree,
                parameters,
                overrides=overrides,
                included_cost_bearers=included_cost_bearers,
                cost_discount_rate=cost_discount_rate,
                outcome_discount_rate=outcome_discount_rate,
            )
        except DecisionTreeValidationError as exc:
            raise PSAConfigurationError(
                f"PSA draw {iteration + 1} produced an invalid tree: {exc}. For mutually exclusive probabilities that must sum to 1, configure them as a Dirichlet group rather than independent scalar distributions."
            ) from exc

        for row in run.strategies:
            costs[row.strategy_id][iteration] = row.expected_cost
            outcomes[row.strategy_id][iteration] = row.expected_outcome

    return PSAResult(strategy_ids, costs, outcomes, parameter_draws, iterations, seed, plan.warnings)


def incremental_plane(result: PSAResult, *, intervention_id: str, comparator_id: str) -> tuple[np.ndarray, np.ndarray]:
    if intervention_id not in result.strategy_ids or comparator_id not in result.strategy_ids:
        raise ValueError("CE-plane comparison references an unknown strategy.")
    return (
        result.outcomes[intervention_id] - result.outcomes[comparator_id],
        result.costs[intervention_id] - result.costs[comparator_id],
    )


def ceac(result: PSAResult, thresholds: Sequence[float]) -> CEACResult:
    threshold_array = np.asarray(tuple(float(value) for value in thresholds), dtype=float)
    if threshold_array.size == 0:
        raise ValueError("CEAC requires at least one threshold.")
    if np.any(~np.isfinite(threshold_array)) or np.any(threshold_array < 0):
        raise ValueError("CEAC thresholds must be finite and non-negative.")
    probabilities = {sid: np.zeros(threshold_array.size, dtype=float) for sid in result.strategy_ids}
    for index, threshold in enumerate(threshold_array):
        nmb = np.vstack([threshold * result.outcomes[sid] - result.costs[sid] for sid in result.strategy_ids])
        max_nmb = np.max(nmb, axis=0)
        is_best = np.isclose(nmb, max_nmb, rtol=1e-12, atol=1e-12)
        tie_counts = np.sum(is_best, axis=0)
        for strategy_index, sid in enumerate(result.strategy_ids):
            probabilities[sid][index] = float(np.mean(is_best[strategy_index] / tie_counts))
    return CEACResult(threshold_array, probabilities)


def pairwise_probability_cost_effective(
    result: PSAResult,
    *,
    intervention_id: str,
    comparator_id: str,
    willingness_to_pay: float,
) -> float:
    if willingness_to_pay < 0 or not isfinite(willingness_to_pay):
        raise ValueError("Willingness-to-pay threshold must be finite and non-negative.")
    delta_effect, delta_cost = incremental_plane(
        result, intervention_id=intervention_id, comparator_id=comparator_id
    )
    return float(np.mean(willingness_to_pay * delta_effect - delta_cost > 0))
