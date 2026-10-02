"""Distribution parameterisation and v0.4.1 parameter-row migration helpers."""

from __future__ import annotations

from math import exp, isfinite, log, sqrt
from typing import Any, Mapping


class ParameterisationError(ValueError):
    pass


DISTRIBUTION_PARAMETERISATIONS: dict[str, tuple[str, ...]] = {
    "beta": ("Alpha + Beta", "Mean + SE"),
    "gamma": ("Shape + Scale", "Mean + SD"),
    "normal": ("Mean + SD", "Estimate + 95% CI"),
    "lognormal": ("Meanlog + SDlog", "Arithmetic mean + SD"),
    "uniform": ("Minimum + Maximum",),
    "dirichlet": ("Alpha concentration",),
}


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ParameterisationError(f"{name} must be numeric.") from exc
    if not isfinite(number):
        raise ParameterisationError(f"{name} must be finite.")
    return number


def beta_from_mean_se(mean: float, se: float) -> dict[str, float]:
    mean = _finite(mean, "Mean")
    se = _finite(se, "SE")
    if not 0 < mean < 1 or se <= 0:
        raise ParameterisationError("Beta mean must be between 0 and 1 and SE must be positive.")
    variance = se * se
    concentration = mean * (1 - mean) / variance - 1
    if concentration <= 0:
        raise ParameterisationError("Mean and SE imply an invalid Beta distribution; uncertainty is too large for this mean.")
    return {"alpha": mean * concentration, "beta": (1 - mean) * concentration}


def gamma_from_mean_sd(mean: float, sd: float) -> dict[str, float]:
    mean = _finite(mean, "Mean")
    sd = _finite(sd, "SD")
    if mean <= 0 or sd <= 0:
        raise ParameterisationError("Gamma mean and SD must be positive.")
    return {"shape": (mean / sd) ** 2, "scale": (sd * sd) / mean}


def lognormal_from_mean_sd(mean: float, sd: float) -> dict[str, float]:
    mean = _finite(mean, "Arithmetic mean")
    sd = _finite(sd, "SD")
    if mean <= 0 or sd <= 0:
        raise ParameterisationError("Lognormal arithmetic mean and SD must be positive.")
    sigma2 = log(1 + (sd * sd) / (mean * mean))
    return {"meanlog": log(mean) - sigma2 / 2, "sdlog": sqrt(sigma2)}


def normal_from_estimate_ci(estimate: float, lower: float, upper: float) -> dict[str, float]:
    estimate = _finite(estimate, "Estimate")
    lower = _finite(lower, "Lower 95% CI")
    upper = _finite(upper, "Upper 95% CI")
    if upper <= lower:
        raise ParameterisationError("Upper 95% CI must exceed lower 95% CI.")
    sd = (upper - lower) / (2 * 1.959963984540054)
    return {"mean": estimate, "sd": sd}


def canonical_distribution_parameters(
    family: str,
    parameterisation: str,
    values: Mapping[str, Any],
    *,
    base_value: float | None = None,
) -> dict[str, float]:
    """Convert friendly UI inputs to the canonical parameters used by the PSA engine."""
    family = family.strip().lower()
    if family == "beta":
        if parameterisation == "Mean + SE":
            return beta_from_mean_se(values.get("mean", base_value), values.get("se"))
        alpha = _finite(values.get("alpha"), "Alpha")
        beta = _finite(values.get("beta"), "Beta")
        if alpha <= 0 or beta <= 0:
            raise ParameterisationError("Beta alpha and beta must be positive.")
        return {"alpha": alpha, "beta": beta}
    if family == "gamma":
        if parameterisation == "Mean + SD":
            return gamma_from_mean_sd(values.get("mean", base_value), values.get("sd"))
        shape = _finite(values.get("shape"), "Shape")
        scale = _finite(values.get("scale"), "Scale")
        if shape <= 0 or scale <= 0:
            raise ParameterisationError("Gamma shape and scale must be positive.")
        return {"shape": shape, "scale": scale}
    if family == "normal":
        if parameterisation == "Estimate + 95% CI":
            return normal_from_estimate_ci(
                values.get("estimate", base_value), values.get("lower_ci"), values.get("upper_ci")
            )
        mean = _finite(values.get("mean", base_value), "Mean")
        sd = _finite(values.get("sd"), "SD")
        if sd <= 0:
            raise ParameterisationError("Normal SD must be positive.")
        return {"mean": mean, "sd": sd}
    if family == "lognormal":
        if parameterisation == "Arithmetic mean + SD":
            return lognormal_from_mean_sd(values.get("mean", base_value), values.get("sd"))
        meanlog = _finite(values.get("meanlog"), "Meanlog")
        sdlog = _finite(values.get("sdlog"), "SDlog")
        if sdlog <= 0:
            raise ParameterisationError("Lognormal SDlog must be positive.")
        return {"meanlog": meanlog, "sdlog": sdlog}
    if family == "uniform":
        low = _finite(values.get("low"), "Minimum")
        high = _finite(values.get("high"), "Maximum")
        if high <= low:
            raise ParameterisationError("Uniform maximum must exceed minimum.")
        return {"low": low, "high": high}
    if family == "dirichlet":
        alpha = _finite(values.get("alpha"), "Alpha concentration")
        if alpha <= 0:
            raise ParameterisationError("Dirichlet alpha concentration must be positive.")
        return {"alpha": alpha}
    raise ParameterisationError(f"Unsupported distribution family '{family}'.")


def parse_legacy_distribution_parameters(value: Any) -> dict[str, float]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return {str(key).lower(): float(item) for key, item in value.items()}
    parsed: dict[str, float] = {}
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ParameterisationError(f"Distribution parameter '{token}' must use name=value syntax.")
        name, raw = token.split("=", 1)
        parsed[name.strip().lower()] = float(raw.strip())
    return parsed


def distribution_parameters_text(parameters: Mapping[str, float]) -> str:
    return ",".join(f"{name}={value:.12g}" for name, value in parameters.items())


def migrate_parameter_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert pre-v0.4.1 table rows to the split DSA/PSA representation."""
    migrated = dict(row)
    if "dsa_enabled" in migrated or "psa_enabled" in migrated:
        migrated.setdefault("dsa_enabled", False)
        migrated.setdefault("psa_enabled", False)
        migrated.setdefault("dsa_rationale", migrated.get("uncertainty_rationale") or "Not represented in DSA.")
        migrated.setdefault("psa_rationale", migrated.get("uncertainty_rationale") or "Not represented in PSA.")
        migrated.setdefault("dsa_lower", migrated.get("lower"))
        migrated.setdefault("dsa_upper", migrated.get("upper"))
        return migrated

    kind = str(migrated.get("uncertainty_kind") or "none").lower()
    rationale = str(migrated.get("uncertainty_rationale") or "Legacy model uncertainty setting.")
    migrated["dsa_enabled"] = kind in {"range", "range_and_distribution"}
    migrated["dsa_lower"] = migrated.get("lower") if migrated["dsa_enabled"] else None
    migrated["dsa_upper"] = migrated.get("upper") if migrated["dsa_enabled"] else None
    migrated["dsa_rationale"] = rationale
    migrated["psa_enabled"] = kind in {"distribution", "range_and_distribution"}
    migrated["psa_rationale"] = rationale
    migrated.setdefault("distribution_parameterisation", "")
    if migrated["psa_enabled"]:
        migrated["distribution_parameters"] = parse_legacy_distribution_parameters(
            migrated.get("distribution_parameters")
        )
    else:
        migrated.setdefault("distribution_parameters", {})
    return migrated
