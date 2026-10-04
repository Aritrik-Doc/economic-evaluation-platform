"""Reproducibility guard for cross-workspace resource/capacity results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


class CapacityReproducibilityError(ValueError):
    pass


def _canonical(value: Any):
    if is_dataclass(value):
        return _canonical(asdict(value))
    if hasattr(value, "to_dict"):
        try:
            return _canonical(value.to_dict("records"))
        except TypeError:
            return _canonical(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _include_key(key: str, *, population_source: str | None, requirement_source: str | None, clinical_model_type: str | None) -> bool:
    if key.startswith("rc_last_"):
        return False
    if key.startswith("rc_"):
        return True
    if population_source == "Shared Population & Uptake" and key.startswith("pu_"):
        return True
    if population_source == "Budget Impact Analysis" and key.startswith("bia_"):
        return True
    if requirement_source in {"Linked clinical model", "Hybrid — clinical + manual"}:
        if clinical_model_type == "Decision Tree" and key.startswith("dt_"):
            return not key.startswith(("dt_psa_", "dt_audit_"))
        if clinical_model_type == "Cohort Markov" and (key.startswith("markov_") or key.startswith("mk_")):
            return not key.startswith(("markov_psa_", "mk_psa_"))
        if clinical_model_type == "Advanced Markov" and key.startswith("adv_"):
            return not key.startswith("adv_psa_")
    return False


def capacity_source_fingerprint(
    session_state: Mapping[str, Any],
    *,
    population_source: str | None,
    requirement_source: str | None,
    clinical_model_type: str | None = None,
) -> str:
    """Hash substantive session inputs that feed the most recent capacity run.

    The hash is an invalidation aid, not a digital signature. It intentionally
    excludes stored capacity outputs themselves and stochastic-analysis outputs
    that do not affect a deterministic capacity calculation.
    """
    payload = {
        key: _canonical(value)
        for key, value in session_state.items()
        if _include_key(
            str(key),
            population_source=population_source,
            requirement_source=requirement_source,
            clinical_model_type=clinical_model_type,
        )
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def capacity_result_is_current(
    session_state: Mapping[str, Any],
    context: Mapping[str, Any] | None,
    stored_fingerprint: str | None,
) -> bool:
    if not context or not stored_fingerprint:
        return False
    current = capacity_source_fingerprint(
        session_state,
        population_source=context.get("population_source"),
        requirement_source=context.get("requirement_source"),
        clinical_model_type=context.get("clinical_model_type"),
    )
    return current == stored_fingerprint
