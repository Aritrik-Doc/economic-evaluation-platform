"""Versioned save/load/export/audit for cohort and semi-Markov models."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence
from uuid import uuid4

from model.markov_builder import compile_markov_tables
from model.semi_markov_builder import compile_semi_markov_tables


STATE_TRANSITION_SCHEMA_VERSION = "0.2"
STATE_TRANSITION_AUDIT_VERSION = "0.2"
STATE_TRANSITION_PLATFORM_VERSION = "0.7"
StateTransitionModelType = Literal["cohort_markov", "semi_markov"]


class StateTransitionPersistenceError(ValueError):
    pass


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise StateTransitionPersistenceError("Model files cannot contain infinite values.")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_json_safe(dict(row)) for row in rows]


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def state_transition_content_hash(bundle: Mapping[str, Any]) -> str:
    content = deepcopy(dict(bundle))
    content.pop("saved_at_utc", None)
    content.pop("content_hash_sha256", None)
    return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def _validate_methods(methods: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "reference_case_code",
        "outcome_code",
        "currency_code",
        "threshold",
        "perspective_label",
        "included_cost_bearers",
        "cost_discount_rate",
        "outcome_discount_rate",
    }
    missing = required - set(methods)
    if missing:
        raise StateTransitionPersistenceError(
            "State-transition model is missing method fields: " + ", ".join(sorted(missing)) + "."
        )
    cleaned = _json_safe(dict(methods))
    threshold = cleaned.get("threshold")
    if threshold is not None and (not math.isfinite(float(threshold)) or float(threshold) < 0):
        raise StateTransitionPersistenceError("Decision threshold must be finite and non-negative.")
    for key in ("cost_discount_rate", "outcome_discount_rate"):
        value = float(cleaned[key])
        if not math.isfinite(value) or not 0 <= value < 1:
            raise StateTransitionPersistenceError(f"{key} must be a finite proportion in [0, 1).")
    if not str(cleaned["currency_code"]).strip():
        raise StateTransitionPersistenceError("Analysis currency code is required.")
    return cleaned


def _validate_engine(engine: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "cycle_length_years",
        "max_cycles",
        "state_accrual_timing",
        "transition_reward_timing",
        "termination_mode",
        "depletion_threshold",
    }
    missing = required - set(engine)
    if missing:
        raise StateTransitionPersistenceError(
            "State-transition model is missing engine settings: " + ", ".join(sorted(missing)) + "."
        )
    cleaned = _json_safe(dict(engine))
    cycle_length = float(cleaned["cycle_length_years"])
    if not math.isfinite(cycle_length) or cycle_length <= 0:
        raise StateTransitionPersistenceError("Cycle length must be positive and finite.")
    max_cycles = int(cleaned["max_cycles"])
    if max_cycles < 1:
        raise StateTransitionPersistenceError("Maximum cycles must be positive.")
    return cleaned


def build_state_transition_bundle(
    *,
    model_type: StateTransitionModelType,
    model_name: str,
    methods: Mapping[str, Any],
    engine: Mapping[str, Any],
    parameter_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    strategy_rows: Sequence[Mapping[str, Any]],
    initial_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    state_reward_rows: Sequence[Mapping[str, Any]],
    transition_reward_rows: Sequence[Mapping[str, Any]],
    mortality_table_rows: Sequence[Mapping[str, Any]] = (),
    mortality_rule_rows: Sequence[Mapping[str, Any]] = (),
    author: str = "",
    notes: str = "",
) -> dict[str, Any]:
    if model_type not in {"cohort_markov", "semi_markov"}:
        raise StateTransitionPersistenceError("Unsupported state-transition model type.")
    if not model_name.strip():
        raise StateTransitionPersistenceError("Model name is required before saving.")

    methods_clean = _validate_methods(methods)
    engine_clean = _validate_engine(engine)
    structure = {
        "parameters": _rows(parameter_rows),
        "states": _rows(state_rows),
        "strategies": _rows(strategy_rows),
        "initial_distribution": _rows(initial_rows),
        "transitions": _rows(transition_rows),
        "state_rewards": _rows(state_reward_rows),
        "transition_rewards": _rows(transition_reward_rows),
        "mortality_table": _rows(mortality_table_rows),
        "mortality_rules": _rows(mortality_rule_rows),
    }

    _compile_for_validation(model_type, structure, engine_clean)

    bundle: dict[str, Any] = {
        "schema_version": STATE_TRANSITION_SCHEMA_VERSION,
        "platform_version": STATE_TRANSITION_PLATFORM_VERSION,
        "model_family": "state_transition",
        "model_type": model_type,
        "model_name": model_name.strip(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": methods_clean,
        "engine": engine_clean,
        "model": structure,
        "metadata": {"author": author.strip(), "notes": notes.strip()},
    }
    bundle["content_hash_sha256"] = state_transition_content_hash(bundle)
    return bundle


def _compile_for_validation(model_type: str, model: Mapping[str, Any], engine: Mapping[str, Any]):
    common = dict(
        cycle_length_years=float(engine["cycle_length_years"]),
        max_cycles=int(engine["max_cycles"]),
        state_accrual_timing=str(engine["state_accrual_timing"]),
        transition_reward_timing=str(engine["transition_reward_timing"]),
        termination_mode=str(engine["termination_mode"]),
        depletion_threshold=float(engine["depletion_threshold"]),
    )
    if model_type == "cohort_markov":
        if model.get("mortality_table") or model.get("mortality_rules"):
            raise StateTransitionPersistenceError(
                "Homogeneous cohort Markov files cannot contain advanced mortality-table rules."
            )
        return compile_markov_tables(
            model["parameters"],
            model["states"],
            model["strategies"],
            model["initial_distribution"],
            model["transitions"],
            model["state_rewards"],
            model["transition_rewards"],
            **common,
        )
    if model_type == "semi_markov":
        return compile_semi_markov_tables(
            model["parameters"],
            model["states"],
            model["strategies"],
            model["initial_distribution"],
            model["transitions"],
            model["state_rewards"],
            model["transition_rewards"],
            model.get("mortality_table", []),
            model.get("mortality_rules", []),
            **common,
        )
    raise StateTransitionPersistenceError(f"Unsupported model_type '{model_type}'.")


def state_transition_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(bundle), indent=2, ensure_ascii=False, sort_keys=True)


def load_state_transition_bundle(data: str | bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        parsed = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateTransitionPersistenceError("Uploaded model file is not valid UTF-8 JSON.") from exc
    if not isinstance(parsed, dict):
        raise StateTransitionPersistenceError("Model file root must be a JSON object.")
    if parsed.get("schema_version") != STATE_TRANSITION_SCHEMA_VERSION:
        raise StateTransitionPersistenceError(
            f"Unsupported state-transition schema version '{parsed.get('schema_version')}'. "
            f"This build supports {STATE_TRANSITION_SCHEMA_VERSION}."
        )
    if parsed.get("model_family") != "state_transition":
        raise StateTransitionPersistenceError("File is not a state-transition model bundle.")
    model_type = parsed.get("model_type")
    if model_type not in {"cohort_markov", "semi_markov"}:
        raise StateTransitionPersistenceError("File has an unsupported state-transition model type.")
    methods = parsed.get("methods")
    engine = parsed.get("engine")
    model = parsed.get("model")
    if not isinstance(methods, dict) or not isinstance(engine, dict) or not isinstance(model, dict):
        raise StateTransitionPersistenceError("Model file is missing methods, engine or model content.")
    required_model = {
        "parameters",
        "states",
        "strategies",
        "initial_distribution",
        "transitions",
        "state_rewards",
        "transition_rewards",
    }
    missing = required_model - set(model)
    if missing:
        raise StateTransitionPersistenceError(
            "State-transition model is missing structural fields: " + ", ".join(sorted(missing)) + "."
        )
    _validate_methods(methods)
    _validate_engine(engine)
    _compile_for_validation(str(model_type), model, engine)
    stored_hash = parsed.get("content_hash_sha256")
    if not stored_hash:
        raise StateTransitionPersistenceError("State-transition model file is missing its content hash.")
    actual_hash = state_transition_content_hash(parsed)
    if stored_hash != actual_hash:
        raise StateTransitionPersistenceError(
            "Model content hash does not match the file. The file may have been edited or corrupted."
        )
    return _json_safe(parsed)


def compile_loaded_state_transition_bundle(bundle: Mapping[str, Any]):
    """Recompile a loaded bundle into validated engine objects."""
    return _compile_for_validation(
        str(bundle["model_type"]),
        bundle["model"],
        bundle["engine"],
    )


def build_state_transition_audit_record(
    bundle: Mapping[str, Any],
    *,
    analysis_type: str,
    run_settings: Mapping[str, Any],
    results: Mapping[str, Any],
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    if not analysis_type.strip():
        raise StateTransitionPersistenceError("Audit analysis_type is required.")
    snapshot = _json_safe(deepcopy(dict(bundle)))
    # Ensure the snapshot is itself a valid, untampered model before recording it.
    load_state_transition_bundle(state_transition_bundle_json(snapshot))
    return {
        "audit_schema_version": STATE_TRANSITION_AUDIT_VERSION,
        "run_id": str(uuid4()),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform_version": STATE_TRANSITION_PLATFORM_VERSION,
        "model_hash_sha256": state_transition_content_hash(snapshot),
        "model_type": snapshot["model_type"],
        "analysis_type": analysis_type.strip(),
        "run_settings": _json_safe(dict(run_settings)),
        "results": _json_safe(dict(results)),
        "warnings": [str(item) for item in warnings],
        "model_snapshot": snapshot,
    }


def state_transition_audit_json(records: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "audit_schema_version": STATE_TRANSITION_AUDIT_VERSION,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": [_json_safe(dict(record)) for record in records],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
