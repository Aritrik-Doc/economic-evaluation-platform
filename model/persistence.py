"""Save/load/export helpers and auditable analysis records.

The persistence format is intentionally JSON-based, versioned, and human-readable.
A content hash protects against accidental edits to a saved model file. Audit
records embed the model snapshot used for a run so they remain interpretable even
if the working model changes later.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from model.tree_builder import compile_builder_tables


MODEL_FILE_SCHEMA_VERSION = "0.1"
AUDIT_SCHEMA_VERSION = "0.1"
PLATFORM_VERSION = "0.4"


class PersistenceError(ValueError):
    pass


def _json_safe(value: Any) -> Any:
    """Convert table/editor values into portable JSON primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise PersistenceError("Model files cannot contain infinite numeric values.")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _normalise_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_json_safe(dict(row)) for row in rows]


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def model_content_hash(bundle: Mapping[str, Any]) -> str:
    """Hash substantive model content, excluding volatile save metadata."""
    content = deepcopy(dict(bundle))
    content.pop("saved_at_utc", None)
    content.pop("content_hash_sha256", None)
    return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def build_decision_tree_bundle(
    *,
    model_name: str,
    reference_case_code: str,
    outcome_code: str,
    currency_code: str,
    threshold: float | None,
    perspective_label: str,
    included_cost_bearers: Sequence[str],
    time_horizon: str,
    cost_discount_rate: float,
    outcome_discount_rate: float,
    parameter_rows: Sequence[Mapping[str, Any]],
    strategy_rows: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
    author: str = "",
    notes: str = "",
) -> dict[str, Any]:
    if not model_name.strip():
        raise PersistenceError("Model name is required before saving.")
    if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
        raise PersistenceError("Decision threshold must be finite and non-negative.")
    for label, rate in (("Cost", cost_discount_rate), ("Outcome", outcome_discount_rate)):
        if not math.isfinite(rate) or rate < 0 or rate >= 1:
            raise PersistenceError(f"{label} discount rate must be a proportion in [0, 1).")

    parameters = _normalise_rows(parameter_rows)
    strategies = _normalise_rows(strategy_rows)
    nodes = _normalise_rows(node_rows)
    branches = _normalise_rows(branch_rows)

    # Compilation is the validation gate for saved decision-tree files.
    compile_builder_tables(parameters, strategies, nodes, branches)

    bundle: dict[str, Any] = {
        "schema_version": MODEL_FILE_SCHEMA_VERSION,
        "platform_version": PLATFORM_VERSION,
        "model_type": "decision_tree",
        "model_name": model_name.strip(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": {
            "reference_case_code": reference_case_code,
            "outcome_code": outcome_code,
            "currency_code": currency_code,
            "threshold": threshold,
            "perspective_label": perspective_label,
            "included_cost_bearers": list(included_cost_bearers),
            "time_horizon": time_horizon,
            "cost_discount_rate": cost_discount_rate,
            "outcome_discount_rate": outcome_discount_rate,
        },
        "model": {
            "parameters": parameters,
            "strategies": strategies,
            "nodes": nodes,
            "branches": branches,
        },
        "metadata": {
            "author": author.strip(),
            "notes": notes.strip(),
        },
    }
    bundle["content_hash_sha256"] = model_content_hash(bundle)
    return bundle


def model_bundle_json(bundle: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(bundle), indent=2, ensure_ascii=False, sort_keys=True)


def load_decision_tree_bundle(data: str | bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        parsed = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersistenceError("The uploaded model file is not valid UTF-8 JSON.") from exc

    if not isinstance(parsed, dict):
        raise PersistenceError("Model file root must be a JSON object.")
    if parsed.get("schema_version") != MODEL_FILE_SCHEMA_VERSION:
        raise PersistenceError(
            f"Unsupported model schema version '{parsed.get('schema_version')}'. "
            f"This build supports {MODEL_FILE_SCHEMA_VERSION}."
        )
    if parsed.get("model_type") != "decision_tree":
        raise PersistenceError("This builder can currently load decision-tree model files only.")

    methods = parsed.get("methods")
    model = parsed.get("model")
    if not isinstance(methods, dict) or not isinstance(model, dict):
        raise PersistenceError("Model file is missing methods or model content.")

    required_methods = {
        "reference_case_code",
        "outcome_code",
        "currency_code",
        "perspective_label",
        "included_cost_bearers",
        "time_horizon",
        "cost_discount_rate",
        "outcome_discount_rate",
    }
    missing_methods = required_methods - set(methods)
    if missing_methods:
        raise PersistenceError(
            "Model file is missing method fields: " + ", ".join(sorted(missing_methods)) + "."
        )

    required_model = {"parameters", "strategies", "nodes", "branches"}
    missing_model = required_model - set(model)
    if missing_model:
        raise PersistenceError(
            "Model file is missing structural fields: " + ", ".join(sorted(missing_model)) + "."
        )

    compile_builder_tables(
        model["parameters"],
        model["strategies"],
        model["nodes"],
        model["branches"],
    )

    stored_hash = parsed.get("content_hash_sha256")
    if stored_hash:
        actual_hash = model_content_hash(parsed)
        if stored_hash != actual_hash:
            raise PersistenceError(
                "Model content hash does not match the file contents. The file may have been edited or corrupted."
            )

    return _json_safe(parsed)


def build_audit_record(
    bundle: Mapping[str, Any],
    *,
    analysis_type: str,
    run_settings: Mapping[str, Any],
    results: Mapping[str, Any],
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    if not analysis_type.strip():
        raise PersistenceError("Audit analysis_type is required.")
    snapshot = _json_safe(deepcopy(dict(bundle)))
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform_version": PLATFORM_VERSION,
        "model_hash_sha256": model_content_hash(snapshot),
        "analysis_type": analysis_type.strip(),
        "run_settings": _json_safe(dict(run_settings)),
        "results": _json_safe(dict(results)),
        "warnings": [str(item) for item in warnings],
        "model_snapshot": snapshot,
    }


def audit_trail_json(records: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": [_json_safe(dict(record)) for record in records],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def rows_to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    rows = [_json_safe(dict(row)) for row in rows]
    if not rows:
        return ""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
