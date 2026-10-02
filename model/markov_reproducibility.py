"""Reproducibility guards shared by the cohort Markov UI and tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


class MarkovReproducibilityError(ValueError):
    pass


def validate_analysis_currency(
    parameter_rows: Sequence[Mapping[str, Any]],
    analysis_currency: str,
) -> None:
    """Require all cost inputs to already be in the selected analysis currency.

    v0.5 does not silently perform FX conversion. This guard prevents a change in
    the UI's analysis-currency label from re-labelling numeric cost inputs without
    an explicit conversion step and provenance.
    """
    analysis_currency = analysis_currency.strip().upper()
    mismatched: list[str] = []
    missing: list[str] = []
    for row in parameter_rows:
        if str(row.get("category") or "").strip().lower() != "cost":
            continue
        parameter_id = str(row.get("id") or "unnamed cost parameter").strip()
        currency = str(row.get("currency") or "").strip().upper()
        if not currency:
            missing.append(parameter_id)
        elif currency != analysis_currency:
            mismatched.append(f"{parameter_id} ({currency})")

    if missing:
        raise MarkovReproducibilityError(
            "Cost parameter currency is missing for: " + ", ".join(missing) + "."
        )
    if mismatched:
        raise MarkovReproducibilityError(
            "The selected analysis currency is "
            + analysis_currency
            + ", but these cost parameters use a different currency: "
            + ", ".join(mismatched)
            + ". Automatic FX conversion is not applied in v0.5. Convert the values explicitly, document the rate/date/source, then update the parameter currency."
        )


def markov_run_fingerprint(
    *,
    parameter_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    strategy_rows: Sequence[Mapping[str, Any]],
    initial_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    state_reward_rows: Sequence[Mapping[str, Any]],
    transition_reward_rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> str:
    """Hash the substantive model and run settings used by a PSA result."""
    payload = {
        "parameters": list(parameter_rows),
        "states": list(state_rows),
        "strategies": list(strategy_rows),
        "initial": list(initial_rows),
        "transitions": list(transition_rows),
        "state_rewards": list(state_reward_rows),
        "transition_rewards": list(transition_reward_rows),
        "settings": dict(settings),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
