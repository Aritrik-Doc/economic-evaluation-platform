"""Documentation-completeness checks for model transparency.

These checks deliberately assess whether key documentation fields are present.
They do not score scientific quality, evidence validity, risk of bias, or model
credibility. The term "Transparency check" is therefore used throughout the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Any


_PLACEHOLDER_TOKENS = (
    "illustrative",
    "replace with evidence",
    "example input",
    "add the evidence source",
    "complete the modelling rationale",
    "before substantive use",
    "user-entered parameter",
    "user-entered model parameter",
)


@dataclass(frozen=True)
class TransparencyItem:
    parameter_id: str
    label: str
    field: str
    status: str  # complete | provisional | missing
    message: str


@dataclass(frozen=True)
class TransparencySummary:
    total_parameters: int
    complete_parameters: int
    provisional_parameters: int
    incomplete_parameters: int
    items: tuple[TransparencyItem, ...]

    @property
    def requires_attention(self) -> bool:
        return self.provisional_parameters > 0 or self.incomplete_parameters > 0

    @property
    def headline(self) -> str:
        if self.total_parameters == 0:
            return "No parameters available"
        if not self.requires_attention:
            return "Documentation complete"
        return "Documentation needs attention"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_placeholder(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in _PLACEHOLDER_TOKENS)


def _item(pid: str, label: str, field: str, value: Any, message: str, *, allow_placeholder: bool = False) -> TransparencyItem:
    text = _text(value)
    if not text:
        return TransparencyItem(pid, label, field, "missing", message)
    if not allow_placeholder and _is_placeholder(text):
        return TransparencyItem(pid, label, field, "provisional", message)
    return TransparencyItem(pid, label, field, "complete", message)


def parameter_transparency_items(row: Mapping[str, Any]) -> tuple[TransparencyItem, ...]:
    pid = _text(row.get("id")) or "unnamed_parameter"
    label = _text(row.get("label")) or pid
    items = [
        _item(pid, label, "source_citation", row.get("source_citation"), "Document an evidence source or explicitly state that this is a user assumption."),
        _item(pid, label, "assumption", row.get("assumption"), "State the modelling assumption attached to this parameter."),
        _item(pid, label, "assumption_rationale", row.get("assumption_rationale"), "Explain why the modelling assumption is appropriate."),
    ]

    if bool(row.get("dsa_enabled")):
        if row.get("dsa_lower") is None or row.get("dsa_upper") is None:
            items.append(TransparencyItem(pid, label, "dsa_bounds", "missing", "Provide both low and high deterministic sensitivity bounds."))
        else:
            items.append(TransparencyItem(pid, label, "dsa_bounds", "complete", "Deterministic sensitivity bounds documented."))
        items.append(_item(pid, label, "dsa_rationale", row.get("dsa_rationale"), "Explain the evidence or rationale for the DSA range."))

    if bool(row.get("psa_enabled")):
        items.append(_item(pid, label, "distribution_family", row.get("distribution_family"), "Select and document a PSA distribution family.", allow_placeholder=True))
        params = row.get("distribution_parameters")
        if not params:
            items.append(TransparencyItem(pid, label, "distribution_parameters", "missing", "Provide the parameters needed for the selected PSA distribution."))
        else:
            items.append(TransparencyItem(pid, label, "distribution_parameters", "complete", "PSA distribution parameters documented."))
        items.append(_item(pid, label, "psa_rationale", row.get("psa_rationale"), "Explain why the chosen PSA distribution and uncertainty are appropriate."))

    if _text(row.get("category")).lower() == "cost":
        items.append(_item(pid, label, "currency", row.get("currency"), "Document the currency for this cost parameter.", allow_placeholder=True))
        if row.get("price_year") in (None, ""):
            items.append(TransparencyItem(pid, label, "price_year", "missing", "Document the price year for this cost parameter."))
        else:
            items.append(TransparencyItem(pid, label, "price_year", "complete", "Price year documented."))
        items.append(_item(pid, label, "cost_bearers", row.get("cost_bearers"), "Document who bears this cost so perspective can be applied computationally.", allow_placeholder=True))

    return tuple(items)


def transparency_summary(rows: Sequence[Mapping[str, Any]]) -> TransparencySummary:
    all_items: list[TransparencyItem] = []
    complete = provisional = incomplete = 0

    for row in rows:
        if not _text(row.get("id")):
            continue
        items = parameter_transparency_items(row)
        all_items.extend(items)
        statuses = {item.status for item in items}
        if "missing" in statuses:
            incomplete += 1
        elif "provisional" in statuses:
            provisional += 1
        else:
            complete += 1

    total = complete + provisional + incomplete
    return TransparencySummary(
        total_parameters=total,
        complete_parameters=complete,
        provisional_parameters=provisional,
        incomplete_parameters=incomplete,
        items=tuple(all_items),
    )
