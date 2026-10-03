"""Shared rendering helpers for deterministic policy interpretation."""

from __future__ import annotations

import streamlit as st

from model.policy_interpretation import PolicyInterpretation
from ui.design_system import coloured_block


DOMAIN_LABELS = {
    "value": ("Value for money", "blue"),
    "affordability": ("Affordability", "teal"),
    "feasibility": ("Implementation feasibility", "amber"),
}


def render_policy_interpretation(interpretation: PolicyInterpretation) -> None:
    label, tone = DOMAIN_LABELS[interpretation.domain]
    headlines = [item for item in interpretation.statements if item.kind == "headline"]
    if headlines:
        coloured_block(
            headlines[0].title,
            headlines[0].text,
            tone=tone,
            kicker=label,
        )

    for statement in interpretation.statements:
        if statement.kind == "headline":
            continue
        with st.container(border=True):
            st.markdown(f"**{statement.title}**")
            st.write(statement.text)
            st.caption("Based on: " + " · ".join(statement.basis))

    if interpretation.caveats:
        with st.expander("Important qualifications", expanded=False):
            for caveat in interpretation.caveats:
                st.write("• " + caveat)


def render_combined_headlines(interpretations: tuple[PolicyInterpretation, ...]) -> None:
    order = {"value": 0, "affordability": 1, "feasibility": 2}
    for interpretation in sorted(interpretations, key=lambda item: order[item.domain]):
        label, tone = DOMAIN_LABELS[interpretation.domain]
        headline = next((item for item in interpretation.statements if item.kind == "headline"), None)
        if headline is None:
            continue
        coloured_block(headline.title, headline.text, tone=tone, kicker=label)
