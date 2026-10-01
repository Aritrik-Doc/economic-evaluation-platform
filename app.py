"""Streamlit user interface for Economic Evaluation Platform v0.1."""

import streamlit as st

from model.economics import Strategy, evaluate_two_strategies


STATUS_LABELS = {
    "dominant": "Intervention is dominant (less costly, more effective)",
    "dominated": "Intervention is dominated (more costly, less effective)",
    "more_costly_more_effective": "Intervention is more costly and more effective",
    "less_costly_less_effective": "Intervention is less costly and less effective",
    "equal_effect_more_costly": "Equal effect; intervention is more costly",
    "equal_effect_less_costly": "Equal effect; intervention is less costly",
    "equal_cost_more_effective": "Equal cost; intervention is more effective",
    "equal_cost_less_effective": "Equal cost; intervention is less effective",
    "no_difference": "No difference in expected cost or effect",
}


def money(value: float) -> str:
    return f"£{value:,.2f}"


st.set_page_config(page_title="Economic Evaluation Platform", layout="wide")
st.title("Economic Evaluation Platform")
st.caption("Version 0.1 — deterministic two-strategy cost-effectiveness analysis")

with st.sidebar:
    st.header("Analysis settings")
    willingness_to_pay = st.number_input(
        "Willingness-to-pay threshold (£ per QALY)",
        min_value=0.0,
        value=30_000.0,
        step=1_000.0,
    )

left, right = st.columns(2)

with left:
    st.subheader("Comparator")
    comparator_name = st.text_input("Comparator name", value="Standard care")
    comparator_cost = st.number_input("Comparator cost (£)", value=10_000.0, step=100.0)
    comparator_effect = st.number_input(
        "Comparator effect (QALYs)", value=4.00, step=0.01, format="%.4f"
    )

with right:
    st.subheader("Intervention")
    intervention_name = st.text_input("Intervention name", value="New treatment")
    intervention_cost = st.number_input("Intervention cost (£)", value=15_000.0, step=100.0)
    intervention_effect = st.number_input(
        "Intervention effect (QALYs)", value=4.30, step=0.01, format="%.4f"
    )

try:
    comparator = Strategy(comparator_name, comparator_cost, comparator_effect)
    intervention = Strategy(intervention_name, intervention_cost, intervention_effect)
    result = evaluate_two_strategies(comparator, intervention, willingness_to_pay)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.divider()
st.subheader("Base-case results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Incremental cost", money(result.incremental_cost))
m2.metric("Incremental QALYs", f"{result.incremental_effect:,.4f}")

if result.icer is None:
    icer_display = "Not defined"
else:
    icer_display = f"{money(result.icer)}/QALY"
m3.metric("ICER", icer_display)
m4.metric("INMB", money(result.incremental_nmb))

st.info(STATUS_LABELS[result.status])

st.subheader("Net monetary benefit")
nmb_left, nmb_right = st.columns(2)
nmb_left.metric(f"NMB — {result.comparator.name}", money(result.comparator_nmb))
nmb_right.metric(f"NMB — {result.intervention.name}", money(result.intervention_nmb))

if result.preferred_by_nmb == "Tie":
    st.write("At the selected threshold, both strategies have the same NMB.")
else:
    st.write(
        f"At a threshold of **{money(result.willingness_to_pay)}/QALY**, "
        f"**{result.preferred_by_nmb}** has the higher NMB."
    )

with st.expander("Calculation definitions"):
    st.markdown(
        """
- **Incremental cost** = intervention cost − comparator cost
- **Incremental effect** = intervention QALYs − comparator QALYs
- **ICER** = incremental cost ÷ incremental effect, when incremental effect is non-zero
- **NMB** = willingness-to-pay × QALYs − cost
- **INMB** = intervention NMB − comparator NMB

Dominance is shown separately because the sign of an ICER alone can be misleading.
        """
    )
