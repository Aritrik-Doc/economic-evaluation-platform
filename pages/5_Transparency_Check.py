"""Documentation-completeness review for model parameters."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model.transparency import transparency_summary
from ui.design_system import apply_design_system, coloured_block, hero, status_bar


st.set_page_config(page_title="Transparency Check", page_icon="🔎", layout="wide")
apply_design_system()

hero(
    "Transparency check",
    "Review whether the model documents the evidence, assumptions, uncertainty and cost metadata needed for transparent appraisal. This is a documentation-completeness check, not a scientific quality score.",
    eyebrow="Review and reproducibility",
)

WORKSPACES = {
    "Decision Tree": "dt_parameter_rows",
    "Cohort Markov": "markov_parameters",
    "Advanced Markov Dynamics": "adv_parameters",
}

available = [name for name, key in WORKSPACES.items() if key in st.session_state]
if not available:
    coloured_block(
        "No active model found",
        "Open a modelling workspace first so its parameter library is available in this Streamlit session. You can then return here to review documentation completeness.",
        tone="amber",
        kicker="Transparency check",
    )
    st.page_link("pages/1_Decision_Tree_Builder.py", label="Open Decision Tree Modeller →")
    st.page_link("pages/2_Cohort_Markov_Builder.py", label="Open Cohort Markov Modeller →")
    st.page_link("pages/3_Advanced_Markov_Dynamics.py", label="Open Advanced Markov Dynamics →")
    st.stop()

workspace = st.selectbox("Model workspace", available)
rows = st.session_state[WORKSPACES[workspace]]
summary = transparency_summary(rows)

status_bar(
    [
        (f"{summary.total_parameters} parameters", "blue"),
        (f"{summary.complete_parameters} documented", "green"),
        (f"{summary.provisional_parameters} provisional", "amber"),
        (f"{summary.incomplete_parameters} incomplete", "red" if summary.incomplete_parameters else "neutral"),
    ]
)

if not summary.requires_attention and summary.total_parameters:
    coloured_block(
        "Documentation complete for the fields checked",
        "All active parameters contain the documentation fields currently assessed by the Transparency check. This does not establish that the evidence is valid, unbiased, relevant, or sufficient for decision making.",
        tone="green",
        kicker="Current model",
    )
else:
    coloured_block(
        "Documentation needs attention",
        "Review the items below before treating the model as ready for substantive appraisal. Provisional items often contain illustrative/example text that should be replaced with the actual evidence or rationale used in the analysis.",
        tone="amber",
        kicker="Current model",
    )

attention = [item for item in summary.items if item.status != "complete"]
if attention:
    table = pd.DataFrame(
        [
            {
                "Parameter": item.label,
                "Parameter ID": item.parameter_id,
                "Documentation field": item.field.replace("_", " ").title(),
                "Status": item.status.title(),
                "What to address": item.message,
            }
            for item in attention
        ]
    )
    st.subheader("Items requiring review")
    st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.success("No missing or provisional documentation fields were identified.")

with st.expander("What the Transparency check does and does not assess", expanded=True):
    st.markdown(
        """
**It checks documentation completeness**, including whether a parameter has an evidence source, assumption statement and rationale; whether enabled DSA/PSA uncertainty is documented; and whether cost parameters contain currency, price-year and cost-bearer information.

**It does not assess scientific quality.** It does not determine whether a study is unbiased, whether a utility source is appropriate, whether a distribution is statistically correct, whether an assumption is clinically plausible, or whether the overall model is valid for a particular decision problem.

A model can therefore pass the Transparency check and still require substantial critical appraisal.
        """
    )

st.caption(
    "Transparency check is deliberately descriptive rather than scored. Missing documentation is made visible without converting complex methodological judgement into an artificial model-quality rating."
)
