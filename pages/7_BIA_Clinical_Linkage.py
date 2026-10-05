"""Guided clinical-model linkage for Budget Impact Analysis."""

from __future__ import annotations

import streamlit as st

from ui.bia_clinical_linkage_page import render_page


st.set_page_config(
    page_title="Clinical model → BIA linkage",
    page_icon="🔗",
    layout="wide",
)

if st.session_state.get("bia_context_valid") is not True:
    st.title("Clinical model → Budget Impact linkage")
    st.error(
        "A current validated Budget Impact Analysis is required before clinical linkage. "
        "Open BIA and resolve any population, treatment-mix or costing validation messages first."
    )
    st.caption(
        "The linkage page will not reconstruct an older BIA result after the current BIA inputs have become invalid."
    )
    st.page_link("pages/6_Budget_Impact_Analysis.py", label="Open Budget Impact Analysis →", icon="💷")
    st.stop()

render_page()
