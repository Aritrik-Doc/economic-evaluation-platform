"""Guided clinical-model linkage for Budget Impact Analysis."""

from __future__ import annotations

import streamlit as st

from ui.bia_clinical_linkage_page import render_page


st.set_page_config(
    page_title="Clinical model → BIA linkage",
    page_icon="🔗",
    layout="wide",
)
render_page()
