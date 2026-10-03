"""Application router and shared product shell for the Economic Evaluation Platform."""

from __future__ import annotations

import streamlit as st

from ui.design_system import apply_design_system


st.set_page_config(
    page_title="Economic Evaluation Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Start": [
        st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
    ],
    "Economic & affordability models": [
        st.Page("pages/1_Decision_Tree_Builder.py", title="Decision Tree Modeller", icon="🌿"),
        st.Page("pages/2_Cohort_Markov_Builder.py", title="Cohort Markov Modeller", icon="🔁"),
        st.Page("pages/3_Advanced_Markov_Dynamics.py", title="Advanced Markov Dynamics", icon="🧭"),
        st.Page("pages/6_Budget_Impact_Analysis.py", title="Budget Impact Analysis", icon="💰"),
    ],
    "Review & reproducibility": [
        st.Page("pages/4_State_Transition_Save_Load_Audit.py", title="Save / Load / Audit", icon="💾"),
        st.Page("pages/5_Transparency_Check.py", title="Transparency Check", icon="🔎"),
    ],
}

current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()
apply_design_system()
