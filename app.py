"""Application router and shared product shell for the Economic Evaluation Platform."""

from __future__ import annotations

import streamlit as st

from model.capacity_reproducibility import capacity_result_is_current
from ui.design_system import apply_design_system


st.set_page_config(
    page_title="Economic Evaluation Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _invalidate_bia_context() -> None:
    st.session_state["bia_context_valid"] = False
    st.session_state.pop("bia_population_rows", None)
    st.session_state.pop("bia_treatment_mix_rows", None)


def _manual_bia_differs_from_validated_context() -> bool:
    """Detect a new/edited manual snapshot before it is revalidated on the BIA page."""

    if st.session_state.get("bia_population_entry_mode") != "Edit annual values manually":
        return False
    population = st.session_state.get("bia_population_rows") or []
    treatment_mix = st.session_state.get("bia_treatment_mix_rows") or []
    if not population or not treatment_mix:
        return False

    horizon = len(population)
    if st.session_state.get("bia_profile") == "CUSTOM":
        if int(st.session_state.get("bia_horizon_custom", horizon)) != horizon:
            return True

    canonical_ids = {str(row["intervention_id"]) for row in treatment_mix}
    current_ids = {
        str(row.get("id"))
        for row in (st.session_state.get("bia_interventions") or [])
        if row.get("id")
    }
    if current_ids and current_ids != canonical_ids:
        return True

    for index, row in enumerate(population):
        eligible_key = f"bia_eligible_{horizon}_{index}"
        covered_key = f"bia_covered_{horizon}_{index}"
        if eligible_key in st.session_state:
            if abs(float(st.session_state[eligible_key]) - float(row["eligible_population"])) > 1e-9:
                return True
        if covered_key in st.session_state:
            raw_covered = float(st.session_state.get(covered_key, 0.0) or 0.0)
            canonical_covered = float(row.get("covered_lives") or 0.0)
            if abs(raw_covered - canonical_covered) > 1e-9:
                return True
    for row in treatment_mix:
        key = f"bia_share_{row['scenario']}_{int(row['year'])}_{row['intervention_id']}"
        if key in st.session_state:
            if abs(float(st.session_state[key]) - float(row["share"])) > 1e-9:
                return True
    return False


def _sync_validated_bia_legacy_keys() -> None:
    """Mirror the canonical validated BIA export to legacy cross-page widget keys.

    BIA v0.15 publishes canonical population/mix rows. Some older downstream page
    code still reads the historical widget keys, so keep those values synchronized
    only after the BIA itself has validated. If a manual snapshot or manual annual
    edit differs from that canonical export, invalidate the canonical result first
    instead of overwriting the newer manual values with an earlier valid analysis.
    """

    if st.session_state.get("bia_context_valid") is not True:
        return
    if _manual_bia_differs_from_validated_context():
        _invalidate_bia_context()
        return
    population = st.session_state.get("bia_population_rows") or []
    treatment_mix = st.session_state.get("bia_treatment_mix_rows") or []
    if not population or not treatment_mix:
        return
    horizon = len(population)
    for index, row in enumerate(population):
        st.session_state[f"bia_eligible_{horizon}_{index}"] = float(row["eligible_population"])
        st.session_state[f"bia_covered_{horizon}_{index}"] = float(row.get("covered_lives") or 0.0)
    for row in treatment_mix:
        st.session_state[
            f"bia_share_{row['scenario']}_{int(row['year'])}_{row['intervention_id']}"
        ] = float(row["share"])


_sync_validated_bia_legacy_keys()

# Capacity results may depend on shared population inputs and/or a clinical model
# edited on another page. Invalidate the stored result before any page can consume
# it when those substantive upstream inputs no longer match the validated run.
if "rc_last_result" in st.session_state:
    if not capacity_result_is_current(
        dict(st.session_state),
        st.session_state.get("rc_last_context"),
        st.session_state.get("rc_last_fingerprint"),
    ):
        for key in (
            "rc_last_result",
            "rc_last_definition",
            "rc_last_context",
            "rc_last_fingerprint",
        ):
            st.session_state.pop(key, None)

pages = {
    "Start": [
        st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
    ],
    "Economic & affordability models": [
        st.Page("pages/1_Decision_Tree_Builder.py", title="Decision Tree Modeller", icon="🌿"),
        st.Page("pages/2_Cohort_Markov_Builder.py", title="Cohort Markov Modeller", icon="🔁"),
        st.Page("pages/3_Advanced_Markov_Dynamics.py", title="Advanced Markov Dynamics", icon="🧭"),
        st.Page("pages/10_Population_Uptake.py", title="Population & Uptake", icon="👥"),
        st.Page("pages/6_Budget_Impact_Analysis.py", title="Budget Impact Analysis", icon="💰"),
        st.Page("pages/7_BIA_Clinical_Linkage.py", title="Clinical model → BIA linkage", icon="🔗"),
        st.Page("pages/8_Resource_Capacity_Planning.py", title="Resource & Capacity Planning", icon="🏥"),
    ],
    "Interpretation & review": [
        st.Page("pages/9_Policy_Interpretation.py", title="Policy Interpretation", icon="🧾"),
        st.Page("pages/5_Transparency_Check.py", title="Transparency Check", icon="🔎"),
        st.Page("pages/4_State_Transition_Save_Load_Audit.py", title="Save / Load / Audit", icon="💾"),
    ],
}

current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()
apply_design_system()
