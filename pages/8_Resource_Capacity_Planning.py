"""Registered Resource & Capacity Planning page with validated-result handoff."""

from __future__ import annotations

from pathlib import Path
import runpy

import streamlit as st

from model.capacity_reproducibility import capacity_source_fingerprint


# Never leave a previous feasible result available after the current page becomes invalid.
for key in ("rc_last_result", "rc_last_definition", "rc_last_context", "rc_last_fingerprint"):
    st.session_state.pop(key, None)

# If an upstream reusable source is selected, require its current validated handoff.
# This prevents an older valid population/BIA state from being reused after the
# user has made the current upstream configuration invalid.
selected_population_source = st.session_state.get("rc_population_source")
if selected_population_source == "Shared Population & Uptake":
    if st.session_state.get("pu_context_valid") is not True:
        st.title("Resource & Capacity Planning")
        st.error(
            "The selected shared Population & Uptake scenario is not currently valid. "
            "Resolve its population or treatment-mix validation messages before reusing it here."
        )
        st.page_link("pages/10_Population_Uptake.py", label="Open Population & Uptake →", icon="👥")
        st.stop()
elif selected_population_source == "Budget Impact Analysis":
    if st.session_state.get("bia_context_valid") is not True:
        st.title("Resource & Capacity Planning")
        st.error(
            "The selected Budget Impact Analysis is not currently valid. "
            "Resolve its population, treatment-mix or costing validation messages before using its population here."
        )
        st.page_link("pages/6_Budget_Impact_Analysis.py", label="Open Budget Impact Analysis →", icon="💷")
        st.stop()

# Remove resource mappings whose target resource has since been deleted. Other mapping
# validity is checked by the implementation against the active clinical model.
valid_resource_ids = {
    str(row.get("id"))
    for row in st.session_state.get("rc_resources", [])
    if row.get("id")
}
if valid_resource_ids and "rc_clinical_mappings" in st.session_state:
    st.session_state.rc_clinical_mappings = [
        row
        for row in st.session_state.rc_clinical_mappings
        if str(row.get("resource_id") or "") in valid_resource_ids
    ]

namespace = runpy.run_path(
    str(Path(__file__).with_name("_8_Resource_Capacity_Planning_impl.py"))
)

if namespace.get("compile_error") is None and namespace.get("capacity_run") is not None:
    requirement_source = namespace.get("requirement_source")
    clinical_model_type = (
        st.session_state.get("rc_clinical_model_type")
        if requirement_source in {"Linked clinical model", "Hybrid — clinical + manual"}
        else None
    )
    context = {
        "population_source": namespace.get("population_source"),
        "population_source_label": namespace.get("source_label"),
        "requirement_source": requirement_source,
        "demand_basis": namespace.get("demand_basis"),
        "clinical_model_type": clinical_model_type,
    }
    st.session_state.rc_last_result = namespace["capacity_run"]
    st.session_state.rc_last_definition = namespace.get("compiled_definition")
    st.session_state.rc_last_context = context
    st.session_state.rc_last_fingerprint = capacity_source_fingerprint(
        dict(st.session_state),
        population_source=context["population_source"],
        requirement_source=context["requirement_source"],
        clinical_model_type=context["clinical_model_type"],
    )
