"""Registered Resource & Capacity Planning page with validated-result handoff."""

from __future__ import annotations

from pathlib import Path
import runpy

import streamlit as st


# Never leave a previous feasible result available after the current page becomes invalid.
for key in ("rc_last_result", "rc_last_definition", "rc_last_context"):
    st.session_state.pop(key, None)

# Remove resource mappings whose target resource has since been deleted. Other mapping
# validity is checked by the implementation against the active clinical model.
valid_resource_ids = {str(row.get("id")) for row in st.session_state.get("rc_resources", []) if row.get("id")}
if valid_resource_ids and "rc_clinical_mappings" in st.session_state:
    st.session_state.rc_clinical_mappings = [
        row for row in st.session_state.rc_clinical_mappings
        if str(row.get("resource_id") or "") in valid_resource_ids
    ]

namespace = runpy.run_path(str(Path(__file__).with_name("_8_Resource_Capacity_Planning_impl.py")))

if namespace.get("compile_error") is None and namespace.get("capacity_run") is not None:
    st.session_state.rc_last_result = namespace["capacity_run"]
    st.session_state.rc_last_definition = namespace.get("compiled_definition")
    st.session_state.rc_last_context = {
        "population_source": namespace.get("population_source"),
        "population_source_label": namespace.get("source_label"),
        "requirement_source": namespace.get("requirement_source"),
        "demand_basis": namespace.get("demand_basis"),
    }
