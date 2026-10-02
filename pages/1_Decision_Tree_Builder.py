"""Guided decision-tree modeller for Economic Evaluation Platform v0.4.1."""

from __future__ import annotations

import hashlib
import json
from math import exp, sqrt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import TreeLayout
from streamlit_flow.state import StreamlitFlowState

from model.currency import CURRENCIES
from model.decision_tree import DecisionTreeValidationError, run_decision_tree
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.guided_tree import (
    GuidedTreeError,
    add_branch_with_child,
    add_strategy,
    append_reward,
    delete_leaf_node,
    slugify,
    unique_id,
    update_node,
)
from model.parameterisation import (
    DISTRIBUTION_PARAMETERISATIONS,
    ParameterisationError,
    canonical_distribution_parameters,
    distribution_parameters_text,
    migrate_parameter_row,
    parse_legacy_distribution_parameters,
)
from model.persistence import (
    PersistenceError,
    audit_trail_json,
    build_audit_record,
    build_decision_tree_bundle,
    load_decision_tree_bundle,
    model_bundle_json,
    rows_to_csv,
)
from model.psa import (
    PSAConfigurationError,
    ceac,
    incremental_plane,
    pairwise_probability_cost_effective,
    run_tree_psa,
)
from model.reference_cases import REFERENCE_CASES
from model.sensitivity import OneWaySensitivitySpec, ThresholdAnalysisSpec, TwoWaySensitivitySpec
from model.tree_builder import BuilderValidationError, compile_builder_tables
from model.tree_sensitivity import (
    one_way_tree_inmb,
    threshold_tree_inmb,
    tornado_tree_inmb,
    two_way_tree_inmb,
)
from model.uncertainty_defaults import suggest_distribution_from_fields


st.set_page_config(page_title="Decision Tree Modeller", layout="wide")
st.title("Decision Tree Modeller")
st.caption("Version 0.4.1 — guided visual construction with independent DSA and PSA settings")

SOURCE_TYPES = [
    "systematic_review", "meta_analysis", "randomised_trial", "observational_study",
    "registry", "database", "tariff", "cost_study", "expert_elicitation",
    "guideline", "user_assumption", "other",
]
PARAMETER_CATEGORIES = ["clinical", "cost", "utility", "resource_use", "survival", "epidemiology", "other"]
DISTRIBUTION_FAMILIES = ["beta", "gamma", "lognormal", "normal", "uniform", "dirichlet"]
COST_BEARER_OPTIONS = [
    "health_system", "personal_social_services", "patient_direct_medical",
    "patient_direct_non_medical", "productivity", "carer", "custom",
]


def money(value: float, symbol: str) -> str:
    return f"{symbol}{value:,.2f}"


def parse_threshold(raw: str) -> float | None:
    text = raw.replace(",", "").strip()
    if not text:
        return None
    value = float(text)
    if value < 0:
        raise ValueError("Decision threshold cannot be negative.")
    return value


def default_parameter_rows() -> list[dict]:
    common = {
        "source_type": "user_assumption",
        "source_citation": "Illustrative example input — replace with evidence",
        "source_url": "",
        "publication_year": None,
        "source_details": "",
        "assumption": "Illustrative value for demonstration.",
        "assumption_rationale": "Replace with a documented modelling assumption before substantive use.",
        "dsa_enabled": True,
        "dsa_rationale": "Illustrative deterministic range; replace with evidence-based bounds.",
        "psa_enabled": True,
        "psa_rationale": "Illustrative probability distribution; replace with evidence-based uncertainty.",
        "correlation_group": "",
        "currency": "",
        "price_year": None,
        "cost_bearers": "",
        "notes": "",
    }
    rows = [
        {
            **common, "id": "p_success_a", "label": "Probability of success — A", "value": 0.8,
            "unit": "probability", "category": "clinical", "dsa_lower": 0.65, "dsa_upper": 0.90,
            "distribution_family": "beta", "distribution_parameterisation": "Alpha + Beta",
            "distribution_parameters": {"alpha": 8.0, "beta": 2.0},
        },
        {
            **common, "id": "p_success_b", "label": "Probability of success — B", "value": 0.6,
            "unit": "probability", "category": "clinical", "dsa_lower": 0.45, "dsa_upper": 0.75,
            "distribution_family": "beta", "distribution_parameterisation": "Alpha + Beta",
            "distribution_parameters": {"alpha": 6.0, "beta": 4.0},
        },
    ]
    for pid, label, value, low, high in [
        ("cost_a", "Strategy A acquisition cost", 1000.0, 800.0, 1200.0),
        ("cost_b", "Strategy B acquisition cost", 500.0, 400.0, 600.0),
        ("cost_success", "Cost after success", 100.0, 80.0, 120.0),
        ("cost_failure", "Cost after failure", 1000.0, 800.0, 1200.0),
    ]:
        rows.append(
            {
                **common, "id": pid, "label": label, "value": value, "unit": "currency",
                "category": "cost", "dsa_lower": low, "dsa_upper": high,
                "distribution_family": "gamma", "distribution_parameterisation": "Mean + SD",
                "distribution_parameters": {"shape": 25.0, "scale": value / 25.0},
                "currency": "GBP", "price_year": 2026, "cost_bearers": "health_system",
            }
        )
    for pid, label, value, low, high, sd in [
        ("outcome_success", "Outcome after success", 2.0, 1.7, 2.2, 0.15),
        ("outcome_failure", "Outcome after failure", 1.0, 0.8, 1.2, 0.10),
    ]:
        rows.append(
            {
                **common, "id": pid, "label": label, "value": value, "unit": "QALY",
                "category": "utility", "dsa_lower": low, "dsa_upper": high,
                "distribution_family": "normal", "distribution_parameterisation": "Mean + SD",
                "distribution_parameters": {"mean": value, "sd": sd},
            }
        )
    return rows


DEFAULT_STRATEGIES = [
    {"strategy_id": "A", "strategy_name": "Strategy A", "root_node_id": "a_root"},
    {"strategy_id": "B", "strategy_name": "Strategy B", "root_node_id": "b_root"},
]
DEFAULT_NODES = [
    {"id": "a_root", "label": "Does the patient respond?", "type": "chance", "cost_rewards": "cost_a@0", "outcome_rewards": ""},
    {"id": "a_success", "label": "Response", "type": "terminal", "cost_rewards": "cost_success@1", "outcome_rewards": "outcome_success@1"},
    {"id": "a_failure", "label": "No response", "type": "terminal", "cost_rewards": "cost_failure@1", "outcome_rewards": "outcome_failure@1"},
    {"id": "b_root", "label": "Does the patient respond?", "type": "chance", "cost_rewards": "cost_b@0", "outcome_rewards": ""},
    {"id": "b_success", "label": "Response", "type": "terminal", "cost_rewards": "cost_success@1", "outcome_rewards": "outcome_success@1"},
    {"id": "b_failure", "label": "No response", "type": "terminal", "cost_rewards": "cost_failure@1", "outcome_rewards": "outcome_failure@1"},
]
DEFAULT_BRANCHES = [
    {"from_node": "a_root", "label": "Responds", "probability_parameter_id": "p_success_a", "probability_mode": "direct", "to_node": "a_success"},
    {"from_node": "a_root", "label": "Does not respond", "probability_parameter_id": "p_success_a", "probability_mode": "complement", "to_node": "a_failure"},
    {"from_node": "b_root", "label": "Responds", "probability_parameter_id": "p_success_b", "probability_mode": "direct", "to_node": "b_success"},
    {"from_node": "b_root", "label": "Does not respond", "probability_parameter_id": "p_success_b", "probability_mode": "complement", "to_node": "b_failure"},
]


def stable_signature(*values) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def distribution_defaults(family: str, params: dict[str, float], base: float, mode: str) -> dict[str, float]:
    params = params or {}
    if family == "beta":
        alpha = max(float(params.get("alpha", max(base * 100, 1))), 1e-9)
        beta = max(float(params.get("beta", max((1 - base) * 100, 1))), 1e-9)
        total = alpha + beta
        mean = alpha / total
        se = sqrt(alpha * beta / (total * total * (total + 1)))
        return {"alpha": alpha, "beta": beta, "mean": mean, "se": se}
    if family == "gamma":
        shape = max(float(params.get("shape", 25.0)), 1e-9)
        scale = max(float(params.get("scale", max(abs(base), 1.0) / shape)), 1e-9)
        return {"shape": shape, "scale": scale, "mean": shape * scale, "sd": sqrt(shape) * scale}
    if family == "normal":
        mean = float(params.get("mean", base))
        sd = max(float(params.get("sd", max(abs(base) * 0.1, 0.1))), 1e-9)
        return {"mean": mean, "sd": sd, "estimate": mean, "lower_ci": mean - 1.96 * sd, "upper_ci": mean + 1.96 * sd}
    if family == "lognormal":
        meanlog = float(params.get("meanlog", np.log(base) if base > 0 else 0.0))
        sdlog = max(float(params.get("sdlog", 0.1)), 1e-9)
        arithmetic_mean = exp(meanlog + sdlog * sdlog / 2)
        arithmetic_sd = sqrt((exp(sdlog * sdlog) - 1) * exp(2 * meanlog + sdlog * sdlog))
        return {"meanlog": meanlog, "sdlog": sdlog, "mean": arithmetic_mean, "sd": arithmetic_sd}
    if family == "uniform":
        width = max(abs(base) * 0.2, 0.1)
        return {"low": float(params.get("low", base - width)), "high": float(params.get("high", base + width))}
    if family == "dirichlet":
        return {"alpha": max(float(params.get("alpha", max(base * 100, 1))), 1e-9)}
    return {}


def flow_objects(strategies: list[dict], nodes: list[dict], branches: list[dict]):
    flow_nodes = []
    flow_edges = []
    node_map = {str(row["id"]): row for row in nodes}
    for strategy in strategies:
        sid = str(strategy["strategy_id"])
        visual_id = f"strategy::{sid}"
        flow_nodes.append(
            StreamlitFlowNode(
                id=visual_id, pos=(0, 0), data={"content": f"**Strategy**\n{strategy['strategy_name']}"},
                node_type="input", source_position="right",
            )
        )
        flow_edges.append(
            StreamlitFlowEdge(
                id=f"{visual_id}->{strategy['root_node_id']}", source=visual_id,
                target=str(strategy["root_node_id"]), marker_end={"type": "arrowclosed"},
            )
        )
    for row in nodes:
        node_id = str(row["id"])
        kind = str(row.get("type", "chance")).lower()
        icon = "◆" if kind == "chance" else "●"
        subtitle = "Chance event" if kind == "chance" else "Terminal outcome"
        flow_nodes.append(
            StreamlitFlowNode(
                id=node_id, pos=(0, 0),
                data={"content": f"{icon} **{row.get('label', node_id)}**\n\n{subtitle}"},
                node_type="default", source_position="right", target_position="left",
            )
        )
    for index, row in enumerate(branches):
        probability = str(row.get("probability_parameter_id") or "")
        mode = str(row.get("probability_mode") or "direct")
        p_text = f"1−{probability}" if mode == "complement" else probability
        flow_edges.append(
            StreamlitFlowEdge(
                id=f"branch::{index}::{row.get('from_node')}->{row.get('to_node')}",
                source=str(row.get("from_node")), target=str(row.get("to_node")),
                marker_end={"type": "arrowclosed"}, label=f"{row.get('label', '')}  [{p_text}]",
            )
        )
    return flow_nodes, flow_edges


def reset_analysis_state():
    for key in ["dt_psa_result", "dt_psa_signature", "dt_audit_trail"]:
        st.session_state.pop(key, None)


def initialise_workspace(bundle: dict | None = None):
    model = bundle.get("model", {}) if bundle else {}
    st.session_state["dt_parameter_rows"] = [migrate_parameter_row(row) for row in model.get("parameters", default_parameter_rows())]
    st.session_state["dt_strategy_rows"] = [dict(row) for row in model.get("strategies", DEFAULT_STRATEGIES)]
    st.session_state["dt_node_rows"] = [dict(row) for row in model.get("nodes", DEFAULT_NODES)]
    st.session_state["dt_branch_rows"] = [dict(row) for row in model.get("branches", DEFAULT_BRANCHES)]
    st.session_state["dt_workspace_revision"] = st.session_state.get("dt_workspace_revision", 0) + 1
    st.session_state.pop("dt_flow_state", None)
    st.session_state.pop("dt_flow_signature", None)
    reset_analysis_state()


if "dt_workspace_revision" not in st.session_state:
    st.session_state["dt_workspace_revision"] = 0
if "dt_audit_trail" not in st.session_state:
    st.session_state["dt_audit_trail"] = []
if "dt_parameter_rows" not in st.session_state:
    initialise_workspace()

with st.expander("Open, reset, or resume a model", expanded=False):
    uploaded_model = st.file_uploader("Upload a saved decision-tree model (.json)", type=["json"], key="dt_model_upload_v041")
    c1, c2 = st.columns(2)
    if c1.button("Load uploaded model", disabled=uploaded_model is None):
        try:
            bundle = load_decision_tree_bundle(uploaded_model.getvalue())
        except (PersistenceError, BuilderValidationError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["dt_loaded_bundle"] = bundle
            methods = bundle["methods"]
            st.session_state["dt_reference_case"] = methods["reference_case_code"]
            st.session_state["dt_outcome_code"] = methods["outcome_code"]
            st.session_state["dt_threshold_text"] = "" if methods.get("threshold") is None else str(methods.get("threshold"))
            st.session_state["dt_custom_currency"] = methods["currency_code"]
            st.session_state["dt_custom_perspective"] = methods["perspective_label"]
            st.session_state["dt_custom_cost_bearers"] = methods["included_cost_bearers"]
            st.session_state["dt_custom_horizon"] = methods["time_horizon"]
            st.session_state["dt_custom_cost_discount"] = methods["cost_discount_rate"] * 100
            st.session_state["dt_custom_outcome_discount"] = methods["outcome_discount_rate"] * 100
            initialise_workspace(bundle)
            st.rerun()
    if c2.button("Start new example model"):
        st.session_state.pop("dt_loaded_bundle", None)
        for key in [
            "dt_reference_case", "dt_outcome_code", "dt_threshold_text", "dt_custom_currency",
            "dt_custom_perspective", "dt_custom_cost_bearers", "dt_custom_horizon",
            "dt_custom_cost_discount", "dt_custom_outcome_discount",
        ]:
            st.session_state.pop(key, None)
        initialise_workspace()
        st.rerun()

revision = int(st.session_state["dt_workspace_revision"])

with st.sidebar:
    st.header("Methods")
    reference_case_code = st.selectbox(
        "Reference case", ["NICE_TA", "HTAIN_2023", "CUSTOM"],
        format_func=lambda code: REFERENCE_CASES[code].name if code in REFERENCE_CASES else "Custom",
        key="dt_reference_case",
    )
    if reference_case_code in REFERENCE_CASES:
        profile = REFERENCE_CASES[reference_case_code]
        outcome_code = st.selectbox(
            "Economic outcome", list(OUTCOME_MEASURES),
            index=list(OUTCOME_MEASURES).index(profile.preferred_outcome_code),
            format_func=lambda code: OUTCOME_MEASURES[code].label, key="dt_outcome_code",
        )
        currency_code = profile.analysis_currency
        st.text_input("Analysis currency", currency_code, disabled=True)
        perspective_label = profile.perspective.label
        included_cost_bearers = list(profile.perspective.included_cost_bearers)
        time_horizon = profile.time_horizon_rule
        cost_discount_rate = profile.cost_discount_rate
        outcome_discount_rate = profile.outcome_discount_rate
        st.write(f"**Perspective:** {perspective_label}")
        if profile.threshold_range and outcome_code == profile.threshold_range.outcome_code:
            default_threshold = str(int(profile.threshold_range.lower))
            st.caption(
                f"Reference range: {CURRENCIES[currency_code].symbol}{profile.threshold_range.lower:,.0f}–"
                f"{CURRENCIES[currency_code].symbol}{profile.threshold_range.upper:,.0f} per {OUTCOME_MEASURES[outcome_code].unit}."
            )
        else:
            default_threshold = ""
        threshold_raw = st.text_input("Decision threshold", default_threshold, key="dt_threshold_text")
        status = profile.outcome_status(outcome_code)
        if status in {"conditional", "supplementary", "not_specified"}:
            st.warning(f"{OUTCOME_MEASURES[outcome_code].label} is {status.replace('_', ' ')} for this reference case.")
    else:
        outcome_code = st.selectbox("Economic outcome", list(OUTCOME_MEASURES), format_func=lambda code: OUTCOME_MEASURES[code].label, key="dt_outcome_code")
        currency_code = st.selectbox("Analysis currency", list(CURRENCIES), format_func=lambda code: f"{code} — {CURRENCIES[code].name}", key="dt_custom_currency")
        perspective_label = st.text_input("Perspective", "Healthcare payer", key="dt_custom_perspective")
        included_cost_bearers = st.multiselect("Include cost bearers", COST_BEARER_OPTIONS, default=["health_system"], key="dt_custom_cost_bearers")
        time_horizon = st.text_input("Time horizon", "Lifetime", key="dt_custom_horizon")
        cost_discount_rate = st.number_input("Cost discount rate (%)", 0.0, 99.0, 3.5, key="dt_custom_cost_discount") / 100
        outcome_discount_rate = st.number_input("Outcome discount rate (%)", 0.0, 99.0, 3.5, key="dt_custom_outcome_discount") / 100
        threshold_raw = st.text_input("Decision threshold", "30000", key="dt_threshold_text")
    st.caption("DSA and PSA are configured independently for each parameter. The analysis selector changes the output view; it does not overwrite parameter uncertainty.")

try:
    threshold = parse_threshold(threshold_raw)
except ValueError as exc:
    st.error(str(exc))
    threshold = None
outcome = OUTCOME_MEASURES[outcome_code]
currency = CURRENCIES[currency_code]

model_tab, parameter_tab, result_tab, analyse_tab, files_tab = st.tabs(
    ["1 · Model design", "2 · Parameters", "3 · Base case", "4 · Analyse", "5 · Save / export"]
)

# ---------------------------------------------------------------------------
# Parameters: guided cards rather than a spreadsheet-first workflow
# ---------------------------------------------------------------------------
with parameter_tab:
    st.subheader("Parameter library")
    st.write("Each parameter stores its base value, deterministic range, probabilistic distribution, and evidence separately. Configure DSA and PSA once; you can switch analyses later without changing the parameter.")
    parameter_rows = st.session_state["dt_parameter_rows"]
    delete_parameter_id = None

    for idx, original in enumerate(parameter_rows):
        row = migrate_parameter_row(original)
        pid = str(row.get("id"))
        dsa_badge = "DSA ✓" if row.get("dsa_enabled") else "DSA —"
        psa_badge = "PSA ✓" if row.get("psa_enabled") else "PSA —"
        with st.expander(f"{row.get('label', pid)}  ·  {row.get('value')} {row.get('unit', '')}  ·  {dsa_badge}  ·  {psa_badge}", expanded=False):
            base_tab, dsa_tab, psa_param_tab, evidence_tab = st.tabs(["Base case", "DSA", "PSA", "Evidence & assumptions"])
            prefix = f"p_{revision}_{pid}"
            with base_tab:
                b1, b2 = st.columns(2)
                row["label"] = b1.text_input("Parameter name", str(row.get("label", "")), key=f"{prefix}_label")
                row["value"] = b2.number_input("Base-case value", value=float(row.get("value", 0.0)), format="%.8g", key=f"{prefix}_value")
                b3, b4 = st.columns(2)
                row["unit"] = b3.text_input("Unit", str(row.get("unit", "")), key=f"{prefix}_unit")
                current_category = str(row.get("category", "other"))
                row["category"] = b4.selectbox("Category", PARAMETER_CATEGORIES, index=PARAMETER_CATEGORIES.index(current_category) if current_category in PARAMETER_CATEGORIES else len(PARAMETER_CATEGORIES)-1, key=f"{prefix}_category")
                st.caption(f"Stable parameter ID: `{pid}`")
                if row["category"] == "cost":
                    c1, c2 = st.columns(2)
                    row["currency"] = c1.text_input("Currency code", str(row.get("currency") or currency_code), key=f"{prefix}_currency").upper()
                    row["price_year"] = int(c2.number_input("Price year", min_value=1900, max_value=2200, value=int(row.get("price_year") or 2026), step=1, key=f"{prefix}_priceyear"))
                    row["cost_bearers"] = st.text_input("Cost bearer(s), comma separated", str(row.get("cost_bearers") or "health_system"), key=f"{prefix}_bearers")
                if st.button("Delete parameter", key=f"{prefix}_delete"):
                    delete_parameter_id = pid
            with dsa_tab:
                row["dsa_enabled"] = st.toggle("Include this parameter in deterministic sensitivity analysis", value=bool(row.get("dsa_enabled")), key=f"{prefix}_dsa_on")
                if row["dsa_enabled"]:
                    d1, d2 = st.columns(2)
                    row["dsa_lower"] = d1.number_input("Low value", value=float(row.get("dsa_lower") if row.get("dsa_lower") is not None else row["value"] * 0.8), format="%.8g", key=f"{prefix}_dsa_low")
                    row["dsa_upper"] = d2.number_input("High value", value=float(row.get("dsa_upper") if row.get("dsa_upper") is not None else row["value"] * 1.2), format="%.8g", key=f"{prefix}_dsa_high")
                    row["dsa_rationale"] = st.text_area("Why are these bounds appropriate?", str(row.get("dsa_rationale") or ""), key=f"{prefix}_dsa_rationale")
                else:
                    row["dsa_lower"] = None
                    row["dsa_upper"] = None
                    row["dsa_rationale"] = st.text_area("Why is DSA not represented for this parameter?", str(row.get("dsa_rationale") or "Not included in DSA."), key=f"{prefix}_dsa_rationale_off")
            with psa_param_tab:
                row["psa_enabled"] = st.toggle("Include this parameter in probabilistic sensitivity analysis", value=bool(row.get("psa_enabled")), key=f"{prefix}_psa_on")
                suggestion = suggest_distribution_from_fields(label=str(row.get("label") or pid), category=str(row.get("category") or "other"), unit=str(row.get("unit") or ""), value=float(row.get("value", 0.0)))
                st.info(f"Suggested starting family: **{suggestion.family.title()}** — {suggestion.rationale}")
                if suggestion.caution:
                    st.caption("Caution: " + suggestion.caution)
                if row["psa_enabled"]:
                    existing_family = str(row.get("distribution_family") or suggestion.family).lower()
                    if existing_family not in DISTRIBUTION_FAMILIES:
                        existing_family = suggestion.family if suggestion.family in DISTRIBUTION_FAMILIES else "normal"
                    family = st.selectbox("Distribution family", DISTRIBUTION_FAMILIES, index=DISTRIBUTION_FAMILIES.index(existing_family), key=f"{prefix}_family")
                    row["distribution_family"] = family
                    modes = DISTRIBUTION_PARAMETERISATIONS[family]
                    existing_mode = str(row.get("distribution_parameterisation") or modes[0])
                    if existing_mode not in modes:
                        existing_mode = modes[0]
                    mode = st.selectbox("How would you like to enter the distribution?", modes, index=modes.index(existing_mode), key=f"{prefix}_parammode")
                    row["distribution_parameterisation"] = mode
                    existing_params = parse_legacy_distribution_parameters(row.get("distribution_parameters"))
                    defaults = distribution_defaults(family, existing_params, float(row["value"]), mode)
                    values = {}
                    if family == "beta" and mode == "Alpha + Beta":
                        p1, p2 = st.columns(2); values["alpha"] = p1.number_input("Alpha", min_value=1e-9, value=float(defaults["alpha"]), format="%.8g", key=f"{prefix}_alpha"); values["beta"] = p2.number_input("Beta", min_value=1e-9, value=float(defaults["beta"]), format="%.8g", key=f"{prefix}_beta")
                    elif family == "beta":
                        p1, p2 = st.columns(2); values["mean"] = p1.number_input("Mean", min_value=1e-9, max_value=1-1e-9, value=float(defaults["mean"]), format="%.8g", key=f"{prefix}_bmean"); values["se"] = p2.number_input("Standard error", min_value=1e-9, value=float(defaults["se"]), format="%.8g", key=f"{prefix}_bse")
                    elif family == "gamma" and mode == "Shape + Scale":
                        p1, p2 = st.columns(2); values["shape"] = p1.number_input("Shape", min_value=1e-9, value=float(defaults["shape"]), format="%.8g", key=f"{prefix}_shape"); values["scale"] = p2.number_input("Scale", min_value=1e-9, value=float(defaults["scale"]), format="%.8g", key=f"{prefix}_scale")
                    elif family == "gamma":
                        p1, p2 = st.columns(2); values["mean"] = p1.number_input("Mean", min_value=1e-9, value=float(defaults["mean"]), format="%.8g", key=f"{prefix}_gmean"); values["sd"] = p2.number_input("Standard deviation", min_value=1e-9, value=float(defaults["sd"]), format="%.8g", key=f"{prefix}_gsd")
                    elif family == "normal" and mode == "Mean + SD":
                        p1, p2 = st.columns(2); values["mean"] = p1.number_input("Mean", value=float(defaults["mean"]), format="%.8g", key=f"{prefix}_nmean"); values["sd"] = p2.number_input("Standard deviation", min_value=1e-9, value=float(defaults["sd"]), format="%.8g", key=f"{prefix}_nsd")
                    elif family == "normal":
                        p1, p2, p3 = st.columns(3); values["estimate"] = p1.number_input("Estimate", value=float(defaults["estimate"]), format="%.8g", key=f"{prefix}_nest"); values["lower_ci"] = p2.number_input("Lower 95% CI", value=float(defaults["lower_ci"]), format="%.8g", key=f"{prefix}_nlci"); values["upper_ci"] = p3.number_input("Upper 95% CI", value=float(defaults["upper_ci"]), format="%.8g", key=f"{prefix}_nuci")
                    elif family == "lognormal" and mode == "Meanlog + SDlog":
                        p1, p2 = st.columns(2); values["meanlog"] = p1.number_input("Mean on log scale", value=float(defaults["meanlog"]), format="%.8g", key=f"{prefix}_meanlog"); values["sdlog"] = p2.number_input("SD on log scale", min_value=1e-9, value=float(defaults["sdlog"]), format="%.8g", key=f"{prefix}_sdlog")
                    elif family == "lognormal":
                        p1, p2 = st.columns(2); values["mean"] = p1.number_input("Arithmetic mean", min_value=1e-9, value=float(defaults["mean"]), format="%.8g", key=f"{prefix}_lmean"); values["sd"] = p2.number_input("Arithmetic SD", min_value=1e-9, value=float(defaults["sd"]), format="%.8g", key=f"{prefix}_lsd")
                    elif family == "uniform":
                        p1, p2 = st.columns(2); values["low"] = p1.number_input("Minimum", value=float(defaults["low"]), format="%.8g", key=f"{prefix}_ulow"); values["high"] = p2.number_input("Maximum", value=float(defaults["high"]), format="%.8g", key=f"{prefix}_uhigh")
                    elif family == "dirichlet":
                        values["alpha"] = st.number_input("Alpha concentration for this component", min_value=1e-9, value=float(defaults["alpha"]), format="%.8g", key=f"{prefix}_dalpha")
                        st.caption("All components in the same Dirichlet group are sampled jointly and sum to 1 in every simulation.")
                    try:
                        row["distribution_parameters"] = canonical_distribution_parameters(family, mode, values, base_value=float(row["value"]))
                    except ParameterisationError as exc:
                        st.error(str(exc))
                    row["correlation_group"] = st.text_input("Correlation / joint-sampling group (optional)", str(row.get("correlation_group") or ""), key=f"{prefix}_corr")
                    row["psa_rationale"] = st.text_area("Why is this distribution appropriate?", str(row.get("psa_rationale") or ""), key=f"{prefix}_psa_rationale")
                    st.caption("Internal canonical parameters: " + distribution_parameters_text(row.get("distribution_parameters") or {}))
                else:
                    row["distribution_family"] = ""
                    row["distribution_parameterisation"] = ""
                    row["distribution_parameters"] = {}
                    row["correlation_group"] = ""
                    row["psa_rationale"] = st.text_area("Why is PSA not represented for this parameter?", str(row.get("psa_rationale") or "Not included in PSA."), key=f"{prefix}_psa_rationale_off")
            with evidence_tab:
                e1, e2 = st.columns(2)
                row["source_type"] = e1.selectbox("Source type", SOURCE_TYPES, index=SOURCE_TYPES.index(str(row.get("source_type") or "other")) if str(row.get("source_type") or "other") in SOURCE_TYPES else len(SOURCE_TYPES)-1, key=f"{prefix}_stype")
                publication = row.get("publication_year")
                row["publication_year"] = int(e2.number_input("Publication year", min_value=1800, max_value=2200, value=int(publication or 2026), step=1, key=f"{prefix}_year"))
                row["source_citation"] = st.text_input("Citation / source", str(row.get("source_citation") or ""), key=f"{prefix}_citation")
                row["source_url"] = st.text_input("Source URL (optional)", str(row.get("source_url") or ""), key=f"{prefix}_url")
                row["source_details"] = st.text_area("Source details (optional)", str(row.get("source_details") or ""), key=f"{prefix}_sdetails")
                row["assumption"] = st.text_area("Assumption statement", str(row.get("assumption") or ""), key=f"{prefix}_assumption")
                row["assumption_rationale"] = st.text_area("Assumption rationale", str(row.get("assumption_rationale") or ""), key=f"{prefix}_arationale")
                row["notes"] = st.text_area("Notes", str(row.get("notes") or ""), key=f"{prefix}_notes")
        parameter_rows[idx] = row

    if delete_parameter_id:
        used_text = json.dumps([st.session_state["dt_node_rows"], st.session_state["dt_branch_rows"]])
        if delete_parameter_id in used_text:
            st.error("This parameter is currently referenced by the tree. Remove those references before deleting it.")
        else:
            st.session_state["dt_parameter_rows"] = [row for row in parameter_rows if str(row.get("id")) != delete_parameter_id]
            reset_analysis_state(); st.rerun()
    else:
        st.session_state["dt_parameter_rows"] = parameter_rows

    with st.form("add_parameter_form"):
        st.markdown("#### Add parameter")
        a1, a2, a3 = st.columns(3)
        new_label = a1.text_input("Name")
        new_value = a2.number_input("Base value", value=0.0, format="%.8g")
        new_category = a3.selectbox("Category", PARAMETER_CATEGORIES)
        new_unit = st.text_input("Unit", value="probability" if new_category == "clinical" else "unit")
        if st.form_submit_button("Add parameter"):
            if not new_label.strip():
                st.error("Enter a parameter name.")
            else:
                existing = [str(row.get("id")) for row in parameter_rows]
                new_id = unique_id(slugify(new_label, "parameter"), existing)
                parameter_rows.append(
                    {
                        "id": new_id, "label": new_label.strip(), "value": new_value, "unit": new_unit,
                        "category": new_category, "source_type": "user_assumption",
                        "source_citation": "Source not yet entered", "source_url": "", "publication_year": 2026,
                        "source_details": "", "assumption": "Assumption not yet documented",
                        "assumption_rationale": "Complete before final analysis", "dsa_enabled": False,
                        "dsa_lower": None, "dsa_upper": None, "dsa_rationale": "Not included in DSA.",
                        "psa_enabled": False, "psa_rationale": "Not included in PSA.",
                        "distribution_family": "", "distribution_parameterisation": "", "distribution_parameters": {},
                        "correlation_group": "", "currency": currency_code if new_category == "cost" else "",
                        "price_year": 2026 if new_category == "cost" else None,
                        "cost_bearers": "health_system" if new_category == "cost" else "", "notes": "",
                    }
                )
                st.session_state["dt_parameter_rows"] = parameter_rows
                reset_analysis_state(); st.rerun()

# ---------------------------------------------------------------------------
# Guided model design and constrained canvas
# ---------------------------------------------------------------------------
with model_tab:
    st.subheader("Build the decision tree")
    st.write("A **chance event** represents something uncertain that can lead to different pathways. A **terminal outcome** ends a pathway. The canvas is deliberately constrained: drag nodes to organise the view, but create branches through the guided controls so invalid freehand connections are not introduced.")
    strategies = st.session_state["dt_strategy_rows"]
    nodes = st.session_state["dt_node_rows"]
    branches = st.session_state["dt_branch_rows"]
    structural_signature = stable_signature(strategies, nodes, branches)
    if st.session_state.get("dt_flow_signature") != structural_signature:
        flow_nodes, flow_edges = flow_objects(strategies, nodes, branches)
        st.session_state["dt_flow_state"] = StreamlitFlowState(flow_nodes, flow_edges)
        st.session_state["dt_flow_signature"] = structural_signature
    st.session_state["dt_flow_state"] = streamlit_flow(
        "guided_decision_tree_canvas",
        st.session_state["dt_flow_state"],
        layout=TreeLayout(direction="right"), fit_view=True, height=520,
        enable_node_menu=False, enable_edge_menu=False, enable_pane_menu=False,
        get_node_on_click=True, get_edge_on_click=False, show_minimap=True,
        hide_watermark=True, allow_new_edges=False, min_zoom=0.2,
    )
    selected_from_canvas = getattr(st.session_state["dt_flow_state"], "selected_id", None)
    node_ids = [str(row["id"]) for row in nodes]
    if selected_from_canvas in node_ids:
        st.session_state["dt_selected_node"] = selected_from_canvas

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Guided construction")
        with st.form("add_strategy_form"):
            new_strategy_name = st.text_input("Add another strategy")
            first_event = st.text_input("First event/question (optional)", placeholder="e.g. Does the patient respond?")
            if st.form_submit_button("+ Add strategy"):
                try:
                    new_strategies, new_nodes = add_strategy(strategies, nodes, strategy_name=new_strategy_name, first_event_label=first_event or None)
                except GuidedTreeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["dt_strategy_rows"] = new_strategies
                    st.session_state["dt_node_rows"] = new_nodes
                    st.session_state.pop("dt_flow_signature", None); reset_analysis_state(); st.rerun()

        if node_ids:
            selected_default = st.session_state.get("dt_selected_node", node_ids[0])
            if selected_default not in node_ids:
                selected_default = node_ids[0]
            selected_node = st.selectbox(
                "Selected node", node_ids, index=node_ids.index(selected_default),
                format_func=lambda nid: next((f"{row['label']} ({row['type']})" for row in nodes if str(row['id']) == nid), nid),
                key="dt_selected_node_select",
            )
            st.session_state["dt_selected_node"] = selected_node
            selected_row = next(row for row in nodes if str(row["id"]) == selected_node)
            st.caption("Tip: click a node on the canvas or choose it above.")
            new_node_label = st.text_input("Node label", str(selected_row.get("label") or ""), key=f"node_label_{revision}_{selected_node}")
            if st.button("Update node label", key=f"node_update_{selected_node}"):
                try:
                    st.session_state["dt_node_rows"] = update_node(nodes, selected_node, label=new_node_label)
                except GuidedTreeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pop("dt_flow_signature", None); reset_analysis_state(); st.rerun()

            if str(selected_row.get("type")) == "chance":
                st.markdown("**Add a possible pathway from this chance event**")
                probability_candidates = [
                    row for row in st.session_state["dt_parameter_rows"]
                    if 0 <= float(row.get("value", -1)) <= 1 and str(row.get("category")) != "cost"
                ]
                if not probability_candidates:
                    st.warning("Create a probability parameter in the Parameters tab before adding a branch.")
                else:
                    with st.form(f"add_branch_{selected_node}"):
                        branch_label = st.text_input("Pathway label", placeholder="e.g. Responds")
                        probability_id = st.selectbox("Probability parameter", [str(row["id"]) for row in probability_candidates], format_func=lambda pid: next(str(row["label"]) for row in probability_candidates if str(row["id"]) == pid))
                        probability_mode = st.radio("Probability rule", ["direct", "complement"], horizontal=True, help="Direct uses p. Complement uses 1−p, useful for the second branch of a binary event.")
                        child_type_label = st.radio("What comes next?", ["Another chance event", "Terminal outcome"], horizontal=True)
                        child_label = st.text_input("Name the next event/outcome")
                        if st.form_submit_button("+ Add pathway"):
                            try:
                                new_nodes, new_branches, child_id = add_branch_with_child(
                                    nodes, branches, parent_id=selected_node, branch_label=branch_label,
                                    probability_parameter_id=probability_id, probability_mode=probability_mode,
                                    child_label=child_label,
                                    child_type="chance" if child_type_label == "Another chance event" else "terminal",
                                )
                            except GuidedTreeError as exc:
                                st.error(str(exc))
                            else:
                                st.session_state["dt_node_rows"] = new_nodes
                                st.session_state["dt_branch_rows"] = new_branches
                                st.session_state["dt_selected_node"] = child_id
                                st.session_state.pop("dt_flow_signature", None); reset_analysis_state(); st.rerun()
                outgoing = [row for row in branches if str(row.get("from_node")) == selected_node]
                if outgoing:
                    st.write("Current pathways:")
                    for branch in outgoing:
                        p = branch.get("probability_parameter_id")
                        expr = f"1 − {p}" if branch.get("probability_mode") == "complement" else p
                        st.write(f"• **{branch.get('label')}** → {branch.get('to_node')} ({expr})")
        
    with right:
        st.markdown("#### Costs and health outcomes at the selected node")
        if node_ids:
            selected_node = st.session_state.get("dt_selected_node", node_ids[0])
            if selected_node not in node_ids:
                selected_node = node_ids[0]
            selected_row = next(row for row in st.session_state["dt_node_rows"] if str(row["id"]) == selected_node)
            st.write(f"**{selected_row.get('label')}**")
            st.caption(f"Costs: {selected_row.get('cost_rewards') or 'none'}")
            st.caption(f"Health outcomes: {selected_row.get('outcome_rewards') or 'none'}")
            cost_params = [row for row in st.session_state["dt_parameter_rows"] if row.get("category") == "cost"]
            outcome_params = [row for row in st.session_state["dt_parameter_rows"] if row.get("category") != "cost"]
            if cost_params:
                with st.form(f"cost_reward_{selected_node}"):
                    cp = st.selectbox("Cost parameter", [str(row["id"]) for row in cost_params], format_func=lambda pid: next(str(row["label"]) for row in cost_params if str(row["id"]) == pid))
                    ct = st.number_input("When does this cost occur? (years from model start)", min_value=0.0, value=0.0, step=0.25)
                    if st.form_submit_button("+ Attach cost"):
                        st.session_state["dt_node_rows"] = append_reward(st.session_state["dt_node_rows"], selected_node, parameter_id=cp, time_years=ct, reward_kind="cost")
                        reset_analysis_state(); st.session_state.pop("dt_flow_signature", None); st.rerun()
            if outcome_params:
                with st.form(f"outcome_reward_{selected_node}"):
                    ep = st.selectbox("Health-outcome parameter", [str(row["id"]) for row in outcome_params], format_func=lambda pid: next(str(row["label"]) for row in outcome_params if str(row["id"]) == pid))
                    et = st.number_input("When does this outcome accrue? (years from model start)", min_value=0.0, value=0.0, step=0.25)
                    if st.form_submit_button("+ Attach outcome"):
                        st.session_state["dt_node_rows"] = append_reward(st.session_state["dt_node_rows"], selected_node, parameter_id=ep, time_years=et, reward_kind="outcome")
                        reset_analysis_state(); st.session_state.pop("dt_flow_signature", None); st.rerun()
            clear1, clear2 = st.columns(2)
            if clear1.button("Clear node costs", key=f"clear_costs_{selected_node}"):
                st.session_state["dt_node_rows"] = update_node(st.session_state["dt_node_rows"], selected_node, cost_rewards="")
                reset_analysis_state(); st.session_state.pop("dt_flow_signature", None); st.rerun()
            if clear2.button("Clear node outcomes", key=f"clear_outcomes_{selected_node}"):
                st.session_state["dt_node_rows"] = update_node(st.session_state["dt_node_rows"], selected_node, outcome_rewards="")
                reset_analysis_state(); st.session_state.pop("dt_flow_signature", None); st.rerun()
            is_root = selected_node in {str(row.get("root_node_id")) for row in strategies}
            has_children = any(str(row.get("from_node")) == selected_node for row in branches)
            if st.button("Delete selected leaf", disabled=is_root or has_children, key=f"delete_leaf_{selected_node}"):
                try:
                    new_nodes, new_branches = delete_leaf_node(strategies, st.session_state["dt_node_rows"], st.session_state["dt_branch_rows"], selected_node)
                except GuidedTreeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["dt_node_rows"] = new_nodes; st.session_state["dt_branch_rows"] = new_branches
                    st.session_state.pop("dt_selected_node", None); st.session_state.pop("dt_flow_signature", None); reset_analysis_state(); st.rerun()

    with st.expander("Advanced — edit underlying model tables", expanded=False):
        st.warning("Advanced mode exposes the raw structure. Invalid edits will be caught by model validation, but guided construction is safer for most users.")
        advanced_params = []
        for row in st.session_state["dt_parameter_rows"]:
            flat = dict(row)
            flat["distribution_parameters"] = distribution_parameters_text(parse_legacy_distribution_parameters(flat.get("distribution_parameters")))
            advanced_params.append(flat)
        adv_p = st.data_editor(pd.DataFrame(advanced_params), num_rows="dynamic", hide_index=True, key=f"adv_params_{revision}")
        adv_s = st.data_editor(pd.DataFrame(st.session_state["dt_strategy_rows"]), num_rows="dynamic", hide_index=True, key=f"adv_strat_{revision}")
        adv_n = st.data_editor(pd.DataFrame(st.session_state["dt_node_rows"]), num_rows="dynamic", hide_index=True, key=f"adv_nodes_{revision}")
        adv_b = st.data_editor(pd.DataFrame(st.session_state["dt_branch_rows"]), num_rows="dynamic", hide_index=True, key=f"adv_branches_{revision}")
        if st.button("Apply advanced table edits"):
            migrated = []
            for row in adv_p.to_dict("records"):
                item = migrate_parameter_row(row)
                item["distribution_parameters"] = parse_legacy_distribution_parameters(item.get("distribution_parameters"))
                migrated.append(item)
            st.session_state["dt_parameter_rows"] = migrated
            st.session_state["dt_strategy_rows"] = adv_s.to_dict("records")
            st.session_state["dt_node_rows"] = adv_n.to_dict("records")
            st.session_state["dt_branch_rows"] = adv_b.to_dict("records")
            st.session_state.pop("dt_flow_signature", None); reset_analysis_state(); st.rerun()

# ---------------------------------------------------------------------------
# Compile / base-case run. During construction validation messages are gentle.
# ---------------------------------------------------------------------------
compiled = run = decision = None
model_error = None
if threshold is None:
    model_error = "Enter a decision threshold in the sidebar to run cost-effectiveness analysis."
else:
    try:
        compiled = compile_builder_tables(
            st.session_state["dt_parameter_rows"], st.session_state["dt_strategy_rows"],
            st.session_state["dt_node_rows"], st.session_state["dt_branch_rows"],
        )
        run = run_decision_tree(
            compiled.tree, compiled.parameters, included_cost_bearers=included_cost_bearers or None,
            cost_discount_rate=cost_discount_rate, outcome_discount_rate=outcome_discount_rate,
        )
        model_strategies = [Strategy(compiled.strategy_names[row.strategy_id], row.expected_cost, row.expected_outcome) for row in run.strategies]
        decision = fully_incremental_analysis(model_strategies, threshold)
    except (BuilderValidationError, DecisionTreeValidationError, ValueError) as exc:
        model_error = str(exc)

with result_tab:
    st.subheader("Base-case results")
    if model_error:
        st.info("The model is not ready to run yet: " + model_error)
    else:
        st.success("The decision tree is structurally valid and the base case ran successfully.")
        expected = [
            {"Strategy": compiled.strategy_names[row.strategy_id], f"Expected cost ({currency.code})": row.expected_cost, f"Expected {outcome.unit}": row.expected_outcome}
            for row in run.strategies
        ]
        st.dataframe(pd.DataFrame(expected), hide_index=True, use_container_width=True)
        rows = []
        for row in decision.rows:
            rows.append({
                "Strategy": row.strategy.name, "Cost": row.strategy.cost, outcome.label: row.strategy.effect,
                "Status": row.status.replace("_", " ").title(), "Compared with": row.compared_with or "—",
                "Incremental cost": row.incremental_cost, f"Incremental {outcome.unit}": row.incremental_effect,
                f"ICER ({currency.code}/{outcome.unit})": row.icer, "NMB": row.nmb,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        preferred = decision.preferred_by_nmb
        st.info(
            f"At {money(threshold, currency.symbol)} per {outcome.unit}, **{preferred[0]}** has the highest NMB."
            if len(preferred) == 1 else "Highest NMB is tied between: " + ", ".join(preferred)
        )
        st.caption(f"Discount rates applied: costs {cost_discount_rate*100:.2f}%, outcomes {outcome_discount_rate*100:.2f}% annually.")

# ---------------------------------------------------------------------------
# Analysis switch: DSA and PSA coexist in parameters; this only changes view.
# ---------------------------------------------------------------------------
with analyse_tab:
    st.subheader("Sensitivity and decision uncertainty")
    if model_error:
        st.info("Complete and validate the model first. " + model_error)
    else:
        analysis_mode = st.segmented_control("Analysis", ["Deterministic (DSA)", "Probabilistic (PSA)"], default="Deterministic (DSA)")
        strategy_ids = list(compiled.strategy_names)
        c1, c2 = st.columns(2)
        comparator_id = c1.selectbox("Comparator", strategy_ids, index=0, format_func=lambda sid: compiled.strategy_names[sid], key="analysis_comp")
        intervention_options = [sid for sid in strategy_ids if sid != comparator_id]
        intervention_id = c2.selectbox("Intervention", intervention_options, index=0, format_func=lambda sid: compiled.strategy_names[sid], key="analysis_int")
        context = dict(
            intervention_id=intervention_id, comparator_id=comparator_id, willingness_to_pay=threshold,
            included_cost_bearers=included_cost_bearers or None, cost_discount_rate=cost_discount_rate,
            outcome_discount_rate=outcome_discount_rate,
        )
        parameter_map = {p.id: p for p in compiled.parameters}

        if analysis_mode == "Deterministic (DSA)":
            st.caption("Only parameters with DSA enabled are varied. PSA settings remain untouched.")
            dsa_parameters = [p for p in compiled.parameters if p.dsa is not None and p.dsa.enabled]
            tornado_tab, one_tab, two_tab, threshold_tab = st.tabs(["Tornado", "One-way", "Two-way decision map", "Threshold"])
            with tornado_tab:
                tornado = tornado_tree_inmb(compiled.tree, compiled.parameters, **context)
                if not tornado:
                    st.info("No parameters are enabled for DSA.")
                else:
                    tornado_df = pd.DataFrame([{"Parameter": r.label, "Low INMB": r.low_inmb, "High INMB": r.high_inmb, "Base INMB": r.base_inmb, "Impact": r.impact} for r in tornado])
                    st.dataframe(tornado_df, hide_index=True, use_container_width=True)
                    chart = tornado_df.iloc[::-1]
                    left_values = np.minimum(chart["Low INMB"], chart["High INMB"])
                    right_values = np.maximum(chart["Low INMB"], chart["High INMB"])
                    fig = go.Figure(go.Bar(y=chart["Parameter"], x=right_values-left_values, base=left_values, orientation="h"))
                    fig.add_vline(x=float(tornado[0].base_inmb), line_dash="dash")
                    fig.update_layout(title="Tornado diagram — INMB", xaxis_title=f"INMB ({currency.code})", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
            with one_tab:
                if not dsa_parameters:
                    st.info("Enable DSA for at least one parameter in the Parameters tab.")
                else:
                    pid = st.selectbox("Parameter", [p.id for p in dsa_parameters], format_func=lambda x: parameter_map[x].label, key="dsa_one_pid")
                    p = parameter_map[pid]; values = (p.dsa.lower, p.value, p.dsa.upper)
                    try:
                        result = one_way_tree_inmb(compiled.tree, compiled.parameters, OneWaySensitivitySpec(pid, values), **context)
                        st.dataframe(pd.DataFrame([{"Parameter value": v, "INMB": metric} for v, metric in result]), hide_index=True)
                    except (DecisionTreeValidationError, ValueError) as exc:
                        st.warning(str(exc))
            with two_tab:
                if len(dsa_parameters) < 2:
                    st.info("Enable DSA for at least two parameters to create a two-way decision map.")
                else:
                    x_id = st.selectbox("First parameter", [p.id for p in dsa_parameters], format_func=lambda x: parameter_map[x].label, key="dsa_two_x")
                    y_ids = [p.id for p in dsa_parameters if p.id != x_id]
                    y_id = st.selectbox("Second parameter", y_ids, format_func=lambda x: parameter_map[x].label, key="dsa_two_y")
                    pxp, pyp = parameter_map[x_id], parameter_map[y_id]
                    x_values = tuple(float(v) for v in np.linspace(pxp.dsa.lower, pxp.dsa.upper, 9))
                    y_values = tuple(float(v) for v in np.linspace(pyp.dsa.lower, pyp.dsa.upper, 9))
                    try:
                        tw = two_way_tree_inmb(compiled.tree, compiled.parameters, TwoWaySensitivitySpec(x_id, x_values, y_id, y_values), **context)
                        grid = pd.DataFrame(tw, columns=[x_id, y_id, "INMB"])
                        matrix = grid.pivot(index=y_id, columns=x_id, values="INMB")
                        decision_text = np.where(matrix.values >= 0, compiled.strategy_names[intervention_id], compiled.strategy_names[comparator_id])
                        fig = go.Figure(go.Heatmap(z=np.where(matrix.values >= 0, 1, 0), x=matrix.columns, y=matrix.index, text=decision_text, customdata=matrix.values, showscale=False, hovertemplate="X: %{x}<br>Y: %{y}<br>Preferred: %{text}<br>INMB: %{customdata:.3g}<extra></extra>"))
                        fig.update_layout(title="Two-way pairwise decision map", xaxis_title=pxp.label, yaxis_title=pyp.label)
                        st.plotly_chart(fig, use_container_width=True)
                    except (DecisionTreeValidationError, ValueError) as exc:
                        st.warning(str(exc))
            with threshold_tab:
                if not dsa_parameters:
                    st.info("Enable DSA for at least one parameter.")
                else:
                    pid = st.selectbox("Parameter", [p.id for p in dsa_parameters], format_func=lambda x: parameter_map[x].label, key="dsa_threshold_pid")
                    p = parameter_map[pid]
                    try:
                        switching = threshold_tree_inmb(compiled.tree, compiled.parameters, ThresholdAnalysisSpec(pid, p.dsa.lower, p.dsa.upper), **context)
                        st.success(f"Switching value: **{switching:,.6g}** (INMB ≈ 0).")
                    except (DecisionTreeValidationError, ValueError) as exc:
                        st.info("No valid switching point was found inside the DSA range: " + str(exc))
        else:
            st.caption("Only parameters with PSA enabled are sampled. DSA ranges remain stored and unchanged.")
            psa_parameters = [p for p in compiled.parameters if p.psa is not None and p.psa.enabled]
            if not psa_parameters:
                st.info("No parameters are enabled for PSA. Configure them in the Parameters tab.")
            else:
                st.write("Sampling: " + ", ".join(f"{p.label} ({p.psa.distribution.family})" for p in psa_parameters))
                r1, r2, r3 = st.columns(3)
                iterations = int(r1.number_input("Iterations", min_value=100, max_value=50000, value=2000, step=100))
                seed = int(r2.number_input("Random seed", min_value=0, value=12345, step=1))
                ceac_max = float(r3.number_input("CEAC maximum threshold", min_value=0.0, value=float(max(threshold*2, threshold+1))))
                signature = stable_signature(st.session_state["dt_parameter_rows"], st.session_state["dt_strategy_rows"], st.session_state["dt_node_rows"], st.session_state["dt_branch_rows"], iterations, seed, cost_discount_rate, outcome_discount_rate, included_cost_bearers)
                if st.button("Run PSA", type="primary"):
                    try:
                        psa_result = run_tree_psa(compiled.tree, compiled.parameters, iterations=iterations, seed=seed, included_cost_bearers=included_cost_bearers or None, cost_discount_rate=cost_discount_rate, outcome_discount_rate=outcome_discount_rate)
                    except (PSAConfigurationError, DecisionTreeValidationError, ValueError) as exc:
                        st.error(str(exc))
                        st.session_state.pop("dt_psa_result", None)
                    else:
                        st.session_state["dt_psa_result"] = psa_result; st.session_state["dt_psa_signature"] = signature
                psa_result = st.session_state.get("dt_psa_result") if st.session_state.get("dt_psa_signature") == signature else None
                if psa_result is not None:
                    for warning in psa_result.warnings:
                        st.warning("PSA completed with methodological warning: " + warning)
                    probability = pairwise_probability_cost_effective(psa_result, intervention_id=intervention_id, comparator_id=comparator_id, willingness_to_pay=threshold)
                    st.metric(f"Probability {compiled.strategy_names[intervention_id]} is cost-effective vs {compiled.strategy_names[comparator_id]}", f"{probability*100:.1f}%")
                    delta_effect, delta_cost = incremental_plane(psa_result, intervention_id=intervention_id, comparator_id=comparator_id)
                    plane = pd.DataFrame({"Incremental effect": delta_effect, "Incremental cost": delta_cost})
                    fig = px.scatter(plane, x="Incremental effect", y="Incremental cost", opacity=0.35, title="Cost-effectiveness plane")
                    fig.add_hline(y=0); fig.add_vline(x=0)
                    xmin, xmax = float(np.min(delta_effect)), float(np.max(delta_effect))
                    if xmin == xmax: xmin -= 1; xmax += 1
                    fig.add_trace(go.Scatter(x=[xmin, xmax], y=[threshold*xmin, threshold*xmax], mode="lines", name="Decision threshold"))
                    st.plotly_chart(fig, use_container_width=True)
                    curve = ceac(psa_result, np.linspace(0, ceac_max, 101))
                    ceac_rows = []
                    for sid in psa_result.strategy_ids:
                        for value, prob in zip(curve.thresholds, curve.probabilities[sid]):
                            ceac_rows.append({"Threshold": value, "Probability cost-effective": prob, "Strategy": compiled.strategy_names[sid]})
                    fig = px.line(pd.DataFrame(ceac_rows), x="Threshold", y="Probability cost-effective", color="Strategy", title="Cost-effectiveness acceptability curve")
                    fig.add_vline(x=threshold, line_dash="dash"); fig.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"PSA used {psa_result.iterations:,} simulations with seed {psa_result.seed}. Review Monte Carlo stability before final use.")

# ---------------------------------------------------------------------------
# Persistence, exports and audit
# ---------------------------------------------------------------------------
with files_tab:
    st.subheader("Save, export and audit")
    loaded_bundle = st.session_state.get("dt_loaded_bundle") or {}
    metadata = loaded_bundle.get("metadata", {})
    m1, m2 = st.columns(2)
    model_name = m1.text_input("Model name", loaded_bundle.get("model_name", "Untitled decision tree"))
    author = m2.text_input("Author / analyst", metadata.get("author", ""))
    notes = st.text_area("Model notes", metadata.get("notes", ""))
    if model_error:
        st.warning("Resolve the model validation issue before saving a reproducible model file: " + model_error)
    else:
        try:
            current_bundle = build_decision_tree_bundle(
                model_name=model_name, reference_case_code=reference_case_code, outcome_code=outcome_code,
                currency_code=currency_code, threshold=threshold, perspective_label=perspective_label,
                included_cost_bearers=included_cost_bearers, time_horizon=time_horizon,
                cost_discount_rate=cost_discount_rate, outcome_discount_rate=outcome_discount_rate,
                parameter_rows=st.session_state["dt_parameter_rows"], strategy_rows=st.session_state["dt_strategy_rows"],
                node_rows=st.session_state["dt_node_rows"], branch_rows=st.session_state["dt_branch_rows"],
                author=author, notes=notes,
            )
        except (PersistenceError, BuilderValidationError, ValueError) as exc:
            st.error(str(exc))
        else:
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_name).strip("_") or "decision_tree"
            st.caption(f"Model content hash: `{current_bundle['content_hash_sha256']}`")
            st.download_button("Download model JSON", model_bundle_json(current_bundle), file_name=f"{safe_name}.json", mime="application/json")
            deterministic_export = [
                {"strategy": row.strategy.name, "cost": row.strategy.cost, "outcome": row.strategy.effect, "status": row.status, "compared_with": row.compared_with, "incremental_cost": row.incremental_cost, "incremental_effect": row.incremental_effect, "icer": row.icer, "nmb": row.nmb}
                for row in decision.rows
            ]
            st.download_button("Download deterministic results CSV", rows_to_csv(deterministic_export), file_name=f"{safe_name}_deterministic_results.csv", mime="text/csv")
            a1, a2 = st.columns(2)
            if a1.button("Record deterministic run"):
                record = build_audit_record(
                    current_bundle, analysis_type="deterministic_decision_tree",
                    run_settings={"decision_threshold": threshold, "currency": currency_code, "outcome": outcome_code, "cost_discount_rate": cost_discount_rate, "outcome_discount_rate": outcome_discount_rate, "included_cost_bearers": included_cost_bearers},
                    results={"expected_values": [{"strategy_id": row.strategy_id, "strategy_name": compiled.strategy_names[row.strategy_id], "expected_cost": row.expected_cost, "expected_outcome": row.expected_outcome} for row in run.strategies], "incremental_analysis": deterministic_export, "preferred_by_nmb": list(decision.preferred_by_nmb)},
                )
                st.session_state["dt_audit_trail"].append(record); st.success("Deterministic run recorded.")
            current_psa = st.session_state.get("dt_psa_result")
            if current_psa is not None:
                if a2.button("Record latest PSA run"):
                    record = build_audit_record(
                        current_bundle, analysis_type="probabilistic_sensitivity_analysis",
                        run_settings={"iterations": current_psa.iterations, "seed": current_psa.seed, "decision_threshold": threshold, "currency": currency_code, "outcome": outcome_code},
                        results={"mean_costs": {sid: float(np.mean(current_psa.costs[sid])) for sid in current_psa.strategy_ids}, "mean_outcomes": {sid: float(np.mean(current_psa.outcomes[sid])) for sid in current_psa.strategy_ids}}, warnings=current_psa.warnings,
                    )
                    st.session_state["dt_audit_trail"].append(record); st.success("PSA run recorded.")
                simulation_data = {"iteration": np.arange(1, current_psa.iterations + 1)}
                for sid in current_psa.strategy_ids:
                    simulation_data[f"cost_{sid}"] = current_psa.costs[sid]; simulation_data[f"outcome_{sid}"] = current_psa.outcomes[sid]
                for pid, draws in current_psa.parameter_draws.items():
                    simulation_data[f"draw_{pid}"] = draws
                st.download_button("Download latest PSA simulations CSV", pd.DataFrame(simulation_data).to_csv(index=False), file_name=f"{safe_name}_psa_simulations.csv", mime="text/csv")
            trail = st.session_state.get("dt_audit_trail", [])
            st.write(f"**Audit records in this session:** {len(trail)}")
            if trail:
                st.dataframe(pd.DataFrame([{"Run ID": record["run_id"], "Time (UTC)": record["run_at_utc"], "Analysis": record["analysis_type"], "Model hash": record["model_hash_sha256"], "Warnings": len(record.get("warnings", []))} for record in trail]), hide_index=True, use_container_width=True)
                st.download_button("Download audit trail JSON", audit_trail_json(trail), file_name=f"{safe_name}_audit_trail.json", mime="application/json")
