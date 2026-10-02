"""Hybrid decision-tree builder for Economic Evaluation Platform v0.4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model.currency import CURRENCIES
from model.decision_tree import DecisionTreeValidationError, run_decision_tree
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
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
    psa_configuration_warnings,
    run_tree_psa,
)
from model.reference_cases import REFERENCE_CASES
from model.sensitivity import OneWaySensitivitySpec, ThresholdAnalysisSpec, TwoWaySensitivitySpec
from model.tree_builder import BuilderValidationError, compile_builder_tables, structure_to_dot
from model.tree_sensitivity import (
    one_way_tree_inmb,
    threshold_tree_inmb,
    tornado_tree_inmb,
    two_way_tree_inmb,
)
from model.uncertainty_defaults import suggest_distribution_from_fields


st.set_page_config(page_title="Decision Tree Builder", layout="wide")
st.title("Decision Tree Builder")
st.caption(
    "Version 0.4 — hybrid builder, discounting, deterministic sensitivity analysis, "
    "PSA, model files and audit trails"
)

SOURCE_TYPES = [
    "systematic_review",
    "meta_analysis",
    "randomised_trial",
    "observational_study",
    "registry",
    "database",
    "tariff",
    "cost_study",
    "expert_elicitation",
    "guideline",
    "user_assumption",
    "other",
]
PARAMETER_CATEGORIES = [
    "clinical",
    "cost",
    "utility",
    "resource_use",
    "survival",
    "epidemiology",
    "other",
]
UNCERTAINTY_KINDS = ["none", "range", "distribution"]
DISTRIBUTION_FAMILIES = ["", "beta", "gamma", "lognormal", "normal", "uniform", "dirichlet"]
COST_BEARER_OPTIONS = [
    "health_system",
    "personal_social_services",
    "patient_direct_medical",
    "patient_direct_non_medical",
    "productivity",
    "carer",
    "custom",
]


def default_parameter_rows():
    common = {
        "unit": "proportion",
        "category": "clinical",
        "source_type": "user_assumption",
        "source_citation": "Illustrative example input",
        "source_url": "",
        "publication_year": None,
        "assumption": "Illustrative value for builder demonstration.",
        "assumption_rationale": "Replace with sourced evidence before substantive use.",
        "uncertainty_kind": "range",
        "uncertainty_rationale": "Illustrative range; replace with evidence-based uncertainty.",
        "lower": None,
        "upper": None,
        "distribution_family": "",
        "distribution_parameters": "",
        "correlation_group": "",
        "currency": "",
        "price_year": None,
        "cost_bearers": "",
        "notes": "",
    }
    rows = [
        {**common, "id": "p_success_a", "label": "Probability of success — A", "value": 0.8, "lower": 0.65, "upper": 0.90},
        {**common, "id": "p_success_b", "label": "Probability of success — B", "value": 0.6, "lower": 0.45, "upper": 0.75},
    ]
    for pid, label, value, low, high in [
        ("cost_a", "Strategy A acquisition cost", 1000.0, 800.0, 1200.0),
        ("cost_b", "Strategy B acquisition cost", 500.0, 400.0, 600.0),
        ("cost_success", "Cost after success", 100.0, 80.0, 120.0),
        ("cost_failure", "Cost after failure", 1000.0, 800.0, 1200.0),
    ]:
        rows.append(
            {
                **common,
                "id": pid,
                "label": label,
                "value": value,
                "lower": low,
                "upper": high,
                "unit": "currency",
                "category": "cost",
                "currency": "GBP",
                "price_year": 2026,
                "cost_bearers": "health_system",
            }
        )
    for pid, label, value, low, high in [
        ("outcome_success", "Outcome after success", 2.0, 1.7, 2.2),
        ("outcome_failure", "Outcome after failure", 1.0, 0.8, 1.2),
    ]:
        rows.append(
            {
                **common,
                "id": pid,
                "label": label,
                "value": value,
                "lower": low,
                "upper": high,
                "unit": "QALY",
                "category": "utility",
            }
        )
    return rows


DEFAULT_STRATEGIES = [
    {"strategy_id": "A", "strategy_name": "Strategy A", "root_node_id": "a_root"},
    {"strategy_id": "B", "strategy_name": "Strategy B", "root_node_id": "b_root"},
]
DEFAULT_NODES = [
    {"id": "a_root", "label": "Outcome under A", "type": "chance", "cost_rewards": "cost_a@0", "outcome_rewards": ""},
    {"id": "b_root", "label": "Outcome under B", "type": "chance", "cost_rewards": "cost_b@0", "outcome_rewards": ""},
    {"id": "success", "label": "Success", "type": "terminal", "cost_rewards": "cost_success@1", "outcome_rewards": "outcome_success@1"},
    {"id": "failure", "label": "Failure", "type": "terminal", "cost_rewards": "cost_failure@1", "outcome_rewards": "outcome_failure@1"},
]
DEFAULT_BRANCHES = [
    {"from_node": "a_root", "label": "Success", "probability_parameter_id": "p_success_a", "probability_mode": "direct", "to_node": "success"},
    {"from_node": "a_root", "label": "Failure", "probability_parameter_id": "p_success_a", "probability_mode": "complement", "to_node": "failure"},
    {"from_node": "b_root", "label": "Success", "probability_parameter_id": "p_success_b", "probability_mode": "direct", "to_node": "success"},
    {"from_node": "b_root", "label": "Failure", "probability_parameter_id": "p_success_b", "probability_mode": "complement", "to_node": "failure"},
]


def money(value, symbol):
    return f"{symbol}{value:,.2f}"


def parse_threshold(raw):
    text = raw.replace(",", "").strip()
    if not text:
        return None
    value = float(text)
    if value < 0:
        raise ValueError("Threshold cannot be negative.")
    return value


def default_bounds(parameter):
    if (
        parameter.uncertainty.kind == "range"
        and parameter.uncertainty.lower is not None
        and parameter.uncertainty.upper is not None
    ):
        return float(parameter.uncertainty.lower), float(parameter.uncertainty.upper)
    base = float(parameter.value)
    if 0 <= base <= 1:
        return max(0.0, base * 0.8), min(1.0, base * 1.2 if base else 0.2)
    width = abs(base) * 0.2 if base else 1.0
    return base - width, base + width


def linear_grid(low, high, points=7):
    return tuple(float(value) for value in np.linspace(low, high, max(2, points)))


def advisory_distribution_rows(df):
    records = df.to_dict("records")
    group_counts = {}
    for row in records:
        group = str(row.get("correlation_group") or "").strip()
        if group:
            group_counts[group] = group_counts.get(group, 0) + 1

    rows = []
    for row in records:
        if not str(row.get("id") or "").strip():
            continue
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        group = str(row.get("correlation_group") or "").strip()
        probability_like = (
            "probab" in str(row.get("label") or "").lower()
            or str(row.get("unit") or "").lower() in {"probability", "proportion", "%"}
        )
        if group and group_counts.get(group, 0) > 1 and probability_like and 0 <= value <= 1:
            suggested_family = "dirichlet"
            rationale = (
                "A Dirichlet distribution is appropriate when this parameter is one component of a set "
                "of mutually exclusive probabilities that must sum to 1."
            )
            parameterisation = (
                "Use the same correlation_group for all components and supply one positive alpha "
                "concentration for each component."
            )
            caution = "Use Dirichlet only when the grouped quantities form one probability simplex."
        else:
            suggestion = suggest_distribution_from_fields(
                label=str(row.get("label") or row.get("id") or ""),
                category=str(row.get("category") or "other"),
                unit=str(row.get("unit") or ""),
                value=value,
            )
            suggested_family = suggestion.family
            rationale = suggestion.rationale
            parameterisation = suggestion.parameterisation
            caution = suggestion.caution
        rows.append(
            {
                "Parameter": row.get("label") or row.get("id"),
                "Suggested family": suggested_family,
                "Why": rationale,
                "Suggested parameterisation": parameterisation,
                "Caution": caution,
                "User-selected family": row.get("distribution_family") or "—",
            }
        )
    return rows


def _clear_analysis_state():
    for key in ["dt_psa_result", "dt_psa_signature", "dt_audit_trail"]:
        st.session_state.pop(key, None)


if "dt_model_revision" not in st.session_state:
    st.session_state["dt_model_revision"] = 0
if "dt_audit_trail" not in st.session_state:
    st.session_state["dt_audit_trail"] = []

loaded_bundle = st.session_state.get("dt_loaded_bundle")

with st.expander("Load or start a model", expanded=False):
    uploaded_model = st.file_uploader(
        "Upload a saved decision-tree model (.json)",
        type=["json"],
        key="dt_model_upload",
    )
    load_col, new_col = st.columns(2)
    if load_col.button("Load uploaded model", disabled=uploaded_model is None):
        try:
            bundle = load_decision_tree_bundle(uploaded_model.getvalue())
        except (PersistenceError, BuilderValidationError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["dt_loaded_bundle"] = bundle
            st.session_state["dt_model_revision"] += 1
            _clear_analysis_state()
            methods = bundle["methods"]
            st.session_state["dt_reference_case"] = methods["reference_case_code"]
            st.session_state["dt_outcome_code"] = methods["outcome_code"]
            st.session_state["dt_threshold_text"] = (
                "" if methods.get("threshold") is None else str(methods.get("threshold"))
            )
            st.session_state["dt_custom_currency"] = methods["currency_code"]
            st.session_state["dt_custom_perspective"] = methods["perspective_label"]
            st.session_state["dt_custom_cost_bearers"] = methods["included_cost_bearers"]
            st.session_state["dt_custom_horizon"] = methods["time_horizon"]
            st.session_state["dt_custom_cost_discount"] = methods["cost_discount_rate"] * 100
            st.session_state["dt_custom_outcome_discount"] = methods["outcome_discount_rate"] * 100
            st.rerun()
    if new_col.button("Start new example model"):
        st.session_state.pop("dt_loaded_bundle", None)
        st.session_state["dt_model_revision"] += 1
        _clear_analysis_state()
        for key in [
            "dt_reference_case",
            "dt_outcome_code",
            "dt_threshold_text",
            "dt_custom_currency",
            "dt_custom_perspective",
            "dt_custom_cost_bearers",
            "dt_custom_horizon",
            "dt_custom_cost_discount",
            "dt_custom_outcome_discount",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

loaded_bundle = st.session_state.get("dt_loaded_bundle")
revision = int(st.session_state["dt_model_revision"])
loaded_model = loaded_bundle.get("model", {}) if loaded_bundle else {}
parameter_initial = loaded_model.get("parameters", default_parameter_rows())
strategy_initial = loaded_model.get("strategies", DEFAULT_STRATEGIES)
node_initial = loaded_model.get("nodes", DEFAULT_NODES)
branch_initial = loaded_model.get("branches", DEFAULT_BRANCHES)

with st.sidebar:
    st.header("Methods")
    reference_case_code = st.selectbox(
        "Reference case",
        ["NICE_TA", "HTAIN_2023", "CUSTOM"],
        format_func=lambda code: REFERENCE_CASES[code].name if code in REFERENCE_CASES else "Custom",
        key="dt_reference_case",
    )
    if reference_case_code in REFERENCE_CASES:
        profile = REFERENCE_CASES[reference_case_code]
        outcome_code = st.selectbox(
            "Economic outcome",
            list(OUTCOME_MEASURES),
            index=list(OUTCOME_MEASURES).index(profile.preferred_outcome_code),
            format_func=lambda code: OUTCOME_MEASURES[code].label,
            key="dt_outcome_code",
        )
        currency_code = profile.analysis_currency
        st.text_input("Analysis currency", currency_code, disabled=True)
        cost_discount_rate = profile.cost_discount_rate
        outcome_discount_rate = profile.outcome_discount_rate
        perspective_label = profile.perspective.label
        time_horizon = profile.time_horizon_rule
        st.write(f"**Perspective:** {perspective_label}")
        st.caption(f"Time horizon rule: {time_horizon}")
        st.caption(
            f"Discounting: costs {cost_discount_rate*100:.1f}%, "
            f"outcomes {outcome_discount_rate*100:.1f}% per year."
        )
        if profile.threshold_range and outcome_code == profile.threshold_range.outcome_code:
            default_threshold = str(int(profile.threshold_range.lower))
            st.caption(
                f"Reference range: {CURRENCIES[currency_code].symbol}"
                f"{profile.threshold_range.lower:,.0f}–{CURRENCIES[currency_code].symbol}"
                f"{profile.threshold_range.upper:,.0f} per {OUTCOME_MEASURES[outcome_code].unit}."
            )
        else:
            default_threshold = ""
        if "dt_threshold_text" not in st.session_state:
            st.session_state["dt_threshold_text"] = default_threshold
        threshold_raw = st.text_input("Decision threshold", key="dt_threshold_text")
        included_cost_bearers = list(profile.perspective.included_cost_bearers)
        status = profile.outcome_status(outcome_code)
        if status in {"conditional", "supplementary", "not_specified"}:
            st.warning(
                f"{OUTCOME_MEASURES[outcome_code].label} is classified as "
                f"**{status.replace('_', ' ')}** for this reference case."
            )
    else:
        outcome_code = st.selectbox(
            "Economic outcome",
            list(OUTCOME_MEASURES),
            format_func=lambda code: OUTCOME_MEASURES[code].label,
            key="dt_outcome_code",
        )
        currency_code = st.selectbox(
            "Analysis currency",
            list(CURRENCIES),
            format_func=lambda code: f"{code} — {CURRENCIES[code].name}",
            key="dt_custom_currency",
        )
        perspective_label = st.text_input(
            "Perspective label", "Healthcare payer", key="dt_custom_perspective"
        )
        included_cost_bearers = st.multiselect(
            "Include cost bearers",
            COST_BEARER_OPTIONS,
            default=["health_system"],
            key="dt_custom_cost_bearers",
        )
        time_horizon = st.text_input(
            "Time horizon specification",
            "Long enough to capture relevant costs and outcomes",
            key="dt_custom_horizon",
        )
        cost_discount_rate = st.number_input(
            "Cost discount rate (%)", 0.0, 99.0, 3.5, key="dt_custom_cost_discount"
        ) / 100
        outcome_discount_rate = st.number_input(
            "Outcome discount rate (%)", 0.0, 99.0, 3.5, key="dt_custom_outcome_discount"
        ) / 100
        threshold_raw = st.text_input(
            "Decision threshold", "30000", key="dt_threshold_text"
        )
    st.caption(
        "Reward timing uses absolute years from model start. Enter rewards as `parameter@years`, "
        "for example `followup_cost@2.5`. Annual discrete discounting is applied automatically."
    )

try:
    threshold = parse_threshold(threshold_raw)
except ValueError as exc:
    st.error(str(exc))
    threshold = None
outcome = OUTCOME_MEASURES[outcome_code]
currency = CURRENCIES[currency_code]

st.markdown(
    "Edit structured tables; the live diagram is generated from the same structure. "
    "Distribution suggestions are advisory: the modeller must confirm the family and "
    "enter parameters derived from the evidence."
)
parameters_tab, structure_tab, results_tab, sensitivity_tab, psa_tab, files_tab = st.tabs(
    [
        "1 · Parameters",
        "2 · Structure + visual tree",
        "3 · Validate + run",
        "4 · Deterministic sensitivity",
        "5 · PSA + CE outputs",
        "6 · Save / export / audit",
    ]
)

with parameters_tab:
    st.subheader("Parameter library")
    st.caption(
        "For PSA, set uncertainty to `distribution`, choose a family, and enter evidence-based "
        "parameters using `name=value` syntax, e.g. `alpha=20,beta=80`."
    )
    parameter_df = st.data_editor(
        pd.DataFrame(parameter_initial),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"dt_parameter_editor_{revision}",
        column_config={
            "category": st.column_config.SelectboxColumn(
                "Category", options=PARAMETER_CATEGORIES, required=True
            ),
            "source_type": st.column_config.SelectboxColumn(
                "Source type", options=SOURCE_TYPES, required=True
            ),
            "uncertainty_kind": st.column_config.SelectboxColumn(
                "Uncertainty", options=UNCERTAINTY_KINDS, required=True
            ),
            "distribution_family": st.column_config.SelectboxColumn(
                "PSA family", options=DISTRIBUTION_FAMILIES
            ),
            "value": st.column_config.NumberColumn("Value", format="%.6f"),
            "lower": st.column_config.NumberColumn("Lower", format="%.6f"),
            "upper": st.column_config.NumberColumn("Upper", format="%.6f"),
            "price_year": st.column_config.NumberColumn("Price year", step=1, format="%d"),
        },
    )
    st.subheader("Evidence-informed distribution suggestions")
    st.caption(
        "Suggestions reflect parameter support and common health-economic practice; they are not "
        "automatically applied. The chosen distribution should match the source evidence and be justified."
    )
    st.dataframe(advisory_distribution_rows(parameter_df), hide_index=True, width="stretch")
    st.caption(
        "For mutually exclusive probabilities that must sum to 1, use `dirichlet` for every component, "
        "give all components the same `correlation_group`, and enter one positive `alpha` per component. "
        "Other declared correlation groups are sampled independently with a prominent warning until a "
        "joint covariance structure is implemented."
    )

with structure_tab:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Strategies")
        strategies_df = st.data_editor(
            pd.DataFrame(strategy_initial),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"dt_strategy_editor_{revision}",
        )
        st.subheader("Nodes")
        st.caption(
            "Timed reward syntax: `parameter@years`; multiple rewards are comma-separated. "
            "A missing time means year 0."
        )
        nodes_df = st.data_editor(
            pd.DataFrame(node_initial),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"dt_node_editor_{revision}",
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "Type", options=["chance", "terminal"], required=True
                )
            },
        )
        st.subheader("Branches")
        branches_df = st.data_editor(
            pd.DataFrame(branch_initial),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"dt_branch_editor_{revision}",
            column_config={
                "probability_mode": st.column_config.SelectboxColumn(
                    "Probability mode", options=["direct", "complement"], required=True
                )
            },
        )
    with right:
        st.subheader("Live tree")
        st.caption(
            "Chance nodes are circles; terminal nodes are double circles. Timed rewards and "
            "branch probability rules are shown on the diagram."
        )
        dot = structure_to_dot(
            strategies_df.to_dict("records"),
            nodes_df.to_dict("records"),
            branches_df.to_dict("records"),
        )
        st.graphviz_chart(dot, width="stretch")

compiled = None
run = None
decision = None
if threshold is not None:
    try:
        compiled = compile_builder_tables(
            parameter_df.to_dict("records"),
            strategies_df.to_dict("records"),
            nodes_df.to_dict("records"),
            branches_df.to_dict("records"),
        )
        run = run_decision_tree(
            compiled.tree,
            compiled.parameters,
            included_cost_bearers=included_cost_bearers or None,
            cost_discount_rate=cost_discount_rate,
            outcome_discount_rate=outcome_discount_rate,
        )
        model_strategies = [
            Strategy(
                compiled.strategy_names[row.strategy_id],
                row.expected_cost,
                row.expected_outcome,
            )
            for row in run.strategies
        ]
        decision = fully_incremental_analysis(model_strategies, threshold)
    except (BuilderValidationError, DecisionTreeValidationError, ValueError) as exc:
        model_error = str(exc)
    else:
        model_error = None
else:
    model_error = "Enter a decision threshold in the sidebar before running the model."

with results_tab:
    st.subheader("Validation and discounted expected values")
    if model_error:
        st.error(model_error)
    else:
        st.success("Tree structure, parameter links, reward timing and probabilities are valid.")
        expected_rows = [
            {
                "Strategy": compiled.strategy_names[row.strategy_id],
                f"Expected cost ({currency.code})": money(row.expected_cost, currency.symbol),
                f"Expected {outcome.unit}": f"{row.expected_outcome:,.4f}",
            }
            for row in run.strategies
        ]
        st.dataframe(expected_rows, hide_index=True, width="stretch")
        st.caption(
            f"Applied annual discount rates: costs {cost_discount_rate*100:.2f}%, "
            f"outcomes {outcome_discount_rate*100:.2f}%."
        )
        st.subheader("Cost-effectiveness results")
        result_rows = []
        for row in decision.rows:
            result_rows.append(
                {
                    "Strategy": row.strategy.name,
                    "Cost": money(row.strategy.cost, currency.symbol),
                    outcome.label: f"{row.strategy.effect:,.4f}",
                    "Status": row.status.replace("_", " ").title(),
                    "Compared with": row.compared_with or "—",
                    "Incremental cost": (
                        "—" if row.incremental_cost is None else money(row.incremental_cost, currency.symbol)
                    ),
                    f"Incremental {outcome.unit}": (
                        "—" if row.incremental_effect is None else f"{row.incremental_effect:,.4f}"
                    ),
                    f"ICER ({currency.code}/{outcome.unit})": (
                        "—" if row.icer is None else money(row.icer, currency.symbol)
                    ),
                    "NMB": money(row.nmb, currency.symbol),
                }
            )
        st.dataframe(result_rows, hide_index=True, width="stretch")
        preferred = decision.preferred_by_nmb
        st.info(
            f"At {money(threshold, currency.symbol)} per {outcome.unit}, **{preferred[0]}** has the highest NMB."
            if len(preferred) == 1
            else "At the selected threshold, highest NMB is tied between: " + ", ".join(preferred) + "."
        )

with sensitivity_tab:
    st.subheader("Deterministic sensitivity analysis")
    if model_error:
        st.info(
            "Resolve the model validation issue first; sensitivity analysis always reruns the validated base model."
        )
    else:
        strategy_ids = list(compiled.strategy_names)
        c1, c2 = st.columns(2)
        comparator_id = c1.selectbox(
            "Comparator",
            strategy_ids,
            index=0,
            format_func=lambda sid: compiled.strategy_names[sid],
            key="dsa_comp",
        )
        intervention_candidates = [sid for sid in strategy_ids if sid != comparator_id]
        intervention_id = c2.selectbox(
            "Intervention",
            intervention_candidates,
            index=0,
            format_func=lambda sid: compiled.strategy_names[sid],
            key="dsa_int",
        )
        st.caption(
            "Outputs use INMB. Positive INMB favours the intervention; negative INMB favours "
            "the comparator; zero is the switching point."
        )
        context = dict(
            intervention_id=intervention_id,
            comparator_id=comparator_id,
            willingness_to_pay=threshold,
            included_cost_bearers=included_cost_bearers or None,
            cost_discount_rate=cost_discount_rate,
            outcome_discount_rate=outcome_discount_rate,
        )
        parameter_map = {p.id: p for p in compiled.parameters}
        parameter_ids = list(parameter_map)

        tornado_tab, ow_tab, tw_tab, th_tab = st.tabs(
            ["Tornado", "One-way", "Two-way decision map", "Threshold"]
        )
        with tornado_tab:
            try:
                tornado = tornado_tree_inmb(compiled.tree, compiled.parameters, **context)
                if not tornado:
                    st.info(
                        "No parameters currently have explicit range uncertainty. Add low/high ranges "
                        "to make the tornado diagram available."
                    )
                else:
                    tornado_df = pd.DataFrame(
                        [
                            {
                                "Parameter": row.label,
                                "Low INMB": row.low_inmb,
                                "High INMB": row.high_inmb,
                                "Base INMB": row.base_inmb,
                                "Impact": row.impact,
                                "Low input": row.low_value,
                                "High input": row.high_value,
                            }
                            for row in tornado
                        ]
                    )
                    st.dataframe(tornado_df, hide_index=True, width="stretch")
                    chart_df = tornado_df.iloc[::-1]
                    left = np.minimum(chart_df["Low INMB"], chart_df["High INMB"])
                    right = np.maximum(chart_df["Low INMB"], chart_df["High INMB"])
                    fig = go.Figure(
                        go.Bar(
                            y=chart_df["Parameter"],
                            x=right - left,
                            base=left,
                            orientation="h",
                            customdata=np.stack(
                                [chart_df["Low input"], chart_df["High input"]], axis=-1
                            ),
                            hovertemplate=(
                                "%{y}<br>INMB range starts at %{base:.3g}"
                                "<br>Input low/high: %{customdata[0]:.3g} / %{customdata[1]:.3g}<extra></extra>"
                            ),
                        )
                    )
                    fig.add_vline(x=float(tornado[0].base_inmb), line_dash="dash")
                    fig.update_layout(
                        title="Tornado diagram — INMB range",
                        xaxis_title=f"INMB ({currency.code})",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except (DecisionTreeValidationError, ValueError) as exc:
                st.warning(str(exc))

        with ow_tab:
            pid = st.selectbox(
                "Parameter", parameter_ids, key="ow_param", format_func=lambda x: parameter_map[x].label
            )
            low_default, high_default = default_bounds(parameter_map[pid])
            a, b = st.columns(2)
            low = a.number_input("Low value", value=float(low_default), key="ow_low")
            high = b.number_input("High value", value=float(high_default), key="ow_high")
            try:
                ow = one_way_tree_inmb(
                    compiled.tree,
                    compiled.parameters,
                    OneWaySensitivitySpec(pid, (low, float(parameter_map[pid].value), high)),
                    **context,
                )
                st.dataframe(
                    pd.DataFrame([{"Parameter value": v, "INMB": metric} for v, metric in ow]),
                    hide_index=True,
                    width="stretch",
                )
            except (DecisionTreeValidationError, ValueError) as exc:
                st.warning(f"This variation is not valid for the current tree: {exc}")

        with tw_tab:
            x_id = st.selectbox(
                "First parameter", parameter_ids, key="tw_x", format_func=lambda x: parameter_map[x].label
            )
            y_options = [pid for pid in parameter_ids if pid != x_id]
            y_id = st.selectbox(
                "Second parameter", y_options, key="tw_y", format_func=lambda x: parameter_map[x].label
            )
            xlo, xhi = default_bounds(parameter_map[x_id])
            ylo, yhi = default_bounds(parameter_map[y_id])
            x1, x2, y1, y2 = st.columns(4)
            x_low = x1.number_input("X low", value=float(xlo), key="tw_x_low")
            x_high = x2.number_input("X high", value=float(xhi), key="tw_x_high")
            y_low = y1.number_input("Y low", value=float(ylo), key="tw_y_low")
            y_high = y2.number_input("Y high", value=float(yhi), key="tw_y_high")
            try:
                x_values = linear_grid(x_low, x_high)
                y_values = linear_grid(y_low, y_high)
                tw = two_way_tree_inmb(
                    compiled.tree,
                    compiled.parameters,
                    TwoWaySensitivitySpec(x_id, x_values, y_id, y_values),
                    **context,
                )
                grid = pd.DataFrame(tw, columns=[x_id, y_id, "INMB"])
                matrix = grid.pivot(index=y_id, columns=x_id, values="INMB")
                st.dataframe(matrix, width="stretch")
                decision_text = np.where(
                    matrix.values >= 0,
                    compiled.strategy_names[intervention_id],
                    compiled.strategy_names[comparator_id],
                )
                z = np.where(matrix.values >= 0, 1, 0)
                fig = go.Figure(
                    go.Heatmap(
                        z=z,
                        x=matrix.columns,
                        y=matrix.index,
                        text=decision_text,
                        customdata=matrix.values,
                        showscale=False,
                        hovertemplate=(
                            f"{x_id}: %{{x}}<br>{y_id}: %{{y}}<br>Preferred: %{{text}}"
                            "<br>INMB: %{customdata:.3g}<extra></extra>"
                        ),
                    )
                )
                fig.update_layout(
                    title="Two-way pairwise decision map",
                    xaxis_title=parameter_map[x_id].label,
                    yaxis_title=parameter_map[y_id].label,
                )
                st.plotly_chart(fig, use_container_width=True)
            except (DecisionTreeValidationError, ValueError) as exc:
                st.warning(f"Part of this two-way grid is not valid for the current tree: {exc}")

        with th_tab:
            th_id = st.selectbox(
                "Parameter to search",
                parameter_ids,
                key="th_param",
                format_func=lambda x: parameter_map[x].label,
            )
            tlo, thi = default_bounds(parameter_map[th_id])
            q1, q2 = st.columns(2)
            lower = q1.number_input("Search lower bound", value=float(tlo), key="th_low")
            upper = q2.number_input("Search upper bound", value=float(thi), key="th_high")
            try:
                switching = threshold_tree_inmb(
                    compiled.tree,
                    compiled.parameters,
                    ThresholdAnalysisSpec(th_id, lower, upper),
                    **context,
                )
                st.success(
                    f"Switching value: **{switching:,.6g}**. At this value, INMB is approximately "
                    "zero at the selected decision threshold."
                )
            except (DecisionTreeValidationError, ValueError) as exc:
                st.info(f"No valid switching value was found within these bounds: {exc}")

with psa_tab:
    st.subheader("Probabilistic sensitivity analysis")
    if model_error:
        st.info("Resolve the model validation issue first.")
    else:
        distributed = [p for p in compiled.parameters if p.uncertainty.kind == "distribution"]
        if not distributed:
            st.info(
                "No PSA distributions are configured yet. In the Parameters tab, change selected "
                "parameters to `distribution`, confirm a family, and enter evidence-based distribution parameters."
            )
        else:
            st.write("**Parameters included in PSA:** " + ", ".join(p.label for p in distributed))
            st.caption(
                "Parameters with `range` uncertainty remain deterministic. The engine samples only "
                "explicitly confirmed probability distributions."
            )
            try:
                configuration_warnings = psa_configuration_warnings(compiled.parameters)
            except PSAConfigurationError as exc:
                configuration_warnings = ()
                st.error(str(exc))
            else:
                for warning in configuration_warnings:
                    st.warning("Correlation warning: " + warning)

            strategy_ids = list(compiled.strategy_names)
            p1, p2 = st.columns(2)
            psa_comparator = p1.selectbox(
                "CE-plane comparator",
                strategy_ids,
                index=0,
                format_func=lambda sid: compiled.strategy_names[sid],
                key="psa_comp",
            )
            psa_int_options = [sid for sid in strategy_ids if sid != psa_comparator]
            psa_intervention = p2.selectbox(
                "CE-plane intervention",
                psa_int_options,
                index=0,
                format_func=lambda sid: compiled.strategy_names[sid],
                key="psa_int",
            )
            r1, r2, r3 = st.columns(3)
            iterations = int(
                r1.number_input("Iterations", min_value=100, max_value=50000, value=2000, step=100)
            )
            seed = int(r2.number_input("Random seed", min_value=0, value=12345, step=1))
            ceac_max = float(
                r3.number_input(
                    "CEAC maximum threshold",
                    min_value=0.0,
                    value=float(max(threshold * 2, threshold + 1)),
                    step=float(max(threshold / 10, 1)),
                )
            )

            signature = hash(
                (
                    parameter_df.to_json(),
                    strategies_df.to_json(),
                    nodes_df.to_json(),
                    branches_df.to_json(),
                    iterations,
                    seed,
                    cost_discount_rate,
                    outcome_discount_rate,
                    tuple(included_cost_bearers or []),
                )
            )
            if st.button("Run PSA", type="primary"):
                try:
                    psa_result = run_tree_psa(
                        compiled.tree,
                        compiled.parameters,
                        iterations=iterations,
                        seed=seed,
                        included_cost_bearers=included_cost_bearers or None,
                        cost_discount_rate=cost_discount_rate,
                        outcome_discount_rate=outcome_discount_rate,
                    )
                    st.session_state["dt_psa_result"] = psa_result
                    st.session_state["dt_psa_signature"] = signature
                except (PSAConfigurationError, DecisionTreeValidationError, ValueError) as exc:
                    st.error(str(exc))
                    st.session_state.pop("dt_psa_result", None)

            psa_result = (
                st.session_state.get("dt_psa_result")
                if st.session_state.get("dt_psa_signature") == signature
                else None
            )
            if psa_result is not None:
                for warning in psa_result.warnings:
                    st.warning("PSA completed with methodological warning: " + warning)
                probability = pairwise_probability_cost_effective(
                    psa_result,
                    intervention_id=psa_intervention,
                    comparator_id=psa_comparator,
                    willingness_to_pay=threshold,
                )
                st.metric(
                    f"Probability {compiled.strategy_names[psa_intervention]} is cost-effective vs "
                    f"{compiled.strategy_names[psa_comparator]}",
                    f"{probability*100:.1f}%",
                )

                delta_effect, delta_cost = incremental_plane(
                    psa_result,
                    intervention_id=psa_intervention,
                    comparator_id=psa_comparator,
                )
                plane_df = pd.DataFrame(
                    {"Incremental effect": delta_effect, "Incremental cost": delta_cost}
                )
                fig = px.scatter(
                    plane_df,
                    x="Incremental effect",
                    y="Incremental cost",
                    opacity=0.35,
                    title=(
                        f"Cost-effectiveness plane: {compiled.strategy_names[psa_intervention]} "
                        f"vs {compiled.strategy_names[psa_comparator]}"
                    ),
                )
                fig.add_hline(y=0)
                fig.add_vline(x=0)
                xmin = float(np.min(delta_effect))
                xmax = float(np.max(delta_effect))
                if xmin == xmax:
                    xmin -= 1
                    xmax += 1
                fig.add_trace(
                    go.Scatter(
                        x=[xmin, xmax],
                        y=[threshold * xmin, threshold * xmax],
                        mode="lines",
                        name=f"Threshold {money(threshold, currency.symbol)}/{outcome.unit}",
                    )
                )
                fig.update_xaxes(title=f"Incremental {outcome.unit}")
                fig.update_yaxes(title=f"Incremental cost ({currency.code})")
                st.plotly_chart(fig, use_container_width=True)

                thresholds = np.linspace(0, ceac_max, 101)
                curve = ceac(psa_result, thresholds)
                ceac_rows = []
                for sid in psa_result.strategy_ids:
                    for value, prob in zip(curve.thresholds, curve.probabilities[sid]):
                        ceac_rows.append(
                            {
                                "Threshold": value,
                                "Probability cost-effective": prob,
                                "Strategy": compiled.strategy_names[sid],
                            }
                        )
                ceac_df = pd.DataFrame(ceac_rows)
                fig = px.line(
                    ceac_df,
                    x="Threshold",
                    y="Probability cost-effective",
                    color="Strategy",
                    title="Cost-effectiveness acceptability curve",
                )
                fig.add_vline(x=threshold, line_dash="dash")
                fig.update_xaxes(title=f"Decision threshold ({currency.code}/{outcome.unit})")
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"PSA used {psa_result.iterations:,} simulations with seed {psa_result.seed}. "
                    "Review Monte Carlo stability before treating the result as final."
                )

with files_tab:
    st.subheader("Save, load, export and audit")
    default_name = loaded_bundle.get("model_name", "Untitled decision tree") if loaded_bundle else "Untitled decision tree"
    metadata = loaded_bundle.get("metadata", {}) if loaded_bundle else {}
    meta1, meta2 = st.columns(2)
    model_name = meta1.text_input("Model name", default_name, key=f"dt_model_name_{revision}")
    author = meta2.text_input("Author / analyst", metadata.get("author", ""), key=f"dt_author_{revision}")
    notes = st.text_area("Model notes", metadata.get("notes", ""), key=f"dt_notes_{revision}")

    if model_error:
        st.warning("Resolve the model validation issue before saving or creating an audit record.")
    else:
        try:
            current_bundle = build_decision_tree_bundle(
                model_name=model_name,
                reference_case_code=reference_case_code,
                outcome_code=outcome_code,
                currency_code=currency_code,
                threshold=threshold,
                perspective_label=perspective_label,
                included_cost_bearers=included_cost_bearers,
                time_horizon=time_horizon,
                cost_discount_rate=cost_discount_rate,
                outcome_discount_rate=outcome_discount_rate,
                parameter_rows=parameter_df.to_dict("records"),
                strategy_rows=strategies_df.to_dict("records"),
                node_rows=nodes_df.to_dict("records"),
                branch_rows=branches_df.to_dict("records"),
                author=author,
                notes=notes,
            )
        except (PersistenceError, BuilderValidationError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.caption(f"Model content hash: `{current_bundle['content_hash_sha256']}`")
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_name).strip("_") or "decision_tree"
            st.download_button(
                "Download model JSON",
                data=model_bundle_json(current_bundle),
                file_name=f"{safe_name}.json",
                mime="application/json",
            )

            deterministic_export = []
            for row in decision.rows:
                deterministic_export.append(
                    {
                        "strategy": row.strategy.name,
                        "cost": row.strategy.cost,
                        "outcome": row.strategy.effect,
                        "status": row.status,
                        "compared_with": row.compared_with,
                        "incremental_cost": row.incremental_cost,
                        "incremental_effect": row.incremental_effect,
                        "icer": row.icer,
                        "nmb": row.nmb,
                    }
                )
            st.download_button(
                "Download deterministic results CSV",
                data=rows_to_csv(deterministic_export),
                file_name=f"{safe_name}_deterministic_results.csv",
                mime="text/csv",
            )

            record_col1, record_col2 = st.columns(2)
            if record_col1.button("Record current deterministic run"):
                deterministic_record = build_audit_record(
                    current_bundle,
                    analysis_type="deterministic_decision_tree",
                    run_settings={
                        "decision_threshold": threshold,
                        "currency": currency_code,
                        "outcome": outcome_code,
                        "cost_discount_rate": cost_discount_rate,
                        "outcome_discount_rate": outcome_discount_rate,
                        "included_cost_bearers": included_cost_bearers,
                    },
                    results={
                        "expected_values": [
                            {
                                "strategy_id": row.strategy_id,
                                "strategy_name": compiled.strategy_names[row.strategy_id],
                                "expected_cost": row.expected_cost,
                                "expected_outcome": row.expected_outcome,
                            }
                            for row in run.strategies
                        ],
                        "incremental_analysis": deterministic_export,
                        "preferred_by_nmb": list(decision.preferred_by_nmb),
                    },
                )
                st.session_state["dt_audit_trail"].append(deterministic_record)
                st.success("Deterministic run added to the in-session audit trail.")

            current_psa = st.session_state.get("dt_psa_result")
            current_signature = st.session_state.get("dt_psa_signature")
            if current_psa is not None and current_signature is not None:
                if record_col2.button("Record latest PSA run"):
                    psa_record = build_audit_record(
                        current_bundle,
                        analysis_type="probabilistic_sensitivity_analysis",
                        run_settings={
                            "iterations": current_psa.iterations,
                            "seed": current_psa.seed,
                            "decision_threshold": threshold,
                            "currency": currency_code,
                            "outcome": outcome_code,
                            "cost_discount_rate": cost_discount_rate,
                            "outcome_discount_rate": outcome_discount_rate,
                            "included_cost_bearers": included_cost_bearers,
                        },
                        results={
                            "mean_costs": {
                                sid: float(np.mean(current_psa.costs[sid]))
                                for sid in current_psa.strategy_ids
                            },
                            "mean_outcomes": {
                                sid: float(np.mean(current_psa.outcomes[sid]))
                                for sid in current_psa.strategy_ids
                            },
                        },
                        warnings=current_psa.warnings,
                    )
                    st.session_state["dt_audit_trail"].append(psa_record)
                    st.success("PSA run added to the in-session audit trail.")

                simulation_data = {"iteration": np.arange(1, current_psa.iterations + 1)}
                for sid in current_psa.strategy_ids:
                    simulation_data[f"cost_{sid}"] = current_psa.costs[sid]
                    simulation_data[f"outcome_{sid}"] = current_psa.outcomes[sid]
                for pid, draws in current_psa.parameter_draws.items():
                    simulation_data[f"draw_{pid}"] = draws
                st.download_button(
                    "Download latest PSA simulations CSV",
                    data=pd.DataFrame(simulation_data).to_csv(index=False),
                    file_name=f"{safe_name}_psa_simulations.csv",
                    mime="text/csv",
                )

            trail = st.session_state.get("dt_audit_trail", [])
            st.write(f"**Audit records in this session:** {len(trail)}")
            if trail:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Run ID": record["run_id"],
                                "Time (UTC)": record["run_at_utc"],
                                "Analysis": record["analysis_type"],
                                "Model hash": record["model_hash_sha256"],
                                "Warnings": len(record.get("warnings", [])),
                            }
                            for record in trail
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.download_button(
                    "Download audit trail JSON",
                    data=audit_trail_json(trail),
                    file_name=f"{safe_name}_audit_trail.json",
                    mime="application/json",
                )
                if st.button("Clear in-session audit trail"):
                    st.session_state["dt_audit_trail"] = []
                    st.rerun()
