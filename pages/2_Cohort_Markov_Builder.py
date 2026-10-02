"""Guided cohort state-transition modeller for Economic Evaluation Platform v0.5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model.currency import CURRENCIES
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.markov import MarkovValidationError, run_cohort_markov
from model.markov_builder import compile_markov_tables, markov_structure_to_dot
from model.markov_psa import markov_psa_configuration_warnings, run_markov_psa
from model.markov_reproducibility import (
    MarkovReproducibilityError,
    markov_run_fingerprint,
    validate_analysis_currency,
)
from model.markov_sensitivity import tornado_markov_inmb
from model.parameterisation import (
    DISTRIBUTION_PARAMETERISATIONS,
    ParameterisationError,
    canonical_distribution_parameters,
)
from model.psa import ceac, incremental_plane, pairwise_probability_cost_effective
from model.reference_cases import REFERENCE_CASES
from model.tree_builder import BuilderValidationError


st.set_page_config(page_title="Cohort Markov Builder", page_icon="🔁", layout="wide")

PARAMETER_CATEGORIES = [
    "clinical",
    "cost",
    "utility",
    "resource_use",
    "survival",
    "epidemiology",
    "other",
]
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


def _parameter_row(
    pid,
    label,
    value,
    unit,
    category,
    *,
    currency="",
    price_year=None,
    cost_bearers="",
    dsa=False,
    low=None,
    high=None,
    psa=False,
    family="",
    distribution_parameters=None,
):
    return {
        "id": pid,
        "label": label,
        "value": value,
        "unit": unit,
        "category": category,
        "currency": currency,
        "price_year": price_year,
        "cost_bearers": cost_bearers,
        "source_citation": "Illustrative input — replace with model evidence",
        "source_type": "user_assumption",
        "publication_year": None,
        "source_url": "",
        "source_details": "",
        "assumption": "Illustrative v0.5 example value.",
        "assumption_rationale": "Replace with a documented evidence-based assumption before decision use.",
        "dsa_enabled": dsa,
        "dsa_lower": low,
        "dsa_upper": high,
        "dsa_rationale": "Illustrative low/high range." if dsa else "Not represented in DSA.",
        "psa_enabled": psa,
        "psa_rationale": "Illustrative sampling uncertainty." if psa else "Not represented in PSA.",
        "distribution_family": family,
        "distribution_parameterisation": "Alpha + Beta" if family == "beta" else "",
        "distribution_parameters": distribution_parameters or {},
        "correlation_group": "",
        "notes": "",
    }


def _defaults():
    parameters = [
        _parameter_row(
            "p_prog_standard",
            "Progression probability — standard care",
            0.12,
            "probability/cycle",
            "clinical",
            dsa=True,
            low=0.08,
            high=0.16,
            psa=True,
            family="beta",
            distribution_parameters={"alpha": 12.0, "beta": 88.0},
        ),
        _parameter_row(
            "p_prog_new",
            "Progression probability — new treatment",
            0.08,
            "probability/cycle",
            "clinical",
            dsa=True,
            low=0.05,
            high=0.12,
            psa=True,
            family="beta",
            distribution_parameters={"alpha": 8.0, "beta": 92.0},
        ),
        _parameter_row(
            "p_death_stable",
            "Death probability — stable",
            0.03,
            "probability/cycle",
            "clinical",
            dsa=True,
            low=0.02,
            high=0.05,
        ),
        _parameter_row(
            "p_death_progressed",
            "Death probability — progressed",
            0.15,
            "probability/cycle",
            "clinical",
            dsa=True,
            low=0.10,
            high=0.20,
        ),
        _parameter_row(
            "cost_stable_standard",
            "Stable-state cost — standard care",
            1000.0,
            "currency/year",
            "cost",
            currency="GBP",
            price_year=2026,
            cost_bearers="health_system",
            dsa=True,
            low=800,
            high=1200,
        ),
        _parameter_row(
            "cost_stable_new",
            "Stable-state cost — new treatment",
            3000.0,
            "currency/year",
            "cost",
            currency="GBP",
            price_year=2026,
            cost_bearers="health_system",
            dsa=True,
            low=2400,
            high=3600,
        ),
        _parameter_row(
            "cost_progressed",
            "Progressed-state cost",
            6000.0,
            "currency/year",
            "cost",
            currency="GBP",
            price_year=2026,
            cost_bearers="health_system",
            dsa=True,
            low=4800,
            high=7200,
        ),
        _parameter_row(
            "utility_stable",
            "Stable-state utility",
            0.82,
            "utility",
            "utility",
            dsa=True,
            low=0.75,
            high=0.88,
        ),
        _parameter_row(
            "utility_progressed",
            "Progressed-state utility",
            0.55,
            "utility",
            "utility",
            dsa=True,
            low=0.45,
            high=0.65,
        ),
    ]
    states = [
        {"state_id": "stable", "state_name": "Stable", "absorbing": False},
        {"state_id": "progressed", "state_name": "Progressed", "absorbing": False},
        {"state_id": "dead", "state_name": "Dead", "absorbing": True},
    ]
    strategies = [
        {"strategy_id": "standard", "strategy_name": "Standard care"},
        {"strategy_id": "new", "strategy_name": "New treatment"},
    ]
    initial = [
        {"strategy_id": "standard", "state_id": "stable", "proportion": 1.0},
        {"strategy_id": "new", "state_id": "stable", "proportion": 1.0},
    ]
    transitions = []
    for sid, progression_parameter in (("standard", "p_prog_standard"), ("new", "p_prog_new")):
        transitions.extend(
            [
                {
                    "strategy_id": sid,
                    "origin_state": "stable",
                    "destination_state": "progressed",
                    "probability_parameter_id": progression_parameter,
                    "probability_mode": "direct",
                },
                {
                    "strategy_id": sid,
                    "origin_state": "stable",
                    "destination_state": "dead",
                    "probability_parameter_id": "p_death_stable",
                    "probability_mode": "direct",
                },
                {
                    "strategy_id": sid,
                    "origin_state": "stable",
                    "destination_state": "stable",
                    "probability_parameter_id": "",
                    "probability_mode": "residual",
                },
                {
                    "strategy_id": sid,
                    "origin_state": "progressed",
                    "destination_state": "dead",
                    "probability_parameter_id": "p_death_progressed",
                    "probability_mode": "direct",
                },
                {
                    "strategy_id": sid,
                    "origin_state": "progressed",
                    "destination_state": "progressed",
                    "probability_parameter_id": "",
                    "probability_mode": "residual",
                },
            ]
        )
    state_rewards = [
        {
            "strategy_id": "standard",
            "state_id": "stable",
            "parameter_id": "cost_stable_standard",
            "reward_type": "cost",
            "accrual": "per_year",
        },
        {
            "strategy_id": "new",
            "state_id": "stable",
            "parameter_id": "cost_stable_new",
            "reward_type": "cost",
            "accrual": "per_year",
        },
    ]
    for sid in ("standard", "new"):
        state_rewards.extend(
            [
                {
                    "strategy_id": sid,
                    "state_id": "progressed",
                    "parameter_id": "cost_progressed",
                    "reward_type": "cost",
                    "accrual": "per_year",
                },
                {
                    "strategy_id": sid,
                    "state_id": "stable",
                    "parameter_id": "utility_stable",
                    "reward_type": "outcome",
                    "accrual": "per_year",
                },
                {
                    "strategy_id": sid,
                    "state_id": "progressed",
                    "parameter_id": "utility_progressed",
                    "reward_type": "outcome",
                    "accrual": "per_year",
                },
            ]
        )
    return parameters, states, strategies, initial, transitions, state_rewards, []


def _ensure_state():
    keys = [
        "markov_parameters",
        "markov_states",
        "markov_strategies",
        "markov_initial",
        "markov_transitions",
        "markov_state_rewards",
        "markov_transition_rewards",
    ]
    if not all(key in st.session_state for key in keys):
        values = _defaults()
        for key, value in zip(keys, values):
            st.session_state[key] = value


def _records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return [dict(row) for row in value]


def _row_index(parameter_id):
    for index, row in enumerate(st.session_state.markov_parameters):
        if row.get("id") == parameter_id:
            return index
    return None


def _distribution_editor(row, key_prefix):
    family_options = ["beta", "gamma", "normal", "lognormal", "uniform", "dirichlet"]
    current_family = row.get("distribution_family") or "beta"
    family = st.selectbox(
        "Distribution family",
        family_options,
        index=family_options.index(current_family) if current_family in family_options else 0,
        key=f"{key_prefix}_family",
    )
    parameterisations = DISTRIBUTION_PARAMETERISATIONS[family]
    current_parameterisation = row.get("distribution_parameterisation") or parameterisations[0]
    if current_parameterisation not in parameterisations:
        current_parameterisation = parameterisations[0]
    parameterisation = st.selectbox(
        "Parameterisation",
        parameterisations,
        index=parameterisations.index(current_parameterisation),
        key=f"{key_prefix}_parameterisation",
    )
    stored = row.get("distribution_parameters")
    stored = stored if isinstance(stored, dict) else {}
    base = float(row.get("value") or 0.0)
    values = {}

    if family == "beta":
        if parameterisation == "Alpha + Beta":
            c1, c2 = st.columns(2)
            values["alpha"] = c1.number_input(
                "Alpha", min_value=0.000001, value=float(stored.get("alpha", 2.0)), key=f"{key_prefix}_alpha"
            )
            values["beta"] = c2.number_input(
                "Beta", min_value=0.000001, value=float(stored.get("beta", 2.0)), key=f"{key_prefix}_beta"
            )
        else:
            c1, c2 = st.columns(2)
            values["mean"] = c1.number_input(
                "Mean",
                min_value=0.000001,
                max_value=0.999999,
                value=min(max(base, 0.000001), 0.999999),
                key=f"{key_prefix}_mean",
            )
            values["se"] = c2.number_input(
                "Standard error", min_value=0.000001, value=0.02, key=f"{key_prefix}_se"
            )
    elif family == "gamma":
        if parameterisation == "Shape + Scale":
            c1, c2 = st.columns(2)
            values["shape"] = c1.number_input(
                "Shape", min_value=0.000001, value=float(stored.get("shape", 10.0)), key=f"{key_prefix}_shape"
            )
            values["scale"] = c2.number_input(
                "Scale",
                min_value=0.000001,
                value=float(stored.get("scale", max(base / 10.0, 1.0))),
                key=f"{key_prefix}_scale",
            )
        else:
            c1, c2 = st.columns(2)
            values["mean"] = c1.number_input(
                "Mean", min_value=0.000001, value=max(base, 0.000001), key=f"{key_prefix}_mean"
            )
            values["sd"] = c2.number_input(
                "Standard deviation",
                min_value=0.000001,
                value=max(abs(base) * 0.2, 0.1),
                key=f"{key_prefix}_sd",
            )
    elif family == "normal":
        if parameterisation == "Estimate + 95% CI":
            c1, c2, c3 = st.columns(3)
            values["estimate"] = c1.number_input("Estimate", value=base, key=f"{key_prefix}_estimate")
            values["lower_ci"] = c2.number_input(
                "Lower 95% CI", value=base - 0.1, key=f"{key_prefix}_lowerci"
            )
            values["upper_ci"] = c3.number_input(
                "Upper 95% CI", value=base + 0.1, key=f"{key_prefix}_upperci"
            )
        else:
            c1, c2 = st.columns(2)
            values["mean"] = c1.number_input(
                "Mean", value=float(stored.get("mean", base)), key=f"{key_prefix}_normalmean"
            )
            values["sd"] = c2.number_input(
                "Standard deviation",
                min_value=0.000001,
                value=float(stored.get("sd", max(abs(base) * 0.1, 0.1))),
                key=f"{key_prefix}_normalsd",
            )
    elif family == "lognormal":
        if parameterisation == "Arithmetic mean + SD":
            c1, c2 = st.columns(2)
            values["mean"] = c1.number_input(
                "Arithmetic mean",
                min_value=0.000001,
                value=max(base, 0.000001),
                key=f"{key_prefix}_logmean",
            )
            values["sd"] = c2.number_input(
                "Standard deviation",
                min_value=0.000001,
                value=max(abs(base) * 0.2, 0.1),
                key=f"{key_prefix}_logsd",
            )
        else:
            c1, c2 = st.columns(2)
            values["meanlog"] = c1.number_input(
                "Meanlog", value=float(stored.get("meanlog", 0.0)), key=f"{key_prefix}_meanlog"
            )
            values["sdlog"] = c2.number_input(
                "SDlog", min_value=0.000001, value=float(stored.get("sdlog", 0.2)), key=f"{key_prefix}_sdlog"
            )
    elif family == "uniform":
        c1, c2 = st.columns(2)
        values["low"] = c1.number_input(
            "Minimum", value=float(stored.get("low", base * 0.8)), key=f"{key_prefix}_low"
        )
        values["high"] = c2.number_input(
            "Maximum",
            value=float(stored.get("high", base * 1.2 if base else 1.0)),
            key=f"{key_prefix}_high",
        )
    else:
        values["alpha"] = st.number_input(
            "Alpha concentration",
            min_value=0.000001,
            value=float(stored.get("alpha", 10.0)),
            key=f"{key_prefix}_diralpha",
        )
        st.caption(
            "Dirichlet components must share a correlation group and represent one mutually exclusive probability vector."
        )

    try:
        canonical = canonical_distribution_parameters(
            family, parameterisation, values, base_value=base
        )
        return family, parameterisation, canonical, None
    except ParameterisationError as exc:
        return family, parameterisation, stored, str(exc)


def _parameter_editor(currency_code):
    rows = st.session_state.markov_parameters
    ids = [row.get("id") for row in rows if row.get("id")]
    if not ids:
        st.error("Add at least one parameter before editing parameter details.")
        return

    c1, c2 = st.columns([4, 1])
    selected = c1.selectbox("Parameter", ids, key="markov_selected_parameter")
    if c2.button("Add parameter", use_container_width=True):
        base = "new_parameter"
        suffix = 1
        new_id = base
        while new_id in ids:
            suffix += 1
            new_id = f"{base}_{suffix}"
        rows.append(_parameter_row(new_id, "New parameter", 0.0, "unit", "clinical"))
        st.session_state.markov_parameters = rows
        st.rerun()

    index = _row_index(selected)
    if index is None:
        return
    row = dict(rows[index])
    tabs = st.tabs(["Base case", "DSA", "PSA", "Evidence & assumptions"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        row["label"] = c1.text_input(
            "Label", value=str(row.get("label") or ""), key=f"mk_{selected}_label"
        )
        row["value"] = c2.number_input(
            "Base value", value=float(row.get("value") or 0.0), key=f"mk_{selected}_value"
        )
        c1, c2 = st.columns(2)
        row["unit"] = c1.text_input(
            "Unit", value=str(row.get("unit") or "unit"), key=f"mk_{selected}_unit"
        )
        current_category = row.get("category") or "clinical"
        row["category"] = c2.selectbox(
            "Category",
            PARAMETER_CATEGORIES,
            index=PARAMETER_CATEGORIES.index(current_category)
            if current_category in PARAMETER_CATEGORIES
            else 0,
            key=f"mk_{selected}_category",
        )
        if row["category"] == "cost":
            c1, c2 = st.columns(2)
            selected_currency = row.get("currency") or currency_code
            row["currency"] = c1.selectbox(
                "Currency",
                list(CURRENCIES),
                index=list(CURRENCIES).index(selected_currency)
                if selected_currency in CURRENCIES
                else 0,
                key=f"mk_{selected}_currency",
            )
            row["price_year"] = int(
                c2.number_input(
                    "Price year",
                    min_value=1900,
                    max_value=2100,
                    value=int(row.get("price_year") or 2026),
                    step=1,
                    key=f"mk_{selected}_priceyear",
                )
            )
            row["cost_bearers"] = st.text_input(
                "Cost bearers (comma-separated)",
                value=str(row.get("cost_bearers") or "health_system"),
                key=f"mk_{selected}_bearers",
            )
        else:
            row["currency"] = ""
            row["price_year"] = None
            row["cost_bearers"] = ""
        row["notes"] = st.text_area(
            "Notes", value=str(row.get("notes") or ""), key=f"mk_{selected}_notes"
        )

    with tabs[1]:
        row["dsa_enabled"] = st.toggle(
            "Include this parameter in deterministic sensitivity analysis",
            value=bool(row.get("dsa_enabled")),
            key=f"mk_{selected}_dsa",
        )
        if row["dsa_enabled"]:
            c1, c2 = st.columns(2)
            row["dsa_lower"] = c1.number_input(
                "Low value",
                value=float(
                    row.get("dsa_lower")
                    if row.get("dsa_lower") is not None
                    else row["value"] * 0.8
                ),
                key=f"mk_{selected}_dsalow",
            )
            row["dsa_upper"] = c2.number_input(
                "High value",
                value=float(
                    row.get("dsa_upper")
                    if row.get("dsa_upper") is not None
                    else row["value"] * 1.2
                ),
                key=f"mk_{selected}_dsahigh",
            )
            row["dsa_rationale"] = st.text_area(
                "Range rationale",
                value=str(row.get("dsa_rationale") or "Evidence-based deterministic range."),
                key=f"mk_{selected}_dsarationale",
            )
        else:
            row["dsa_lower"] = None
            row["dsa_upper"] = None
            row["dsa_rationale"] = str(
                row.get("dsa_rationale") or "Not represented in DSA."
            )

    with tabs[2]:
        row["psa_enabled"] = st.toggle(
            "Include this parameter in probabilistic sensitivity analysis",
            value=bool(row.get("psa_enabled")),
            key=f"mk_{selected}_psa",
        )
        if row["psa_enabled"]:
            family, parameterisation, canonical, error = _distribution_editor(
                row, f"mk_{selected}_psa"
            )
            row["distribution_family"] = family
            row["distribution_parameterisation"] = parameterisation
            row["distribution_parameters"] = canonical
            row["correlation_group"] = st.text_input(
                "Correlation / joint-sampling group",
                value=str(row.get("correlation_group") or ""),
                key=f"mk_{selected}_corr",
            )
            row["psa_rationale"] = st.text_area(
                "Distribution rationale",
                value=str(row.get("psa_rationale") or "Evidence-based sampling uncertainty."),
                key=f"mk_{selected}_psarationale",
            )
            if error:
                st.error(error)
        else:
            row["distribution_family"] = ""
            row["distribution_parameters"] = {}
            row["correlation_group"] = ""
            row["psa_rationale"] = str(
                row.get("psa_rationale") or "Not represented in PSA."
            )

    with tabs[3]:
        row["source_citation"] = st.text_area(
            "Citation / evidence source",
            value=str(row.get("source_citation") or ""),
            key=f"mk_{selected}_citation",
        )
        current_source = row.get("source_type") or "user_assumption"
        row["source_type"] = st.selectbox(
            "Source type",
            SOURCE_TYPES,
            index=SOURCE_TYPES.index(current_source)
            if current_source in SOURCE_TYPES
            else 0,
            key=f"mk_{selected}_sourcetype",
        )
        c1, c2 = st.columns(2)
        row["publication_year"] = (
            c1.number_input(
                "Publication year (0 = unspecified)",
                min_value=0,
                max_value=2100,
                value=int(row.get("publication_year") or 0),
                step=1,
                key=f"mk_{selected}_pubyear",
            )
            or None
        )
        row["source_url"] = c2.text_input(
            "Source URL", value=str(row.get("source_url") or ""), key=f"mk_{selected}_url"
        )
        row["assumption"] = st.text_area(
            "Assumption statement",
            value=str(row.get("assumption") or ""),
            key=f"mk_{selected}_assumption",
        )
        row["assumption_rationale"] = st.text_area(
            "Assumption rationale",
            value=str(row.get("assumption_rationale") or ""),
            key=f"mk_{selected}_assumptionwhy",
        )

    rows[index] = row
    st.session_state.markov_parameters = rows
    if st.button("Delete selected parameter", type="secondary"):
        st.session_state.markov_parameters = [
            item for item in rows if item.get("id") != selected
        ]
        st.rerun()


def _fingerprint(settings):
    return markov_run_fingerprint(
        parameter_rows=st.session_state.markov_parameters,
        state_rows=st.session_state.markov_states,
        strategy_rows=st.session_state.markov_strategies,
        initial_rows=st.session_state.markov_initial,
        transition_rows=st.session_state.markov_transitions,
        state_reward_rows=st.session_state.markov_state_rewards,
        transition_reward_rows=st.session_state.markov_transition_rewards,
        settings=settings,
    )


_ensure_state()

st.title("🔁 Cohort Markov / State-Transition Builder")
st.caption(
    "Version 0.5 — closed-cohort, discrete-time state-transition modelling with deterministic and probabilistic uncertainty"
)
st.info(
    "v0.5 deliberately starts with a transparent time-homogeneous cohort model. "
    "Transition inputs are probabilities for the selected model cycle. The app does not silently convert "
    "hazards/rates or infer competing-risk adjustments. Tunnel states, time-varying probabilities and "
    "semi-Markov history dependence are planned next."
)

reference_case_code = st.sidebar.selectbox(
    "Methods profile",
    list(REFERENCE_CASES),
    format_func=lambda code: REFERENCE_CASES[code].name,
)
profile = REFERENCE_CASES[reference_case_code]
outcome_code = st.sidebar.selectbox(
    "Economic outcome",
    list(OUTCOME_MEASURES),
    index=list(OUTCOME_MEASURES).index(profile.preferred_outcome_code)
    if profile.preferred_outcome_code in OUTCOME_MEASURES
    else 0,
)
currency_code = st.sidebar.selectbox(
    "Analysis currency",
    list(CURRENCIES),
    index=list(CURRENCIES).index(profile.analysis_currency)
    if profile.analysis_currency in CURRENCIES
    else 0,
)
threshold_default = (
    0.0
    if profile.threshold_range is None
    else float((profile.threshold_range.lower + profile.threshold_range.upper) / 2)
)
threshold = st.sidebar.number_input(
    "Decision threshold", min_value=0.0, value=threshold_default, step=1000.0
)
cost_discount_rate = st.sidebar.number_input(
    "Annual cost discount rate",
    min_value=0.0,
    max_value=0.99,
    value=float(profile.cost_discount_rate),
    format="%.4f",
)
outcome_discount_rate = st.sidebar.number_input(
    "Annual outcome discount rate",
    min_value=0.0,
    max_value=0.99,
    value=float(profile.outcome_discount_rate),
    format="%.4f",
)
included_cost_bearers_text = st.sidebar.text_input(
    "Included cost bearers", value=", ".join(profile.perspective.included_cost_bearers)
)
included_cost_bearers = tuple(
    item.strip() for item in included_cost_bearers_text.split(",") if item.strip()
)
if profile.threshold_range is None:
    st.sidebar.caption(
        "This reference case does not prescribe a single monetary threshold; the value above is used only for NMB/INMB analysis."
    )

methods_tab, parameters_tab, structure_tab, rewards_tab, analyse_tab = st.tabs(
    ["1 · Methods", "2 · Parameters", "3 · States & transitions", "4 · Rewards", "5 · Analyse"]
)

with methods_tab:
    st.subheader("Cycle structure and time horizon")
    c1, c2, c3 = st.columns(3)
    cycle_months = c1.selectbox(
        "Cycle length",
        [1, 3, 6, 12],
        index=3,
        format_func=lambda value: f"{value} month" if value == 1 else f"{value} months",
    )
    horizon_years = int(
        c2.number_input(
            "Maximum horizon (years)", min_value=1, max_value=200, value=20, step=1
        )
    )
    termination_label = c3.selectbox(
        "Termination", ["Fixed horizon", "Cohort depletion"]
    )
    cycle_length_years = cycle_months / 12.0
    max_cycles = int(round(horizon_years / cycle_length_years))
    termination_mode = (
        "fixed_cycles" if termination_label == "Fixed horizon" else "cohort_depletion"
    )
    depletion_threshold = st.number_input(
        "Non-absorbing cohort threshold for depletion stopping",
        min_value=0.0,
        max_value=0.5,
        value=0.0001,
        format="%.6f",
        disabled=termination_mode != "cohort_depletion",
    )

    st.subheader("Within-cycle accrual")
    c1, c2 = st.columns(2)
    state_accrual_timing = c1.selectbox(
        "State reward accrual",
        ["half_cycle", "start", "end"],
        format_func=lambda value: {
            "half_cycle": "Half-cycle / trapezoidal",
            "start": "Start of cycle",
            "end": "End of cycle",
        }[value],
    )
    transition_reward_timing = c2.selectbox(
        "Transition-event reward timing",
        ["mid_cycle", "start", "end"],
        format_func=lambda value: {
            "mid_cycle": "Mid-cycle",
            "start": "Start of cycle",
            "end": "End of cycle",
        }[value],
    )
    st.caption(
        "Half-cycle accrual averages start- and end-of-cycle state occupancy. It is an explicit approximation, "
        "not an automatic requirement; document why the selected approach is appropriate for the cycle length and clinical process."
    )
    st.metric("Maximum cycles", max_cycles)
    if cycle_months != 12:
        st.warning(
            "The illustrative transition probabilities were entered for the default 12-month cycle. "
            "If you change cycle length, replace them with evidence-appropriate probabilities for the new cycle; v0.5 does not silently rescale them."
        )

with parameters_tab:
    st.subheader("Parameter library")
    st.caption(
        "Each parameter keeps one base value plus independent DSA and PSA specifications. "
        "Transition probabilities must correspond to the selected cycle length."
    )
    _parameter_editor(currency_code)
    with st.expander("Advanced · inspect complete parameter table"):
        st.dataframe(
            pd.DataFrame(st.session_state.markov_parameters), use_container_width=True
        )

with structure_tab:
    st.subheader("Health states")
    states_df = st.data_editor(
        pd.DataFrame(st.session_state.markov_states),
        num_rows="dynamic",
        use_container_width=True,
        key="markov_states_editor",
    )
    st.session_state.markov_states = _records(states_df)

    st.subheader("Strategies")
    strategies_df = st.data_editor(
        pd.DataFrame(st.session_state.markov_strategies),
        num_rows="dynamic",
        use_container_width=True,
        key="markov_strategies_editor",
    )
    st.session_state.markov_strategies = _records(strategies_df)

    st.subheader("Initial cohort distribution")
    initial_df = st.data_editor(
        pd.DataFrame(st.session_state.markov_initial),
        num_rows="dynamic",
        use_container_width=True,
        key="markov_initial_editor",
    )
    st.session_state.markov_initial = _records(initial_df)

    st.subheader("Strategy-specific transitions")
    st.caption(
        "Use one residual transition per origin state when a destination probability should equal 1 minus the other outgoing probabilities. "
        "Absorbing states may omit their self-transition; the engine supplies probability 1 automatically."
    )
    transitions_df = st.data_editor(
        pd.DataFrame(st.session_state.markov_transitions),
        num_rows="dynamic",
        use_container_width=True,
        key="markov_transitions_editor",
        column_config={
            "probability_mode": st.column_config.SelectboxColumn(
                "probability_mode", options=["direct", "complement", "residual"]
            ),
        },
    )
    st.session_state.markov_transitions = _records(transitions_df)

    strategy_ids_for_diagram = [
        row.get("strategy_id")
        for row in st.session_state.markov_strategies
        if row.get("strategy_id")
    ]
    if strategy_ids_for_diagram:
        diagram_strategy = st.selectbox(
            "Diagram strategy", strategy_ids_for_diagram, key="markov_diagram_strategy"
        )
        st.graphviz_chart(
            markov_structure_to_dot(
                st.session_state.markov_states,
                st.session_state.markov_transitions,
                strategy_id=diagram_strategy,
            ),
            use_container_width=True,
        )

with rewards_tab:
    st.subheader("State rewards")
    st.caption(
        "Per-year rewards are multiplied by cycle length. Utilities used for QALYs should normally be entered as per-year state rewards; "
        "a value of 1 per year on living states can generate life-years."
    )
    state_rewards_df = st.data_editor(
        pd.DataFrame(st.session_state.markov_state_rewards),
        num_rows="dynamic",
        use_container_width=True,
        key="markov_state_rewards_editor",
        column_config={
            "reward_type": st.column_config.SelectboxColumn(
                "reward_type", options=["cost", "outcome"]
            ),
            "accrual": st.column_config.SelectboxColumn(
                "accrual", options=["per_year", "per_cycle"]
            ),
        },
    )
    st.session_state.markov_state_rewards = _records(state_rewards_df)

    st.subheader("Transition-event rewards")
    st.caption(
        "Use this for costs or health effects that occur when a specific transition occurs, rather than for time spent in a state."
    )
    transition_rewards_df = st.data_editor(
        pd.DataFrame(
            st.session_state.markov_transition_rewards,
            columns=[
                "strategy_id",
                "origin_state",
                "destination_state",
                "parameter_id",
                "reward_type",
            ],
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="markov_transition_rewards_editor",
        column_config={
            "reward_type": st.column_config.SelectboxColumn(
                "reward_type", options=["cost", "outcome"]
            )
        },
    )
    st.session_state.markov_transition_rewards = _records(transition_rewards_df)

run_settings = {
    "reference_case_code": reference_case_code,
    "outcome_code": outcome_code,
    "analysis_currency": currency_code,
    "decision_threshold": threshold,
    "cost_discount_rate": cost_discount_rate,
    "outcome_discount_rate": outcome_discount_rate,
    "included_cost_bearers": included_cost_bearers,
    "cycle_length_years": cycle_length_years,
    "max_cycles": max_cycles,
    "state_accrual_timing": state_accrual_timing,
    "transition_reward_timing": transition_reward_timing,
    "termination_mode": termination_mode,
    "depletion_threshold": depletion_threshold,
}
current_fingerprint = _fingerprint(run_settings)

compiled = None
base_result = None
compile_error = None
try:
    validate_analysis_currency(st.session_state.markov_parameters, currency_code)
    compiled = compile_markov_tables(
        st.session_state.markov_parameters,
        st.session_state.markov_states,
        st.session_state.markov_strategies,
        st.session_state.markov_initial,
        st.session_state.markov_transitions,
        st.session_state.markov_state_rewards,
        st.session_state.markov_transition_rewards,
        cycle_length_years=cycle_length_years,
        max_cycles=max_cycles,
        state_accrual_timing=state_accrual_timing,
        transition_reward_timing=transition_reward_timing,
        termination_mode=termination_mode,
        depletion_threshold=depletion_threshold,
    )
    base_result = run_cohort_markov(
        compiled.model,
        compiled.parameters,
        included_cost_bearers=included_cost_bearers,
        cost_discount_rate=cost_discount_rate,
        outcome_discount_rate=outcome_discount_rate,
    )
except (
    BuilderValidationError,
    MarkovValidationError,
    MarkovReproducibilityError,
    ValueError,
) as exc:
    compile_error = str(exc)

with analyse_tab:
    if compile_error:
        st.error(compile_error)
        st.stop()

    assert compiled is not None and base_result is not None
    st.success("Model structure, currency consistency and transition matrices validated.")
    currency = CURRENCIES[currency_code]
    outcome = OUTCOME_MEASURES[outcome_code]

    economic_strategies = [
        Strategy(row.label, row.expected_cost, row.expected_outcome)
        for row in base_result.strategies
    ]
    incremental = fully_incremental_analysis(economic_strategies, threshold)
    inc_by_name = {row.strategy.name: row for row in incremental.rows}
    result_rows = []
    for row in base_result.strategies:
        inc = inc_by_name[row.label]
        result_rows.append(
            {
                "Strategy": row.label,
                f"Expected cost ({currency_code})": row.expected_cost,
                f"Expected {outcome.unit}": row.expected_outcome,
                "NMB": threshold * row.expected_outcome - row.expected_cost,
                "Frontier status": inc.status,
                "Incremental cost": inc.incremental_cost,
                "Incremental outcome": inc.incremental_effect,
                "ICER": inc.icer,
                "Cycles run": row.cycles_run,
            }
        )
    st.subheader("Base-case economic results")
    st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Costs shown in {currency_code}; outcomes shown as {outcome.label}. Fully incremental analysis uses a decision threshold of "
        f"{currency.symbol}{threshold:,.0f} per {outcome.unit}."
    )

    st.subheader("Cohort trace")
    strategy_map = {row.strategy_id: row for row in base_result.strategies}
    trace_strategy = st.selectbox(
        "Trace strategy",
        list(strategy_map),
        format_func=lambda sid: strategy_map[sid].label,
    )
    trace = strategy_map[trace_strategy]
    trace_frame = pd.DataFrame(trace.trace, columns=trace.state_ids)
    trace_frame.insert(
        0, "Time (years)", np.arange(len(trace.trace)) * cycle_length_years
    )
    long_trace = trace_frame.melt(
        id_vars="Time (years)", var_name="State", value_name="Proportion"
    )
    st.plotly_chart(
        px.line(
            long_trace,
            x="Time (years)",
            y="Proportion",
            color="State",
            title=f"State occupancy — {trace.label}",
        ),
        use_container_width=True,
    )
    if trace.stopped_early:
        st.info(
            f"Simulation stopped after {trace.cycles_run} cycles because the non-absorbing cohort fell below the depletion threshold."
        )

    analysis_mode = st.radio(
        "Uncertainty analysis",
        ["Deterministic (DSA)", "Probabilistic (PSA)"],
        horizontal=True,
    )
    strategy_ids = [row.strategy_id for row in base_result.strategies]
    labels = {row.strategy_id: row.label for row in base_result.strategies}
    c1, c2 = st.columns(2)
    comparator_id = c1.selectbox(
        "Comparator",
        strategy_ids,
        format_func=lambda sid: labels[sid],
        key="markov_comparator",
    )
    intervention_options = [sid for sid in strategy_ids if sid != comparator_id]
    intervention_id = c2.selectbox(
        "Intervention",
        intervention_options,
        format_func=lambda sid: labels[sid],
        key="markov_intervention",
    )

    context = dict(
        intervention_id=intervention_id,
        comparator_id=comparator_id,
        willingness_to_pay=threshold,
        included_cost_bearers=included_cost_bearers,
        cost_discount_rate=cost_discount_rate,
        outcome_discount_rate=outcome_discount_rate,
    )

    if analysis_mode == "Deterministic (DSA)":
        st.subheader("One-way tornado analysis")
        try:
            tornado = tornado_markov_inmb(compiled.model, compiled.parameters, **context)
            if not tornado:
                st.info("No parameters are enabled for DSA.")
            else:
                tornado_df = pd.DataFrame(
                    [
                        {
                            "Parameter": row.label,
                            "Low value": row.low_value,
                            "High value": row.high_value,
                            "INMB at low": row.low_inmb,
                            "INMB at high": row.high_inmb,
                            "Impact": row.impact,
                        }
                        for row in tornado
                    ]
                )
                st.dataframe(tornado_df, use_container_width=True, hide_index=True)
                chart_df = tornado_df.sort_values("Impact")
                st.plotly_chart(
                    px.bar(
                        chart_df,
                        x="Impact",
                        y="Parameter",
                        orientation="h",
                        title="Tornado ranking by maximum absolute change in INMB",
                    ),
                    use_container_width=True,
                )
        except (MarkovValidationError, ValueError) as exc:
            st.error(str(exc))
    else:
        st.subheader("Probabilistic sensitivity analysis")
        c1, c2 = st.columns(2)
        iterations = int(
            c1.number_input(
                "Iterations", min_value=100, max_value=100000, value=1000, step=100
            )
        )
        seed = int(
            c2.number_input(
                "Random seed",
                min_value=0,
                max_value=2_147_483_647,
                value=2026,
                step=1,
            )
        )
        try:
            warnings = markov_psa_configuration_warnings(compiled.parameters)
            for warning in warnings:
                st.warning(warning)
        except ValueError as exc:
            st.error(str(exc))

        if st.button("Run Markov PSA", type="primary"):
            try:
                st.session_state.markov_psa_result = run_markov_psa(
                    compiled.model,
                    compiled.parameters,
                    iterations=iterations,
                    seed=seed,
                    included_cost_bearers=included_cost_bearers,
                    cost_discount_rate=cost_discount_rate,
                    outcome_discount_rate=outcome_discount_rate,
                )
                st.session_state.markov_psa_fingerprint = current_fingerprint
            except ValueError as exc:
                st.error(str(exc))

        psa_result = st.session_state.get("markov_psa_result")
        psa_fingerprint = st.session_state.get("markov_psa_fingerprint")
        if psa_result is not None and psa_fingerprint != current_fingerprint:
            st.info(
                "The saved in-session PSA result belongs to an earlier model/settings state and is not displayed. Run PSA again for the current model."
            )
            psa_result = None

        if psa_result is not None:
            de, dc = incremental_plane(
                psa_result,
                intervention_id=intervention_id,
                comparator_id=comparator_id,
            )
            plane = pd.DataFrame(
                {"Incremental outcome": de, "Incremental cost": dc}
            )
            st.plotly_chart(
                px.scatter(
                    plane,
                    x="Incremental outcome",
                    y="Incremental cost",
                    opacity=0.45,
                    title=f"Cost-effectiveness plane — {labels[intervention_id]} vs {labels[comparator_id]}",
                ),
                use_container_width=True,
            )
            probability = pairwise_probability_cost_effective(
                psa_result,
                intervention_id=intervention_id,
                comparator_id=comparator_id,
                willingness_to_pay=threshold,
            )
            st.metric(
                "Pairwise probability cost-effective at selected threshold",
                f"{probability:.1%}",
            )
            max_threshold = max(threshold * 2, 1.0)
            thresholds = np.linspace(0.0, max_threshold, 51)
            curve = ceac(psa_result, thresholds)
            ceac_rows = []
            for sid, values in curve.probabilities.items():
                for threshold_value, probability_value in zip(
                    curve.thresholds, values
                ):
                    ceac_rows.append(
                        {
                            "Threshold": threshold_value,
                            "Probability cost-effective": probability_value,
                            "Strategy": labels.get(sid, sid),
                        }
                    )
            st.plotly_chart(
                px.line(
                    pd.DataFrame(ceac_rows),
                    x="Threshold",
                    y="Probability cost-effective",
                    color="Strategy",
                    title="Cost-effectiveness acceptability curve",
                ),
                use_container_width=True,
            )

    st.download_button(
        "Download deterministic results CSV",
        data=pd.DataFrame(result_rows).to_csv(index=False),
        file_name="markov_deterministic_results.csv",
        mime="text/csv",
    )
