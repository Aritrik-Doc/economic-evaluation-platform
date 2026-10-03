"""Reusable guided parameter editor for Streamlit model builders."""

from __future__ import annotations

from math import exp, sqrt
from typing import Any

import numpy as np
import streamlit as st

from model.guided_markov import slugify, unique_id
from model.parameterisation import (
    DISTRIBUTION_PARAMETERISATIONS,
    ParameterisationError,
    canonical_distribution_parameters,
    migrate_parameter_row,
    parse_legacy_distribution_parameters,
)
from model.uncertainty_defaults import suggest_distribution_from_fields


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
DISTRIBUTION_FAMILIES = ["beta", "gamma", "normal", "lognormal", "uniform", "dirichlet"]


def _distribution_defaults(family: str, params: dict[str, float], base: float) -> dict[str, float]:
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
        return {
            "mean": mean,
            "sd": sd,
            "estimate": mean,
            "lower_ci": mean - 1.96 * sd,
            "upper_ci": mean + 1.96 * sd,
        }
    if family == "lognormal":
        meanlog = float(params.get("meanlog", np.log(base) if base > 0 else 0.0))
        sdlog = max(float(params.get("sdlog", 0.1)), 1e-9)
        arithmetic_mean = exp(meanlog + sdlog * sdlog / 2)
        arithmetic_sd = sqrt((exp(sdlog * sdlog) - 1) * exp(2 * meanlog + sdlog * sdlog))
        return {
            "meanlog": meanlog,
            "sdlog": sdlog,
            "mean": arithmetic_mean,
            "sd": arithmetic_sd,
        }
    if family == "uniform":
        width = max(abs(base) * 0.2, 0.1)
        return {
            "low": float(params.get("low", base - width)),
            "high": float(params.get("high", base + width)),
        }
    if family == "dirichlet":
        return {"alpha": max(float(params.get("alpha", max(abs(base) * 100, 1))), 1e-9)}
    return {}


def _default_parameter(label: str, pid: str, value: float, unit: str, category: str, currency_code: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": pid,
        "label": label,
        "value": value,
        "unit": unit,
        "category": category,
        "source_citation": "User-entered parameter — add the evidence source before substantive use",
        "source_type": "user_assumption",
        "publication_year": None,
        "source_url": "",
        "source_details": "",
        "assumption": "User-entered model parameter.",
        "assumption_rationale": "Complete the modelling rationale before substantive use.",
        "dsa_enabled": False,
        "dsa_lower": None,
        "dsa_upper": None,
        "dsa_rationale": "Not represented in DSA.",
        "psa_enabled": False,
        "psa_rationale": "Not represented in PSA.",
        "distribution_family": "",
        "distribution_parameterisation": "",
        "distribution_parameters": {},
        "correlation_group": "",
        "notes": "",
        "currency": "",
        "price_year": None,
        "cost_bearers": "",
    }
    if category == "cost":
        row.update(currency=currency_code, price_year=2026, cost_bearers="health_system")
    return row


def render_parameter_library(*, session_key: str, currency_code: str, key_prefix: str) -> None:
    rows = [migrate_parameter_row(row) for row in st.session_state[session_key]]
    st.session_state[session_key] = rows
    st.subheader("Parameter library")
    st.caption(
        "A parameter keeps one base-case value plus independent DSA and PSA specifications. "
        "Evidence, assumptions and uncertainty remain explicit regardless of model type."
    )

    if rows:
        labels = {str(row["id"]): str(row.get("label") or row["id"]) for row in rows}
        selected = st.selectbox(
            "Parameter to edit",
            list(labels),
            format_func=lambda pid: f"{labels[pid]}  (`{pid}`)",
            key=f"{key_prefix}_selected_parameter",
        )
        index = next(i for i, row in enumerate(rows) if str(row.get("id")) == selected)
        row = dict(rows[index])
        tabs = st.tabs(["Base case", "DSA", "PSA", "Evidence & assumptions"])

        with tabs[0]:
            c1, c2 = st.columns(2)
            row["label"] = c1.text_input(
                "Parameter name", str(row.get("label") or ""), key=f"{key_prefix}_{selected}_label"
            )
            row["value"] = c2.number_input(
                "Base-case value",
                value=float(row.get("value") or 0.0),
                format="%.10g",
                key=f"{key_prefix}_{selected}_value",
            )
            c1, c2 = st.columns(2)
            row["unit"] = c1.text_input(
                "Unit", str(row.get("unit") or ""), key=f"{key_prefix}_{selected}_unit"
            )
            current_category = str(row.get("category") or "other")
            row["category"] = c2.selectbox(
                "Category",
                PARAMETER_CATEGORIES,
                index=PARAMETER_CATEGORIES.index(current_category)
                if current_category in PARAMETER_CATEGORIES
                else len(PARAMETER_CATEGORIES) - 1,
                key=f"{key_prefix}_{selected}_category",
            )
            st.caption(f"Stable parameter ID: `{selected}`")
            if row["category"] == "cost":
                c1, c2 = st.columns(2)
                row["currency"] = c1.text_input(
                    "Currency code",
                    str(row.get("currency") or currency_code),
                    key=f"{key_prefix}_{selected}_currency",
                ).upper()
                row["price_year"] = int(
                    c2.number_input(
                        "Price year",
                        min_value=1900,
                        max_value=2200,
                        value=int(row.get("price_year") or 2026),
                        step=1,
                        key=f"{key_prefix}_{selected}_price_year",
                    )
                )
                row["cost_bearers"] = st.text_input(
                    "Cost bearer(s), comma separated",
                    str(row.get("cost_bearers") or "health_system"),
                    key=f"{key_prefix}_{selected}_bearers",
                )

        with tabs[1]:
            row["dsa_enabled"] = st.toggle(
                "Include this parameter in deterministic sensitivity analysis",
                value=bool(row.get("dsa_enabled")),
                key=f"{key_prefix}_{selected}_dsa_on",
            )
            if row["dsa_enabled"]:
                c1, c2 = st.columns(2)
                base = float(row.get("value") or 0.0)
                row["dsa_lower"] = c1.number_input(
                    "Low value",
                    value=float(row.get("dsa_lower") if row.get("dsa_lower") is not None else base * 0.8),
                    format="%.10g",
                    key=f"{key_prefix}_{selected}_dsa_low",
                )
                row["dsa_upper"] = c2.number_input(
                    "High value",
                    value=float(row.get("dsa_upper") if row.get("dsa_upper") is not None else base * 1.2),
                    format="%.10g",
                    key=f"{key_prefix}_{selected}_dsa_high",
                )
                row["dsa_rationale"] = st.text_area(
                    "Why are these bounds appropriate?",
                    str(row.get("dsa_rationale") or ""),
                    key=f"{key_prefix}_{selected}_dsa_rationale",
                )
            else:
                row["dsa_lower"] = None
                row["dsa_upper"] = None
                row["dsa_rationale"] = st.text_area(
                    "Why is DSA not represented?",
                    str(row.get("dsa_rationale") or "Not represented in DSA."),
                    key=f"{key_prefix}_{selected}_dsa_rationale_off",
                )

        with tabs[2]:
            row["psa_enabled"] = st.toggle(
                "Include this parameter in probabilistic sensitivity analysis",
                value=bool(row.get("psa_enabled")),
                key=f"{key_prefix}_{selected}_psa_on",
            )
            suggestion = suggest_distribution_from_fields(
                label=str(row.get("label") or selected),
                category=str(row.get("category") or "other"),
                unit=str(row.get("unit") or ""),
                value=float(row.get("value") or 0.0),
            )
            st.info(
                f"Suggested starting family: **{suggestion.family.title()}** — {suggestion.rationale}"
            )
            if suggestion.caution:
                st.caption("Caution: " + suggestion.caution)
            if row["psa_enabled"]:
                existing_family = str(row.get("distribution_family") or suggestion.family).lower()
                if existing_family not in DISTRIBUTION_FAMILIES:
                    existing_family = suggestion.family if suggestion.family in DISTRIBUTION_FAMILIES else "normal"
                family = st.selectbox(
                    "Distribution family",
                    DISTRIBUTION_FAMILIES,
                    index=DISTRIBUTION_FAMILIES.index(existing_family),
                    key=f"{key_prefix}_{selected}_family",
                )
                row["distribution_family"] = family
                parameterisations = DISTRIBUTION_PARAMETERISATIONS[family]
                existing_parameterisation = str(row.get("distribution_parameterisation") or parameterisations[0])
                if existing_parameterisation not in parameterisations:
                    existing_parameterisation = parameterisations[0]
                parameterisation = st.selectbox(
                    "How would you like to enter it?",
                    parameterisations,
                    index=parameterisations.index(existing_parameterisation),
                    key=f"{key_prefix}_{selected}_parameterisation",
                )
                row["distribution_parameterisation"] = parameterisation
                stored = parse_legacy_distribution_parameters(row.get("distribution_parameters"))
                defaults = _distribution_defaults(family, stored, float(row["value"]))
                values: dict[str, float] = {}
                if family == "beta" and parameterisation == "Alpha + Beta":
                    c1, c2 = st.columns(2)
                    values["alpha"] = c1.number_input("Alpha", min_value=1e-9, value=float(defaults["alpha"]), format="%.10g", key=f"{key_prefix}_{selected}_alpha")
                    values["beta"] = c2.number_input("Beta", min_value=1e-9, value=float(defaults["beta"]), format="%.10g", key=f"{key_prefix}_{selected}_beta")
                elif family == "beta":
                    c1, c2 = st.columns(2)
                    values["mean"] = c1.number_input("Mean", min_value=1e-9, max_value=1 - 1e-9, value=float(defaults["mean"]), format="%.10g", key=f"{key_prefix}_{selected}_beta_mean")
                    values["se"] = c2.number_input("Standard error", min_value=1e-9, value=float(defaults["se"]), format="%.10g", key=f"{key_prefix}_{selected}_beta_se")
                elif family == "gamma" and parameterisation == "Shape + Scale":
                    c1, c2 = st.columns(2)
                    values["shape"] = c1.number_input("Shape", min_value=1e-9, value=float(defaults["shape"]), format="%.10g", key=f"{key_prefix}_{selected}_shape")
                    values["scale"] = c2.number_input("Scale", min_value=1e-9, value=float(defaults["scale"]), format="%.10g", key=f"{key_prefix}_{selected}_scale")
                elif family == "gamma":
                    c1, c2 = st.columns(2)
                    values["mean"] = c1.number_input("Mean", min_value=1e-9, value=max(float(defaults["mean"]), 1e-9), format="%.10g", key=f"{key_prefix}_{selected}_gamma_mean")
                    values["sd"] = c2.number_input("Standard deviation", min_value=1e-9, value=float(defaults["sd"]), format="%.10g", key=f"{key_prefix}_{selected}_gamma_sd")
                elif family == "normal" and parameterisation == "Mean + SD":
                    c1, c2 = st.columns(2)
                    values["mean"] = c1.number_input("Mean", value=float(defaults["mean"]), format="%.10g", key=f"{key_prefix}_{selected}_normal_mean")
                    values["sd"] = c2.number_input("Standard deviation", min_value=1e-9, value=float(defaults["sd"]), format="%.10g", key=f"{key_prefix}_{selected}_normal_sd")
                elif family == "normal":
                    c1, c2, c3 = st.columns(3)
                    values["estimate"] = c1.number_input("Estimate", value=float(defaults["estimate"]), format="%.10g", key=f"{key_prefix}_{selected}_estimate")
                    values["lower_ci"] = c2.number_input("Lower 95% CI", value=float(defaults["lower_ci"]), format="%.10g", key=f"{key_prefix}_{selected}_lower_ci")
                    values["upper_ci"] = c3.number_input("Upper 95% CI", value=float(defaults["upper_ci"]), format="%.10g", key=f"{key_prefix}_{selected}_upper_ci")
                elif family == "lognormal" and parameterisation == "Meanlog + SDlog":
                    c1, c2 = st.columns(2)
                    values["meanlog"] = c1.number_input("Mean on log scale", value=float(defaults["meanlog"]), format="%.10g", key=f"{key_prefix}_{selected}_meanlog")
                    values["sdlog"] = c2.number_input("SD on log scale", min_value=1e-9, value=float(defaults["sdlog"]), format="%.10g", key=f"{key_prefix}_{selected}_sdlog")
                elif family == "lognormal":
                    c1, c2 = st.columns(2)
                    values["mean"] = c1.number_input("Arithmetic mean", min_value=1e-9, value=max(float(defaults["mean"]), 1e-9), format="%.10g", key=f"{key_prefix}_{selected}_lognormal_mean")
                    values["sd"] = c2.number_input("Standard deviation", min_value=1e-9, value=float(defaults["sd"]), format="%.10g", key=f"{key_prefix}_{selected}_lognormal_sd")
                elif family == "uniform":
                    c1, c2 = st.columns(2)
                    values["low"] = c1.number_input("Minimum", value=float(defaults["low"]), format="%.10g", key=f"{key_prefix}_{selected}_uniform_low")
                    values["high"] = c2.number_input("Maximum", value=float(defaults["high"]), format="%.10g", key=f"{key_prefix}_{selected}_uniform_high")
                else:
                    values["alpha"] = st.number_input("Alpha concentration", min_value=1e-9, value=float(defaults["alpha"]), format="%.10g", key=f"{key_prefix}_{selected}_dirichlet_alpha")
                    st.caption("Use the same correlation group for all mutually exclusive probability components sampled jointly as a Dirichlet vector.")
                try:
                    row["distribution_parameters"] = canonical_distribution_parameters(
                        family,
                        parameterisation,
                        values,
                        base_value=float(row["value"]),
                    )
                except ParameterisationError as exc:
                    st.error(str(exc))
                row["correlation_group"] = st.text_input(
                    "Correlation / joint-sampling group",
                    str(row.get("correlation_group") or ""),
                    key=f"{key_prefix}_{selected}_correlation",
                )
                row["psa_rationale"] = st.text_area(
                    "Distribution rationale",
                    str(row.get("psa_rationale") or ""),
                    key=f"{key_prefix}_{selected}_psa_rationale",
                )
            else:
                row["distribution_family"] = ""
                row["distribution_parameters"] = {}
                row["correlation_group"] = ""
                row["psa_rationale"] = st.text_area(
                    "Why is PSA not represented?",
                    str(row.get("psa_rationale") or "Not represented in PSA."),
                    key=f"{key_prefix}_{selected}_psa_rationale_off",
                )

        with tabs[3]:
            row["source_citation"] = st.text_area(
                "Citation / evidence source",
                str(row.get("source_citation") or ""),
                key=f"{key_prefix}_{selected}_citation",
            )
            current_source = str(row.get("source_type") or "user_assumption")
            row["source_type"] = st.selectbox(
                "Source type",
                SOURCE_TYPES,
                index=SOURCE_TYPES.index(current_source) if current_source in SOURCE_TYPES else SOURCE_TYPES.index("other"),
                key=f"{key_prefix}_{selected}_source_type",
            )
            c1, c2 = st.columns(2)
            year = c1.number_input(
                "Publication year (0 = unspecified)",
                min_value=0,
                max_value=2200,
                value=int(row.get("publication_year") or 0),
                step=1,
                key=f"{key_prefix}_{selected}_publication_year",
            )
            row["publication_year"] = int(year) or None
            row["source_url"] = c2.text_input(
                "Source URL",
                str(row.get("source_url") or ""),
                key=f"{key_prefix}_{selected}_source_url",
            )
            row["source_details"] = st.text_area(
                "Source details",
                str(row.get("source_details") or ""),
                key=f"{key_prefix}_{selected}_source_details",
            )
            row["assumption"] = st.text_area(
                "Assumption statement",
                str(row.get("assumption") or ""),
                key=f"{key_prefix}_{selected}_assumption",
            )
            row["assumption_rationale"] = st.text_area(
                "Assumption rationale",
                str(row.get("assumption_rationale") or ""),
                key=f"{key_prefix}_{selected}_assumption_rationale",
            )
            row["notes"] = st.text_area(
                "Notes",
                str(row.get("notes") or ""),
                key=f"{key_prefix}_{selected}_notes",
            )

        rows[index] = row
        st.session_state[session_key] = rows
        if st.button("Delete selected parameter", key=f"{key_prefix}_{selected}_delete", type="secondary"):
            st.session_state[session_key] = [item for item in rows if str(item.get("id")) != selected]
            st.rerun()
    else:
        st.info("No parameters have been added yet.")

    with st.expander("+ Add parameter", expanded=not bool(rows)):
        with st.form(f"{key_prefix}_add_parameter_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            label = c1.text_input("Parameter name")
            category = c2.selectbox("Category", PARAMETER_CATEGORIES)
            c1, c2, c3 = st.columns(3)
            value = c1.number_input("Base value", value=0.0, format="%.10g")
            unit = c2.text_input("Unit")
            pid = c3.text_input("Parameter ID (optional)")
            submitted = st.form_submit_button("Add parameter")
            if submitted:
                if not label.strip() or not unit.strip():
                    st.error("Parameter name and unit are required.")
                else:
                    existing = [str(row.get("id") or "") for row in st.session_state[session_key]]
                    chosen_id = pid.strip() or unique_id(slugify(label), existing)
                    if chosen_id in existing:
                        st.error(f"Parameter ID '{chosen_id}' already exists.")
                    else:
                        new_row = _default_parameter(label.strip(), chosen_id, float(value), unit.strip(), category, currency_code)
                        st.session_state[session_key] = [*st.session_state[session_key], new_row]
                        st.rerun()

    with st.expander("Advanced · inspect complete parameter rows"):
        st.dataframe(st.session_state[session_key], use_container_width=True)
