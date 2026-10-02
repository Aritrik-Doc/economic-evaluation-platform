"""Save, load, validate and audit state-transition model bundles."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model.currency import CURRENCIES
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.markov import run_cohort_markov
from model.reference_cases import REFERENCE_CASES
from model.semi_markov import run_semi_markov
from model.state_transition_persistence import (
    StateTransitionPersistenceError,
    build_state_transition_audit_record,
    build_state_transition_bundle,
    compile_loaded_state_transition_bundle,
    load_state_transition_bundle,
    state_transition_audit_json,
    state_transition_bundle_json,
)


st.set_page_config(page_title="State-Transition Save / Load / Audit", page_icon="💾", layout="wide")
st.title("💾 State-Transition Save / Load / Audit")
st.caption("Version 0.7 — one versioned file family for homogeneous cohort Markov and advanced semi-Markov models")

st.info(
    "Saved files contain the editable model tables, methods settings, engine settings and a SHA-256 content hash. "
    "Uploaded files are recompiled before they are accepted. Audit records embed the complete model snapshot used for the run."
)


COHORT_KEYS = {
    "parameters": "markov_parameters",
    "states": "markov_states",
    "strategies": "markov_strategies",
    "initial_distribution": "markov_initial",
    "transitions": "markov_transitions",
    "state_rewards": "markov_state_rewards",
    "transition_rewards": "markov_transition_rewards",
}
SEMI_KEYS = {
    "parameters": "adv_parameters",
    "states": "adv_states",
    "strategies": "adv_strategies",
    "initial_distribution": "adv_initial",
    "transitions": "adv_transitions",
    "state_rewards": "adv_state_rewards",
    "transition_rewards": "adv_transition_rewards",
    "mortality_table": "adv_mortality_table",
    "mortality_rules": "adv_mortality_rules",
}


def _session_has(keys):
    return all(value in st.session_state for value in keys.values())


def _session_tables(model_type):
    mapping = COHORT_KEYS if model_type == "cohort_markov" else SEMI_KEYS
    if not _session_has(mapping):
        return None
    data = {name: st.session_state[key] for name, key in mapping.items()}
    data.setdefault("mortality_table", [])
    data.setdefault("mortality_rules", [])
    return data


def _apply_bundle_to_session(bundle):
    model_type = bundle["model_type"]
    mapping = COHORT_KEYS if model_type == "cohort_markov" else SEMI_KEYS
    model = bundle["model"]
    for name, key in mapping.items():
        st.session_state[key] = [dict(row) for row in model.get(name, [])]
    settings_key = "loaded_cohort_markov_settings" if model_type == "cohort_markov" else "loaded_semi_markov_settings"
    st.session_state[settings_key] = {
        "methods": dict(bundle["methods"]),
        "engine": dict(bundle["engine"]),
        "model_name": bundle["model_name"],
        "metadata": dict(bundle.get("metadata") or {}),
    }


def _run_bundle(bundle):
    compiled = compile_loaded_state_transition_bundle(bundle)
    methods = bundle["methods"]
    model_type = bundle["model_type"]
    run_kwargs = dict(
        included_cost_bearers=tuple(methods["included_cost_bearers"]),
        cost_discount_rate=float(methods["cost_discount_rate"]),
        outcome_discount_rate=float(methods["outcome_discount_rate"]),
    )
    if model_type == "cohort_markov":
        result = run_cohort_markov(compiled.model, compiled.parameters, **run_kwargs)
    else:
        result = run_semi_markov(compiled.model, compiled.parameters, **run_kwargs)
    return compiled, result


def _result_rows(bundle, result):
    methods = bundle["methods"]
    threshold = methods.get("threshold")
    output = []
    for row in result.strategies:
        item = {
            "strategy_id": row.strategy_id,
            "strategy": row.label,
            "expected_cost": row.expected_cost,
            "expected_outcome": row.expected_outcome,
            "cycles_run": row.cycles_run,
            "stopped_early": row.stopped_early,
        }
        if threshold is not None:
            item["nmb"] = float(threshold) * row.expected_outcome - row.expected_cost
        output.append(item)
    return output


save_tab, load_tab, audit_tab = st.tabs(["1 · Save / export", "2 · Load / restore", "3 · Validate / audit"])

with save_tab:
    st.subheader("Create a versioned state-transition model file")
    available = []
    if _session_has(COHORT_KEYS):
        available.append("cohort_markov")
    if _session_has(SEMI_KEYS):
        available.append("semi_markov")

    if not available:
        st.warning(
            "No state-transition model tables are currently present in this Streamlit session. Open the Cohort Markov Builder or Advanced Markov Dynamics page first, then return here."
        )
    else:
        model_type = st.selectbox(
            "Model to save",
            available,
            format_func=lambda value: "Cohort Markov" if value == "cohort_markov" else "Advanced semi-Markov",
        )
        model_name = st.text_input("Model name", value="Health-economic state-transition model")
        c1, c2 = st.columns(2)
        author = c1.text_input("Author / modeller", value="")
        notes = c2.text_input("Version / notes", value="")

        st.markdown("#### Methods snapshot")
        c1, c2, c3 = st.columns(3)
        reference_case_code = c1.selectbox("Reference case", [*REFERENCE_CASES.keys(), "CUSTOM"])
        preferred = REFERENCE_CASES[reference_case_code].preferred_outcome_code if reference_case_code in REFERENCE_CASES else "QALY"
        outcome_code = c2.selectbox("Economic outcome", list(OUTCOME_MEASURES), index=list(OUTCOME_MEASURES).index(preferred))
        default_currency = REFERENCE_CASES[reference_case_code].analysis_currency if reference_case_code in REFERENCE_CASES else "GBP"
        currency_code = c3.selectbox("Analysis currency", list(CURRENCIES), index=list(CURRENCIES).index(default_currency) if default_currency in CURRENCIES else 0)

        profile = REFERENCE_CASES.get(reference_case_code)
        default_threshold = 0.0
        if profile and profile.threshold_range is not None:
            default_threshold = float((profile.threshold_range.lower + profile.threshold_range.upper) / 2)
        c1, c2, c3 = st.columns(3)
        threshold = c1.number_input("Decision threshold", min_value=0.0, value=default_threshold, step=1000.0)
        cost_discount = c2.number_input("Cost discount rate", min_value=0.0, max_value=0.99, value=float(profile.cost_discount_rate if profile else 0.035), format="%.4f")
        outcome_discount = c3.number_input("Outcome discount rate", min_value=0.0, max_value=0.99, value=float(profile.outcome_discount_rate if profile else 0.035), format="%.4f")
        perspective_label = st.text_input("Perspective label", value=profile.perspective.label if profile else "Healthcare payer")
        bearers_default = ", ".join(profile.perspective.included_cost_bearers) if profile else "health_system"
        bearers_text = st.text_input("Included cost bearers", value=bearers_default)
        included_cost_bearers = [item.strip() for item in bearers_text.split(",") if item.strip()]

        st.markdown("#### Engine snapshot")
        c1, c2, c3 = st.columns(3)
        cycle_months = c1.selectbox("Cycle length", [1, 3, 6, 12], index=3, format_func=lambda x: f"{x} month" if x == 1 else f"{x} months")
        horizon_years = int(c2.number_input("Maximum horizon (years)", min_value=1, max_value=200, value=20, step=1))
        state_accrual = c3.selectbox("State accrual", ["half_cycle", "start", "end"])
        cycle_length_years = cycle_months / 12.0
        max_cycles = int(round(horizon_years / cycle_length_years))
        c1, c2, c3 = st.columns(3)
        transition_timing = c1.selectbox("Transition reward timing", ["mid_cycle", "start", "end"])
        termination_mode = c2.selectbox("Termination", ["fixed_cycles", "cohort_depletion"])
        depletion_threshold = c3.number_input("Depletion threshold", min_value=0.0, max_value=0.5, value=0.0001, format="%.6f")

        tables = _session_tables(model_type)
        assert tables is not None
        try:
            current_bundle = build_state_transition_bundle(
                model_type=model_type,
                model_name=model_name,
                methods={
                    "reference_case_code": reference_case_code,
                    "outcome_code": outcome_code,
                    "currency_code": currency_code,
                    "threshold": threshold,
                    "perspective_label": perspective_label,
                    "included_cost_bearers": included_cost_bearers,
                    "cost_discount_rate": cost_discount,
                    "outcome_discount_rate": outcome_discount,
                },
                engine={
                    "cycle_length_years": cycle_length_years,
                    "max_cycles": max_cycles,
                    "state_accrual_timing": state_accrual,
                    "transition_reward_timing": transition_timing,
                    "termination_mode": termination_mode,
                    "depletion_threshold": depletion_threshold,
                },
                parameter_rows=tables["parameters"],
                state_rows=tables["states"],
                strategy_rows=tables["strategies"],
                initial_rows=tables["initial_distribution"],
                transition_rows=tables["transitions"],
                state_reward_rows=tables["state_rewards"],
                transition_reward_rows=tables["transition_rewards"],
                mortality_table_rows=tables["mortality_table"],
                mortality_rule_rows=tables["mortality_rules"],
                author=author,
                notes=notes,
            )
            st.success("Model recompiles successfully and is ready to save.")
            st.code(current_bundle["content_hash_sha256"], language=None)
            st.download_button(
                "Download model JSON",
                state_transition_bundle_json(current_bundle),
                file_name=(model_name.strip().replace(" ", "_") or "state_transition_model") + ".json",
                mime="application/json",
                type="primary",
            )
            st.session_state.current_state_transition_bundle = current_bundle
        except Exception as exc:
            st.error(f"Model cannot be saved until it validates: {exc}")

with load_tab:
    st.subheader("Load and restore a state-transition model")
    uploaded = st.file_uploader("Upload model JSON", type=["json"], key="state_transition_upload")
    if uploaded is not None:
        try:
            loaded = load_state_transition_bundle(uploaded.getvalue())
            st.session_state.loaded_state_transition_bundle = loaded
            st.success(
                f"Validated {loaded['model_type'].replace('_', ' ')} model: **{loaded['model_name']}**"
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Schema", loaded["schema_version"])
            c2.metric("Parameters", len(loaded["model"]["parameters"]))
            c3.metric("Strategies", len(loaded["model"]["strategies"]))
            st.code(loaded["content_hash_sha256"], language=None)
            if st.button("Restore editable model tables to this session", type="primary"):
                _apply_bundle_to_session(loaded)
                st.success(
                    "Editable tables restored. Open the corresponding builder page. The saved methods/engine settings are retained in session metadata and are shown below so you can reproduce them exactly."
                )
            with st.expander("Saved methods and engine settings", expanded=True):
                st.json({"methods": loaded["methods"], "engine": loaded["engine"], "metadata": loaded.get("metadata")})
        except StateTransitionPersistenceError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"The uploaded model failed structural validation: {exc}")

with audit_tab:
    st.subheader("Re-run, validate and create an audit record")
    candidates = []
    if st.session_state.get("current_state_transition_bundle") is not None:
        candidates.append("Current saved model")
    if st.session_state.get("loaded_state_transition_bundle") is not None:
        candidates.append("Uploaded model")

    if not candidates:
        st.info("Create or upload a valid model bundle first.")
    else:
        choice = st.radio("Model snapshot", candidates, horizontal=True)
        bundle = st.session_state.current_state_transition_bundle if choice == "Current saved model" else st.session_state.loaded_state_transition_bundle
        try:
            _, result = _run_bundle(bundle)
            rows = _result_rows(bundle, result)
            st.success("Saved snapshot recompiled and reran successfully.")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            threshold = bundle["methods"].get("threshold")
            if threshold is not None:
                economic = [Strategy(row.label, row.expected_cost, row.expected_outcome) for row in result.strategies]
                incremental = fully_incremental_analysis(economic, float(threshold))
                st.caption("Fully incremental frontier status is recalculated from the saved model outputs, not stored as a fixed result.")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Strategy": row.strategy.name,
                            "Status": row.status,
                            "Compared with": row.compared_with,
                            "Incremental cost": row.incremental_cost,
                            "Incremental outcome": row.incremental_effect,
                            "ICER": row.icer,
                            "NMB": row.nmb,
                        }
                        for row in incremental.rows
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

            if st.button("Append base-case audit record", type="primary"):
                record = build_state_transition_audit_record(
                    bundle,
                    analysis_type="base_case",
                    run_settings={
                        "threshold": bundle["methods"].get("threshold"),
                        "cost_discount_rate": bundle["methods"]["cost_discount_rate"],
                        "outcome_discount_rate": bundle["methods"]["outcome_discount_rate"],
                        "included_cost_bearers": bundle["methods"]["included_cost_bearers"],
                        "engine": bundle["engine"],
                    },
                    results={row["strategy_id"]: row for row in rows},
                )
                records = list(st.session_state.get("state_transition_audit_records", []))
                records.append(record)
                st.session_state.state_transition_audit_records = records
                st.success(f"Audit record created: {record['run_id']}")

            records = st.session_state.get("state_transition_audit_records", [])
            if records:
                st.write(f"**Audit records in this session:** {len(records)}")
                st.download_button(
                    "Download audit trail JSON",
                    state_transition_audit_json(records),
                    file_name="state_transition_audit_trail.json",
                    mime="application/json",
                )
        except Exception as exc:
            st.error(f"Saved model could not be rerun: {exc}")

st.divider()
st.caption(
    "Loaded structural tables are restored into the appropriate builder session. The persistence file remains the authoritative snapshot for methods and engine settings; builders will be consolidated to consume those settings directly in the next UI-refinement pass."
)
