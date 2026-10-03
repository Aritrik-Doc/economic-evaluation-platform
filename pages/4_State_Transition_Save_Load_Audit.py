"""Save, load, validate and audit state-transition model bundles."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model.economics import Strategy, fully_incremental_analysis
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
st.caption("Version 0.13 — active settings are part of the reproducible model snapshot")

st.info(
    "Saved files contain the editable model tables, the exact active methods/engine settings and a SHA-256 content hash. "
    "Uploaded files are recompiled before acceptance. Restoring a model also restores its analysis settings and invalidates stale PSA results."
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


def _session_has(mapping):
    return all(key in st.session_state for key in mapping.values())


def _session_tables(model_type):
    mapping = COHORT_KEYS if model_type == "cohort_markov" else SEMI_KEYS
    if not _session_has(mapping):
        return None
    data = {name: st.session_state[key] for name, key in mapping.items()}
    data.setdefault("mortality_table", [])
    data.setdefault("mortality_rules", [])
    return data


def _termination(value: str) -> str:
    return "cohort_depletion" if str(value) in {"Cohort depletion", "cohort_depletion"} else "fixed_cycles"


def _loaded_settings(model_type):
    key = "loaded_cohort_markov_settings" if model_type == "cohort_markov" else "loaded_semi_markov_settings"
    loaded = st.session_state.get(key) or {}
    return dict(loaded.get("methods") or {}), dict(loaded.get("engine") or {})


def _active_settings(model_type):
    """Return the methods and engine settings currently driving the builder."""
    loaded_methods, loaded_engine = _loaded_settings(model_type)
    if model_type == "cohort_markov":
        profile_code = st.session_state.get("mk_reference_case", loaded_methods.get("reference_case_code", "NICE_TA"))
        profile = REFERENCE_CASES.get(profile_code)
        methods = {
            "reference_case_code": profile_code,
            "outcome_code": st.session_state.get("mk_outcome", loaded_methods.get("outcome_code", "QALY")),
            "currency_code": st.session_state.get("mk_currency", loaded_methods.get("currency_code", "GBP")),
            "threshold": float(st.session_state.get("mk_threshold", loaded_methods.get("threshold", 0.0) or 0.0)),
            "perspective_label": loaded_methods.get("perspective_label") or (profile.perspective.label if profile else "Healthcare payer"),
            "included_cost_bearers": [
                item.strip()
                for item in str(st.session_state.get("mk_cost_bearers", ", ".join(loaded_methods.get("included_cost_bearers", ["health_system"])))).split(",")
                if item.strip()
            ],
            "cost_discount_rate": float(st.session_state.get("mk_cost_discount", loaded_methods.get("cost_discount_rate", 0.035))),
            "outcome_discount_rate": float(st.session_state.get("mk_outcome_discount", loaded_methods.get("outcome_discount_rate", 0.035))),
        }
        cycle_months = int(st.session_state.get("mk_cycle_months", round(float(loaded_engine.get("cycle_length_years", 1.0)) * 12)))
        horizon_years = int(st.session_state.get("mk_horizon_years", round(float(loaded_engine.get("max_cycles", 20)) * float(loaded_engine.get("cycle_length_years", 1.0)))))
        cycle_years = cycle_months / 12.0
        engine = {
            "cycle_length_years": cycle_years,
            "max_cycles": int(round(horizon_years / cycle_years)),
            "state_accrual_timing": st.session_state.get("mk_state_accrual", loaded_engine.get("state_accrual_timing", "half_cycle")),
            "transition_reward_timing": st.session_state.get("mk_transition_timing", loaded_engine.get("transition_reward_timing", "mid_cycle")),
            "termination_mode": _termination(st.session_state.get("mk_termination", loaded_engine.get("termination_mode", "fixed_cycles"))),
            "depletion_threshold": float(st.session_state.get("mk_depletion", loaded_engine.get("depletion_threshold", 0.0001))),
        }
        return methods, engine

    methods = {
        "reference_case_code": loaded_methods.get("reference_case_code", "CUSTOM"),
        "outcome_code": st.session_state.get("adv_outcome", loaded_methods.get("outcome_code", "QALY")),
        "currency_code": st.session_state.get("adv_currency", loaded_methods.get("currency_code", "GBP")),
        "threshold": float(st.session_state.get("adv_threshold", loaded_methods.get("threshold", 0.0) or 0.0)),
        "perspective_label": loaded_methods.get("perspective_label", "Healthcare payer"),
        "included_cost_bearers": [
            item.strip()
            for item in str(st.session_state.get("adv_cost_bearers", ", ".join(loaded_methods.get("included_cost_bearers", ["health_system"])))).split(",")
            if item.strip()
        ],
        "cost_discount_rate": float(st.session_state.get("adv_cost_discount", loaded_methods.get("cost_discount_rate", 0.035))),
        "outcome_discount_rate": float(st.session_state.get("adv_outcome_discount", loaded_methods.get("outcome_discount_rate", 0.035))),
    }
    cycle_months = int(st.session_state.get("adv_cycle_months", round(float(loaded_engine.get("cycle_length_years", 1.0)) * 12)))
    horizon_years = int(st.session_state.get("adv_horizon_years", round(float(loaded_engine.get("max_cycles", 20)) * float(loaded_engine.get("cycle_length_years", 1.0)))))
    cycle_years = cycle_months / 12.0
    engine = {
        "cycle_length_years": cycle_years,
        "max_cycles": int(round(horizon_years / cycle_years)),
        "state_accrual_timing": st.session_state.get("adv_state_accrual", loaded_engine.get("state_accrual_timing", "half_cycle")),
        "transition_reward_timing": st.session_state.get("adv_transition_timing", loaded_engine.get("transition_reward_timing", "mid_cycle")),
        "termination_mode": _termination(st.session_state.get("adv_termination", loaded_engine.get("termination_mode", "fixed_cycles"))),
        "depletion_threshold": float(st.session_state.get("adv_depletion", loaded_engine.get("depletion_threshold", 0.0001))),
    }
    return methods, engine


def _seed_builder_settings(bundle):
    methods = bundle["methods"]
    engine = bundle["engine"]
    model_type = bundle["model_type"]
    cycle_months = max(1, int(round(float(engine["cycle_length_years"]) * 12)))
    horizon_years = max(1, int(round(float(engine["max_cycles"]) * float(engine["cycle_length_years"]))))
    termination_label = "Cohort depletion" if engine.get("termination_mode") == "cohort_depletion" else "Fixed horizon"

    if model_type == "cohort_markov":
        if methods.get("reference_case_code") in REFERENCE_CASES:
            st.session_state["mk_reference_case"] = methods["reference_case_code"]
        st.session_state["mk_outcome"] = methods["outcome_code"]
        st.session_state["mk_currency"] = methods["currency_code"]
        st.session_state["mk_threshold"] = float(methods.get("threshold") or 0.0)
        st.session_state["mk_cost_discount"] = float(methods["cost_discount_rate"])
        st.session_state["mk_outcome_discount"] = float(methods["outcome_discount_rate"])
        st.session_state["mk_cost_bearers"] = ", ".join(methods["included_cost_bearers"])
        st.session_state["mk_cycle_months"] = cycle_months
        st.session_state["mk_horizon_years"] = horizon_years
        st.session_state["mk_state_accrual"] = engine["state_accrual_timing"]
        st.session_state["mk_transition_timing"] = engine["transition_reward_timing"]
        st.session_state["mk_termination"] = termination_label
        st.session_state["mk_depletion"] = float(engine["depletion_threshold"])
        st.session_state.pop("markov_loaded_settings_applied", None)
        st.session_state.pop("markov_psa_result", None)
        st.session_state.pop("markov_psa_fingerprint", None)
    else:
        st.session_state["adv_outcome"] = methods["outcome_code"]
        st.session_state["adv_currency"] = methods["currency_code"]
        st.session_state["adv_threshold"] = float(methods.get("threshold") or 0.0)
        st.session_state["adv_cost_discount"] = float(methods["cost_discount_rate"])
        st.session_state["adv_outcome_discount"] = float(methods["outcome_discount_rate"])
        st.session_state["adv_cost_bearers"] = ", ".join(methods["included_cost_bearers"])
        st.session_state["adv_cycle_months"] = cycle_months
        st.session_state["adv_horizon_years"] = horizon_years
        st.session_state["adv_state_accrual"] = engine["state_accrual_timing"]
        st.session_state["adv_transition_timing"] = engine["transition_reward_timing"]
        st.session_state["adv_termination"] = termination_label
        st.session_state["adv_depletion"] = float(engine["depletion_threshold"])
        st.session_state.pop("adv_loaded_settings_applied", None)
        st.session_state.pop("adv_psa_result", None)
        st.session_state.pop("adv_psa_fingerprint", None)


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
        "restore_token": bundle["content_hash_sha256"],
    }
    _seed_builder_settings(bundle)


def _run_bundle(bundle):
    compiled = compile_loaded_state_transition_bundle(bundle)
    methods = bundle["methods"]
    kwargs = dict(
        included_cost_bearers=tuple(methods["included_cost_bearers"]),
        cost_discount_rate=float(methods["cost_discount_rate"]),
        outcome_discount_rate=float(methods["outcome_discount_rate"]),
    )
    if bundle["model_type"] == "cohort_markov":
        return compiled, run_cohort_markov(compiled.model, compiled.parameters, **kwargs)
    return compiled, run_semi_markov(compiled.model, compiled.parameters, **kwargs)


def _result_rows(bundle, result):
    threshold = bundle["methods"].get("threshold")
    rows = []
    for result_row in result.strategies:
        row = {
            "strategy_id": result_row.strategy_id,
            "strategy": result_row.label,
            "expected_cost": result_row.expected_cost,
            "expected_outcome": result_row.expected_outcome,
            "cycles_run": result_row.cycles_run,
            "stopped_early": result_row.stopped_early,
        }
        if threshold is not None:
            row["nmb"] = float(threshold) * result_row.expected_outcome - result_row.expected_cost
        rows.append(row)
    return rows


save_tab, load_tab, audit_tab = st.tabs(["1 · Save / export", "2 · Load / restore", "3 · Validate / audit"])

with save_tab:
    st.subheader("Create a versioned state-transition model file")
    available = []
    if _session_has(COHORT_KEYS):
        available.append("cohort_markov")
    if _session_has(SEMI_KEYS):
        available.append("semi_markov")

    if not available:
        st.warning("Open a Cohort Markov or Advanced Markov builder first so there is an active model to save.")
    else:
        model_type = st.selectbox(
            "Model to save", available,
            format_func=lambda value: "Cohort Markov" if value == "cohort_markov" else "Advanced semi-Markov"
        )
        model_name = st.text_input("Model name", value="Health-economic state-transition model")
        c1, c2 = st.columns(2)
        author = c1.text_input("Author / modeller", value="")
        notes = c2.text_input("Version / notes", value="")
        methods, engine = _active_settings(model_type)
        st.markdown("#### Active analysis settings")
        st.caption(
            "These values are read from the active builder and will be saved exactly with the model. Change them in the modeller rather than creating a second settings copy here."
        )
        st.json({"methods": methods, "engine": engine})
        tables = _session_tables(model_type)
        assert tables is not None
        try:
            current_bundle = build_state_transition_bundle(
                model_type=model_type,
                model_name=model_name,
                methods=methods,
                engine=engine,
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
            st.success("Active model and settings recompile successfully and are ready to save.")
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
            st.success(f"Validated {loaded['model_type'].replace('_', ' ')} model: **{loaded['model_name']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Schema", loaded["schema_version"])
            c2.metric("Parameters", len(loaded["model"]["parameters"]))
            c3.metric("Strategies", len(loaded["model"]["strategies"]))
            st.code(loaded["content_hash_sha256"], language=None)
            if st.button("Restore model and analysis settings to this session", type="primary"):
                _apply_bundle_to_session(loaded)
                st.success(
                    "Model tables and saved analysis settings were restored. Previous PSA results for that modeller were cleared because they belong to the prior model/settings state."
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
        bundle = (
            st.session_state.current_state_transition_bundle
            if choice == "Current saved model"
            else st.session_state.loaded_state_transition_bundle
        )
        try:
            _, result = _run_bundle(bundle)
            rows = _result_rows(bundle, result)
            st.success("Saved snapshot recompiled and reran successfully using the settings stored in that file.")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            threshold = bundle["methods"].get("threshold")
            if threshold is not None:
                economic = [Strategy(row.label, row.expected_cost, row.expected_outcome) for row in result.strategies]
                incremental = fully_incremental_analysis(economic, float(threshold))
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
    "A saved state-transition file is the authoritative reproducibility snapshot: model structure, parameter evidence, cycle settings, horizon, discounting, threshold and perspective travel together."
)
