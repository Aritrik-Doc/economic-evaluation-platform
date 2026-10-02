"""Hybrid decision-tree builder for Economic Evaluation Platform v0.4."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model.currency import CURRENCIES
from model.decision_tree import DecisionTreeValidationError, run_decision_tree
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.reference_cases import REFERENCE_CASES
from model.tree_builder import BuilderValidationError, compile_builder_tables, structure_to_dot

st.set_page_config(page_title="Decision Tree Builder", layout="wide")
st.title("Decision Tree Builder")
st.caption("Version 0.4 — structured editing with live visual preview")

SOURCE_TYPES = ["systematic_review","meta_analysis","randomised_trial","observational_study","registry","database","tariff","cost_study","expert_elicitation","guideline","user_assumption","other"]
PARAMETER_CATEGORIES = ["clinical","cost","utility","resource_use","survival","epidemiology","other"]
UNCERTAINTY_KINDS = ["none","range"]
COST_BEARER_OPTIONS = ["health_system","personal_social_services","patient_direct_medical","patient_direct_non_medical","productivity","carer","custom"]


def default_parameter_rows():
    common = {
        "unit":"proportion","category":"clinical","source_type":"user_assumption","source_citation":"Illustrative example input","source_url":"","publication_year":None,
        "assumption":"Illustrative value for builder demonstration.","assumption_rationale":"Replace with sourced evidence before substantive use.",
        "uncertainty_kind":"none","uncertainty_rationale":"Illustrative deterministic example.","lower":None,"upper":None,"currency":"","price_year":None,"cost_bearers":"","notes":"",
    }
    rows=[]
    for pid,value in [("p_success_a",0.8),("p_failure_a",0.2),("p_success_b",0.6),("p_failure_b",0.4)]:
        rows.append({**common,"id":pid,"label":pid.replace("_"," ").title(),"value":value})
    for pid,label,value in [("cost_a","Strategy A acquisition cost",1000.0),("cost_b","Strategy B acquisition cost",500.0),("cost_success","Cost after success",100.0),("cost_failure","Cost after failure",1000.0)]:
        rows.append({**common,"id":pid,"label":label,"value":value,"unit":"currency","category":"cost","currency":"GBP","price_year":2026,"cost_bearers":"health_system"})
    for pid,label,value in [("outcome_success","Outcome after success",2.0),("outcome_failure","Outcome after failure",1.0)]:
        rows.append({**common,"id":pid,"label":label,"value":value,"unit":"QALY","category":"utility"})
    return rows

DEFAULT_STRATEGIES=[{"strategy_id":"A","strategy_name":"Strategy A","root_node_id":"a_root"},{"strategy_id":"B","strategy_name":"Strategy B","root_node_id":"b_root"}]
DEFAULT_NODES=[
    {"id":"a_root","label":"Outcome under A","type":"chance","cost_parameter_ids":"cost_a","outcome_parameter_ids":""},
    {"id":"b_root","label":"Outcome under B","type":"chance","cost_parameter_ids":"cost_b","outcome_parameter_ids":""},
    {"id":"success","label":"Success","type":"terminal","cost_parameter_ids":"cost_success","outcome_parameter_ids":"outcome_success"},
    {"id":"failure","label":"Failure","type":"terminal","cost_parameter_ids":"cost_failure","outcome_parameter_ids":"outcome_failure"},
]
DEFAULT_BRANCHES=[
    {"from_node":"a_root","label":"Success","probability_parameter_id":"p_success_a","to_node":"success"},
    {"from_node":"a_root","label":"Failure","probability_parameter_id":"p_failure_a","to_node":"failure"},
    {"from_node":"b_root","label":"Success","probability_parameter_id":"p_success_b","to_node":"success"},
    {"from_node":"b_root","label":"Failure","probability_parameter_id":"p_failure_b","to_node":"failure"},
]


def money(value, symbol): return f"{symbol}{value:,.2f}"

def parse_threshold(raw):
    text=raw.replace(",","").strip()
    if not text: return None
    value=float(text)
    if value<0: raise ValueError("Threshold cannot be negative.")
    return value

with st.sidebar:
    st.header("Methods")
    reference_case_code=st.selectbox("Reference case",["NICE_TA","HTAIN_2023","CUSTOM"],format_func=lambda code: REFERENCE_CASES[code].name if code in REFERENCE_CASES else "Custom")
    if reference_case_code in REFERENCE_CASES:
        profile=REFERENCE_CASES[reference_case_code]
        outcome_code=st.selectbox("Economic outcome",list(OUTCOME_MEASURES),index=list(OUTCOME_MEASURES).index(profile.preferred_outcome_code),format_func=lambda code: OUTCOME_MEASURES[code].label)
        currency_code=profile.analysis_currency
        st.text_input("Analysis currency",currency_code,disabled=True)
        if profile.threshold_range and outcome_code==profile.threshold_range.outcome_code:
            default_threshold=str(int(profile.threshold_range.lower))
            st.caption(f"Reference range: {CURRENCIES[currency_code].symbol}{profile.threshold_range.lower:,.0f}–{CURRENCIES[currency_code].symbol}{profile.threshold_range.upper:,.0f} per {OUTCOME_MEASURES[outcome_code].unit}.")
        else:
            default_threshold=""
        threshold_raw=st.text_input("Decision threshold",default_threshold)
        included_cost_bearers=list(profile.perspective.included_cost_bearers)
        st.write(f"**Perspective:** {profile.perspective.label}")
        st.caption(f"Reference discount rates: costs {profile.cost_discount_rate*100:.1f}%, outcomes {profile.outcome_discount_rate*100:.1f}%.")
        status=profile.outcome_status(outcome_code)
        if status in {"conditional","supplementary","not_specified"}:
            st.warning(f"{OUTCOME_MEASURES[outcome_code].label} is classified as **{status.replace('_',' ')}** for this reference case.")
    else:
        outcome_code=st.selectbox("Economic outcome",list(OUTCOME_MEASURES),format_func=lambda code: OUTCOME_MEASURES[code].label)
        currency_code=st.selectbox("Analysis currency",list(CURRENCIES),format_func=lambda code: f"{code} — {CURRENCIES[code].name}")
        st.text_input("Perspective label","Healthcare payer")
        included_cost_bearers=st.multiselect("Include cost bearers",COST_BEARER_OPTIONS,default=["health_system"])
        st.selectbox("Time horizon",["Short horizon (<1 year)","Custom / pre-discounted"])
        st.number_input("Cost discount rate (%)",0.0,99.0,3.5)
        st.number_input("Outcome discount rate (%)",0.0,99.0,3.5)
        threshold_raw=st.text_input("Decision threshold","30000")
    st.warning("Current v0.4 tree rewards are not time-stamped. For multi-year models, enter already-discounted rewards until timed accrual and automatic discounting are added.")

try: threshold=parse_threshold(threshold_raw)
except ValueError as exc:
    st.error(str(exc)); threshold=None
outcome=OUTCOME_MEASURES[outcome_code]; currency=CURRENCIES[currency_code]

st.markdown("The builder separates **what the model means** from **how it looks**. Edit structured tables; the visual tree updates from the same structure. Parameters retain explicit source, assumption, and uncertainty metadata.")
parameters_tab,structure_tab,results_tab=st.tabs(["1 · Parameters","2 · Structure + visual tree","3 · Validate + run"])

with parameters_tab:
    st.subheader("Parameter library")
    st.caption("Rows with blank ids are ignored. For cost parameters, currency, price year, and at least one cost bearer are mandatory.")
    parameter_df=st.data_editor(
        pd.DataFrame(default_parameter_rows()),num_rows="dynamic",hide_index=True,width="stretch",key="dt_parameter_editor",
        column_config={
            "category":st.column_config.SelectboxColumn("Category",options=PARAMETER_CATEGORIES,required=True),
            "source_type":st.column_config.SelectboxColumn("Source type",options=SOURCE_TYPES,required=True),
            "uncertainty_kind":st.column_config.SelectboxColumn("Uncertainty",options=UNCERTAINTY_KINDS,required=True),
            "value":st.column_config.NumberColumn("Value",format="%.6f"),"lower":st.column_config.NumberColumn("Lower",format="%.6f"),"upper":st.column_config.NumberColumn("Upper",format="%.6f"),
            "price_year":st.column_config.NumberColumn("Price year",step=1,format="%d"),
        })

with structure_tab:
    left,right=st.columns([1.1,1])
    with left:
        st.subheader("Strategies")
        strategies_df=st.data_editor(pd.DataFrame(DEFAULT_STRATEGIES),num_rows="dynamic",hide_index=True,width="stretch",key="dt_strategy_editor")
        st.subheader("Nodes")
        nodes_df=st.data_editor(pd.DataFrame(DEFAULT_NODES),num_rows="dynamic",hide_index=True,width="stretch",key="dt_node_editor",column_config={"type":st.column_config.SelectboxColumn("Type",options=["chance","terminal"],required=True)})
        st.subheader("Branches")
        branches_df=st.data_editor(pd.DataFrame(DEFAULT_BRANCHES),num_rows="dynamic",hide_index=True,width="stretch",key="dt_branch_editor")
    with right:
        st.subheader("Live tree")
        st.caption("Chance nodes are circles; terminal nodes are double circles. Branch labels show the linked probability parameter.")
        dot=structure_to_dot(strategies_df.to_dict("records"),nodes_df.to_dict("records"),branches_df.to_dict("records"))
        st.graphviz_chart(dot,width="stretch")

with results_tab:
    st.subheader("Validation and expected values")
    if threshold is None:
        st.info("Enter a decision threshold in the sidebar before running the model.")
    else:
        try:
            compiled=compile_builder_tables(parameter_df.to_dict("records"),strategies_df.to_dict("records"),nodes_df.to_dict("records"),branches_df.to_dict("records"))
            run=run_decision_tree(compiled.tree,compiled.parameters,included_cost_bearers=included_cost_bearers or None)
            model_strategies=[Strategy(compiled.strategy_names[row.strategy_id],row.expected_cost,row.expected_outcome) for row in run.strategies]
            decision=fully_incremental_analysis(model_strategies,threshold)
        except (BuilderValidationError,DecisionTreeValidationError,ValueError) as exc:
            st.error(str(exc))
        else:
            st.success("Tree structure and parameter links are valid.")
            expected_rows=[{"Strategy":compiled.strategy_names[row.strategy_id],f"Expected cost ({currency.code})":money(row.expected_cost,currency.symbol),f"Expected {outcome.unit}":f"{row.expected_outcome:,.4f}"} for row in run.strategies]
            st.dataframe(expected_rows,hide_index=True,width="stretch")
            st.subheader("Cost-effectiveness results")
            result_rows=[]
            for row in decision.rows:
                result_rows.append({"Strategy":row.strategy.name,"Cost":money(row.strategy.cost,currency.symbol),outcome.label:f"{row.strategy.effect:,.4f}","Status":row.status.replace("_"," ").title(),"Compared with":row.compared_with or "—","Incremental cost":"—" if row.incremental_cost is None else money(row.incremental_cost,currency.symbol),f"Incremental {outcome.unit}":"—" if row.incremental_effect is None else f"{row.incremental_effect:,.4f}",f"ICER ({currency.code}/{outcome.unit})":"—" if row.icer is None else money(row.icer,currency.symbol),"NMB":money(row.nmb,currency.symbol)})
            st.dataframe(result_rows,hide_index=True,width="stretch")
            preferred=decision.preferred_by_nmb
            if len(preferred)==1: st.info(f"At {money(threshold,currency.symbol)} per {outcome.unit}, **{preferred[0]}** has the highest net monetary benefit.")
            else: st.info("At the selected threshold, the highest NMB is tied between: "+", ".join(preferred)+".")
            dominated=[row.strategy.name for row in decision.rows if row.status=="strongly_dominated"]
            extended=[row.strategy.name for row in decision.rows if row.status=="extendedly_dominated"]
            if dominated: st.write("**Dominated:** "+", ".join(dominated)+". Each costs more without providing more health benefit than another available strategy.")
            if extended: st.write("**Extendedly dominated:** "+", ".join(extended)+". Moving through other strategies provides health gains at a more favourable incremental cost-effectiveness rate.")
