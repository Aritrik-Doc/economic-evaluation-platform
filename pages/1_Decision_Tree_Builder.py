"""Hybrid decision-tree builder for Economic Evaluation Platform v0.4."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model.currency import CURRENCIES
from model.decision_tree import DecisionTreeValidationError, run_decision_tree
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.reference_cases import REFERENCE_CASES
from model.sensitivity import OneWaySensitivitySpec, ThresholdAnalysisSpec, TwoWaySensitivitySpec
from model.tree_builder import BuilderValidationError, compile_builder_tables, structure_to_dot
from model.tree_sensitivity import one_way_tree_inmb, threshold_tree_inmb, two_way_tree_inmb

st.set_page_config(page_title="Decision Tree Builder", layout="wide")
st.title("Decision Tree Builder")
st.caption("Version 0.4 — hybrid structured editor, live tree, discounting and sensitivity analysis")

SOURCE_TYPES = ["systematic_review","meta_analysis","randomised_trial","observational_study","registry","database","tariff","cost_study","expert_elicitation","guideline","user_assumption","other"]
PARAMETER_CATEGORIES = ["clinical","cost","utility","resource_use","survival","epidemiology","other"]
UNCERTAINTY_KINDS = ["none","range"]
COST_BEARER_OPTIONS = ["health_system","personal_social_services","patient_direct_medical","patient_direct_non_medical","productivity","carer","custom"]


def default_parameter_rows():
    common = {
        "unit":"proportion","category":"clinical","source_type":"user_assumption","source_citation":"Illustrative example input","source_url":"","publication_year":None,
        "assumption":"Illustrative value for builder demonstration.","assumption_rationale":"Replace with sourced evidence before substantive use.",
        "uncertainty_kind":"range","uncertainty_rationale":"Illustrative range; replace with evidence-based uncertainty.","lower":None,"upper":None,"currency":"","price_year":None,"cost_bearers":"","notes":"",
    }
    rows=[]
    rows.append({**common,"id":"p_success_a","label":"Probability of success — A","value":0.8,"lower":0.65,"upper":0.90})
    rows.append({**common,"id":"p_success_b","label":"Probability of success — B","value":0.6,"lower":0.45,"upper":0.75})
    for pid,label,value,low,high in [
        ("cost_a","Strategy A acquisition cost",1000.0,800.0,1200.0),
        ("cost_b","Strategy B acquisition cost",500.0,400.0,600.0),
        ("cost_success","Cost after success",100.0,80.0,120.0),
        ("cost_failure","Cost after failure",1000.0,800.0,1200.0),
    ]:
        rows.append({**common,"id":pid,"label":label,"value":value,"lower":low,"upper":high,"unit":"currency","category":"cost","currency":"GBP","price_year":2026,"cost_bearers":"health_system"})
    for pid,label,value,low,high in [
        ("outcome_success","Outcome after success",2.0,1.7,2.2),
        ("outcome_failure","Outcome after failure",1.0,0.8,1.2),
    ]:
        rows.append({**common,"id":pid,"label":label,"value":value,"lower":low,"upper":high,"unit":"QALY","category":"utility"})
    return rows


DEFAULT_STRATEGIES=[{"strategy_id":"A","strategy_name":"Strategy A","root_node_id":"a_root"},{"strategy_id":"B","strategy_name":"Strategy B","root_node_id":"b_root"}]
DEFAULT_NODES=[
    {"id":"a_root","label":"Outcome under A","type":"chance","cost_rewards":"cost_a@0","outcome_rewards":""},
    {"id":"b_root","label":"Outcome under B","type":"chance","cost_rewards":"cost_b@0","outcome_rewards":""},
    {"id":"success","label":"Success","type":"terminal","cost_rewards":"cost_success@1","outcome_rewards":"outcome_success@1"},
    {"id":"failure","label":"Failure","type":"terminal","cost_rewards":"cost_failure@1","outcome_rewards":"outcome_failure@1"},
]
DEFAULT_BRANCHES=[
    {"from_node":"a_root","label":"Success","probability_parameter_id":"p_success_a","probability_mode":"direct","to_node":"success"},
    {"from_node":"a_root","label":"Failure","probability_parameter_id":"p_success_a","probability_mode":"complement","to_node":"failure"},
    {"from_node":"b_root","label":"Success","probability_parameter_id":"p_success_b","probability_mode":"direct","to_node":"success"},
    {"from_node":"b_root","label":"Failure","probability_parameter_id":"p_success_b","probability_mode":"complement","to_node":"failure"},
]


def money(value, symbol): return f"{symbol}{value:,.2f}"


def parse_threshold(raw):
    text=raw.replace(",","").strip()
    if not text: return None
    value=float(text)
    if value<0: raise ValueError("Threshold cannot be negative.")
    return value


def default_bounds(parameter):
    if parameter.uncertainty.kind == "range" and parameter.uncertainty.lower is not None and parameter.uncertainty.upper is not None:
        return float(parameter.uncertainty.lower), float(parameter.uncertainty.upper)
    base=float(parameter.value)
    if 0 <= base <= 1:
        return max(0.0, base * 0.8), min(1.0, base * 1.2 if base else 0.2)
    width=abs(base)*0.2 if base else 1.0
    return base-width, base+width


def linear_grid(low, high, points=5):
    if points < 2: return (low,)
    step=(high-low)/(points-1)
    return tuple(low+i*step for i in range(points))


with st.sidebar:
    st.header("Methods")
    reference_case_code=st.selectbox("Reference case",["NICE_TA","HTAIN_2023","CUSTOM"],format_func=lambda code: REFERENCE_CASES[code].name if code in REFERENCE_CASES else "Custom")
    if reference_case_code in REFERENCE_CASES:
        profile=REFERENCE_CASES[reference_case_code]
        outcome_code=st.selectbox("Economic outcome",list(OUTCOME_MEASURES),index=list(OUTCOME_MEASURES).index(profile.preferred_outcome_code),format_func=lambda code: OUTCOME_MEASURES[code].label)
        currency_code=profile.analysis_currency
        st.text_input("Analysis currency",currency_code,disabled=True)
        cost_discount_rate=profile.cost_discount_rate
        outcome_discount_rate=profile.outcome_discount_rate
        st.write(f"**Perspective:** {profile.perspective.label}")
        st.caption(f"Discounting: costs {cost_discount_rate*100:.1f}%, outcomes {outcome_discount_rate*100:.1f}% per year.")
        if profile.threshold_range and outcome_code==profile.threshold_range.outcome_code:
            default_threshold=str(int(profile.threshold_range.lower))
            st.caption(f"Reference range: {CURRENCIES[currency_code].symbol}{profile.threshold_range.lower:,.0f}–{CURRENCIES[currency_code].symbol}{profile.threshold_range.upper:,.0f} per {OUTCOME_MEASURES[outcome_code].unit}.")
        else:
            default_threshold=""
        threshold_raw=st.text_input("Decision threshold",default_threshold)
        included_cost_bearers=list(profile.perspective.included_cost_bearers)
        status=profile.outcome_status(outcome_code)
        if status in {"conditional","supplementary","not_specified"}:
            st.warning(f"{OUTCOME_MEASURES[outcome_code].label} is classified as **{status.replace('_',' ')}** for this reference case.")
    else:
        outcome_code=st.selectbox("Economic outcome",list(OUTCOME_MEASURES),format_func=lambda code: OUTCOME_MEASURES[code].label)
        currency_code=st.selectbox("Analysis currency",list(CURRENCIES),format_func=lambda code: f"{code} — {CURRENCIES[code].name}")
        st.text_input("Perspective label","Healthcare payer")
        included_cost_bearers=st.multiselect("Include cost bearers",COST_BEARER_OPTIONS,default=["health_system"])
        cost_discount_rate=st.number_input("Cost discount rate (%)",0.0,99.0,3.5)/100
        outcome_discount_rate=st.number_input("Outcome discount rate (%)",0.0,99.0,3.5)/100
        threshold_raw=st.text_input("Decision threshold","30000")
    st.caption("Reward timing uses absolute years from model start. Enter rewards as `parameter@years`, for example `followup_cost@2.5`. Annual discrete discounting is applied automatically.")

try: threshold=parse_threshold(threshold_raw)
except ValueError as exc:
    st.error(str(exc)); threshold=None
outcome=OUTCOME_MEASURES[outcome_code]; currency=CURRENCIES[currency_code]

st.markdown("Edit structured tables; the live diagram is generated from the same structure. Probability branches can use a parameter directly or its complement **1 − parameter**, which keeps binary chance nodes coherent during sensitivity analysis.")
parameters_tab,structure_tab,results_tab,sensitivity_tab=st.tabs(["1 · Parameters","2 · Structure + visual tree","3 · Validate + run","4 · Sensitivity analysis"])

with parameters_tab:
    st.subheader("Parameter library")
    st.caption("Source, assumption and uncertainty fields are explicit. For cost parameters, currency, price year and cost bearer are mandatory.")
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
        st.caption("Timed reward syntax: `parameter@years`; multiple rewards are comma-separated. A missing time means year 0.")
        nodes_df=st.data_editor(pd.DataFrame(DEFAULT_NODES),num_rows="dynamic",hide_index=True,width="stretch",key="dt_node_editor",column_config={"type":st.column_config.SelectboxColumn("Type",options=["chance","terminal"],required=True)})
        st.subheader("Branches")
        branches_df=st.data_editor(pd.DataFrame(DEFAULT_BRANCHES),num_rows="dynamic",hide_index=True,width="stretch",key="dt_branch_editor",column_config={"probability_mode":st.column_config.SelectboxColumn("Probability mode",options=["direct","complement"],required=True)})
    with right:
        st.subheader("Live tree")
        st.caption("Chance nodes are circles; terminal nodes are double circles. Timed rewards and branch probability rules are shown on the diagram.")
        dot=structure_to_dot(strategies_df.to_dict("records"),nodes_df.to_dict("records"),branches_df.to_dict("records"))
        st.graphviz_chart(dot,width="stretch")

compiled=None
run=None
decision=None
if threshold is not None:
    try:
        compiled=compile_builder_tables(parameter_df.to_dict("records"),strategies_df.to_dict("records"),nodes_df.to_dict("records"),branches_df.to_dict("records"))
        run=run_decision_tree(compiled.tree,compiled.parameters,included_cost_bearers=included_cost_bearers or None,cost_discount_rate=cost_discount_rate,outcome_discount_rate=outcome_discount_rate)
        model_strategies=[Strategy(compiled.strategy_names[row.strategy_id],row.expected_cost,row.expected_outcome) for row in run.strategies]
        decision=fully_incremental_analysis(model_strategies,threshold)
    except (BuilderValidationError,DecisionTreeValidationError,ValueError) as exc:
        model_error=str(exc)
    else:
        model_error=None
else:
    model_error="Enter a decision threshold in the sidebar before running the model."

with results_tab:
    st.subheader("Validation and discounted expected values")
    if model_error:
        st.error(model_error)
    else:
        st.success("Tree structure, parameter links, reward timing and probabilities are valid.")
        expected_rows=[{"Strategy":compiled.strategy_names[row.strategy_id],f"Expected cost ({currency.code})":money(row.expected_cost,currency.symbol),f"Expected {outcome.unit}":f"{row.expected_outcome:,.4f}"} for row in run.strategies]
        st.dataframe(expected_rows,hide_index=True,width="stretch")
        st.caption(f"Applied annual discount rates: costs {cost_discount_rate*100:.2f}%, outcomes {outcome_discount_rate*100:.2f}%.")
        st.subheader("Cost-effectiveness results")
        result_rows=[]
        for row in decision.rows:
            result_rows.append({"Strategy":row.strategy.name,"Cost":money(row.strategy.cost,currency.symbol),outcome.label:f"{row.strategy.effect:,.4f}","Status":row.status.replace("_"," ").title(),"Compared with":row.compared_with or "—","Incremental cost":"—" if row.incremental_cost is None else money(row.incremental_cost,currency.symbol),f"Incremental {outcome.unit}":"—" if row.incremental_effect is None else f"{row.incremental_effect:,.4f}",f"ICER ({currency.code}/{outcome.unit})":"—" if row.icer is None else money(row.icer,currency.symbol),"NMB":money(row.nmb,currency.symbol)})
        st.dataframe(result_rows,hide_index=True,width="stretch")
        preferred=decision.preferred_by_nmb
        st.info((f"At {money(threshold,currency.symbol)} per {outcome.unit}, **{preferred[0]}** has the highest NMB." if len(preferred)==1 else "At the selected threshold, highest NMB is tied between: "+", ".join(preferred)+"."))

with sensitivity_tab:
    st.subheader("Deterministic sensitivity analysis")
    if model_error:
        st.info("Resolve the model validation issue first; sensitivity analysis always reruns the validated base model.")
    else:
        strategy_ids=list(compiled.strategy_names)
        c1,c2=st.columns(2)
        comparator_id=c1.selectbox("Comparator for sensitivity analysis",strategy_ids,index=0,format_func=lambda sid: compiled.strategy_names[sid])
        intervention_candidates=[sid for sid in strategy_ids if sid!=comparator_id]
        intervention_id=c2.selectbox("Intervention for sensitivity analysis",intervention_candidates,index=0,format_func=lambda sid: compiled.strategy_names[sid])
        st.caption("Sensitivity outputs use incremental net monetary benefit (INMB). Positive INMB favours the intervention; negative INMB favours the comparator; zero is the switching point.")
        context=dict(intervention_id=intervention_id,comparator_id=comparator_id,willingness_to_pay=threshold,included_cost_bearers=included_cost_bearers or None,cost_discount_rate=cost_discount_rate,outcome_discount_rate=outcome_discount_rate)
        parameter_map={p.id:p for p in compiled.parameters}
        parameter_ids=list(parameter_map)

        ow_tab,tw_tab,th_tab=st.tabs(["One-way","Two-way","Threshold"])
        with ow_tab:
            pid=st.selectbox("Parameter",parameter_ids,key="ow_param",format_func=lambda x: parameter_map[x].label)
            low_default,high_default=default_bounds(parameter_map[pid])
            a,b=st.columns(2)
            low=a.number_input("Low value",value=float(low_default),key="ow_low")
            high=b.number_input("High value",value=float(high_default),key="ow_high")
            values=(low,float(parameter_map[pid].value),high)
            try:
                ow=one_way_tree_inmb(compiled.tree,compiled.parameters,OneWaySensitivitySpec(pid,values),**context)
                st.dataframe(pd.DataFrame([{"Parameter value":v,"INMB":metric} for v,metric in ow]),hide_index=True,width="stretch")
            except (DecisionTreeValidationError,ValueError) as exc:
                st.warning(f"This variation is not valid for the current tree: {exc}")

        with tw_tab:
            x_id=st.selectbox("First parameter",parameter_ids,key="tw_x",format_func=lambda x: parameter_map[x].label)
            y_options=[pid for pid in parameter_ids if pid!=x_id]
            y_id=st.selectbox("Second parameter",y_options,key="tw_y",format_func=lambda x: parameter_map[x].label)
            xlo,xhi=default_bounds(parameter_map[x_id]); ylo,yhi=default_bounds(parameter_map[y_id])
            x1,x2,y1,y2=st.columns(4)
            x_low=x1.number_input("X low",value=float(xlo),key="tw_x_low"); x_high=x2.number_input("X high",value=float(xhi),key="tw_x_high")
            y_low=y1.number_input("Y low",value=float(ylo),key="tw_y_low"); y_high=y2.number_input("Y high",value=float(yhi),key="tw_y_high")
            try:
                tw=two_way_tree_inmb(compiled.tree,compiled.parameters,TwoWaySensitivitySpec(x_id,linear_grid(x_low,x_high),y_id,linear_grid(y_low,y_high)),**context)
                grid=pd.DataFrame(tw,columns=[x_id,y_id,"INMB"])
                st.dataframe(grid.pivot(index=y_id,columns=x_id,values="INMB"),width="stretch")
                st.caption("Each cell is INMB for one combination of the two parameter values.")
            except (DecisionTreeValidationError,ValueError) as exc:
                st.warning(f"Part of this two-way grid is not valid for the current tree: {exc}")

        with th_tab:
            th_id=st.selectbox("Parameter to search",parameter_ids,key="th_param",format_func=lambda x: parameter_map[x].label)
            tlo,thi=default_bounds(parameter_map[th_id])
            q1,q2=st.columns(2)
            lower=q1.number_input("Search lower bound",value=float(tlo),key="th_low")
            upper=q2.number_input("Search upper bound",value=float(thi),key="th_high")
            try:
                switching=threshold_tree_inmb(compiled.tree,compiled.parameters,ThresholdAnalysisSpec(th_id,lower,upper),**context)
                st.success(f"Switching value: **{switching:,.6g}**. At this value, INMB is approximately zero at the selected decision threshold.")
            except (DecisionTreeValidationError,ValueError) as exc:
                st.info(f"No valid switching value was found within these bounds: {exc}")

        st.caption("For older trees that encode binary outcomes with two independent probability parameters, varying only one can violate the sum-to-one constraint. Use the new `complement` branch mode when one branch is 1 − p.")
