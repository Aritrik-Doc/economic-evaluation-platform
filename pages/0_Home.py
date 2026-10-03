"""Product landing page for the Economic Evaluation Platform."""

from __future__ import annotations

import streamlit as st

from model.currency import CURRENCIES
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.reference_cases import REFERENCE_CASES, custom_reference_case
from ui.design_system import card, coloured_block, hero, workflow_step


STATUS_LABELS = {
    "efficient": "On the cost-effectiveness frontier",
    "strongly_dominated": "Dominated",
    "extendedly_dominated": "Extendedly dominated",
}


def money(value: float | None, symbol: str) -> str:
    if value is None:
        return "—"
    return f"{symbol}{value:,.2f}"


def number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.4f}"


def parse_threshold(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    value = float(cleaned)
    if value < 0:
        raise ValueError("Decision threshold cannot be negative.")
    return value


def explanation(row, outcome_label: str, symbol: str) -> str:
    name = row.strategy.name
    if row.status == "strongly_dominated":
        return (
            f"**{name} is dominated.** Another available strategy provides at least as much "
            "health benefit at a lower cost, or more health benefit without costing more."
        )
    if row.status == "extendedly_dominated":
        return (
            f"**{name} is extendedly dominated.** Moving through other available strategies "
            "provides additional health benefit at a better incremental cost-effectiveness rate."
        )
    if row.compared_with is None:
        return f"**{name} is on the cost-effectiveness frontier** and is the starting strategy."
    return (
        f"**{name} remains on the cost-effectiveness frontier.** Compared with "
        f"**{row.compared_with}**, it costs {money(row.incremental_cost, symbol)} more and "
        f"provides {number(row.incremental_effect)} additional {outcome_label.lower()}."
    )


hero(
    "Build transparent, reproducible health-economic models",
    "Construct decision trees and state-transition models, document the evidence behind every parameter, explore uncertainty, and communicate cost-effectiveness and affordability results within a structured HEOR and HTA workflow.",
    eyebrow="Economic Evaluation Platform",
)

c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("pages/1_Decision_Tree_Builder.py", label="Start a decision-tree model", icon="🌿")
with c2:
    st.page_link("pages/2_Cohort_Markov_Builder.py", label="Start a cohort Markov model", icon="🔁")
with c3:
    st.page_link("pages/4_State_Transition_Save_Load_Audit.py", label="Open / restore a saved model", icon="💾")

st.markdown("## From evidence to decision")
st.caption(
    "The platform separates model structure, evidence, assumptions, uncertainty and decision analysis so each can be reviewed independently."
)
workflow_step(1, "Define the decision problem", "Specify population, interventions, comparators, perspective, reference case, horizon, discounting and outcome measure.")
workflow_step(2, "Build the model", "Represent the clinical pathway using a decision tree, cohort state-transition model, or advanced semi-Markov dynamics.")
workflow_step(3, "Document the evidence", "Attach sources, assumptions, rationale, uncertainty and cost metadata to the parameters that drive the model.")
workflow_step(4, "Analyse value and affordability", "Explore cost effectiveness, uncertainty and—where relevant—budget impact for the population and budget holder.")
workflow_step(5, "Review and reproduce", "Validate structure, inspect outputs, save the full model snapshot and retain reproducible audit information.")

st.markdown("## Designed for transparent analysis")
c1, c2, c3 = st.columns(3)
with c1:
    card(
        "Evidence transparency",
        "Sources, assumptions, uncertainty and methodological rationale are stored alongside model parameters rather than separated from the calculation.",
        kicker="Document",
    )
with c2:
    card(
        "Methodological consistency",
        "Recognised reference-case profiles provide structured defaults and make departures visible without preventing justified alternative assumptions.",
        kicker="Structure",
    )
with c3:
    card(
        "Reproducibility",
        "Model files, seeds, engine settings, hashes and audit records preserve the analytical context needed to reconstruct an analysis.",
        kicker="Reproduce",
    )

coloured_block(
    "Transparency check",
    "The platform checks whether important documentation is present — for example evidence sources, assumption rationales, DSA/PSA specifications, currency, price year and cost bearer. It identifies missing or provisional documentation but does not score scientific quality, risk of bias, or the credibility of a modelling choice.",
    tone="teal",
    kicker="A documentation safeguard — not a model quality score",
)

coloured_block(
    "Software validation does not replace scientific judgement",
    "The platform can validate model structure, calculations and documentation completeness. The validity of an economic evaluation still depends on the quality and relevance of the evidence, assumptions and methodological choices supplied by the analyst. Users should document and justify those choices explicitly.",
    tone="amber",
    kicker="Reliability",
)

st.markdown("## Built around recognised HTA methods")
c1, c2, c3 = st.columns(3)
with c1:
    card(
        "NICE technology appraisal",
        "A structured profile for NICE technology-appraisal economic evaluation, including the preferred health outcome, perspective and reference-case methodological settings.",
        kicker="England",
    )
with c2:
    card(
        "HTAIn / Indian Reference Case 2023",
        "A structured Indian profile using the 2023 reference case, including the abridged societal perspective and India-specific outcome conventions.",
        kicker="India",
    )
with c3:
    card(
        "Custom methods profile",
        "Define perspective, horizon, discounting, currency, outcome and decision threshold when another jurisdiction or analytical specification is required.",
        kicker="Extensible",
    )
st.caption("The profile registry is designed to expand to additional recognised HTA systems as the platform develops.")

st.markdown("## Choose a modelling workspace")
workspace_cols = st.columns(3)
with workspace_cols[0]:
    card(
        "Decision Tree Modeller",
        "Guided visual construction for short-horizon pathways and mutually exclusive events, with DSA, two-way analysis, threshold analysis and PSA.",
        kicker="Pathway models",
    )
    st.page_link("pages/1_Decision_Tree_Builder.py", label="Open Decision Tree Modeller →")
with workspace_cols[1]:
    card(
        "Cohort Markov Modeller",
        "Guided health-state and transition modelling with cohort traces, rewards, uncertainty analysis and fully incremental cost-effectiveness analysis.",
        kicker="State-transition models",
    )
    st.page_link("pages/2_Cohort_Markov_Builder.py", label="Open Cohort Markov Modeller →")
with workspace_cols[2]:
    card(
        "Advanced Markov Dynamics",
        "Semi-Markov state-time memory, time-varying transitions, attained-age mortality and explicit hazard/rate conversion for more complex cohort models.",
        kicker="Advanced dynamics",
    )
    st.page_link("pages/3_Advanced_Markov_Dynamics.py", label="Open Advanced Dynamics →")

with st.expander("Quick incremental analysis", expanded=False):
    st.write(
        "Use this secondary tool when strategy-level expected costs and outcomes are already known. Parameter-driven modelling belongs in the dedicated modelling workspaces above."
    )

    with st.sidebar:
        st.divider()
        st.subheader("Quick-analysis settings")
        profile_code = st.selectbox(
            "Reference case",
            options=["NICE_TA", "HTAIN_2023", "CUSTOM"],
            format_func=lambda code: REFERENCE_CASES[code].name if code in REFERENCE_CASES else "Custom methods profile",
            key="home_profile",
        )

        if profile_code == "CUSTOM":
            custom_name = st.text_input("Custom profile name", value="Custom HTA analysis", key="home_custom_name")
            perspective_label = st.text_input("Perspective", value="Healthcare payer", key="home_perspective")
            outcome_code = st.selectbox("Primary economic outcome", list(OUTCOME_MEASURES), format_func=lambda code: OUTCOME_MEASURES[code].label, key="home_outcome_custom")
            currency_code = st.selectbox("Analysis currency", list(CURRENCIES), format_func=lambda code: f"{code} — {CURRENCIES[code].name}", key="home_currency")
            horizon_rule = st.text_input("Time horizon", value="Lifetime", key="home_horizon")
            cost_discount_pct = st.number_input("Cost discount rate (%)", 0.0, 99.0, 3.5, key="home_cost_disc")
            outcome_discount_pct = st.number_input("Outcome discount rate (%)", 0.0, 99.0, 3.5, key="home_outcome_disc")
            threshold_text = st.text_input("Decision threshold", value="30000", key="home_threshold_custom")
            threshold_value = parse_threshold(threshold_text)
            profile = custom_reference_case(
                name=custom_name,
                perspective_label=perspective_label,
                preferred_outcome_code=outcome_code,
                cost_discount_rate=cost_discount_pct / 100,
                outcome_discount_rate=outcome_discount_pct / 100,
                time_horizon_rule=horizon_rule,
                analysis_currency=currency_code,
                threshold=threshold_value,
            )
        else:
            profile = REFERENCE_CASES[profile_code]
            outcome_code = st.selectbox(
                "Primary economic outcome",
                list(OUTCOME_MEASURES),
                index=list(OUTCOME_MEASURES).index(profile.preferred_outcome_code),
                format_func=lambda code: OUTCOME_MEASURES[code].label,
                key="home_outcome_reference",
            )
            currency_code = profile.analysis_currency
            default_threshold = ""
            if profile.threshold_range is not None and outcome_code == profile.threshold_range.outcome_code:
                default_threshold = f"{profile.threshold_range.lower:.0f}"
            threshold_value = parse_threshold(st.text_input("Analysis threshold", value=default_threshold, key="home_threshold_reference"))

    if threshold_value is None:
        st.info("Enter a decision threshold in the sidebar to run quick incremental analysis.")
    else:
        outcome = OUTCOME_MEASURES[outcome_code]
        currency = CURRENCIES[currency_code]
        strategy_count = int(st.number_input("Number of strategies", min_value=2, max_value=10, value=3, step=1, key="home_strategy_count"))
        defaults = [
            ("Standard care", 10000.0, 4.0),
            ("Treatment A", 14000.0, 4.2),
            ("Treatment B", 18000.0, 4.45),
        ]
        strategies = []
        for idx in range(strategy_count):
            name_default, cost_default, effect_default = defaults[idx] if idx < len(defaults) else (f"Strategy {idx + 1}", 18000.0 + idx * 2000, 4.45 + idx * 0.2)
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Strategy", name_default, key=f"home_name_{idx}")
            cost = c2.number_input(f"Cost ({currency.code})", value=cost_default, key=f"home_cost_{idx}")
            effect = c3.number_input(outcome.label, value=effect_default, format="%.4f", key=f"home_effect_{idx}")
            strategies.append(Strategy(name, cost, effect))

        try:
            result = fully_incremental_analysis(strategies, threshold_value)
        except ValueError as exc:
            st.error(str(exc))
        else:
            rows = []
            for row in result.rows:
                rows.append({
                    "Strategy": row.strategy.name,
                    f"Cost ({currency.code})": money(row.strategy.cost, currency.symbol),
                    outcome.label: f"{row.strategy.effect:,.4f}",
                    "Status": STATUS_LABELS[row.status],
                    "Compared with": row.compared_with or "—",
                    "Incremental cost": money(row.incremental_cost, currency.symbol),
                    f"Incremental {outcome.unit}": number(row.incremental_effect),
                    "ICER": money(row.icer, currency.symbol),
                    "NMB": money(row.nmb, currency.symbol),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            for row in result.rows:
                st.markdown(explanation(row, outcome.label, currency.symbol))

st.divider()
st.caption(
    "Economic Evaluation Platform is a modelling and decision-analysis environment. Its validation and transparency features support review and reproducibility; they do not replace critical appraisal of the underlying clinical, economic and methodological evidence."
)
