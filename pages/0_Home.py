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
    return "—" if value is None else f"{symbol}{value:,.2f}"


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:,.4f}"


def parse_threshold(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError("Decision threshold must be numeric.") from exc
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
    "Health-economic modelling from value to implementation",
    "Build transparent cost-effectiveness models, estimate budget impact, and test whether services have the resources and capacity to deliver change.",
    eyebrow="Economic Evaluation Platform",
)

st.markdown("## Three questions the platform can answer")
value_col, affordability_col, feasibility_col = st.columns(3)
with value_col:
    coloured_block(
        "Is it good value?",
        "Compare costs and health outcomes across strategies using incremental CEA, ICERs, NMB/INMB and uncertainty analysis.",
        tone="blue",
        kicker="Value for money",
    )
with affordability_col:
    coloured_block(
        "What does it do to the budget?",
        "Apply population and uptake assumptions to estimate annual and cumulative payer budget impact.",
        tone="teal",
        kicker="Affordability",
    )
with feasibility_col:
    coloured_block(
        "Can the system deliver it?",
        "Translate treatment into resource demand and compare staff, facilities, equipment and supplies with available capacity.",
        tone="amber",
        kicker="Implementation feasibility",
    )

st.markdown("## A simple workflow")
st.caption("Use only the parts you need, or connect them for a broader assessment.")
workflow_step(
    1,
    "Build the model",
    "Represent the clinical pathway with a Decision Tree, Cohort Markov model or Advanced Markov model where clinical modelling is needed.",
)
workflow_step(
    2,
    "Define evidence, population and uptake",
    "Document model inputs and assumptions, then define the eligible population and how people are allocated across current and future options.",
)
workflow_step(
    3,
    "Run the decision analyses",
    "Assess cost-effectiveness, budget impact and resource/capacity consequences. Link clinical costs or resource use downstream when appropriate.",
)
workflow_step(
    4,
    "Review and reproduce",
    "Interpret results, inspect documentation gaps, save supported models and retain outputs for review.",
)

st.markdown("## Open a workspace")
st.caption("Clinical modelling")
row1 = st.columns(3)
with row1[0]:
    st.page_link("pages/1_Decision_Tree_Builder.py", label="Decision Tree", icon="🌿")
with row1[1]:
    st.page_link("pages/2_Cohort_Markov_Builder.py", label="Cohort Markov", icon="🔁")
with row1[2]:
    st.page_link("pages/3_Advanced_Markov_Dynamics.py", label="Advanced Markov", icon="🧭")

st.caption("Population, affordability and implementation")
row2 = st.columns(3)
with row2[0]:
    st.page_link("pages/10_Population_Uptake.py", label="Population & Uptake", icon="👥")
with row2[1]:
    st.page_link("pages/6_Budget_Impact_Analysis.py", label="Budget Impact Analysis", icon="💷")
with row2[2]:
    st.page_link("pages/8_Resource_Capacity_Planning.py", label="Resource & Capacity Planning", icon="🏥")

st.caption("Link, review and reproduce")
row3 = st.columns(4)
with row3[0]:
    st.page_link("pages/7_BIA_Clinical_Linkage.py", label="Clinical → BIA", icon="🔗")
with row3[1]:
    st.page_link("pages/9_Policy_Interpretation.py", label="Policy Interpretation", icon="🧾")
with row3[2]:
    st.page_link("pages/5_Transparency_Check.py", label="Transparency Check", icon="🔎")
with row3[3]:
    st.page_link("pages/4_State_Transition_Save_Load_Audit.py", label="Save / Load / Audit", icon="💾")

st.markdown("## Review and governance")
review1, review2, review3 = st.columns(3)
with review1:
    card(
        "Interpretation",
        "Bring value, affordability and feasibility results together in plain language without turning them into an automatic adopt/reject recommendation.",
        kicker="Explain",
    )
with review2:
    card(
        "Transparency",
        "Make evidence sources, assumptions and missing or provisional documentation visible without presenting documentation completeness as scientific quality.",
        kicker="Challenge",
    )
with review3:
    card(
        "Reproducibility",
        "Preserve supported model settings, saved files, fingerprints, hashes and audit information so analyses can be reconstructed and checked.",
        kicker="Reproduce",
    )

st.markdown("## Built around recognised methods")
c1, c2, c3 = st.columns(3)
with c1:
    card(
        "NICE technology appraisal",
        "Structured economic-evaluation settings aligned with the NICE technology-appraisal reference case.",
        kicker="England",
    )
with c2:
    card(
        "HTAIn / Indian Reference Case 2023",
        "Structured Indian economic-evaluation settings, with an India-specific Budget Impact Analysis profile.",
        kicker="India",
    )
with c3:
    card(
        "Custom methods",
        "Define local methodological settings for other jurisdictions, payers or analytical specifications.",
        kicker="Extensible",
    )

coloured_block(
    "Software checks support review — they do not establish scientific validity",
    "The platform can validate calculations, structural relationships and documentation requirements. The credibility of the analysis still depends on the evidence, assumptions and modelling choices supplied by the analyst.",
    tone="amber",
    kicker="Reliability boundary",
)

with st.expander("Quick incremental analysis", expanded=False):
    st.write(
        "Use this secondary tool when strategy-level expected costs and outcomes are already known. Parameter-driven modelling belongs in the dedicated workspaces above."
    )

    threshold_value = None
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
            try:
                threshold_value = parse_threshold(threshold_text)
            except ValueError as exc:
                st.error(str(exc))
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
            threshold_text = st.text_input("Analysis threshold", value=default_threshold, key="home_threshold_reference")
            try:
                threshold_value = parse_threshold(threshold_text)
            except ValueError as exc:
                st.error(str(exc))

    if threshold_value is None:
        st.info("Enter a valid decision threshold in the sidebar to run quick incremental analysis.")
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
    "Economic Evaluation Platform is a modelling and decision-analysis environment. Its validation, interpretation and transparency features support review and reproducibility; they do not replace critical appraisal of the underlying clinical, economic, population, resource or methodological evidence."
)
