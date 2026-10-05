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
    "From clinical pathways to value, affordability and implementation feasibility",
    "Build auditable health-economic models, define the population affected by a policy decision, estimate cost-effectiveness and budget impact, translate clinical pathways into physical resource demand, compare demand with available capacity, and review the resulting evidence in one connected HEOR and HTA environment.",
    eyebrow="Economic Evaluation Platform",
)

st.markdown("### Start with the part of the decision you need")
row1 = st.columns(3)
with row1[0]:
    st.page_link("pages/1_Decision_Tree_Builder.py", label="Decision Tree", icon="🌿")
with row1[1]:
    st.page_link("pages/2_Cohort_Markov_Builder.py", label="Cohort Markov", icon="🔁")
with row1[2]:
    st.page_link("pages/3_Advanced_Markov_Dynamics.py", label="Advanced Markov", icon="🧭")
row2 = st.columns(3)
with row2[0]:
    st.page_link("pages/10_Population_Uptake.py", label="Population & Uptake", icon="👥")
with row2[1]:
    st.page_link("pages/6_Budget_Impact_Analysis.py", label="Budget Impact Analysis", icon="💷")
with row2[2]:
    st.page_link("pages/8_Resource_Capacity_Planning.py", label="Resource & Capacity Planning", icon="🏥")

st.markdown("## Three connected policy questions")
value_col, affordability_col, feasibility_col = st.columns(3)
with value_col:
    coloured_block(
        "Is it good value?",
        "Use Decision Tree, Cohort Markov or Advanced Markov models to estimate expected costs and health outcomes, incremental costs and effects, ICERs, NMB/INMB and decision uncertainty.",
        tone="blue",
        kicker="Value for money",
    )
with affordability_col:
    coloured_block(
        "What does adoption do to the budget?",
        "Define the eligible population and current/future treatment mix, then estimate annual and cumulative budget impact, cost-category changes and PMPM where covered lives are available.",
        tone="teal",
        kicker="Affordability",
    )
with feasibility_col:
    coloured_block(
        "Can the system deliver it?",
        "Translate treatment and clinical pathways into natural-unit demand such as staff time, bed-days, treatment-chair time, tests, vials, syringes, ambulance trips, oxygen or blood products, then compare demand with available capacity.",
        tone="amber",
        kicker="Implementation feasibility",
    )

st.markdown("## How the platform connects")
st.caption(
    "The modules are deliberately separable: evaluators can use only the components relevant to the decision, or link them into a broader assessment without forcing one model to answer every policy question."
)
workflow_step(1, "Frame the decision", "Specify the jurisdiction, perspective or budget holder, population, intervention(s), comparator(s), time horizon, outcomes and decision threshold where relevant.")
workflow_step(2, "Build the clinical pathway", "Use a Decision Tree, Cohort Markov model or Advanced Markov Dynamics. Advanced models support state-time memory, time-varying transitions, attained-age mortality and explicit probability/rate conversion.")
workflow_step(3, "Define population and uptake", "Create a reusable annual eligible-population or new-treatment-start scenario and specify current and future allocation across options. This layer can feed BIA and capacity planning independently.")
workflow_step(4, "Assess value and affordability", "Run incremental cost-effectiveness analysis and, where needed, Budget Impact Analysis. Clinical-model linkage can carry downstream condition-related costs into BIA while keeping payer costs explicit.")
workflow_step(5, "Plan resources and capacity", "Enter resources manually, link natural-unit resource use from the clinical model, or combine both. Compare current/future demand with residual capacity and explore capacity-expansion scenarios.")
workflow_step(6, "Interpret and challenge the result", "Use deterministic policy interpretation, uncertainty outputs and the Transparency check to summarise what the model shows, what drives it and what remains dependent on evidence or assumptions.")
workflow_step(7, "Preserve reproducibility", "Save and restore supported state-transition models with active settings, hashes and audit information, and export analysis tables for review.")

st.markdown("## Connected analysis pathways")
path1, path2 = st.columns(2)
with path1:
    card(
        "Clinical model → economic evaluation",
        "Decision Tree and Markov workspaces combine parameterised clinical pathways with costs and health outcomes. DSA, PSA, threshold analysis, two-way analysis and CEAC/CE-plane outputs are available where supported by the selected modeller.",
        kicker="Clinical + economic",
    )
    st.page_link("pages/1_Decision_Tree_Builder.py", label="Open clinical modelling workspaces →")
with path2:
    card(
        "Clinical model → Budget Impact Analysis",
        "The clinical-linkage workspace can project selected downstream clinical costs into BIA. Acquisition, administration, monitoring and other direct payer costs remain explicit so the source of each cost is visible and double counting can be controlled.",
        kicker="Clinical → affordability",
    )
    st.page_link("pages/7_BIA_Clinical_Linkage.py", label="Open Clinical model → BIA linkage →")

path3, path4 = st.columns(2)
with path3:
    card(
        "Population & Uptake → BIA or capacity",
        "Define the population once and reuse it without making BIA a prerequisite for implementation planning. Annual cross-sectional populations and annual new treatment starts are kept distinct because they imply different longitudinal calculations.",
        kicker="Shared policy population",
    )
    st.page_link("pages/10_Population_Uptake.py", label="Open Population & Uptake →")
with path4:
    card(
        "Clinical resource use → capacity",
        "Resource-use parameters can be mapped to clinical states or decision-tree events and converted into annual natural-unit profiles. Manual and linked requirements can also be combined in a hybrid capacity plan.",
        kicker="Clinical → feasibility",
    )
    st.page_link("pages/8_Resource_Capacity_Planning.py", label="Open Resource & Capacity Planning →")

st.markdown("## Review, interpretation and reproducibility")
review1, review2, review3 = st.columns(3)
with review1:
    card(
        "Policy Interpretation",
        "Bring currently configured value-for-money, affordability and feasibility results together in a deterministic plain-language summary. The platform describes the results under the stated assumptions; it does not issue an adoption/rejection instruction.",
        kicker="Explain",
    )
    st.page_link("pages/9_Policy_Interpretation.py", label="Open Policy Interpretation →", icon="🧾")
with review2:
    card(
        "Transparency Check",
        "Identify missing or provisional documentation for evidence, assumptions, uncertainty, population derivation, uptake, costing and other key model inputs. This is a documentation safeguard, not a scientific quality score.",
        kicker="Challenge",
    )
    st.page_link("pages/5_Transparency_Check.py", label="Open Transparency Check →", icon="🔎")
with review3:
    card(
        "Save / Load / Audit",
        "For supported state-transition models, preserve editable model tables together with active methods and engine settings, then revalidate the model when it is restored.",
        kicker="Reproduce",
    )
    st.page_link("pages/4_State_Transition_Save_Load_Audit.py", label="Open Save / Load / Audit →", icon="💾")

st.markdown("## Choose a modelling or planning workspace")
workspaces = [
    (
        "Decision Tree Modeller",
        "Guided visual construction for short-horizon pathways and mutually exclusive events, with deterministic and probabilistic uncertainty analysis.",
        "Clinical + economic model",
        "pages/1_Decision_Tree_Builder.py",
    ),
    (
        "Cohort Markov Modeller",
        "Closed-cohort state-transition modelling with configurable cycle length, accrual timing, discounting, cohort traces, rewards and incremental economic analysis.",
        "State-transition model",
        "pages/2_Cohort_Markov_Builder.py",
    ),
    (
        "Advanced Markov Dynamics",
        "Semi-Markov state-time memory, model-time schedules, competing rates, attained-age mortality and explicit probability-to-rate conversion for more complex cohort models.",
        "Advanced clinical dynamics",
        "pages/3_Advanced_Markov_Dynamics.py",
    ),
    (
        "Population & Uptake",
        "Build a reusable policy population, distinguish annual eligible populations from new treatment starts, and specify current/future option shares for downstream affordability and capacity analyses.",
        "Shared population layer",
        "pages/10_Population_Uptake.py",
    ),
    (
        "Budget Impact Analysis",
        "Estimate annual and cumulative payer budget consequences from population, uptake and intervention-specific cost components, with scenario exploration and optional clinical-cost linkage.",
        "Affordability model",
        "pages/6_Budget_Impact_Analysis.py",
    ),
    (
        "Resource & Capacity Planning",
        "Model natural-unit resource demand and residual service capacity using manual inputs, linked clinical resource profiles, or a hybrid of both; identify utilisation, headroom and shortfalls by year.",
        "Implementation planning",
        "pages/8_Resource_Capacity_Planning.py",
    ),
]
for index in range(0, len(workspaces), 2):
    cols = st.columns(2)
    for col, (title, body, kicker, target) in zip(cols, workspaces[index : index + 2]):
        with col:
            card(title, body, kicker=kicker)
            st.page_link(target, label=f"Open {title} →")

st.markdown("## Designed for transparent analysis")
c1, c2, c3 = st.columns(3)
with c1:
    card(
        "Evidence transparency",
        "Sources, assumptions, uncertainty and methodological rationale stay alongside the parameters, population, costing and resource inputs that drive the analysis.",
        kicker="Document",
    )
with c2:
    card(
        "Methodological consistency",
        "Recognised reference-case and budget-impact profiles provide structured defaults, while important departures and conversion assumptions remain explicit rather than being hidden.",
        kicker="Structure",
    )
with c3:
    card(
        "Reproducibility",
        "Model files, random seeds, engine settings, fingerprints, hashes and audit records are used where supported to reduce the risk of presenting results from an earlier model state.",
        kicker="Reproduce",
    )

coloured_block(
    "Software checks cannot establish scientific validity",
    "The platform can validate many structural relationships, calculation inputs, probability constraints, stale-result conditions and documentation requirements. It cannot determine whether the underlying clinical evidence is unbiased, whether an assumption is scientifically appropriate, or whether a result is transferable to a particular policy setting. Those judgements remain the responsibility of the analyst and reviewer.",
    tone="amber",
    kicker="Reliability boundary",
)

st.markdown("## Built around recognised methods")
c1, c2, c3 = st.columns(3)
with c1:
    card(
        "NICE technology appraisal",
        "A structured profile for NICE technology-appraisal economic evaluation, including preferred outcome, perspective and reference-case methodological settings.",
        kicker="England",
    )
with c2:
    card(
        "HTAIn / Indian Reference Case 2023",
        "A structured Indian profile for economic evaluation, alongside an India-specific Budget Impact Analysis methods profile for affordability assessment.",
        kicker="India",
    )
with c3:
    card(
        "Custom methods",
        "Define local economic-evaluation or budget-impact assumptions when another jurisdiction, payer or analytical specification is required.",
        kicker="Extensible",
    )
st.caption("The methods registries are designed to expand to additional recognised HTA and budget-impact systems as the platform develops.")

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
