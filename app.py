"""Streamlit home and quick decision-analysis interface for Economic Evaluation Platform v0.5."""

import streamlit as st

from model.currency import CURRENCIES
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis
from model.reference_cases import REFERENCE_CASES, custom_reference_case


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


def explanation(row, outcome_label: str, symbol: str) -> str:
    name = row.strategy.name
    if row.status == "strongly_dominated":
        return (
            f"**{name} is dominated.** Another available strategy provides at least as much "
            "health benefit at a lower cost, or more health benefit without costing more. "
            "It is therefore excluded from the incremental cost-effectiveness frontier."
        )
    if row.status == "extendedly_dominated":
        return (
            f"**{name} is extendedly dominated.** Moving through other available strategies "
            "provides additional health benefit at a better incremental cost-effectiveness "
            "rate. It is therefore excluded from the final incremental comparison."
        )
    if row.compared_with is None:
        return (
            f"**{name} is on the cost-effectiveness frontier** and is the starting strategy "
            "for the fully incremental comparison."
        )
    return (
        f"**{name} remains on the cost-effectiveness frontier.** Compared with "
        f"**{row.compared_with}**, it costs {money(row.incremental_cost, symbol)} more "
        f"and provides {number(row.incremental_effect)} additional {outcome_label.lower()}. "
        f"Its sequential ICER is {money(row.icer, symbol)} per outcome unit."
    )


def parse_threshold(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError("Decision threshold must be a number.") from exc
    if value < 0:
        raise ValueError("Decision threshold cannot be negative.")
    return value


st.set_page_config(page_title="Economic Evaluation Platform", layout="wide")
st.title("Economic Evaluation Platform")
st.caption("Version 0.5 — HEOR/HTA decision analysis with decision-tree and cohort Markov modelling")
st.info(
    "Use this home screen for quick strategy-level incremental analysis. For parameter-driven modelling, open the Decision Tree Builder or Cohort Markov / State-Transition Builder from the page navigation."
)

with st.sidebar:
    st.header("Methods profile")
    profile_code = st.selectbox(
        "Reference case",
        options=["NICE_TA", "HTAIN_2023", "CUSTOM"],
        format_func=lambda code: (
            REFERENCE_CASES[code].name if code in REFERENCE_CASES else "Custom methods profile"
        ),
    )

    if profile_code == "CUSTOM":
        custom_name = st.text_input("Custom profile name", value="Custom HTA analysis")
        perspective_label = st.text_input("Perspective", value="Healthcare payer")
        outcome_code = st.selectbox(
            "Primary economic outcome",
            options=list(OUTCOME_MEASURES),
            format_func=lambda code: OUTCOME_MEASURES[code].label,
        )
        currency_code = st.selectbox(
            "Analysis currency",
            options=list(CURRENCIES),
            format_func=lambda code: f"{code} — {CURRENCIES[code].name}",
        )
        horizon_mode = st.selectbox("Time horizon", ["Lifetime", "Fixed number of years"])
        if horizon_mode == "Lifetime":
            horizon_rule = "Lifetime"
        else:
            horizon_years = st.number_input("Time horizon (years)", min_value=0.01, value=10.0)
            horizon_rule = f"{horizon_years:g} years"
        cost_discount_pct = st.number_input("Cost discount rate (%)", min_value=0.0, max_value=99.0, value=3.5)
        outcome_discount_pct = st.number_input("Outcome discount rate (%)", min_value=0.0, max_value=99.0, value=3.5)
        threshold_text = st.text_input("Decision threshold", value="30000")
        try:
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
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    else:
        profile = REFERENCE_CASES[profile_code]
        outcome_code = st.selectbox(
            "Primary economic outcome",
            options=list(OUTCOME_MEASURES),
            index=list(OUTCOME_MEASURES).index(profile.preferred_outcome_code),
            format_func=lambda code: OUTCOME_MEASURES[code].label,
        )
        currency_code = profile.analysis_currency
        st.text_input("Analysis currency", value=currency_code, disabled=True)
        outcome_status = profile.outcome_status(outcome_code)
        if outcome_status == "conditional":
            st.warning(
                "This outcome is conditionally supported rather than the preferred reference-case outcome. Document the reason for using it."
            )
        elif outcome_status == "supplementary":
            st.warning(
                "This is a supplementary/non-reference-case primary outcome for the selected profile. Document and justify the departure."
            )
        elif outcome_status == "not_specified":
            st.warning("This outcome is not specified as a reference-case outcome for this profile.")

        if profile.threshold_range is not None and outcome_code == profile.threshold_range.outcome_code:
            default_threshold = f"{profile.threshold_range.lower:.0f}"
            st.caption(
                f"Current reference range: {CURRENCIES[currency_code].symbol}{profile.threshold_range.lower:,.0f}–"
                f"{CURRENCIES[currency_code].symbol}{profile.threshold_range.upper:,.0f} per {OUTCOME_MEASURES[outcome_code].unit}."
            )
        else:
            default_threshold = ""
            if profile.threshold_range is None:
                st.caption(
                    "This reference case does not prescribe a single monetary decision threshold. Enter the threshold used for this analysis and document its source."
                )
            else:
                st.caption(
                    "The recognised reference-case threshold is defined for a different outcome measure. Enter and justify a threshold if NMB analysis is required."
                )
        threshold_text = st.text_input("Analysis threshold", value=default_threshold)
        try:
            threshold_value = parse_threshold(threshold_text)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    outcome = OUTCOME_MEASURES[outcome_code]
    currency = CURRENCIES[currency_code]

    with st.expander("Reference-case methods", expanded=False):
        st.write(f"**Perspective:** {profile.perspective.label}")
        st.write(f"**Cost discounting:** {profile.cost_discount_rate * 100:.1f}%")
        st.write(f"**Outcome discounting:** {profile.outcome_discount_rate * 100:.1f}%")
        st.write(f"**Time-horizon rule:** {profile.time_horizon_rule}")
        st.write(f"**Comparator rule:** {profile.comparator_rule}")
        st.write(f"**Methods source:** {profile.source_title}")

    strategy_count = st.number_input("Number of strategies", min_value=2, max_value=10, value=3, step=1)
    st.caption(
        "Quick-analysis costs must already be expressed in the selected analysis currency. Parameter-driven model pages retain source, price-year and uncertainty metadata for model inputs."
    )

if threshold_value is None:
    st.warning(
        "Enter an analysis threshold to calculate NMB and run the current decision-analysis screen. The reference-case profile itself has still been selected correctly."
    )
    st.stop()

st.subheader("Strategies")
st.write(
    "Enter expected per-patient costs and outcomes for each mutually exclusive strategy. This quick screen is useful for checking incremental economic logic; the modelling pages calculate these totals from parameters and model structure."
)

strategies = []
default_names = ["Standard care", "Treatment A", "Treatment B"]
default_costs = [10_000.0, 14_000.0, 18_000.0]
default_effects = [4.0, 4.2, 4.45]

for idx in range(int(strategy_count)):
    with st.expander(default_names[idx] if idx < len(default_names) else f"Strategy {idx + 1}", expanded=idx < 3):
        c1, c2, c3 = st.columns(3)
        default_name = default_names[idx] if idx < len(default_names) else f"Strategy {idx + 1}"
        default_cost = default_costs[idx] if idx < len(default_costs) else 20_000.0 + idx * 2_000
        default_effect = default_effects[idx] if idx < len(default_effects) else 4.5 + idx * 0.2
        name = c1.text_input("Strategy name", value=default_name, key=f"name_{idx}")
        cost = c2.number_input(f"Cost ({currency.code})", value=default_cost, step=100.0, key=f"cost_{idx}")
        effect = c3.number_input(outcome.label, value=default_effect, step=0.01, format="%.4f", key=f"effect_{idx}")
        strategies.append(Strategy(name, cost, effect))

try:
    result = fully_incremental_analysis(strategies, threshold_value)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.divider()
st.subheader("Decision summary")
preferred = result.preferred_by_nmb
if len(preferred) == 1:
    st.success(
        f"At a threshold of **{money(threshold_value, currency.symbol)} per {outcome.unit}**, **{preferred[0]}** has the highest net monetary benefit."
    )
else:
    st.info(
        "At the selected threshold, the following strategies have equal highest net monetary benefit: "
        f"**{', '.join(preferred)}**."
    )

st.subheader("Fully incremental analysis")
table_rows = []
for row in result.rows:
    table_rows.append(
        {
            "Strategy": row.strategy.name,
            f"Cost ({currency.code})": money(row.strategy.cost, currency.symbol),
            outcome.label: f"{row.strategy.effect:,.4f}",
            "Status": STATUS_LABELS[row.status],
            "Compared with": row.compared_with or "—",
            "Incremental cost": money(row.incremental_cost, currency.symbol),
            f"Incremental {outcome.unit}": number(row.incremental_effect),
            f"ICER ({currency.code}/{outcome.unit})": money(row.icer, currency.symbol),
            "NMB": money(row.nmb, currency.symbol),
        }
    )
st.dataframe(table_rows, use_container_width=True, hide_index=True)

st.subheader("What the results mean")
for row in result.rows:
    st.markdown(explanation(row, outcome.label, currency.symbol))

with st.expander("Calculation definitions"):
    st.markdown(
        f"""
- **Outcome measure:** {outcome.label}. All supported economic benefit measures are oriented so that a higher value represents more health benefit.
- **NMB** = willingness-to-pay × outcome − cost.
- **Fully incremental analysis** orders non-dominated strategies by effectiveness and compares each efficient strategy with the next less effective strategy on the frontier.
- **Dominated strategy:** another strategy is no more costly and no less effective, with at least one strict advantage.
- **Extended dominance:** the strategy is removed because its incremental cost-effectiveness is less efficient than moving through other available strategies.
- **ICERs shown in the table are sequential ICERs on the final efficiency frontier**, not arbitrary pairwise ICERs.
        """
    )
