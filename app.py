"""Streamlit user interface for Economic Evaluation Platform v0.2."""

import streamlit as st

from model.currency import CURRENCIES
from model.economics import OUTCOME_MEASURES, Strategy, fully_incremental_analysis


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


st.set_page_config(page_title="Economic Evaluation Platform", layout="wide")
st.title("Economic Evaluation Platform")
st.caption("Version 0.2 — multi-strategy incremental cost-effectiveness analysis")

with st.sidebar:
    st.header("Analysis settings")

    outcome_code = st.selectbox(
        "Primary economic outcome",
        options=list(OUTCOME_MEASURES),
        format_func=lambda code: OUTCOME_MEASURES[code].label,
        index=0,
    )
    outcome = OUTCOME_MEASURES[outcome_code]

    currency_code = st.selectbox(
        "Analysis currency",
        options=list(CURRENCIES),
        format_func=lambda code: f"{code} — {CURRENCIES[code].name}",
        index=list(CURRENCIES).index("GBP"),
    )
    currency = CURRENCIES[currency_code]

    willingness_to_pay = st.number_input(
        f"Willingness-to-pay threshold ({currency.code} per {outcome.unit})",
        min_value=0.0,
        value=30_000.0,
        step=1_000.0,
    )

    strategy_count = st.number_input(
        "Number of strategies",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
    )

    st.caption(
        "All costs entered in v0.2 must already be in the selected analysis currency. "
        "Daily market FX conversion will be added when source cost parameters can carry "
        "their own currencies."
    )

st.subheader("Strategies")
st.write(
    "Enter expected per-patient costs and outcomes for each mutually exclusive strategy. "
    "These are direct inputs for now; later model structures will calculate them."
)

strategies = []
default_names = ["Standard care", "Treatment A", "Treatment B"]
default_costs = [10_000.0, 14_000.0, 18_000.0]
default_effects = [4.0, 4.2, 4.45]

for idx in range(int(strategy_count)):
    with st.expander(
        default_names[idx] if idx < len(default_names) else f"Strategy {idx + 1}",
        expanded=idx < 3,
    ):
        c1, c2, c3 = st.columns(3)
        default_name = (
            default_names[idx] if idx < len(default_names) else f"Strategy {idx + 1}"
        )
        default_cost = (
            default_costs[idx]
            if idx < len(default_costs)
            else 20_000.0 + idx * 2_000
        )
        default_effect = (
            default_effects[idx]
            if idx < len(default_effects)
            else 4.5 + idx * 0.2
        )

        name = c1.text_input("Strategy name", value=default_name, key=f"name_{idx}")
        cost = c2.number_input(
            f"Cost ({currency.code})",
            value=default_cost,
            step=100.0,
            key=f"cost_{idx}",
        )
        effect = c3.number_input(
            outcome.label,
            value=default_effect,
            step=0.01,
            format="%.4f",
            key=f"effect_{idx}",
        )
        strategies.append(Strategy(name, cost, effect))

try:
    result = fully_incremental_analysis(strategies, willingness_to_pay)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.divider()
st.subheader("Decision summary")

preferred = result.preferred_by_nmb
if len(preferred) == 1:
    st.success(
        f"At a threshold of **{money(willingness_to_pay, currency.symbol)} per "
        f"{outcome.unit}**, **{preferred[0]}** has the highest net monetary benefit."
    )
else:
    st.info(
        "At the selected threshold, the following strategies have equal highest net "
        f"monetary benefit: **{', '.join(preferred)}**."
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
- **Outcome measure:** {outcome.label}. All v0.2 outcome measures are defined so that a higher value represents more health benefit.
- **NMB** = willingness-to-pay × outcome − cost.
- **Fully incremental analysis** orders non-dominated strategies by effectiveness and compares each efficient strategy with the next less effective strategy on the frontier.
- **Dominated strategy:** another strategy is no more costly and no less effective, with at least one strict advantage.
- **Extended dominance:** the strategy is removed because its incremental cost-effectiveness is less efficient than moving through other available strategies.
- **ICERs shown in the table are sequential ICERs on the final efficiency frontier**, not arbitrary pairwise ICERs.
        """
    )
