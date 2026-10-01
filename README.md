# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform aimed at HEOR, HTA, and eventually payer / market-access workflows.

## Current version: 0.2

Version 0.2 provides a deterministic multi-strategy cost-effectiveness decision-analysis layer. It is intentionally model-agnostic: for now users enter expected per-patient cost and outcome totals directly; later decision-tree, Markov, and survival models will calculate those totals and pass them into the same decision-analysis engine.

### Supported primary economic outcomes

All current economic outcome measures are oriented so that **more is better**:

- QALYs gained
- life-years gained
- DALYs averted

OS and PFS are planned as clinical/model endpoints for survival-based models, rather than being treated as interchangeable primary economic outcome measures.

### Current decision-analysis outputs

- total cost and outcome by strategy
- net monetary benefit (NMB)
- preferred strategy by NMB at the selected threshold
- strong dominance
- extended dominance
- efficient cost-effectiveness frontier
- sequential incremental costs and effects
- sequential ICERs

The interface explains dominance in plain language so users do not have to infer meaning from a negative ICER.

## Currency approach

Version 0.2 lets the user choose a single analysis currency from a catalog of major currencies.

All costs entered in v0.2 must already be expressed in that analysis currency. Daily **market exchange-rate conversion** will be added when the parameter/model layer allows individual cost inputs to carry a source currency and exchange-rate date. The model should then freeze the rate, source, and date used for reproducibility rather than silently updating old analyses.

PPP / international-dollar conversion is not part of the current design.

## Definitions

For intervention `1` versus comparator `0`:

- `ΔC = C1 - C0`
- `ΔE = E1 - E0`
- `ICER = ΔC / ΔE`, where meaningful
- `NMB = λE - C`
- `INMB = NMB1 - NMB0 = λΔE - ΔC`

For three or more mutually exclusive strategies, the engine performs fully incremental analysis rather than presenting every pairwise ICER.

## Methodological behaviour

The multi-strategy engine:

1. checks strategy names and inputs;
2. identifies strongly dominated strategies;
3. orders remaining strategies by increasing health outcome;
4. removes strategies subject to extended dominance;
5. recalculates the final efficient frontier;
6. reports sequential ICERs only between adjacent strategies on that frontier.

Exact duplicate strategies (identical cost and outcome) are rejected in v0.2 because they are economically indistinguishable and do not define a unique incremental frontier.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
pytest -q
```

## Architecture

```text
Streamlit UI
    |
    v
Decision-analysis engine
    |
    +-- outcomes
    +-- dominance
    +-- incremental frontier
    +-- ICER / NMB
    |
    v
Future model engines
    +-- decision tree
    +-- cohort Markov
    +-- partitioned survival
    +-- microsimulation
```

The core economic code is kept independent of Streamlit so it can later sit behind a different web frontend or API without rewriting the analytical logic.

## Planned next milestone

Version 0.3: define the generic model and parameter schema — including strategies, model type, health states, transitions, costs, utilities/outcomes, sources, uncertainty, currencies, and audit metadata — before implementing the first model-building engine.
