# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform aimed at HEOR, HTA, and eventually payer / market-access workflows.

## Current development milestone: 0.4

Version 0.4 introduces the first **parameter-driven model engine**: a deterministic decision tree. The v0.3 methods/reference-case layer remains the methodological foundation beneath it.

The decision-tree engine calculates expected costs and health outcomes from explicit model structure and linked parameters rather than requiring strategy totals to be entered directly.

## Recognised reference cases

### NICE technology appraisal

The current profile uses current NICE PMG36 methods, including:

- QALYs as the reference-case economic outcome
- NHS and Personal Social Services cost perspective
- 3.5% annual discounting for both costs and health outcomes
- a horizon long enough to reflect all important differences in costs and outcomes
- current standard technology-appraisal threshold range of **£25,000–£35,000 per QALY gained**

Source: https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/

### HTAIn / Indian Reference Case (2023)

The platform treats the 2023 Indian Reference Case as the governing economic-evaluation reference case:

- QALYs preferred
- DALYs supported in appropriate circumstances and represented as **DALYs averted**
- life-years gained retained as a supplementary measure
- abridged societal base-case perspective
- separate healthcare-payer results should also be reportable
- 3% annual discounting for costs and outcomes, with 0–5% explored in sensitivity analysis
- current practice as comparator; multiple comparators possible
- horizon long enough to capture all significant costs and consequences
- no single monetary decision threshold hard-coded into the reference case

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10485782/

## Custom methods profile

A `CUSTOM` profile is included for jurisdictions or decision contexts not yet represented in the recognised registry. Users explicitly specify:

- primary outcome
- perspective
- time horizon
- cost discount rate
- outcome discount rate
- analysis currency
- decision threshold when one is required

The registry is deliberately extensible so recognised systems such as CADTH, PBAC, ZIN and others can be added later without changing the analytical engine.

## Supported economic outcomes

All decision-analysis benefit measures are oriented so that **higher = more health benefit**:

- QALYs gained
- life-years gained
- DALYs averted

OS and PFS are represented as clinical/survival endpoints for model structures rather than as interchangeable economic outcome measures.

## Mandatory provenance

Every model parameter must explicitly carry:

- evidence source
- assumption statement and rationale
- uncertainty specification and rationale

Cost parameters additionally require:

- currency
- price year
- cost bearer(s)
- dated market-FX conversion provenance when conversion is used

This supports a **computational perspective**: costs can be automatically included or excluded according to who bears them and which reference case is selected.

## Decision-tree engine

The v0.4 decision-tree engine supports:

- multiple mutually exclusive strategies
- a separate root for each strategy
- chance nodes with parameter-linked branch probabilities
- terminal nodes
- costs and health outcomes accrued at chance or terminal nodes
- shared downstream subtrees
- recursive expected-value calculation
- parameter overrides for later DSA, two-way SA and threshold analysis
- cost inclusion/exclusion by cost bearer so perspective affects calculations
- structural validation before a model runs

Validation currently checks:

- at least two strategies
- globally unique node ids
- one root per strategy
- valid child-node references
- valid parameter references
- probabilities constrained to 0–1
- outgoing chance probabilities summing to 1 within tolerance
- no cycles
- cost rewards linked only to cost parameters
- outcome rewards not linked to cost parameters

## Sensitivity analysis

The methods layer defines first-class specifications and helpers for:

- one-way deterministic sensitivity analysis
- **two-way sensitivity analysis**
- **threshold analysis**
- scenario analysis
- probabilistic sensitivity analysis

The decision-tree runner accepts parameter overrides without mutating the base parameter set, allowing the same engine to be called by these sensitivity-analysis workflows.

## Decision-analysis engine

Model-generated strategy totals feed into the existing decision-analysis layer, which supports:

- multiple mutually exclusive strategies
- net monetary benefit
- strong dominance
- extended dominance
- efficient cost-effectiveness frontier
- sequential ICERs
- plain-language interpretation

## Currency approach

The platform uses **market currency conversion**, not purchasing-power-parity/international-dollar conversion.

The schema preserves source currency, target currency, exchange rate, exchange-rate date, exchange-rate source and original cost price year. Price-year adjustment remains conceptually separate from FX conversion.

## Key files

- `model/reference_cases.py` — recognised NICE/HTAIn profiles and custom profile support
- `model/schema.py` — auditable model, parameter, source, uncertainty and structure definitions
- `model/sensitivity.py` — one-way, two-way, threshold, scenario and PSA specifications/helpers
- `model/decision_tree.py` — deterministic parameter-driven decision-tree engine
- `docs/methods.md` — methodology and source documentation
- `tests/test_decision_tree.py` — decision-tree expected-value and validation tests

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

## Next milestones

1. Build a user-facing decision-tree model builder on top of the v0.4 engine, including parameter entry, provenance capture and visual validation.
2. Connect deterministic, two-way and threshold sensitivity analyses directly to model runs.
3. Add PSA sampling over uncertain tree parameters.
4. Build the **cohort state-transition / Markov engine** using the same parameter, provenance, reference-case, perspective and uncertainty framework.
