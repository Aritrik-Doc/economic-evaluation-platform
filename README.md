# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform aimed at HEOR, HTA, and eventually payer / market-access workflows.

## Current development milestone: 0.4

Version 0.4 provides the first **parameter-driven model engine**: a deterministic decision tree, plus a **hybrid model builder** that combines structured editing with a live visual tree. The v0.3 methods/reference-case layer remains the methodological foundation beneath it.

The decision-tree engine now calculates discounted expected costs and health outcomes from explicit model structure, parameter values, reward timing and perspective-specific cost inclusion. Strategy totals feed directly into the existing fully incremental cost-effectiveness engine.

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

A `CUSTOM` profile is included for jurisdictions or decision contexts not yet represented in the recognised registry. Users explicitly specify primary outcome, perspective, time horizon, discount rates, analysis currency and decision threshold where relevant.

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

Cost parameters additionally require currency, price year, cost bearer(s), and dated market-FX provenance when conversion is used. This supports a **computational perspective** in which costs are included or excluded according to who bears them and which reference case is selected.

## Hybrid decision-tree builder

`pages/1_Decision_Tree_Builder.py` provides the user-facing modelling workflow.

It uses structured editable tables for:

- parameter library and provenance
- strategies and root nodes
- chance and terminal nodes
- branch probabilities and destinations
- timed cost and outcome rewards

A live Graphviz diagram is generated from the same structure, so the visual representation cannot silently diverge from the analytical model.

### Timed rewards and discounting

Rewards are entered using `parameter@years` syntax. Examples:

- `drug_cost@0`
- `followup_cost@2.5`
- `qaly_gain@1`

A reward without an explicit time is treated as occurring at year 0. The engine applies annual discrete discounting separately to costs and outcomes using the selected reference-case or custom rates.

### Probability complements

Binary chance nodes can use one underlying probability parameter twice:

- one branch in `direct` mode = `p`
- the other in `complement` mode = `1 - p`

This keeps branch probabilities coherent during sensitivity analysis and avoids silently renormalising invalid probabilities.

## Decision-tree engine

The engine supports:

- multiple mutually exclusive strategies
- a separate root for each strategy
- chance nodes with parameter-linked branch probabilities
- direct and complement branch probability modes
- terminal nodes
- timed costs and health outcomes at chance or terminal nodes
- separate cost and outcome discount rates
- shared downstream subtrees
- recursive expected-value calculation
- parameter overrides for sensitivity analysis
- cost inclusion/exclusion by cost bearer
- structural validation before a model runs

Validation checks node and strategy uniqueness, child and parameter references, probability bounds and sum-to-one constraints, cycles, reward type consistency, timing, and discount-rate validity.

## Sensitivity analysis

The decision-tree builder now runs sensitivity analysis directly against the validated model:

- one-way deterministic sensitivity analysis
- **two-way sensitivity analysis** with an INMB grid
- **threshold analysis** to identify parameter switching values

Sensitivity calculations use **incremental net monetary benefit (INMB)** for a user-selected intervention and comparator. Positive INMB favours the intervention, negative INMB favours the comparator, and zero is the switching point.

The methods layer also defines scenario and probabilistic sensitivity-analysis specifications; PSA sampling is the next analytical implementation step.

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

The platform uses **market currency conversion**, not purchasing-power-parity/international-dollar conversion. The schema preserves source currency, target currency, exchange rate, exchange-rate date, exchange-rate source and original cost price year. Price-year adjustment remains conceptually separate from FX conversion.

## Key files

- `model/reference_cases.py` — recognised NICE/HTAIn profiles and custom profile support
- `model/schema.py` — auditable model, parameter, source, uncertainty and structure definitions
- `model/sensitivity.py` — generic one-way, two-way, threshold, scenario and PSA specifications/helpers
- `model/decision_tree.py` — deterministic parameter-driven decision-tree engine
- `model/tree_builder.py` — compiler from UI tables to model objects
- `model/tree_sensitivity.py` — decision-tree INMB sensitivity helpers
- `pages/1_Decision_Tree_Builder.py` — hybrid structured/visual decision-tree builder
- `docs/methods.md` — methodology and source documentation
- `tests/test_decision_tree.py`
- `tests/test_tree_builder.py`
- `tests/test_tree_sensitivity.py`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the **Decision Tree Builder** page from the Streamlit page navigation.

## Run tests

```bash
pytest -q
```

## Next milestones

1. Add **PSA sampling** over uncertain decision-tree parameters, including reproducible seeds and appropriate distributions.
2. Add **save/load/export** for model definitions, results and audit metadata.
3. Add scenario workflows and richer sensitivity visualisations such as tornado plots and two-way decision maps.
4. Build the **cohort state-transition / Markov engine** using the same parameter, provenance, reference-case, perspective and uncertainty framework.
