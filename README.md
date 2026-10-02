# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform aimed at HEOR, HTA, and eventually payer / market-access workflows.

## Current version: 0.3

Version 0.3 adds the **methods and model-schema layer** underneath the existing multi-strategy decision-analysis engine.

The platform now distinguishes between recognised HTA reference cases, custom methods profiles, model structure, parameters, evidence provenance, assumptions, uncertainty, clinical endpoints, economic outcomes, and sensitivity-analysis specifications.

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

## Sensitivity analysis

Version 0.3 defines first-class specifications and helpers for:

- one-way deterministic sensitivity analysis
- **two-way sensitivity analysis**
- **threshold analysis**
- scenario analysis
- probabilistic sensitivity analysis

Threshold analysis identifies parameter switching values for metrics such as incremental net monetary benefit. Two-way analysis evaluates the full grid formed by two parameter ranges.

## Decision-analysis engine

The existing deterministic engine continues to support:

- multiple mutually exclusive strategies
- net monetary benefit
- strong dominance
- extended dominance
- efficient cost-effectiveness frontier
- sequential ICERs
- plain-language interpretation

## Currency approach

The platform uses **market currency conversion**, not purchasing-power-parity/international-dollar conversion.

Version 0.3 introduces the schema required to preserve source currency, target currency, exchange rate, exchange-rate date, exchange-rate source and original cost price year. Price-year adjustment remains conceptually separate from FX conversion.

## Key v0.3 files

- `model/reference_cases.py` — recognised NICE/HTAIn profiles and custom profile support
- `model/schema.py` — auditable model, parameter, source, uncertainty and structure definitions
- `model/sensitivity.py` — one-way, two-way, threshold, scenario and PSA specifications/helpers
- `docs/methods.md` — methodology and source documentation
- `tests/test_reference_cases.py`
- `tests/test_schema.py`
- `tests/test_sensitivity.py`

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

## Next milestone

Use the v0.3 schema to build the first **parameter-driven model engine**, so expected costs and health outcomes are calculated from model structure and parameters rather than entered directly. The specific first engine (decision tree or cohort state-transition model) should be agreed methodologically before implementation.
