# Economic Evaluation Platform

Version 0.1 is a deliberately small, auditable starting point for a broader health-economic modelling platform.

## What v0.1 does

It compares one intervention against one comparator using expected per-patient costs and QALYs and calculates:

- incremental cost
- incremental QALYs
- ICER, where mathematically defined
- net monetary benefit (NMB) for each strategy
- incremental net monetary benefit (INMB)
- basic dominance / cost-effectiveness quadrant classification

The economic calculation code is kept separate from the Streamlit interface so it can be tested independently and reused later.

## Definitions

For intervention `1` versus comparator `0`:

- `ΔC = C1 - C0`
- `ΔE = E1 - E0`
- `ICER = ΔC / ΔE` when `ΔE != 0`
- `NMB = λE - C`
- `INMB = NMB1 - NMB0 = λΔE - ΔC`

The app reports dominance separately rather than interpreting a negative ICER by itself.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
pytest -q
```

## Current scope deliberately excludes

- multiple-strategy incremental frontier analysis
- discounting over time
- decision trees or Markov models
- deterministic sensitivity analysis
- probabilistic sensitivity analysis
- CE planes / CEACs
- parameter distributions and correlation
- saved model files / database
- authentication

Those are intended to be added incrementally after the foundational choices are reviewed.

## Proposed next milestone

Version 0.2: support 3+ mutually exclusive strategies and perform correct incremental analysis, including strong and extended dominance.
