# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform aimed at HEOR, HTA, and eventually payer / market-access workflows.

## Current development milestone: 0.5

Version 0.5 adds a **cohort state-transition / Markov modeller** alongside the guided decision-tree modeller. Both model types use the same reference-case layer, parameter provenance, split deterministic/probabilistic uncertainty and common cost-effectiveness decision-analysis engine.

The Markov implementation deliberately begins with a transparent **closed-cohort, discrete-time, time-homogeneous** model. Semi-Markov/tunnel-state and time-varying transition logic are explicit later extensions rather than hidden assumptions in the first release.

## Recognised reference cases

### NICE technology appraisal

The current profile uses NICE PMG36 methods, including QALYs as the reference-case economic outcome, NHS/PSS costs, 3.5% annual discounting for costs and health outcomes, and a horizon long enough to reflect important differences.

Source: https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/

### HTAIn / Indian Reference Case (2023)

The platform treats the 2023 Indian Reference Case as the governing Indian economic-evaluation reference case: QALYs preferred, DALYs supported where appropriate and represented as **DALYs averted**, life-years gained retained as supplementary, abridged-societal base case, 3% discounting for costs and outcomes, current practice as comparator, and no single monetary decision threshold hard-coded into the profile.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10485782/

## Cohort Markov / state-transition modeller

`pages/2_Cohort_Markov_Builder.py` provides five work areas:

1. **Methods** — cycle length, maximum horizon, fixed-horizon versus cohort-depletion stopping, discounting, state-reward accrual timing and transition-event timing.
2. **Parameters** — the same guided Base case / DSA / PSA / Evidence-and-assumptions parameter cards used by the decision-tree workflow.
3. **States & transitions** — health states, absorbing-state designation, strategies, initial cohort distributions, strategy-specific transition rows and a rendered state diagram.
4. **Rewards** — state occupancy costs/outcomes and transition-event costs/outcomes.
5. **Analyse** — base-case totals, fully incremental CEA, cohort trace, tornado DSA, Markov PSA, cost-effectiveness plane and CEAC.

### Transition semantics

Transition inputs in v0.5 are **probabilities for the selected model cycle length**. The modeller can use:

- `direct` — use probability parameter `p`;
- `complement` — use `1-p`;
- `residual` — use `1 - sum(other outgoing probabilities)`.

Every non-absorbing stochastic row must sum to 1 within numerical tolerance. Absorbing states may omit their explicit self-transition; the engine then supplies probability 1.

The platform does **not** silently treat rates, hazards or hazard ratios as transition probabilities. Explicit rate/intensity-matrix and competing-risk-aware conversion is a later extension.

### State and transition rewards

State rewards may accrue `per_cycle` or `per_year`. Per-year rewards are multiplied by cycle length in years. Typical examples are annual state-management costs and utility weights used to generate QALYs.

Transition-event rewards are applied to expected flow between two states and are intended for one-off events such as progression, hospitalisation or treatment initiation.

### Within-cycle accrual

State rewards can use start-of-cycle, end-of-cycle or half-cycle/trapezoidal occupancy. Half-cycle accrual is an explicit modelling choice rather than an automatically imposed correction. Transition-event rewards can be timed at the start, midpoint or end of a cycle for discounting.

### Time horizon

The model can run a fixed number of cycles or use **cohort-depletion stopping**, where simulation ends when the remaining cohort in non-absorbing states falls below a declared threshold, subject to a maximum-cycle safety bound.

## Validation

The Markov engine validates:

- unique states and strategies;
- initial state distributions summing to 1;
- defined state/parameter references;
- no duplicate origin/destination transitions within a strategy;
- probability parameters in `[0,1]`;
- no more than one residual transition per origin state;
- stochastic transition rows summing to 1;
- absorbing states cannot be exited;
- cost/outcome reward parameter types;
- transition rewards only on structurally defined transitions;
- cohort mass conservation during simulation;
- finite valid discount rates.

## Deterministic and probabilistic uncertainty

DSA and PSA remain independent properties of each parameter. A parameter can participate in both at the same time.

Markov DSA reruns the full cohort model under deterministic parameter overrides and currently exposes a one-way INMB tornado view in the v0.5 page. The engine wrappers also support generic one-way, two-way and threshold analyses.

Markov PSA reruns the complete cohort model for every draw. Scalar Beta, Gamma, Lognormal, Normal and Uniform distributions are supported, plus grouped Dirichlet components for coherent probability vectors. Markov PSA outputs reuse the common cost-effectiveness plane and CEAC calculations.

A model/settings fingerprint prevents an old in-session PSA result from being displayed after the current Markov model has changed.

## Currency guard

v0.5 does not silently convert cost inputs between currencies. Every cost parameter must match the selected analysis currency before the Markov model can run. If the modeller changes the analysis currency, cost values must be explicitly converted and their source/date documented before the parameter currency is changed.

## Guided decision-tree modeller

`pages/1_Decision_Tree_Builder.py` remains the guided constrained visual decision-tree modeller introduced in v0.4.1. It supports chance events, terminal outcomes, direct/complement branch probabilities, timed rewards, base-case CEA, DSA, PSA, model persistence and audit exports.

## Economic outcomes

All decision-analysis benefit measures are oriented so that **higher = more health benefit**:

- QALYs gained
- life-years gained
- DALYs averted

OS and PFS remain clinical/survival endpoints for later survival-based model structures.

## Persistence and audit status

Decision-tree save/load/audit remains available. Markov JSON persistence and audit-record export are **not yet claimed as complete in v0.5**; they are the next model-infrastructure sub-step so the Markov model structure can be stored and revalidated with the same rigor as decision trees.

## Key files

- `model/reference_cases.py` — NICE, HTAIn and custom methods profiles
- `model/schema.py` — provenance and split DSA/PSA uncertainty definitions
- `model/parameterisation.py` — guided distribution conversions and legacy migration
- `model/decision_tree.py` — parameter-driven decision-tree engine
- `model/markov.py` — cohort state-transition engine
- `model/markov_builder.py` — compiler from editable Markov structures to validated model objects
- `model/markov_sensitivity.py` — deterministic Markov INMB sensitivity wrappers
- `model/markov_psa.py` — Markov PSA engine
- `model/markov_reproducibility.py` — currency and stale-result reproducibility guards
- `model/psa.py` — shared PSA result structures, CE plane and CEAC calculations
- `pages/1_Decision_Tree_Builder.py` — guided decision-tree modeller
- `pages/2_Cohort_Markov_Builder.py` — guided cohort Markov modeller
- `docs/cohort-markov.md` — detailed Markov methodological conventions and limitations
- `docs/uncertainty-and-audit.md` — uncertainty/correlation/persistence conventions

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open either the **Decision Tree Builder** or **Cohort Markov / State-Transition Builder** page.

## Run tests

```bash
pytest -q
```

## Next modelling extensions

The main state-transition extensions after the first v0.5 UI test are:

1. Markov model save/load/audit parity with the decision-tree workflow.
2. Tunnel-state / semi-Markov support for time-in-state dependence.
3. Time-varying transition matrices and age/time-dependent mortality.
4. Explicit rate/intensity-matrix and competing-risk-aware conversion.
5. Scenario management and richer Markov validation/reporting.
