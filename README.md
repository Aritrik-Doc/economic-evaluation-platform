# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform aimed at HEOR, HTA, and eventually payer / market-access workflows.

## Current development milestone: 0.6

Version 0.6 extends the v0.5 cohort state-transition modeller with **semi-Markov / tunnel-state memory, model-time-varying transitions, age-dependent mortality, and explicit rate/hazard conversion**.

The v0.5 homogeneous cohort engine remains available and unchanged. Advanced dynamics are initially exposed through a separate workbench so they can be tested before being folded into the main guided Markov builder and persistence layer.

## Recognised reference cases

### NICE technology appraisal

The current profile uses NICE PMG36 methods, including QALYs as the reference-case economic outcome, NHS/PSS costs, 3.5% annual discounting for costs and health outcomes, and a horizon long enough to reflect important differences.

Source: https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/

### HTAIn / Indian Reference Case (2023)

The platform treats the 2023 Indian Reference Case as the governing Indian economic-evaluation reference case: QALYs preferred, DALYs supported where appropriate and represented as **DALYs averted**, life-years gained retained as supplementary, abridged-societal base case, 3% discounting for costs and outcomes, current practice as comparator, and no single monetary decision threshold hard-coded into the profile.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10485782/

## Cohort Markov / state-transition modeller

`pages/2_Cohort_Markov_Builder.py` remains the guided v0.5 modeller for transparent, closed-cohort, discrete-time, time-homogeneous models. It supports cycle length, horizon, discounting, state and transition rewards, cohort traces, fully incremental CEA, DSA, PSA, CE plane and CEAC.

Transition inputs in that page remain explicit **probabilities for the selected model cycle** using direct, complement or residual probability rules. The engine does not reinterpret hazards or rates as probabilities.

## Advanced Markov dynamics

`pages/3_Advanced_Markov_Dynamics.py` provides the v0.6 workbench. It separates two clocks:

- **model time** — time since the simulation began;
- **state time** — time since cohort mass entered the current state.

State-time schedules implement semi-Markov / tunnel-state dependence. Internally, cohort occupancy is tracked by `state × tenure`, while economic results continue to use aggregated state occupancy.

### Piecewise schedules

Dynamic transitions use explicit non-overlapping bands `[start_time, end_time)`. A schedule may be based on model time or state time. Missing coverage is an error rather than an implicit carry-forward assumption.

All exits from an origin state must use one coherent representation: probabilities or rates. The engine does not silently mix the two.

### Hazard and rate conversion

`model/transition_dynamics.py` provides:

- single constant rate → interval probability using `p = 1 - exp(-r*t)`;
- interval probability → constant rate;
- probability rescaling under an explicit constant-hazard assumption;
- joint competing-rate conversion so exit probabilities plus remaining in the origin state sum to 1;
- full continuous-time generator conversion `P(t) = exp(Q*t)` using uniformization, without a SciPy runtime dependency.

The generator conversion allows within-cycle multi-step movement such as A→B→C where implied by the continuous-time process.

### Age-specific mortality

Annual age-specific mortality probabilities are converted to forces of mortality. Cycles that cross birthdays integrate the hazard piecewise across age bands. Optional standardized mortality ratios multiply the **mortality rate**, not the probability.

Automatic background mortality is only combined with **rate-based** disease exits. For probability-based rows, death must be represented explicitly within the probability system because there is no unique safe rule for adding an external competing mortality probability.

### Semi-Markov uncertainty

The advanced engine accepts the same parameter override mechanism as the rest of the platform. Dedicated wrappers are provided for deterministic INMB sensitivity analysis and full-model PSA, so time-varying and state-time-dependent transitions can participate in uncertainty analysis without a separate parameter system.

## Validation

Across the v0.5 and v0.6 state-transition layers, validation includes:

- unique states and strategies;
- initial distributions summing to 1;
- valid state and parameter references;
- stochastic probability rows;
- coherent rate-based competing exits;
- absorbing-state behaviour;
- valid piecewise schedule coverage and non-overlap;
- explicit distinction between model-time and state-time schedules;
- age-mortality table coverage;
- cost/outcome reward parameter types;
- cohort mass conservation;
- finite discount rates;
- refusal to combine background mortality heuristically with probability rows.

## Deterministic and probabilistic uncertainty

DSA and PSA remain independent properties of each parameter. A parameter can participate in both simultaneously. Decision-tree, homogeneous Markov and semi-Markov engines all rerun the complete model under parameter overrides rather than applying post-hoc adjustments to totals.

Grouped Dirichlet components remain supported for coherent PSA probability vectors. Other declared correlation groups without a configured joint distribution are sampled independently with a methodological warning.

## Currency guard

The platform does not silently convert cost inputs between currencies. Every cost parameter must match the selected analysis currency before the model can run. Currency conversion, source/date and price-year handling remain explicit provenance concepts.

## Guided decision-tree modeller

`pages/1_Decision_Tree_Builder.py` remains the guided constrained visual decision-tree modeller. It supports chance events, terminal outcomes, direct/complement branch probabilities, timed rewards, base-case CEA, DSA, PSA, model persistence and audit exports.

## Economic outcomes

All decision-analysis benefit measures are oriented so that **higher = more health benefit**:

- QALYs gained
- life-years gained
- DALYs averted

OS and PFS remain clinical/survival endpoints for later survival-based model structures.

## Persistence and audit status

Decision-tree save/load/audit is available. Markov and advanced semi-Markov persistence/audit parity is intentionally the **next milestone** after the v0.6 dynamics are validated, so saved files can capture schedules, mortality tables, hazard semantics and model-time/state-time choices without introducing a second incompatible format.

## Key files

- `model/reference_cases.py` — NICE, HTAIn and custom methods profiles
- `model/schema.py` — provenance and split DSA/PSA uncertainty definitions
- `model/decision_tree.py` — decision-tree engine
- `model/markov.py` — homogeneous cohort state-transition engine
- `model/markov_builder.py` — homogeneous Markov table compiler
- `model/markov_sensitivity.py` / `model/markov_psa.py` — homogeneous Markov uncertainty
- `model/transition_dynamics.py` — hazard/probability conversion, schedules and age mortality
- `model/semi_markov.py` — state-time/model-time dynamic cohort engine
- `model/semi_markov_builder.py` — advanced dynamics table compiler
- `model/semi_markov_sensitivity.py` / `model/semi_markov_psa.py` — advanced-model uncertainty wrappers
- `pages/1_Decision_Tree_Builder.py` — guided decision-tree modeller
- `pages/2_Cohort_Markov_Builder.py` — guided homogeneous cohort Markov modeller
- `pages/3_Advanced_Markov_Dynamics.py` — v0.6 advanced dynamics workbench
- `docs/cohort-markov.md` — homogeneous Markov conventions
- `docs/advanced-markov-dynamics.md` — semi-Markov, time dependence, mortality and hazard conversion

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the Decision Tree Builder, Cohort Markov Builder, or Advanced Markov Dynamics page.

## Run tests

```bash
pytest -q
```

## Next milestone

After v0.6 dynamics are tested in the UI, the next milestone is **Markov/semi-Markov save, load, export and audit parity** with the decision-tree workflow, including versioned JSON schemas, model hashes, run fingerprints and migration handling for v0.5 homogeneous models.
