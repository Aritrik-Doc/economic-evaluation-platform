# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform for HEOR, HTA, and future payer / market-access workflows.

## Current development milestone: 0.8

Version 0.8 introduces the **product shell and visual design system**: a dedicated landing page, a restrained scientific colour palette, clearer workflow orientation, and the **Transparency check** concept for documenting evidence, assumptions and uncertainty without pretending to grade scientific quality.

The analytical engines remain unchanged by this milestone.

## Product principles

The platform is designed around four principles:

1. **Model structure should be understandable.** Decision trees and state-transition models use guided construction rather than requiring users to manipulate internal tables directly.
2. **Evidence should remain attached to the model.** Sources, assumptions, uncertainty, currency, price year and cost bearer are stored with parameters rather than only described elsewhere.
3. **Uncertainty should be explicit.** DSA and PSA are independent parameter properties, with dedicated two-way, threshold, tornado, CE-plane and CEAC outputs where supported.
4. **Reproducibility should be built in.** Saved model files, seeds, settings, hashes and audit records preserve the analytical context needed to reconstruct a run.

## Transparency check

The **Transparency check** assesses documentation completeness, not model quality.

It is designed to flag items such as:

- missing or provisional evidence sources;
- undocumented modelling assumptions or rationales;
- incomplete DSA bounds or rationale;
- incomplete PSA distribution specification or rationale;
- missing currency, price year or cost-bearer metadata for costs.

A completed Transparency check does **not** mean that an evidence source is high quality, that an assumption is valid, or that a model is scientifically credible. Those judgements remain the responsibility of the analyst and reviewer.

## Recognised reference cases

### NICE technology appraisal

The current profile uses NICE PMG36 methods, including QALYs as the reference-case economic outcome, NHS/PSS costs, 3.5% annual discounting for costs and health outcomes, and a horizon long enough to reflect important differences.

Source: https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/

### HTAIn / Indian Reference Case (2023)

The platform treats the 2023 Indian Reference Case as the governing Indian economic-evaluation profile: QALYs preferred, DALYs supported where appropriate and represented as **DALYs averted**, life-years gained retained as supplementary, abridged-societal base case, 3% discounting for costs and outcomes, current practice as comparator, and no single monetary decision threshold hard-coded into the profile.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10485782/

A custom methods profile is also available, and the reference-case registry is designed to expand to other recognised HTA systems.

## Modelling workspaces

### Decision Tree Modeller

`pages/1_Decision_Tree_Builder.py`

Guided constrained visual decision-tree construction with chance events, terminal outcomes, direct/complement branch probabilities, timed rewards, deterministic analysis, two-way analysis, threshold analysis, PSA, CE plane, CEAC, save/load and audit export.

### Cohort Markov / State-Transition Modeller

`pages/2_Cohort_Markov_Builder.py`

Closed-cohort, discrete-time state-transition modelling with guided state and transition construction, initial cohort allocation, state/transition rewards, cycle length, horizon, within-cycle accrual, cohort traces, fully incremental CEA, DSA and PSA.

Transition inputs in this modeller are explicit probabilities for the selected cycle. Direct, complement and residual probability rules are supported.

### Advanced Markov Dynamics

`pages/3_Advanced_Markov_Dynamics.py`

Adds semi-Markov / tunnel-state memory, model-time-varying transitions, state-time-varying transitions, age-dependent mortality, competing-risk rate conversion and explicit hazard/probability conversion.

The engine distinguishes:

- **model time** — time since simulation start;
- **state time** — time since entry to the current state.

State-time schedules provide the semi-Markov / tunnel-state mechanism.

## Hazard and mortality conventions

`model/transition_dynamics.py` supports:

- constant rate → interval probability using `p = 1 - exp(-r*t)`;
- probability → constant rate;
- probability rescaling under an explicit constant-hazard assumption;
- joint competing-rate conversion;
- full continuous-time generator conversion `P(t) = exp(Q*t)` using uniformization.

Age-specific mortality probabilities are converted to forces of mortality. Cycles that cross birthdays integrate the hazard across age bands, and SMRs multiply the mortality **rate**, not the probability.

Automatic background mortality is only combined with rate-based exits. Probability-based rows must include death coherently inside the probability system rather than having an external mortality probability added heuristically.

## Economic outcomes

All supported decision-analysis benefit measures are oriented so that **higher = more health benefit**:

- QALYs gained
- life-years gained
- DALYs averted

OS and PFS remain clinical/survival endpoints rather than being treated as interchangeable economic outcomes.

## Uncertainty

A parameter can participate in DSA and PSA simultaneously.

Evidence-informed distribution families are suggested but never silently imposed. Supported PSA families include Beta, Gamma, Lognormal, Normal, Uniform and grouped Dirichlet sampling. Other declared correlation groups without a configured joint sampling structure are sampled independently with a methodological warning.

## Persistence and audit

Decision-tree and state-transition model families support versioned JSON persistence and audit records. State-transition persistence covers homogeneous cohort Markov and advanced semi-Markov models within one file family.

Saved files retain model structure, parameters, provenance, uncertainty, methods settings and engine settings. SHA-256 content hashes detect unexpected modification; they are integrity checks, not digital signatures of authorship.

## Visual design system

The v0.8 product shell uses a shared visual language built around:

- deep navy for structure and headings;
- teal as the primary scientific/health accent;
- muted blue for secondary emphasis;
- soft neutral backgrounds and white cards;
- green for completed/validated software states;
- amber for methodological attention;
- muted red for invalid model states.

Colour is intended to communicate status and hierarchy rather than scientific certainty.

## Key files

- `app.py` — product landing page and secondary quick incremental analysis
- `ui/design_system.py` — shared visual language and landing-page components
- `model/transparency.py` — documentation-completeness Transparency check
- `model/reference_cases.py` — NICE, HTAIn and custom methods profiles
- `model/schema.py` — parameter provenance and split DSA/PSA definitions
- `model/decision_tree.py` — decision-tree engine
- `model/markov.py` — homogeneous cohort state-transition engine
- `model/semi_markov.py` — advanced dynamic cohort engine
- `model/transition_dynamics.py` — hazard/probability conversion and age mortality
- `model/state_transition_persistence.py` — state-transition persistence and audit

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

## Next UI refinement

The shared design system can now be propagated further into individual modelling pages with consistent page headers, model-status bars and live Transparency check summaries. Analytical development can continue independently because the underlying engines and persisted model schemas are not changed by v0.8 styling.
