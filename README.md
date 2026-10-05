# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform for HEOR, HTA, payer affordability and implementation planning.

## Current development milestone: 0.15 — stakeholder-readiness pass

The platform now supports a connected analytical pathway from **clinical modelling** through **cost-effectiveness**, **population and uptake**, **Budget Impact Analysis**, **clinical-cost linkage**, **physical resource/capacity planning**, **policy interpretation**, **transparency review**, and **reproducibility/audit**.

Version 0.15 focuses on stakeholder readiness rather than adding another analytical method. It expands the Home page to represent the whole platform, hardens cross-page handoffs, makes BIA population projections reactive, prevents stale invalid upstream analyses from being reused downstream, adds end-to-end integration regressions, verifies navigation targets, and adds a real Streamlit startup smoke test to CI.

## The three policy questions

The platform deliberately keeps three related but distinct questions visible:

1. **Value for money** — what are the expected incremental costs and health outcomes, and how do they compare under the stated decision rule?
2. **Affordability** — what annual and cumulative financial consequences follow from changing the treatment mix for a defined budget holder and eligible population?
3. **Implementation feasibility** — what physical resources are required, what capacity is available, and where do utilisation, headroom or shortfalls arise?

These domains can be used independently or connected. They are not collapsed into a single adoption/rejection score.

## Main workspaces

### Decision Tree Modeller

`pages/1_Decision_Tree_Builder.py`

Guided decision-tree construction for short-horizon pathways and mutually exclusive events, with parameter provenance, timed rewards, deterministic sensitivity analysis, two-way and threshold analysis, PSA, cost-effectiveness plane and CEAC outputs.

### Cohort Markov Modeller

`pages/2_Cohort_Markov_Builder.py`

Closed-cohort, discrete-time state-transition modelling with configurable cycle length, horizon, initial cohort allocation, state/transition rewards, within-cycle timing, discounting, cohort traces, incremental CEA, DSA and PSA.

### Advanced Markov Dynamics

`pages/3_Advanced_Markov_Dynamics.py`

Semi-Markov/state-time memory, model-time-varying transitions, attained-age mortality, competing-rate conversion and explicit probability-to-rate conversion for more complex cohort models. Model time and state time are kept distinct.

### Population & Uptake

`pages/10_Population_Uptake.py`

A reusable policy-population layer for BIA and capacity planning. It supports:

- top-down population funnels;
- direct eligible-population projections;
- separate eligible-population and covered-lives growth;
- explicit **annual eligible population** versus **annual new treatment starts** semantics;
- formula-derived annual values or explicit manual annual overrides;
- current and future allocation across options.

The shared population scenario contains no costs or clinical outcomes. BIA receives an explicit editable snapshot; capacity planning can consume the validated shared population directly.

### Budget Impact Analysis

`pages/6_Budget_Impact_Analysis.py`

Annual payer-budget modelling with eligible population, current/future treatment mix, uptake, intervention-specific cost components, annual and cumulative budget impact, PMPM where covered lives are supplied, category-level expenditure and scenario exploration.

Population projections are reactive: formula-derived annual values change when the population drivers change, while manual annual overrides are explicitly separated from the formula.

### Clinical model → BIA linkage

`pages/7_BIA_Clinical_Linkage.py`

Links selected downstream condition-related clinical costs from Decision Tree, Cohort Markov or Advanced Markov models into BIA. Direct payer-facing acquisition, administration, monitoring and other BIA costs remain separate to reduce double counting.

Longitudinal linked trajectories require annual **new treatment starts**; the platform does not silently interpret a prevalence stock as a fresh cohort each year.

### Resource & Capacity Planning

`pages/8_Resource_Capacity_Planning.py`

Physical implementation planning in natural units. Population can come from:

- the shared Population & Uptake workspace;
- an active validated BIA;
- local capacity-workspace inputs.

Resource requirements can be:

- manual;
- linked from explicit clinical `resource_use` parameters;
- hybrid — linked clinical requirements plus additional manual service-planning requirements.

Resources are unit-agnostic. Examples include staff-hours, treatment-chair hours, bed-days, appointments, tests, scans, syringes, vials, devices, ambulance trips, vehicle-hours, oxygen supply and blood products. Demand and capacity must use the same documented natural unit.

Outputs include current/future demand, residual available capacity, utilisation, headroom, shortfall and simple capacity-expansion scenarios.

### Policy Interpretation

`pages/9_Policy_Interpretation.py`

A deterministic, rules-based communication layer across:

- **Value** — cost-effectiveness results under the stated threshold;
- **Affordability** — validated BIA results;
- **Feasibility** — the latest validated capacity result.

The interpretation explains calculated results in plain language but does not make an adoption, reimbursement or service-allocation recommendation.

### Transparency Check

`pages/5_Transparency_Check.py`

Checks whether important evidence, assumptions, uncertainty specifications and modelling rationales are documented. It is a documentation-completeness safeguard, **not** a scientific quality score and not an evidence-risk-of-bias assessment.

### Save / Load / Audit

`pages/4_State_Transition_Save_Load_Audit.py`

Versioned persistence for cohort Markov and advanced semi-Markov state-transition models, including active methods/engine settings, model tables, provenance and SHA-256 integrity hashes. Loaded models are recompiled and revalidated rather than blindly trusted. Hashes are integrity checks, not signatures of authorship.

## Core methodological safeguards

- No silent FX conversion.
- No silent normalisation of treatment shares or incoherent probabilities.
- DSA and PSA can coexist independently for the same parameter.
- Stored PSA results are invalidated when the substantive model/settings fingerprint changes.
- Automatic background mortality is combined with disease exits only under a coherent rate-based representation; annual probabilities can be explicitly converted to constant rates when that modelling assumption is intended.
- Competing hazards are converted jointly rather than by adding probabilities.
- SMRs act on rates/hazards rather than probabilities.
- Population stock and new-treatment-start cohort semantics are kept distinct.
- Natural-resource demand and capacity must use compatible units.
- Cross-page validated handoffs are invalidated when the upstream configuration becomes invalid or materially changes.
- Policy Interpretation and Transparency Check never imply that software validation establishes scientific correctness.

## Outcomes and economics

Supported economic benefit measures are oriented so that higher values indicate more health benefit:

- QALYs gained;
- life-years gained;
- DALYs averted.

The platform supports fully incremental economic analysis, dominance/extended dominance, ICERs, NMB/INMB and uncertainty outputs where supported by the modeller.

## Reference-case profiles

The repository includes structured profiles for:

- NICE technology-appraisal economic evaluation;
- HTAIn / Indian Reference Case 2023;
- Custom economic-evaluation methods;
- BIA-specific methods profiles including an India-oriented profile and NICE resource-impact framing.

Methodological profiles provide structured defaults and guidance; they do not remove the need to verify the current governing methods in the relevant jurisdiction.

## Stakeholder-readiness validation

The automated workflow now performs:

1. Python compilation of `app.py`, all `pages`, `model` and `ui` modules;
2. the complete pytest regression suite;
3. navigation/page-link integrity tests;
4. a numerical Population → BIA → Capacity integration regression;
5. validated BIA-session handoff tests;
6. a Streamlit server startup/health smoke test.

These tests increase confidence in software consistency. They are not a formal model-validation certificate and cannot establish the quality, bias, relevance or transferability of user-supplied evidence.

## Current boundaries

The platform is suitable for **initial stakeholder review**, but several boundaries remain intentional:

- BIA receives Population & Uptake as an explicit snapshot rather than a live two-way synchronised object.
- Prevalent-stock clinical/capacity linkage requires an explicit time-since-treatment/state distribution and is not inferred automatically.
- Capacity planning is deterministic demand-versus-capacity analysis, not queue/scheduling optimisation or automatic rationing.
- Scenario/uncertainty management for BIA/capacity is less mature than the CEA uncertainty framework.
- Policy Interpretation is deterministic and rules-based; it does not yet automatically integrate every stochastic result across every workspace.
- There is no account/database layer yet; Streamlit session state remains session-scoped.
- Software checks do not establish that model structure, evidence sources or assumptions are scientifically valid.

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

The CI workflow also boots the Streamlit server and checks its health endpoint so a repository state that compiles but cannot start the application is caught before merge.
