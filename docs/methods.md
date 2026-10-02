# Methods specification — v0.3

This document records the methodological choices encoded in the platform. It is intended to make model behaviour auditable and to separate recognised HTA reference cases from user-defined methods.

## Recognised reference cases

### NICE technology appraisal (England)

Source: [NICE technology appraisal and highly specialised technologies guidance: the manual (PMG36)](https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/)

Current platform defaults:

- primary reference-case outcome: QALYs gained
- costs perspective: NHS and Personal Social Services
- costs and health effects discounted at 3.5% annually
- time horizon: long enough to reflect all important differences in costs and health outcomes
- comparators: those specified in the appraisal scope
- analysis currency: GBP
- current standard technology-appraisal cost-effectiveness range: £25,000–£35,000 per QALY gained
- fully incremental analysis with dominance and extended dominance where appropriate
- deterministic, scenario and probabilistic uncertainty analysis; threshold analysis and multi-parameter sensitivity analysis where informative

The £25,000–£35,000 range is the current NICE range and supersedes the previous £20,000–£30,000 range. The platform stores the guideline/version metadata so historical analyses can remain reproducible when methods change.

### HTAIn / Indian Reference Case (2023)

Source: [Development of the Indian Reference Case for undertaking economic evaluation for health technology assessment](https://pmc.ncbi.nlm.nih.gov/articles/PMC10485782/)

Current platform defaults:

- preferred outcome: QALYs gained
- DALYs permitted in special scenarios; the platform represents these as **DALYs averted** so that larger values consistently mean greater health benefit
- life-years gained available as a supplementary outcome
- base-case perspective: abridged societal
- separate healthcare-payer results should also be reportable
- direct medical and direct non-medical costs borne by the health system and patients are included in the abridged-societal base case
- indirect/productivity costs are considered in sensitivity analysis rather than the base case
- costs and outcomes discounted at 3% annually; 0–5% explored in sensitivity analysis
- comparator: current practice in use; multiple comparators may be included
- time horizon: long enough to capture all significant costs and consequences
- analysis currency: INR
- no single monetary cost-effectiveness threshold is prescribed by the 2023 Indian Reference Case; an analysis threshold must therefore be explicitly supplied and sourced when NMB/decision-threshold calculations are performed
- deterministic, scenario and probabilistic sensitivity analysis are expected

The 2023 Indian Reference Case is treated as superseding the older platform interpretation based on the 2018 HTAIn manual.

## Custom methods profile

A `CUSTOM` profile is available for jurisdictions or decision contexts not yet represented in the recognised registry. The user must explicitly specify at least:

- primary outcome
- perspective
- time horizon
- discount rate for costs
- discount rate for outcomes
- analysis currency
- decision threshold, when one is being used

Using `CUSTOM` does not imply compliance with any recognised HTA reference case.

The registry is intentionally extensible so additional recognised systems (for example CADTH, PBAC, ZIN or other national HTA methods) can be added later without changing the model engine.

## Core model concepts

The platform separates the **decision problem**, **model structure**, **parameters**, **clinical endpoints**, **economic outcomes** and **decision analysis**.

Primary economic outcomes currently supported are:

- QALYs gained
- life-years gained
- DALYs averted

OS and PFS are represented as clinical/survival endpoints that may drive model state occupancy and health outcomes; they are not treated as interchangeable with QALYs or DALYs averted.

## Mandatory provenance and uncertainty

Every model parameter must explicitly include:

- an evidence source
- an assumption statement and rationale (including an explicit statement when no additional modelling assumption is being made)
- an uncertainty specification and rationale (including an explicit `none` when uncertainty is intentionally not represented)

Cost parameters additionally require:

- currency
- price year
- cost bearer(s), so perspective can be applied computationally
- any FX conversion provenance, including source currency, target currency, exchange rate, rate date and source

The platform keeps **price-year adjustment** conceptually separate from **market FX conversion**.

## Perspective is computational

Perspective is not just report metadata. Cost parameters carry one or more cost bearers (for example health system, patient direct medical, patient direct non-medical, personal social services). A reference case then determines which cost bearers are included in each analysis.

This allows one underlying model to support, for example, the NICE NHS/PSS perspective and the Indian abridged-societal and healthcare-payer perspectives without rebuilding the clinical model.

## Sensitivity analysis

Version 0.3 defines first-class specifications for:

- one-way deterministic sensitivity analysis
- **two-way sensitivity analysis**
- **threshold analysis**
- scenario analysis
- probabilistic sensitivity analysis

The two-way helper evaluates the full Cartesian grid of two parameter values. Threshold analysis searches for a parameter switching value where a decision metric (for example incremental net monetary benefit) reaches a specified target.

For NICE, current PMG36 explicitly describes threshold analysis as useful for identifying switching values and recognises deterministic analyses exploring individual or multiple correlated parameters. Threshold analysis should not be used when the parameter is highly correlated with other influential parameters.

## Outcome direction

All economic benefit measures used by the decision engine are oriented so that **higher = more health benefit**. Therefore the platform uses `DALYs averted`, not raw DALYs, in incremental decision analysis.
