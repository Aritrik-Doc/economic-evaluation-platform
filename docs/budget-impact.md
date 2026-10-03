# Budget Impact Analysis (v0.9)

## Purpose

Budget Impact Analysis (BIA) estimates the financial consequences of introducing or changing a health intervention for a defined budget holder and eligible population. It complements cost-effectiveness analysis; it is not a substitute for evidence on value for money.

The first platform implementation uses an auditable annual **cost-calculator** framework:

`eligible population × treatment share × cost per treated person`

The current-treatment scenario is compared with the future-treatment scenario after introduction or implementation of the intervention. Eligible population, treatment mix and costs can change in every model year.

## Methodological profiles

### ISPOR Budget Impact Good Practice

The generic profile is based on Sullivan et al., *Principles of Good Practice for Budget Impact Analysis II* (Value in Health 2014). Key implementation principles are:

- decision-maker / budget-holder perspective;
- local eligible-population and treatment-pattern data where possible;
- explicit current and future intervention mixes;
- explicit anticipated uptake and market effects;
- treatment, administration, monitoring, adverse-event and condition-related costs where relevant;
- period-by-period budget consequences rather than a single discounted net-present-value result;
- sensitivity/scenario analysis appropriate to the decision maker;
- sufficient input and calculation detail for another analyst to replicate the analysis.

Source: https://www.ispor.org/heor-resources/good-practices/article/principles-of-good-practice-for-budget-impact-analysis-ii

### India National BIA Guidelines (2021)

The India profile is based on the national methodological guidelines commissioned in the HTAIn context. The profile encodes:

- healthcare payer / budget-holder perspective;
- consideration of multi-payer and single-payer/UHC scenarios;
- recommended 1–4 year horizon;
- top-down eligible-population estimation;
- open population over the analysis horizon;
- current and future utilisation/coverage mix;
- no discounting of annual budget flows;
- year-wise and resource-wise reporting;
- deterministic sensitivity and scenario analysis.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC8238667/

### NICE resource-impact framing (2026)

NICE's resource-impact material is represented as a **resource-impact framing**, not labelled as a complete BIA reference case. NICE states that its resource-impact tools changed to three future modelling years in January 2026. The profile therefore defaults to a 3-year implementation/resource-planning horizon.

Source: https://www.nice.org.uk/About/What-we-do/Into-practice/resource-impact-assessment/

## Inputs

v0.9 supports:

- budget holder / payer;
- currency;
- 1–10 year custom horizon, constrained by recognised profiles where applicable;
- direct annual eligible-population input or a top-down population funnel;
- covered lives for optional PMPM output;
- multiple interventions/options;
- current and future annual treatment/utilisation shares;
- six cost categories: acquisition, administration/procedure, monitoring, adverse events, disease management, and other;
- annual intervention-specific price/resource-cost change;
- inclusion/exclusion of cost categories;
- evidence source and rationale fields for population, treatment mix/uptake and intervention costs.

Treatment shares must sum to 1 separately for every scenario and year. The engine does not silently normalise invalid shares.

## Outputs

The model reports:

- eligible population by year;
- current-scenario cost by year;
- future-scenario cost by year;
- annual net budget impact;
- cumulative budget impact;
- PMPM budget impact when covered lives are provided;
- treated population by intervention/scenario/year;
- cost per treated person;
- budget disaggregated by cost category;
- CSV export of annual results.

## Scenario explorer

The first scenario explorer can vary:

- eligible-population size;
- cost of a selected intervention;
- future uptake of that intervention.

When target uptake is changed, the remaining future shares are redistributed proportionally across other interventions. This is an explicit scenario rule and should not be interpreted as an evidence-based substitution pattern unless justified by the modeller.

## Transparency check

The BIA workspace checks whether the modeller has documented:

- eligible-population source and rationale;
- current-mix source and rationale;
- future uptake/substitution source and rationale;
- intervention-specific cost/resource sources and costing rationale.

This is a documentation-completeness safeguard only. It does not judge the quality, relevance or bias of the evidence and does not provide a model quality score.

## Deliberate v0.9 boundaries

The first release is an annual cost-calculator BIA. It does not yet include:

- linked Markov/decision-tree clinical-event projections;
- one-time initiation costs based on incident starts as a distinct accounting rule;
- capacity constraints, workforce/equipment natural-unit planning beyond treated counts;
- saved multi-scenario libraries;
- full BIA DSA tornado workflow;
- BIA save/load/audit bundles;
- jurisdiction-specific budget-impact thresholds.

These should be added incrementally after the basic payer workflow is tested with real examples.
