# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform aimed at HEOR, HTA, and eventually payer / market-access workflows.

## Current development milestone: 0.4

Version 0.4 provides the first **parameter-driven model engine**: a decision tree with a **hybrid structured/visual builder**, automatic discounting, deterministic sensitivity analysis, probabilistic sensitivity analysis, model persistence and audit exports.

The v0.3 methods/reference-case layer remains the methodological foundation beneath it.

## Recognised reference cases

### NICE technology appraisal

The current profile uses current NICE PMG36 methods, including:

- QALYs as the reference-case economic outcome
- NHS and Personal Social Services cost perspective
- 3.5% annual discounting for costs and health outcomes
- a horizon long enough to reflect all important differences in costs and outcomes
- current standard technology-appraisal threshold range of **£25,000–£35,000 per QALY gained**

Source: https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/

### HTAIn / Indian Reference Case (2023)

The platform treats the 2023 Indian Reference Case as the governing Indian economic-evaluation reference case:

- QALYs preferred
- DALYs supported in appropriate circumstances and represented as **DALYs averted**
- life-years gained retained as a supplementary measure
- abridged societal base-case perspective
- separate healthcare-payer results should also be reportable
- 3% annual discounting for costs and outcomes, with 0–5% explored in sensitivity analysis
- current practice as comparator; multiple comparators possible
- no single monetary decision threshold hard-coded into the reference case

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10485782/

## Hybrid decision-tree builder

`pages/1_Decision_Tree_Builder.py` combines structured editable tables with a live Graphviz tree. The same structure drives both the visual diagram and the analytical model.

The builder supports:

- parameter library with source, assumption and uncertainty provenance
- multiple strategies
- chance and terminal nodes
- direct and complement branch probabilities (`p` and `1-p`)
- timed cost and health-outcome rewards using `parameter@years`
- automatic annual discounting with separate cost/outcome rates
- computational perspective through cost bearers
- fully incremental cost-effectiveness analysis

## Economic outcomes

All decision-analysis benefit measures are oriented so that **higher = more health benefit**:

- QALYs gained
- life-years gained
- DALYs averted

OS and PFS remain clinical/survival endpoints for later survival-based model structures rather than interchangeable economic outcome measures.

## Deterministic sensitivity analysis

The builder provides:

- one-way sensitivity analysis
- **tornado diagram** using explicit low/high parameter ranges
- **two-way sensitivity analysis**
- **two-way pairwise decision map** showing which strategy is preferred across the parameter grid
- **threshold analysis** identifying the parameter value at which INMB crosses zero

These analyses rerun the complete decision tree and use incremental net monetary benefit for a selected intervention/comparator pair.

## Probabilistic sensitivity analysis

PSA is implemented for decision trees.

The platform deliberately does **not** choose distributions silently. Instead, it shows evidence-informed family suggestions and requires the user to confirm the family and supply distribution parameters derived from the evidence.

Current advisory starting points include:

- probabilities/proportions → Beta for a single bounded probability
- a vector of mutually exclusive probabilities constrained to sum to 1 → Dirichlet
- non-negative costs → Gamma
- hazard ratios / risk ratios / odds ratios → Lognormal
- utilities genuinely bounded 0–1 → Beta, with a warning that utility values may be negative in some systems
- approximately symmetric continuous parameters → Normal where its support is appropriate

Uniform is available as a user-selected option but is not automatically inferred from a low/high range.

Supported PSA distributions are:

- Beta (`alpha`, `beta`)
- Gamma (`shape`, `scale`, or equivalent alpha/rate parameterisation)
- Lognormal (`meanlog`, `sdlog`)
- Normal (`mean`, `sd`)
- Uniform (`low`, `high`)
- grouped Dirichlet (one positive `alpha` per component within a shared `correlation_group`)

PSA sampling is reproducible through an explicit random seed. Only parameters explicitly marked with distribution uncertainty are sampled.

### Correlation behavior

Parameters can carry a `correlation_group`.

For probability vectors configured as grouped Dirichlet components, the engine performs one joint Dirichlet draw per iteration, so the simulated probabilities remain non-negative and sum to 1.

For other declared correlation groups where no covariance or joint sampling structure has yet been provided, PSA is allowed to run but those parameters are sampled independently. The platform displays a prominent methodological warning and carries that warning into the exported audit record. This is intended to support exploratory analysis without pretending that the correlation has been handled correctly.

If independent probability draws make the tree structurally invalid, the run still stops and recommends using a coherent joint probability model such as Dirichlet.

### Probabilistic outputs

The builder provides:

- **cost-effectiveness plane** for a selected intervention/comparator pair
- **cost-effectiveness acceptability curve (CEAC)** across all strategies
- pairwise probability of cost effectiveness at the chosen threshold
- exportable simulation-level PSA draws

The PSA engine reruns the complete model for every draw, including perspective filtering and discounting.

## Save / load / export and audit trail

Decision-tree models can be saved as versioned, human-readable JSON and loaded back into the builder.

A saved model includes methods settings, perspective, time horizon, discounting, threshold, all parameters and provenance, distributions, strategies, nodes and branches, plus a SHA-256 content hash. The file is structurally revalidated on load and a hash mismatch is treated as possible editing or corruption.

The builder can export:

- complete model JSON
- deterministic results CSV
- latest PSA simulation-level CSV
- audit-trail JSON

Audit records are explicit user-created run records. Each contains a unique run id, UTC timestamp, platform version, model hash, analysis type, run settings, results summary, warnings and the complete model snapshot used for that run.

See `docs/uncertainty-and-audit.md` for the detailed conventions.

## Mandatory provenance

Every model parameter must explicitly carry:

- evidence source
- assumption statement and rationale
- uncertainty specification and rationale

Cost parameters additionally require currency, price year and cost bearer(s), plus dated market-FX provenance when conversion is used.

## Currency approach

The platform uses **market currency conversion**, not PPP/international-dollar conversion. Price-year adjustment is treated separately from currency conversion.

## Key files

- `model/reference_cases.py` — NICE, HTAIn and custom methods profiles
- `model/schema.py` — model, parameter, provenance and uncertainty definitions
- `model/decision_tree.py` — parameter-driven decision-tree engine
- `model/tree_builder.py` — compiler from editable UI tables to model objects
- `model/tree_sensitivity.py` — deterministic INMB sensitivity helpers and tornado summaries
- `model/uncertainty_defaults.py` — advisory PSA distribution suggestions
- `model/psa.py` — PSA simulation, grouped Dirichlet sampling, CE-plane data and CEAC calculations
- `model/persistence.py` — versioned model files, hashes, CSV helpers and audit records
- `pages/1_Decision_Tree_Builder.py` — hybrid builder, visual outputs and file/audit UI
- `docs/methods.md` — core methodological specification
- `docs/uncertainty-and-audit.md` — PSA/correlation/persistence/audit conventions

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the **Decision Tree Builder** page.

## Run tests

```bash
pytest -q
```

## Next milestones

1. Add explicit covariance-matrix / multivariate-normal joint sampling and other joint distributions where methodologically justified.
2. Add scenario-management workflows and richer reporting/export.
3. Improve audit/version comparison, including model-difference summaries between saved versions.
4. Build the **cohort state-transition / Markov engine** using the same parameter, provenance, reference-case, perspective, uncertainty, persistence and visualisation framework.
