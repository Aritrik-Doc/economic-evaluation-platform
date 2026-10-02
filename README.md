# Economic Evaluation Platform

An open, auditable health-economic modelling and decision-analysis platform aimed at HEOR, HTA, and eventually payer / market-access workflows.

## Current development milestone: 0.4.1

Version 0.4.1 focuses on making the decision-tree modeller usable by people who do not routinely build decision trees. The analytical engine remains parameter-driven, but the primary interface is now a **guided, constrained visual modeller** rather than a spreadsheet-first workflow.

The v0.3 methods/reference-case layer remains the methodological foundation beneath it.

## Recognised reference cases

### NICE technology appraisal

The current profile uses current NICE PMG36 methods, including QALYs as the reference-case economic outcome, NHS/PSS costs, 3.5% annual discounting for costs and health outcomes, a horizon long enough to reflect important differences, and the current £25,000–£35,000 per QALY standard technology-appraisal range.

Source: https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/

### HTAIn / Indian Reference Case (2023)

The platform treats the 2023 Indian Reference Case as the governing Indian economic-evaluation reference case: QALYs preferred, DALYs supported where appropriate and represented as **DALYs averted**, life-years gained retained as supplementary, abridged-societal base case, 3% discounting for costs and outcomes, current practice as comparator, and no single monetary decision threshold hard-coded into the profile.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10485782/

## Guided decision-tree modeller

`pages/1_Decision_Tree_Builder.py` now provides five main work areas:

1. **Model design** — interactive constrained canvas with guided strategy, chance-event and terminal-outcome creation.
2. **Parameters** — parameter cards with separate Base case, DSA, PSA, and Evidence/assumption sections.
3. **Base case** — discounted expected costs/outcomes and fully incremental cost-effectiveness analysis.
4. **Analyse** — a DSA/PSA segmented selector that changes the analysis view without rewriting parameter settings.
5. **Save / export** — model JSON, results CSV, PSA simulations and audit records.

The visual canvas allows node movement and selection but deliberately disables freehand edge creation and editing menus. New pathways are created through constrained controls so the modeller specifies the event, probability rule and next node together. The raw table representation remains available under **Advanced** for experienced users, bulk edits and debugging.

### Decision-tree concepts exposed in the UI

- **Chance event**: an uncertain event with two or more possible pathways.
- **Terminal outcome**: the end of a modelled pathway.
- **Direct probability**: a branch uses parameter `p`.
- **Complement probability**: a branch uses `1-p`, useful for binary events.
- **Timed reward**: a cost or health outcome attached to a node at a specified number of years from model start.

The engine supports multiple mutually exclusive strategies, shared or separate downstream structures, timed costs/outcomes, automatic discounting, computational perspective through cost bearers, and structural validation before analysis.

## Parameter uncertainty: DSA and PSA are independent

Version 0.4.1 separates deterministic and probabilistic uncertainty in the schema. A parameter can therefore participate in **both DSA and PSA at the same time**.

Each parameter stores:

- base-case value and unit
- DSA enabled/disabled, low value, high value and rationale
- PSA enabled/disabled, distribution, distribution parameters, optional correlation/joint-sampling group and rationale
- evidence source
- assumption statement and rationale
- cost metadata where applicable

The DSA/PSA control in the Analyse tab only changes which analysis is shown; it does not overwrite the stored uncertainty specification.

Older saved decision-tree files using the previous combined `range` / `distribution` uncertainty field are migrated at load/compile time into the split representation.

## Guided PSA distribution parameterisation

The modeller still provides evidence-informed distribution-family suggestions but never silently applies them. Once the user selects a family, the UI asks for the variables relevant to that family rather than requiring raw `name=value` strings.

Supported guided entry modes currently include:

- **Beta** — Alpha + Beta, or Mean + SE
- **Gamma** — Shape + Scale, or Mean + SD
- **Normal** — Mean + SD, or Estimate + 95% CI
- **Lognormal** — Meanlog + SDlog, or arithmetic Mean + SD
- **Uniform** — Minimum + Maximum
- **Dirichlet** — one positive Alpha concentration per component in a shared joint-sampling group

The UI converts friendly parameterisations into a canonical internal distribution representation before PSA.

Current advisory starting points include single bounded probabilities → Beta; mutually exclusive probabilities summing to one → Dirichlet; non-negative costs → Gamma; positive relative-effect measures → Lognormal; genuinely 0–1 utilities → Beta; and approximately symmetric continuous parameters → Normal where its support is appropriate.

## Deterministic sensitivity analysis

The builder provides one-way analysis, tornado diagrams, two-way sensitivity analysis with a pairwise decision map, and threshold analysis. These analyses rerun the full decision tree and use incremental net monetary benefit for the selected intervention/comparator pair.

## Probabilistic sensitivity analysis

PSA reruns the complete model for every draw, including discounting and perspective filtering. Outputs include a cost-effectiveness plane, CEAC across all strategies, pairwise probability of cost effectiveness, and simulation-level exports.

Grouped Dirichlet components are sampled jointly so simulated probabilities remain non-negative and sum to one. Other declared correlation groups without a configured joint distribution are sampled independently with a prominent methodological warning that is retained in the audit record. Structurally invalid probability draws still stop the run.

## Economic outcomes

All decision-analysis benefit measures are oriented so that **higher = more health benefit**:

- QALYs gained
- life-years gained
- DALYs averted

OS and PFS remain clinical/survival endpoints for later survival-based model structures.

## Save / load / export and audit trail

Decision-tree models can be saved as versioned human-readable JSON and loaded back into the guided builder. Saved models include methods settings, perspective, horizon, discounting, threshold, parameters/provenance, DSA and PSA specifications, strategies, nodes and branches, plus a SHA-256 content hash.

Exports include complete model JSON, deterministic results CSV, PSA simulation-level CSV, and audit-trail JSON. Explicit audit records contain a unique run ID, UTC timestamp, platform version, model hash, run settings, results summary, warnings and the complete model snapshot used for that run.

## Currency and provenance

The platform uses **market currency conversion**, not PPP/international-dollar conversion. Price-year adjustment remains separate from currency conversion. Evidence source, assumption/rationale and uncertainty/rationale are mandatory model concepts; costs additionally carry currency, price year and cost bearer(s), with FX provenance when conversion is used.

## Key files

- `model/reference_cases.py` — NICE, HTAIn and custom methods profiles
- `model/schema.py` — model, provenance and split DSA/PSA uncertainty definitions
- `model/parameterisation.py` — guided distribution conversions and legacy-row migration
- `model/guided_tree.py` — constrained tree-construction operations
- `model/decision_tree.py` — parameter-driven decision-tree engine
- `model/tree_builder.py` — compiler from UI structures to model objects
- `model/tree_sensitivity.py` — deterministic INMB sensitivity helpers
- `model/uncertainty_defaults.py` — advisory PSA distribution suggestions
- `model/psa.py` — PSA, grouped Dirichlet sampling, CE-plane and CEAC calculations
- `model/persistence.py` — model files, hashes, exports and audit records
- `pages/1_Decision_Tree_Builder.py` — guided visual modeller and analyses
- `docs/methods.md` — methodological specification
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

## Next milestone

After v0.4.1 is tested in the UI, the next model engine is the **cohort state-transition / Markov modeller**, reusing the same reference-case, parameter, provenance, uncertainty, persistence and visualisation framework.
