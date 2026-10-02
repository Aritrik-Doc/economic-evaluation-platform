# State-transition persistence and audit (v0.7)

Version 0.7 introduces one versioned persistence family for both homogeneous cohort Markov and advanced semi-Markov models.

## File identity

Every saved file declares:

- `schema_version`
- `platform_version`
- `model_family = state_transition`
- `model_type = cohort_markov | semi_markov`
- model name and save timestamp
- methods snapshot
- engine snapshot
- editable model tables
- author/notes metadata
- SHA-256 content hash

The model type is explicit because the transition semantics differ. A homogeneous cohort model stores per-cycle probability rows; a semi-Markov model can additionally store model-time/state-time schedules, rate semantics and mortality-table rules.

## Methods snapshot

The saved methods section includes:

- reference-case identifier
- economic outcome code
- analysis currency
- decision threshold
- perspective label
- included cost bearers
- cost discount rate
- outcome discount rate

These values are stored with the model rather than inferred from whatever reference-case defaults happen to be current when the file is reopened.

## Engine snapshot

The engine section includes:

- cycle length in years
- maximum number of cycles
- state reward accrual timing
- transition reward timing
- termination mode
- depletion threshold

For semi-Markov models, the transition table itself additionally stores the time basis and schedule bands.

## Model tables

Both model types preserve editable rows for parameters, states, strategies, initial cohort allocation, transitions, state rewards and transition-event rewards.

Semi-Markov bundles also preserve age-specific mortality tables and strategy-specific mortality application rules.

This table-oriented representation is intentional: a loaded model can be restored to the guided editor rather than only deserialized into opaque engine objects.

## Revalidation on load

Loading is not blind deserialization. The loader:

1. parses UTF-8 JSON;
2. checks the supported schema and state-transition family;
3. checks required methods, engine and structure fields;
4. recompiles the saved tables through the appropriate cohort-Markov or semi-Markov compiler;
5. checks the required SHA-256 content hash.

A structurally invalid or hash-mismatched file is rejected.

## Content hash

The SHA-256 hash is calculated from canonical JSON after excluding only volatile save metadata (`saved_at_utc` and the hash field itself). Changing a parameter, transition, schedule, mortality value, method setting or engine setting therefore changes the model hash.

The hash is an integrity/audit aid, not a cryptographic signature of authorship. A person deliberately editing a file and recalculating the hash can produce a new internally consistent file; provenance still depends on evidence fields, authorship metadata and organizational controls.

## Audit records

A state-transition audit record contains:

- audit schema version
- unique run UUID
- UTC run timestamp
- platform version
- model content hash
- model type
- analysis type
- run settings
- results summary
- warnings
- complete model snapshot

The complete model snapshot is embedded so the audit record remains interpretable if the editable working model is later changed.

Before an audit record is created, the embedded snapshot is loaded and validated again.

## UI workflow

`pages/4_State_Transition_Save_Load_Audit.py` provides three areas:

1. **Save / export** — captures the current homogeneous or advanced model tables plus an explicit methods and engine snapshot and downloads versioned JSON.
2. **Load / restore** — validates an uploaded bundle and restores its editable tables to the matching Streamlit session.
3. **Validate / audit** — recompiles and reruns the exact saved snapshot, recalculates economic results and can append a base-case audit record for export.

The saved file is the authoritative source for methods and engine settings. During this first persistence release, restored structural tables are applied directly to the relevant builder session and the saved methods/engine settings are retained as session metadata for exact reproduction. A later UI consolidation can make each builder consume those settings automatically without changing the file schema.

## Compatibility strategy

Decision-tree persistence remains on its existing schema. State-transition persistence starts at schema `0.2` because its structure and engine semantics are materially different from the original decision-tree `0.1` bundle.

No v0.5 Markov file migration is required because v0.5 never claimed a completed Markov save/load format.

Future schema changes should use explicit migration functions rather than silently interpreting old fields with new semantics.
