# Uncertainty, correlation, model files and audit trail

## DSA and PSA are separate parameter properties

From v0.4.1, deterministic and probabilistic uncertainty are stored independently. A parameter may therefore have a base-case value, DSA low/high bounds, and a PSA distribution simultaneously.

The analysis selector in the UI does not alter parameter definitions. Choosing **Deterministic (DSA)** uses the stored deterministic bounds; choosing **Probabilistic (PSA)** samples the stored probability distributions.

Older decision-tree files created with the previous combined `uncertainty_kind` representation are migrated at compile/load time. A legacy `range` becomes DSA; a legacy `distribution` becomes PSA; the compatibility layer also supports models that carry both.

## Distribution suggestions are advisory

The platform provides evidence-informed suggestions for PSA distribution families, but it does not assign a distribution automatically. The modeller remains responsible for confirming that the distribution support and parameterisation represent the source evidence.

The guided editor asks for variables appropriate to the selected family instead of requiring raw `name=value` text:

- Beta: Alpha + Beta, or Mean + SE
- Gamma: Shape + Scale, or Mean + SD
- Normal: Mean + SD, or Estimate + 95% CI
- Lognormal: Meanlog + SDlog, or arithmetic Mean + SD
- Uniform: Minimum + Maximum
- Dirichlet: Alpha concentration for each jointly sampled component

Friendly inputs are converted to a canonical internal representation before PSA. Raw distribution text remains available only through the advanced model-table view and for backward compatibility.

## Dirichlet sampling

A Dirichlet distribution is available for a vector of mutually exclusive probabilities that must sum to 1.

In the guided parameter editor, enable PSA for every component, select **Dirichlet**, assign the same correlation/joint-sampling group to all components, and enter one positive Alpha concentration per component.

The PSA engine draws the full probability vector jointly using one Dirichlet draw per iteration. It does not independently sample the components.

Dirichlet should only be used when the grouped parameters genuinely form one probability simplex. It is not a generic solution for all forms of correlation.

## Other declared correlations

If two or more non-Dirichlet parameters share a correlation group but no joint covariance/sampling structure has been specified, PSA is allowed to run so exploratory work can continue. The parameters are sampled independently and the platform emits a prominent warning.

This warning is substantive: independent sampling can understate or overstate decision uncertainty when correlation is material. The warning is also carried into exported audit records.

A future extension should support explicit covariance matrices / multivariate-normal sampling and other evidence-appropriate joint distributions. The platform should not infer a correlation coefficient merely because parameters share a group label.

If independent probability draws make a decision tree structurally invalid, the simulation still stops. Probability vectors constrained to sum to 1 should normally use a coherent joint model such as Dirichlet.

## Reproducibility

Every PSA run records a user-visible random seed. Using the same saved model, distributions, run settings and seed reproduces the same random draws with the same engine implementation.

PSA outputs include parameter draws, strategy costs/outcomes by simulation, CE-plane data, CEAC probabilities, pairwise probability of cost-effectiveness, and methodological warnings.

## Saved model files

Decision-tree models can be saved as human-readable JSON. The model file includes schema and platform versions; reference-case/custom methods settings; outcome, currency and threshold; perspective and cost bearers; time horizon; discount rates; the complete parameter definitions including separate DSA/PSA settings, evidence and assumptions; strategies, nodes and branches; author/notes; and a SHA-256 content hash.

The model is structurally validated before it can be saved. When a saved file is loaded, the model is revalidated and its content hash is checked when present. A hash mismatch is treated as possible editing or corruption rather than silently ignored.

## Audit records

Audit records are explicit run records rather than background telemetry. A user can record a deterministic model run or the latest PSA run and export the in-session audit trail as JSON.

Each record contains a unique run ID, UTC timestamp, platform version, model SHA-256 hash, analysis type, run settings, results summary, methodological warnings, and the complete model snapshot used by that record.

Embedding the model snapshot means an audit record remains interpretable even if the working model is subsequently changed. The platform also allows CSV export of deterministic results and the latest PSA simulation-level outputs.
