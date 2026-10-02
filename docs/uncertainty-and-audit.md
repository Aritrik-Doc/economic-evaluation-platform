# Uncertainty, correlation, model files and audit trail

## Distribution suggestions are advisory

The platform provides evidence-informed suggestions for PSA distribution families, but it does not assign a distribution automatically. The modeller remains responsible for confirming that the distribution support and parameterisation represent the source evidence.

Current scalar families are beta, gamma, lognormal, normal and uniform.

## Dirichlet sampling

A Dirichlet distribution is available for a vector of mutually exclusive probabilities that must sum to 1.

In the decision-tree parameter table:

1. set each component parameter to `uncertainty_kind=distribution`;
2. set each component to `distribution_family=dirichlet`;
3. give all components the same `correlation_group`;
4. provide one positive concentration for each component using `alpha=<value>`.

The PSA engine draws the full probability vector jointly using one Dirichlet draw per iteration. It does not independently sample the components.

Dirichlet should only be used when the grouped parameters genuinely form one probability simplex. It is not a generic solution for all forms of correlation.

## Other declared correlations

If two or more non-Dirichlet parameters share a `correlation_group` but no joint covariance/sampling structure has been specified, PSA is allowed to run so exploratory work can continue. The parameters are sampled independently and the platform emits a prominent warning before and after the run.

This warning is substantive: independent sampling can understate or overstate decision uncertainty when correlation is material. The warning is also carried into exported audit records.

A future extension should support explicit covariance matrices / multivariate-normal sampling and other evidence-appropriate joint distributions. The platform should not infer a correlation coefficient merely because parameters share a group label.

If independent probability draws make a decision tree structurally invalid (for example because competing branch probabilities no longer sum to 1), the simulation still stops. Such probability vectors should normally be represented using a Dirichlet group or another coherent joint probability model.

## Reproducibility

Every PSA run records a user-visible random seed. Using the same saved model, distributions, run settings and seed reproduces the same random draws with the same engine implementation.

PSA outputs include:

- parameter draws;
- strategy costs and outcomes by simulation;
- cost-effectiveness plane data;
- CEAC probabilities;
- pairwise probability of cost-effectiveness;
- methodological warnings.

## Saved model files

Decision-tree models can be saved as human-readable JSON. The model file includes:

- schema and platform versions;
- reference-case / custom methods settings;
- outcome, currency and threshold;
- perspective and included cost bearers;
- time-horizon specification;
- discount rates;
- the complete parameter table, including evidence, assumptions and uncertainty;
- strategies, nodes and branches;
- author/analyst and notes;
- a SHA-256 content hash.

The model is structurally validated before it can be saved. When a saved file is loaded, the model is revalidated and its content hash is checked when present. A hash mismatch is treated as possible editing or corruption rather than silently ignored.

## Audit records

Audit records are explicit run records rather than background telemetry. A user can record a deterministic model run or the latest PSA run and export the in-session audit trail as JSON.

Each record contains:

- a unique run id;
- UTC timestamp;
- platform version;
- model SHA-256 hash;
- analysis type;
- run settings, including discount rates and decision threshold;
- results summary;
- methodological warnings;
- the complete model snapshot used by that record.

Embedding the model snapshot means an audit record remains interpretable even if the working model is subsequently changed.

The platform also allows CSV export of deterministic results and the latest PSA simulation-level outputs.
