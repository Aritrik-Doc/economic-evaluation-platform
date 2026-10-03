# Policy Interpretation Foundation

Version 0.12 introduces a deterministic interpretation layer across the platform's three decision-support domains:

1. **Value for money** — cost-effectiveness analysis;
2. **Affordability** — Budget Impact Analysis;
3. **Implementation feasibility** — natural-resource and capacity planning.

The interpretation layer is intended to simplify communication without replacing appraisal or policy judgement.

## Core principle

Interpretation is generated from validated numerical outputs using explicit rules. The software does **not** use a generative model to infer a recommendation, fill missing evidence, or collapse the domains into a single score.

Every detailed statement carries a `basis` field identifying the analytical output used to generate it. This is intended to support auditability and future export into policy briefs or reports.

## Value-for-money interpretation

The cost-effectiveness interpretation can report:

- the strategy or strategies with the highest NMB at the stated threshold;
- incremental cost and effect for the relevant efficient-frontier comparison;
- the sequential ICER when defined;
- strong and extended dominance;
- an optional PSA probability that a strategy has the highest NMB at the same threshold;
- optional leading DSA drivers based on the configured INMB swing.

The language is explicitly threshold-specific. It does not state that a technology is universally 'cost-effective' without reference to the decision rule used.

The interpretation also states that cost-effectiveness does not establish affordability or implementation feasibility.

## Affordability interpretation

The BIA interpretation can report:

- cumulative additional expenditure, cumulative savings, or approximately neutral budget impact;
- whether annual budget effects are consistently positive, consistently negative, or change direction;
- the year with the largest additional expenditure and/or saving;
- PMPM where covered lives are supplied;
- the cost category with the largest absolute cumulative change.

The interpretation does not label a budget impact as affordable unless an explicit payer affordability rule is supplied in a future extension. It instead reports the financial consequence and states that affordability depends on available budget and competing commitments.

## Feasibility interpretation

The capacity interpretation can report:

- whether any modelled resource exceeds residual available capacity;
- the first year in which a shortfall occurs;
- the largest absolute shortfall in natural units;
- the highest modelled utilisation ratio where defined;
- the largest current-versus-future change in resource demand;
- cases where demand is positive but residual capacity is zero.

A shortfall is described as an implementation constraint. The software does not prescribe whether the response should be capacity expansion, phased uptake, treatment restriction, service redesign, reallocation, or another policy action.

## Combined policy summary

The consolidated workspace presents the three domains in the order:

**Value → Affordability → Feasibility**

This is not a weighting system. The domains remain separate because a technology can, for example, have favourable value-for-money results while increasing payer expenditure or exceeding service capacity.

If a domain has not been configured, the workspace reports it as unavailable rather than extrapolating from other analyses.

## Current uncertainty boundary

The core interpretation engine already accepts threshold-specific PSA probabilities and DSA-driver summaries. The initial consolidated workspace prioritises reproducible base-case reconstruction from active model inputs. Deeper automatic transfer of stale-safe PSA/DSA results into the consolidated page should use the same model fingerprints already used elsewhere in the platform.

Richer BIA and capacity uncertainty is intentionally a later milestone. When added, the policy interpretation layer should report alternative scenario ranges and probabilistic capacity/affordability findings without converting them into unqualified recommendations.

## Transparency and reporting context

The framework is consistent with the platform's broader Transparency Check philosophy: reporting completeness is not a scientific quality score. CHEERS 2022 emphasises reporting main results, effects of uncertainty, limitations and practical relevance, while ISPOR BIA guidance emphasises decision-maker-specific outputs and transparent assumptions/scenarios.

Policy Interpretation therefore sits **after** the analytical engines and **alongside** Transparency Check. It cannot improve weak evidence by wording the result more clearly.

## Deliberate exclusions

Version 0.12 does not:

- recommend adoption, rejection or reimbursement;
- assign an overall policy score or traffic-light verdict;
- apply an unstated affordability threshold;
- monetise capacity shortfalls unless explicitly modelled elsewhere;
- infer equity, ethical, legal or distributional conclusions that were not modelled;
- infer missing PSA/DSA evidence;
- replace stakeholder or committee judgement.
