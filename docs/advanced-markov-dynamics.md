# Advanced Markov dynamics (v0.6)

Version 0.6 extends the cohort state-transition framework with explicit time dependence and semi-Markov state memory while keeping the v0.5 homogeneous engine intact.

## Two different clocks

The platform treats two clocks separately:

- **Model time**: time since the simulation began. This is appropriate for effects such as treatment waning after a fixed calendar/model duration, secular changes, or externally imposed time-varying transition matrices.
- **State time**: time since cohort mass entered the current health state. This is the semi-Markov/tunnel-state clock and is appropriate when transition risk depends on duration in a state.

The semi-Markov engine tracks occupancy by state and completed cycles since state entry. Users therefore do not need to manually create dozens of tunnel states just to represent time-since-entry dependence. Aggregated state occupancy is still reported for economic results and traces.

Reference: ISPOR-SMDM State-Transition Modeling Good Research Practices Task Force 3: https://www.ispor.org/heor-resources/good-practices/article/state-transition-modeling

A practical tutorial distinguishing simulation-time and state-residence-time dependence is available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC9844995/

## Piecewise transition schedules

A dynamic transition is represented by one or more non-overlapping time bands. Each band identifies the parameter used during the interval `[start_time, end_time)`. The final band may be open-ended.

Schedules may use either `model_time` or `state_time`. Missing coverage is an error; the engine does not silently carry the last value forward across a gap.

All outgoing transitions from a state must use one coherent representation in a given dynamic model:

- probabilities, or
- cause-specific rates / hazards.

The platform does not silently mix the two.

## Tunnel / semi-Markov implementation

For each strategy, cohort mass is tracked internally as `state × tenure`. Remaining in the same state advances tenure by one cycle. Entering another state resets tenure to zero. This is mathematically equivalent to an automatically managed set of tunnel substates for the supported piecewise-constant state-time transitions.

The model also reports mean time since entry among current occupants of each state as a diagnostic.

## Single-event rate to probability

For one constant hazard `r` over interval `t`, the platform uses:

`p = 1 - exp(-r*t)`

The inverse finite-rate conversion is:

`r = -ln(1-p)/t`

Probability rescaling between intervals therefore assumes an underlying constant hazard. It is not treated as a generic arithmetic rescaling rule.

## Competing cause-specific rates

For competing exits with cause-specific rates `h_j`, let `H = sum(h_j)`. Over interval `t`:

- probability of remaining in the origin state = `exp(-H*t)`
- probability of exit to cause `j` = `(h_j/H) * (1 - exp(-H*t))`

This conversion is performed jointly so the resulting row remains stochastic. Independent conversion of each competing hazard is not used.

## Full continuous-time generator conversion

For a continuous-time Markov chain with generator matrix `Q`, the exact interval transition matrix is:

`P(t) = exp(Q*t)`

The platform computes this using **uniformization**, so no additional SciPy runtime dependency is required. The method preserves the possibility of multiple transitions within one model interval (for example A→B→C within a cycle), which a simple row-wise competing-risk conversion does not represent.

Background/theoretical references:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC6289421/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9181506/

## Age-specific background mortality

Annual age-specific mortality probabilities are converted to annual forces of mortality. If a cycle crosses a birthday, the integrated hazard is calculated piecewise across age bands. A standardized mortality ratio (SMR), when used, multiplies the mortality rate rather than directly multiplying the annual probability.

Background mortality is only added automatically to **rate-based** disease exits. If an origin row is probability-based, mortality must be represented explicitly as part of that probability system. This prevents an arbitrary combination rule from being applied to probabilities that may already include competing death risk.

The current v0.6 workbench uses one mortality table for an analysis and allows strategy-specific application rules, initial age, applicable states and an optional SMR parameter.

## Current limitations

The v0.6 advanced dynamics layer is intentionally bounded:

- transition schedules are piecewise constant within model cycles;
- state-time memory is cohort-level and does not model arbitrary individual history beyond time since current-state entry;
- background mortality currently joins rate-based exits only;
- age/sex-specific table selection beyond one supplied mortality table is not yet automated;
- no interpolation between mortality ages is performed beyond the constant-force assumption within each age year;
- explicit continuous-time generator conversion is available as a conversion tool but is not yet a selectable engine for every dynamic semi-Markov row;
- advanced-model save/load/audit is deferred to the next milestone by design.

## Validation

Tests cover:

- single-event rate/probability round trips;
- probability interval rescaling;
- competing-risk row sums;
- generator-matrix conversion and within-cycle jump-over paths;
- model-time vs state-time schedules;
- age-specific mortality across birthdays;
- SMR application to rates;
- semi-Markov state-time memory;
- background mortality competing with disease rates;
- refusal to combine background mortality heuristically with probability rows;
- reward accrual, discounting and parameter overrides;
- compilation from editable table structures.
