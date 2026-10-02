# Cohort state-transition / Markov modelling conventions

Version 0.5 introduces a closed-cohort, discrete-time state-transition engine.

## Scope of v0.5

The first implementation is deliberately time-homogeneous: within a strategy, the same transition matrix is used in every cycle. This makes the model transparent, easy to validate, and appropriate for decision problems where transition probabilities do not depend materially on prior history or time already spent in a state.

The core elements are:

- mutually exclusive health states;
- one initial state-distribution vector per strategy;
- one strategy-specific transition matrix;
- explicit cycle length;
- state rewards for costs and health outcomes;
- optional transition-event rewards;
- separate discounting of costs and outcomes;
- deterministic and probabilistic parameter uncertainty;
- either a fixed number of cycles or bounded cohort-depletion stopping.

This follows the standard state-transition framing described by the ISPOR-SMDM Modeling Good Research Practices Task Force. State-transition models represent a clinical process using states, transitions, an initial state vector, transition probabilities, cycle length and rewards.

Source: https://www.ispor.org/heor-resources/good-practices/article/state-transition-modeling

## Transition probabilities

Transition inputs in v0.5 are **probabilities for the selected model cycle length**.

Each non-absorbing state must have outgoing probabilities that sum to 1 within numerical tolerance. The builder supports:

- `direct`: use the referenced probability parameter;
- `complement`: use `1 - p` for the referenced parameter;
- `residual`: use `1 - sum(other outgoing probabilities)`.

Only one residual transition is allowed per origin state. This is useful for a self-transition when other outgoing probabilities are separately parameterised.

Absorbing states may omit their self-transition. The engine then supplies a self-transition probability of 1. If an absorbing-state transition row is supplied explicitly, it must still represent a self-transition probability of 1 with no probability of leaving the state.

### Rates and hazards are not silently converted

The platform does not currently treat a rate, hazard or hazard ratio as if it were a transition probability. A simple conversion such as `p = 1 - exp(-r*t)` can be useful for a single constant event rate, but applying scalar conversions independently to competing events can distort the transition process or generate invalid probability rows.

Future versions should add explicit rate/intensity-matrix parameterisation and competing-risk-aware conversion rather than guessing from units.

## Health-state rewards

State rewards can be marked as:

- `per_cycle`: the parameter value is accrued once per occupied model cycle;
- `per_year`: the parameter value is multiplied by the cycle length in years.

Examples:

- annual state-management cost -> cost reward, `per_year`;
- utility weight -> outcome reward, `per_year` to generate QALYs;
- value 1 on living states -> outcome reward, `per_year` to generate life-years;
- fixed administration cost each cycle -> cost reward, `per_cycle`.

The current engine assumes rewards are constant within a state and cycle unless the modeller creates separate states/parameters.

## Transition-event rewards

Transition rewards are applied to the expected flow from an origin state to a destination state during each cycle. They are intended for one-off events such as hospitalisation, progression, treatment initiation or an adverse event that occurs on a transition rather than continuously while occupying a state.

Transition-event rewards can be timed at the start, midpoint or end of the cycle for discounting.

## Within-cycle state accrual and half-cycle correction

The modeller must explicitly choose one of:

- start-of-cycle state occupancy;
- end-of-cycle state occupancy;
- half-cycle / trapezoidal state occupancy.

The half-cycle option averages the start- and end-of-cycle occupancy vectors before accruing state rewards. It is not automatically forced by the engine.

NICE submission guidance asks modellers to report the cycle length and whether a half-cycle correction has been applied where appropriate. Current NICE economic analyses show both use and non-use depending on cycle length and structure, so the platform treats this as a documented modelling choice rather than a universal default requirement.

Sources:
- https://www.nice.org.uk/process/pmg24/chapter/cost-effectiveness
- https://www.nice.org.uk/guidance/NG256/documents/economic-report-2

## Discounting

Costs and outcomes are discounted separately using annual discrete discount rates. State rewards are discounted at the selected state-accrual time; transition rewards are discounted at the selected transition-event time.

For a reward at time `t` years and annual discount rate `r`, present value is:

`PV = value / (1 + r)^t`

The reference-case layer supplies jurisdiction-specific defaults, but model settings remain explicit and editable.

## Time horizon and termination

Two termination modes are available:

### Fixed horizon

The model runs the configured number of cycles exactly.

### Cohort depletion

The model stops early when the total proportion remaining in all non-absorbing states falls below a specified threshold, subject to the configured maximum number of cycles. This can approximate a lifetime horizon when an absorbing death state exists while retaining a bounded safety limit.

The maximum horizon and depletion threshold should be reported because they can affect results when material cohort mass remains at truncation.

## Uncertainty

Markov parameters use the same split uncertainty schema as the decision-tree modeller:

- base value;
- DSA low/high range;
- PSA distribution and parameterisation.

The same parameter can participate in DSA and PSA simultaneously.

PSA reruns the complete cohort model for every parameter draw. Grouped Dirichlet sampling is supported for mutually exclusive probability components that genuinely form one probability simplex. Other declared correlation groups still generate a warning when no covariance/joint-sampling structure is supplied.

## Validation checks

The engine validates at least the following before running:

- unique state and strategy ids;
- initial cohort distributions sum to 1;
- transition state references exist;
- no duplicate origin/destination pair within a strategy;
- direct/complement probability parameters lie in [0,1];
- no more than one residual transition per origin state;
- each transition row sums to 1;
- absorbing states cannot be exited;
- state and transition rewards reference parameters of the correct cost/outcome type;
- transition rewards reference structurally defined transitions;
- cohort mass is conserved during simulation;
- discount rates are finite proportions in [0,1).

## Current limitations and next extension

Version 0.5 does **not** yet implement:

- tunnel states generated automatically from state duration;
- time-varying transition matrices;
- semi-Markov / state-time-dependent transitions;
- patient-history-dependent transitions;
- transition-intensity matrices or competing-risk-aware rate conversion;
- calibration;
- individual-level microsimulation.

The next methodological extension should be semi-Markov/tunnel-state support plus explicit rate-to-transition handling, because those address the main limitation of a time-homogeneous cohort Markov model: the Markov property that future movement depends only on the current state rather than prior history or duration in state.
