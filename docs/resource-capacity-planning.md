# Resource and capacity planning

Version 0.11 adds natural-unit resource and service-capacity planning alongside Budget Impact Analysis (BIA).

## Purpose

Budget impact answers a financial affordability question. Capacity planning asks a different implementation question: **can the health system deliver the projected activity with the workforce, facilities, equipment and other physical resources available?**

The module therefore keeps natural resource quantities separate from monetary costs. Examples include:

- clinician, nurse, pharmacist or technician hours;
- infusion-chair or treatment-room hours;
- outpatient visits;
- laboratory tests and imaging investigations;
- procedures;
- bed-days;
- machine time;
- pharmacy preparation time;
- devices or consumable units.

The active BIA provides the annual eligible population and current/future treatment mix. The resource planner adds per-person natural-unit requirements and annual capacity.

## Methodological context

NICE describes resource impact as including not only financial costs/savings but also capacity and demand, patient flows, workforce, training and facilities. Current NICE resource-impact templates can use natural resource inputs such as appointment counts, appointment length and staff type when estimating capacity consequences.

ISPOR BIA good practice recommends decision-maker-specific resource use and locally relevant data, using the simplest transparent model that can generate credible estimates.

The platform uses those principles as design guidance rather than presenting a single universal capacity-planning reference case.

## Two population/resource-demand bases

A core modelling issue is whether annual population values are a cross-sectional stock or annual new treatment starts.

### Annual treated population

`annual_treated_population` interprets each BIA year's eligible population and treatment mix as the patients relevant to that budget year. Resource requirement Period 1, 2, 3… therefore means Budget Year 1, 2, 3….

For intervention `i`, resource `r`, year `t` and scenario `s`:

`required units = eligible population(t) × treatment share(s,t,i) × units per treated person(i,r,t)`

This is appropriate when annual BIA populations represent the treated population in each year and resource-use inputs are annual cross-sectional requirements.

### New treatment starts / longitudinal resource profile

`new_treatment_starts` interprets the annual BIA population as new initiations. Resource requirement Period 1, 2, 3… means Year 1, Year 2, Year 3 since treatment initiation.

Each initiation cohort retains its intervention and contributes later-year resource requirements in subsequent budget periods. The cohorts are stacked across the planning horizon. This supports patterns such as a resource-intensive initiation year followed by lighter maintenance/follow-up requirements.

The software does not infer which population basis is correct. The modeller must select the interpretation that matches the epidemiology and service pathway.

## Capacity available to the modelled population

For each resource and year the modeller enters:

- **total capacity**; and
- **capacity already committed to other services/populations outside the BIA**.

The capacity available to the modelled population is:

`available capacity = total capacity - committed other demand`

This distinction avoids assuming that nominal organisational capacity is fully available to the intervention under evaluation.

Committed other demand must not exceed total capacity.

## Outputs

For each resource, year and scenario the engine reports:

- required natural units;
- total capacity;
- committed other demand;
- available capacity;
- utilisation ratio;
- headroom; and
- shortfall.

The current and future treatment-mix scenarios are then compared to show:

- current demand;
- future demand;
- change in demand;
- current and future utilisation;
- future headroom; and
- future shortfall.

When available capacity is zero and demand is positive, utilisation is deliberately reported as undefined rather than dividing by zero; the entire requirement is reported as a shortfall.

## Capacity expansion planner

The first capacity scenario tool allows the modeller to select a resource and, from a stated year onward, apply:

- a multiplier to total capacity; and/or
- an additional fixed number of resource units per year.

Demand already committed to other services remains unchanged. This lets users test concrete service-expansion plans without treating them as uncertainty distributions.

Richer BIA/resource scenarios and uncertainty management are intentionally deferred to a later milestone.

## Transparency

The workspace includes documentation checks for:

- resource/capacity evidence source;
- resource definition rationale;
- resource-use evidence and rationale for each non-zero intervention-resource profile; and
- capacity assumption rationale.

The Transparency check is a documentation-completeness safeguard only. It does not establish that capacity estimates are accurate, locally transferable, operationally achievable or free of bias.

## Current boundaries

Version 0.11 does not yet:

- optimise schedules or queueing/waiting times;
- allocate scarce capacity between competing indications;
- automatically turn a capacity shortfall into reduced treatment uptake;
- monetise capacity expansion automatically;
- infer resource use from clinical states/transitions;
- probabilistically sample capacity parameters; or
- generate policy recommendations from the resource results.

Those boundaries are deliberate. The next planned layer is a deterministic, rules-based policy interpretation framework that can describe cost-effectiveness, budget-impact and capacity findings together without making the policy decision for the user.
