"""Auditable health-economic model schema.

Version 0.4.1 separates deterministic and probabilistic uncertainty so the same
parameter can participate in DSA and PSA without being reconfigured. The legacy
``UncertaintySpec`` is retained as a compatibility layer for older saved models
and tests; new code should use ``dsa`` and ``psa`` on ``Parameter``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import isfinite
from typing import Literal


ParameterCategory = Literal[
    "clinical",
    "cost",
    "utility",
    "resource_use",
    "survival",
    "epidemiology",
    "other",
]
SourceType = Literal[
    "systematic_review",
    "meta_analysis",
    "randomised_trial",
    "observational_study",
    "registry",
    "database",
    "tariff",
    "cost_study",
    "expert_elicitation",
    "guideline",
    "user_assumption",
    "other",
]
ModelType = Literal[
    "trial_based",
    "decision_tree",
    "cohort_state_transition",
    "partitioned_survival",
    "microsimulation",
    "other",
]
StrategyRole = Literal["intervention", "comparator", "other"]


@dataclass(frozen=True)
class EvidenceSource:
    citation: str
    source_type: SourceType
    url: str | None = None
    publication_year: int | None = None
    details: str = ""

    def __post_init__(self) -> None:
        if not self.citation.strip():
            raise ValueError("Evidence source citation is mandatory.")
        if self.publication_year is not None and self.publication_year < 1800:
            raise ValueError("Publication year is implausible.")


@dataclass(frozen=True)
class AssumptionSpec:
    statement: str
    rationale: str
    material: bool = True

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("Assumption statement is mandatory; use an explicit 'none' statement if applicable.")
        if not self.rationale.strip():
            raise ValueError("Assumption rationale is mandatory.")


@dataclass(frozen=True)
class DistributionSpec:
    family: str
    parameters: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.family.strip():
            raise ValueError("Distribution family cannot be empty.")
        if not self.parameters:
            raise ValueError("Distribution parameters are required.")
        if any(not isfinite(value) for _, value in self.parameters):
            raise ValueError("Distribution parameters must be finite.")


@dataclass(frozen=True)
class DeterministicUncertaintySpec:
    enabled: bool
    rationale: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("DSA uncertainty rationale is mandatory.")
        if self.enabled:
            if self.lower is None or self.upper is None:
                raise ValueError("Enabled DSA requires lower and upper values.")
            if not (isfinite(self.lower) and isfinite(self.upper)) or self.lower > self.upper:
                raise ValueError("Invalid DSA uncertainty range.")
        elif self.lower is not None or self.upper is not None:
            raise ValueError("Disabled DSA should not carry low/high values.")


@dataclass(frozen=True)
class ProbabilisticUncertaintySpec:
    enabled: bool
    rationale: str
    distribution: DistributionSpec | None = None
    correlation_group: str | None = None

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("PSA uncertainty rationale is mandatory.")
        if self.enabled and self.distribution is None:
            raise ValueError("Enabled PSA requires a probability distribution.")
        if not self.enabled and self.distribution is not None:
            raise ValueError("Disabled PSA should not carry a distribution.")


@dataclass(frozen=True)
class UncertaintySpec:
    """Legacy combined uncertainty specification.

    This remains supported so models created before v0.4.1 continue to compile.
    New model-builder code stores separate deterministic and probabilistic specs.
    """

    kind: Literal["none", "range", "distribution", "range_and_distribution", "scenario"]
    rationale: str
    lower: float | None = None
    upper: float | None = None
    distribution: DistributionSpec | None = None
    correlation_group: str | None = None

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("Uncertainty rationale is mandatory.")
        if self.kind in {"range", "range_and_distribution"}:
            if self.lower is None or self.upper is None:
                raise ValueError("Range uncertainty requires lower and upper values.")
            if not (isfinite(self.lower) and isfinite(self.upper)) or self.lower > self.upper:
                raise ValueError("Invalid uncertainty range.")
        if self.kind in {"distribution", "range_and_distribution"} and self.distribution is None:
            raise ValueError("Distribution uncertainty requires a distribution specification.")


@dataclass(frozen=True)
class FXConversionSpec:
    from_currency: str
    to_currency: str
    rate: float
    rate_date: date
    source: EvidenceSource

    def __post_init__(self) -> None:
        if len(self.from_currency) != 3 or len(self.to_currency) != 3:
            raise ValueError("Currencies must use three-letter ISO-style codes.")
        if not isfinite(self.rate) or self.rate <= 0:
            raise ValueError("FX rate must be positive and finite.")


@dataclass(frozen=True)
class Parameter:
    id: str
    label: str
    value: float
    unit: str
    category: ParameterCategory
    source: EvidenceSource
    assumption: AssumptionSpec
    uncertainty: UncertaintySpec | None = None
    dsa: DeterministicUncertaintySpec | None = None
    psa: ProbabilisticUncertaintySpec | None = None
    formula: str | None = None
    currency: str | None = None
    price_year: int | None = None
    cost_bearers: tuple[str, ...] = ()
    fx_conversion: FXConversionSpec | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.label.strip() or not self.unit.strip():
            raise ValueError("Parameter id, label and unit are mandatory.")
        if not isfinite(self.value):
            raise ValueError("Parameter value must be finite.")

        # Convert legacy uncertainty to the v0.4.1 split representation.
        if self.dsa is None or self.psa is None:
            legacy = self.uncertainty
            if legacy is None:
                raise ValueError("Parameter uncertainty must explicitly specify both DSA and PSA handling.")
            if self.dsa is None:
                dsa_enabled = legacy.kind in {"range", "range_and_distribution"}
                object.__setattr__(
                    self,
                    "dsa",
                    DeterministicUncertaintySpec(
                        enabled=dsa_enabled,
                        rationale=legacy.rationale,
                        lower=legacy.lower if dsa_enabled else None,
                        upper=legacy.upper if dsa_enabled else None,
                    ),
                )
            if self.psa is None:
                psa_enabled = legacy.kind in {"distribution", "range_and_distribution"}
                object.__setattr__(
                    self,
                    "psa",
                    ProbabilisticUncertaintySpec(
                        enabled=psa_enabled,
                        rationale=legacy.rationale,
                        distribution=legacy.distribution if psa_enabled else None,
                        correlation_group=legacy.correlation_group if psa_enabled else None,
                    ),
                )

        # Build a legacy view when a new-style parameter did not provide one.
        if self.uncertainty is None:
            assert self.dsa is not None and self.psa is not None
            if self.dsa.enabled and self.psa.enabled:
                kind = "range_and_distribution"
            elif self.dsa.enabled:
                kind = "range"
            elif self.psa.enabled:
                kind = "distribution"
            else:
                kind = "none"
            object.__setattr__(
                self,
                "uncertainty",
                UncertaintySpec(
                    kind=kind,
                    rationale=f"DSA: {self.dsa.rationale} PSA: {self.psa.rationale}",
                    lower=self.dsa.lower if self.dsa.enabled else None,
                    upper=self.dsa.upper if self.dsa.enabled else None,
                    distribution=self.psa.distribution if self.psa.enabled else None,
                    correlation_group=self.psa.correlation_group if self.psa.enabled else None,
                ),
            )

        if self.category == "cost":
            if self.currency is None or len(self.currency) != 3:
                raise ValueError("Cost parameters require a three-letter currency code.")
            if self.price_year is None or self.price_year < 1900:
                raise ValueError("Cost parameters require an explicit price year.")
            if not self.cost_bearers:
                raise ValueError("Cost parameters require at least one explicit cost bearer.")
        elif any(value is not None for value in (self.currency, self.price_year, self.fx_conversion)):
            raise ValueError("Currency, price year and FX conversion are only valid for cost parameters.")


@dataclass(frozen=True)
class StrategyDefinition:
    id: str
    name: str
    role: StrategyRole
    description: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip() or not self.description.strip():
            raise ValueError("Strategy id, name and description are mandatory.")


@dataclass(frozen=True)
class DecisionProblem:
    policy_question: str
    condition: str
    population: str
    setting: str
    strategies: tuple[StrategyDefinition, ...]

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.policy_question, self.condition, self.population, self.setting)):
            raise ValueError("Decision problem fields cannot be empty.")
        if len(self.strategies) < 2:
            raise ValueError("A decision problem requires at least two strategies.")


@dataclass(frozen=True)
class TimeHorizon:
    kind: Literal["years", "lifetime"]
    years: float | None = None
    justification: str = ""

    def __post_init__(self) -> None:
        if self.kind == "years":
            if self.years is None or not isfinite(self.years) or self.years <= 0:
                raise ValueError("A finite time horizon requires a positive number of years.")
        elif self.years is not None:
            raise ValueError("Lifetime horizons should not also specify years.")
        if not self.justification.strip():
            raise ValueError("Time-horizon justification is mandatory.")


@dataclass(frozen=True)
class Discounting:
    cost_rate: float
    outcome_rate: float
    source: EvidenceSource

    def __post_init__(self) -> None:
        for value in (self.cost_rate, self.outcome_rate):
            if not isfinite(value) or value < 0 or value >= 1:
                raise ValueError("Discount rates must be finite proportions in [0, 1).")


@dataclass(frozen=True)
class HealthState:
    id: str
    name: str
    description: str
    absorbing: bool = False

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip() or not self.description.strip():
            raise ValueError("Health-state id, name and description are mandatory.")


@dataclass(frozen=True)
class Transition:
    origin_state: str
    destination_state: str
    parameter_id: str | None = None
    formula: str | None = None

    def __post_init__(self) -> None:
        if not self.origin_state.strip() or not self.destination_state.strip():
            raise ValueError("Transition origin and destination are mandatory.")
        if bool(self.parameter_id) == bool(self.formula):
            raise ValueError("A transition must specify exactly one of parameter_id or formula.")


@dataclass(frozen=True)
class ClinicalEndpoint:
    code: str
    label: str
    definition: str
    source: EvidenceSource

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.label.strip() or not self.definition.strip():
            raise ValueError("Clinical endpoint code, label and definition are mandatory.")


@dataclass(frozen=True)
class ModelSpecification:
    id: str
    name: str
    model_type: ModelType
    reference_case_code: str
    decision_problem: DecisionProblem
    primary_outcome_code: Literal["QALY", "LYG", "DALY_AVERTED"]
    perspective_code: str
    time_horizon: TimeHorizon
    discounting: Discounting
    parameters: tuple[Parameter, ...]
    health_states: tuple[HealthState, ...] = ()
    transitions: tuple[Transition, ...] = ()
    clinical_endpoints: tuple[ClinicalEndpoint, ...] = ()
    structural_assumptions: tuple[AssumptionSpec, ...] = ()
    non_reference_case_reasons: tuple[str, ...] = ()
    created_by: str = ""
    version: str = "0.1"
    created_on: date = field(default_factory=date.today)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip() or not self.reference_case_code.strip():
            raise ValueError("Model id, name and reference-case code are mandatory.")
        if not self.perspective_code.strip():
            raise ValueError("Perspective code is mandatory.")
        parameter_ids = [p.id for p in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("Parameter ids must be unique within a model.")
        state_ids = {state.id for state in self.health_states}
        for transition in self.transitions:
            if transition.origin_state not in state_ids or transition.destination_state not in state_ids:
                raise ValueError("Transitions must reference health states defined in the model.")
            if transition.parameter_id and transition.parameter_id not in set(parameter_ids):
                raise ValueError("Transition parameter_id must reference a defined parameter.")
