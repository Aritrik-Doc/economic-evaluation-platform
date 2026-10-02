"""Recognised health-economic reference-case profiles.

The registry is deliberately data-driven so additional HTA systems can be added
without changing the model engine. The values here describe methods guidance;
actual model settings are stored separately in ``model.schema``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PerspectiveSpec:
    code: str
    label: str
    included_cost_bearers: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class ThresholdRange:
    lower: float
    upper: float
    currency: str
    outcome_code: str
    notes: str

    def __post_init__(self) -> None:
        if not (isfinite(self.lower) and isfinite(self.upper)):
            raise ValueError("Threshold bounds must be finite.")
        if self.lower < 0 or self.upper < self.lower:
            raise ValueError("Threshold range is invalid.")


@dataclass(frozen=True)
class ReferenceCase:
    code: str
    name: str
    jurisdiction: str
    source_title: str
    source_url: str
    source_year: int
    guideline_version: str
    preferred_outcome_code: str
    reference_outcome_codes: tuple[str, ...]
    conditional_outcome_codes: tuple[str, ...]
    supplementary_outcome_codes: tuple[str, ...]
    perspective: PerspectiveSpec
    cost_discount_rate: float
    outcome_discount_rate: float
    time_horizon_rule: str
    comparator_rule: str
    analysis_currency: str
    threshold_range: ThresholdRange | None
    sensitivity_expectations: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        for value in (self.cost_discount_rate, self.outcome_discount_rate):
            if not isfinite(value) or value < 0 or value >= 1:
                raise ValueError("Discount rates must be finite proportions in [0, 1).")

    def outcome_status(self, outcome_code: str) -> str:
        if outcome_code in self.reference_outcome_codes:
            return "reference"
        if outcome_code in self.conditional_outcome_codes:
            return "conditional"
        if outcome_code in self.supplementary_outcome_codes:
            return "supplementary"
        return "not_specified"


NICE_TA = ReferenceCase(
    code="NICE_TA",
    name="NICE technology appraisal",
    jurisdiction="England",
    source_title="NICE technology appraisal and highly specialised technologies guidance: the manual (PMG36)",
    source_url="https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/",
    source_year=2026,
    guideline_version="current PMG36 methods (thresholds updated in 2026)",
    preferred_outcome_code="QALY",
    reference_outcome_codes=("QALY",),
    conditional_outcome_codes=(),
    supplementary_outcome_codes=("LYG",),
    perspective=PerspectiveSpec(
        code="NHS_PSS",
        label="NHS and Personal Social Services",
        included_cost_bearers=("health_system", "personal_social_services"),
        notes=(
            "Reference-case costs relate to NHS and PSS resources; productivity "
            "costs are outside the reference case."
        ),
    ),
    cost_discount_rate=0.035,
    outcome_discount_rate=0.035,
    time_horizon_rule="Long enough to reflect all important differences in costs and health outcomes.",
    comparator_rule="Use the comparator(s) specified in the NICE appraisal scope.",
    analysis_currency="GBP",
    threshold_range=ThresholdRange(
        lower=25_000,
        upper=35_000,
        currency="GBP",
        outcome_code="QALY",
        notes=(
            "Current standard technology-appraisal cost-effectiveness range. "
            "Committee decisions also consider uncertainty, uncaptured benefits, "
            "non-health factors and health inequalities."
        ),
    ),
    sensitivity_expectations=(
        "deterministic",
        "two_way_where_informative",
        "threshold_where_informative",
        "scenario",
        "probabilistic",
    ),
)


HTAIN_2023 = ReferenceCase(
    code="HTAIN_2023",
    name="HTAIn / Indian Reference Case (2023)",
    jurisdiction="India",
    source_title="Development of the Indian Reference Case for undertaking economic evaluation for health technology assessment",
    source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10485782/",
    source_year=2023,
    guideline_version="Indian Reference Case 2023",
    preferred_outcome_code="QALY",
    reference_outcome_codes=("QALY",),
    conditional_outcome_codes=("DALY_AVERTED",),
    supplementary_outcome_codes=("LYG",),
    perspective=PerspectiveSpec(
        code="ABRIDGED_SOCIETAL",
        label="Abridged societal",
        included_cost_bearers=("health_system", "patient_direct_medical", "patient_direct_non_medical"),
        notes=(
            "Base case includes direct medical and direct non-medical costs borne by "
            "the health system and patients. Results should also be reported separately "
            "from the healthcare-payer perspective. Indirect/productivity costs are "
            "addressed in sensitivity analysis rather than the base case."
        ),
    ),
    cost_discount_rate=0.03,
    outcome_discount_rate=0.03,
    time_horizon_rule="Long enough to capture all significant costs and consequences.",
    comparator_rule="Current practice in use; multiple comparators may be included.",
    analysis_currency="INR",
    threshold_range=None,
    sensitivity_expectations=(
        "deterministic",
        "two_way_where_informative",
        "threshold_where_informative",
        "scenario",
        "probabilistic",
        "discount_rate_0_to_5_percent",
    ),
    notes=(
        "QALYs are preferred. DALYs may be used in special scenarios; the platform "
        "represents benefit as DALYs averted so that higher values consistently mean "
        "more health benefit. The 2023 reference case does not prescribe a single "
        "monetary cost-effectiveness threshold."
    ),
)


REFERENCE_CASES: dict[str, ReferenceCase] = {
    NICE_TA.code: NICE_TA,
    HTAIN_2023.code: HTAIN_2023,
}


def custom_reference_case(
    *,
    name: str,
    perspective_label: str,
    preferred_outcome_code: str,
    cost_discount_rate: float,
    outcome_discount_rate: float,
    time_horizon_rule: str,
    analysis_currency: str,
    threshold: float | None,
) -> ReferenceCase:
    """Create a user-defined methods profile.

    Custom profiles intentionally do not imply compliance with a recognised HTA
    reference case. The user must explicitly supply the key methodological choices.
    """
    if not name.strip() or not perspective_label.strip() or not time_horizon_rule.strip():
        raise ValueError("Custom reference-case text fields cannot be empty.")
    threshold_range = None
    if threshold is not None:
        if not isfinite(threshold) or threshold < 0:
            raise ValueError("Custom threshold must be a finite non-negative number.")
        threshold_range = ThresholdRange(
            lower=threshold,
            upper=threshold,
            currency=analysis_currency,
            outcome_code=preferred_outcome_code,
            notes="User-specified custom decision threshold.",
        )

    return ReferenceCase(
        code="CUSTOM",
        name=name,
        jurisdiction="Custom",
        source_title="User-defined methods profile",
        source_url="",
        source_year=0,
        guideline_version="custom",
        preferred_outcome_code=preferred_outcome_code,
        reference_outcome_codes=(preferred_outcome_code,),
        conditional_outcome_codes=(),
        supplementary_outcome_codes=(),
        perspective=PerspectiveSpec(
            code="CUSTOM",
            label=perspective_label,
            included_cost_bearers=("custom",),
            notes="User-defined perspective; cost inclusion rules must be documented explicitly.",
        ),
        cost_discount_rate=cost_discount_rate,
        outcome_discount_rate=outcome_discount_rate,
        time_horizon_rule=time_horizon_rule,
        comparator_rule="User-defined comparator specification.",
        analysis_currency=analysis_currency,
        threshold_range=threshold_range,
        sensitivity_expectations=(
            "deterministic",
            "two_way_where_informative",
            "threshold_where_informative",
            "scenario",
            "probabilistic",
        ),
        notes="Custom methods profile; recognised-reference-case compliance is not implied.",
    )
