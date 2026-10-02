"""Advisory distribution suggestions for probabilistic sensitivity analysis.

These are deliberately suggestions, not automatic assignments. NICE guidance
requires PSA distributions to represent the available evidence and to be
justified rather than selected mechanically. The functions below therefore
recommend a commonly used family and explain why, while leaving the modeller
responsible for confirming the family and entering evidence-based parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from model.schema import Parameter


@dataclass(frozen=True)
class DistributionSuggestion:
    family: str
    rationale: str
    parameterisation: str
    caution: str = ""


RATIO_TERMS = (
    "hazard ratio",
    "odds ratio",
    "risk ratio",
    "relative risk",
    "rate ratio",
    "incidence rate ratio",
)


def suggest_distribution_from_fields(
    *,
    label: str,
    category: str,
    unit: str,
    value: float,
) -> DistributionSuggestion:
    """Return a non-binding distribution-family suggestion.

    The suggestion reflects common health-economic practice and parameter
    support, but does not infer an uncertainty magnitude. Distribution
    parameters must still come from the source evidence or justified elicitation.
    """

    label_lower = label.lower()
    unit_lower = unit.lower()

    if any(term in label_lower or term in unit_lower for term in RATIO_TERMS):
        return DistributionSuggestion(
            family="lognormal",
            rationale="Relative treatment-effect measures are positive and are commonly modelled on the log scale.",
            parameterisation="meanlog and sdlog, preferably derived from the log estimate and its standard error or confidence interval.",
        )

    probability_like = (
        "probab" in label_lower
        or "proportion" in label_lower
        or unit_lower in {"probability", "proportion", "%"}
    )
    if probability_like and 0 <= value <= 1:
        return DistributionSuggestion(
            family="beta",
            rationale="Probabilities and proportions are bounded between 0 and 1; beta distributions are commonly used for their parameter uncertainty.",
            parameterisation="alpha and beta from event counts, or derived from an evidence-based mean and standard error.",
        )

    if category == "cost" and value >= 0:
        return DistributionSuggestion(
            family="gamma",
            rationale="Non-negative costs are often right-skewed; gamma distributions preserve non-negative support.",
            parameterisation="shape and scale (or equivalent rate parameterisation) derived from the evidence-based mean and standard error.",
        )

    if category == "resource_use" and value >= 0:
        return DistributionSuggestion(
            family="gamma",
            rationale="Positive resource-use quantities are commonly right-skewed and require non-negative support.",
            parameterisation="shape and scale derived from the evidence; use a count-specific model where the source data require it.",
            caution="For discrete counts, Poisson/negative-binomial or empirical sampling may be more faithful than a gamma distribution.",
        )

    if category == "utility":
        if 0 <= value <= 1:
            return DistributionSuggestion(
                family="beta",
                rationale="A beta distribution can be suitable when the utility quantity is genuinely bounded between 0 and 1.",
                parameterisation="alpha and beta derived from the evidence-based mean and uncertainty.",
                caution="Do not use beta if the relevant utility scale permits values below 0 or if the uncertainty support extends outside 0–1.",
            )
        return DistributionSuggestion(
            family="normal",
            rationale="A normal distribution can be a pragmatic starting point for an approximately symmetric utility parameter that may include negative values.",
            parameterisation="mean and standard deviation from the source evidence.",
            caution="Check the implied support; transformed, bounded, or empirical distributions may be preferable for the actual utility evidence.",
        )

    return DistributionSuggestion(
        family="normal",
        rationale="For a continuous parameter with approximately symmetric uncertainty, a normal distribution is a common starting point.",
        parameterisation="mean and standard deviation from the source evidence.",
        caution="Confirm that unbounded symmetric support is appropriate. Choose a different family when the evidence or natural parameter bounds require it.",
    )


def suggest_distribution(parameter: Parameter) -> DistributionSuggestion:
    return suggest_distribution_from_fields(
        label=parameter.label,
        category=parameter.category,
        unit=parameter.unit,
        value=parameter.value,
    )
