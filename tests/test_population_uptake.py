import pytest

from model.population_uptake import (
    PopulationOption,
    PopulationUptakeDefinition,
    PopulationUptakeValidationError,
    PopulationYear,
    TreatmentMixShare,
    run_population_uptake,
)


def _definition(basis="annual_eligible_population"):
    options = (PopulationOption("A", "Current"), PopulationOption("B", "New"))
    population = (PopulationYear(1, 1000), PopulationYear(2, 1200))
    mix = (
        TreatmentMixShare("current", 1, "A", 1.0),
        TreatmentMixShare("current", 1, "B", 0.0),
        TreatmentMixShare("future", 1, "A", 0.7),
        TreatmentMixShare("future", 1, "B", 0.3),
        TreatmentMixShare("current", 2, "A", 1.0),
        TreatmentMixShare("current", 2, "B", 0.0),
        TreatmentMixShare("future", 2, "A", 0.5),
        TreatmentMixShare("future", 2, "B", 0.5),
    )
    return PopulationUptakeDefinition(options, population, mix, population_basis=basis)


def test_shared_population_returns_treated_counts():
    result = run_population_uptake(_definition())
    assert result.treated_people("future", 1, "B") == pytest.approx(300)
    assert result.treated_people("future", 2, "B") == pytest.approx(600)
    assert result.population_basis == "annual_eligible_population"


def test_population_basis_is_explicit():
    result = run_population_uptake(_definition("new_treatment_starts"))
    assert result.population_basis == "new_treatment_starts"


def test_mix_is_not_silently_normalised():
    definition = _definition()
    broken = PopulationUptakeDefinition(
        definition.options,
        definition.population,
        tuple(
            TreatmentMixShare(row.scenario, row.year, row.intervention_id, 0.4)
            if row.scenario == "future" and row.year == 1
            else row
            for row in definition.treatment_mix
        ),
    )
    with pytest.raises(PopulationUptakeValidationError, match="sum"):
        run_population_uptake(broken)
