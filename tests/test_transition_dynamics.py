import numpy as np
import pytest

from model.transition_dynamics import (
    AgeSpecificMortalityTable,
    ParameterBand,
    PiecewiseParameterSchedule,
    TransitionConversionError,
    competing_rates_to_probabilities,
    generator_to_transition_matrix,
    probability_to_rate,
    rate_to_probability,
    rescale_probability,
)


def test_rate_probability_round_trip():
    rate = 0.2
    probability = rate_to_probability(rate, 0.5)
    assert probability_to_rate(probability, 0.5) == pytest.approx(rate)


def test_probability_rescaling_uses_constant_hazard_assumption():
    annual = 0.36
    half_year = rescale_probability(annual, from_duration=1.0, to_duration=0.5)
    assert half_year == pytest.approx(0.2)


def test_competing_rates_are_converted_jointly_and_sum_to_one():
    result = competing_rates_to_probabilities({"progressed": 0.10, "dead": 0.05}, 1.0)
    assert result.total_probability == pytest.approx(1.0)
    assert result.destination_probabilities["progressed"] / result.destination_probabilities["dead"] == pytest.approx(2.0)
    assert result.stay_probability == pytest.approx(np.exp(-0.15))


def test_zero_competing_rates_leave_everyone_in_origin_state():
    result = competing_rates_to_probabilities({"event_a": 0.0, "event_b": 0.0})
    assert result.stay_probability == 1.0
    assert result.destination_probabilities == {"event_a": 0.0, "event_b": 0.0}


def test_generator_conversion_matches_two_state_closed_form():
    # Alive -> dead at rate 0.2; dead is absorbing.
    q = np.array([[-0.2, 0.2], [0.0, 0.0]])
    p = generator_to_transition_matrix(q, duration=1.0)
    assert p[0, 0] == pytest.approx(np.exp(-0.2), abs=1e-10)
    assert p[0, 1] == pytest.approx(1 - np.exp(-0.2), abs=1e-10)
    assert p[1, 1] == pytest.approx(1.0)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_generator_conversion_allows_jump_over_paths_within_cycle():
    # A -> B -> C. Matrix exponential permits some A -> C occupancy after one
    # interval even though no direct A -> C intensity exists.
    q = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    p = generator_to_transition_matrix(q, duration=1.0)
    assert p[0, 2] > 0
    assert np.allclose(p.sum(axis=1), 1.0)


def test_invalid_generator_is_rejected():
    q = np.array([[-0.1, 0.2], [0.0, 0.0]])
    with pytest.raises(TransitionConversionError, match="row must sum to zero|row must sum"):
        generator_to_transition_matrix(q)


def test_piecewise_schedule_distinguishes_model_and_state_time():
    schedule = PiecewiseParameterSchedule(
        basis="state_time",
        bands=(
            ParameterBand(0.0, "p_early", 1.0),
            ParameterBand(1.0, "p_late", None),
        ),
    )
    assert schedule.parameter_id(model_time=10.0, state_time=0.5) == "p_early"
    assert schedule.parameter_id(model_time=10.0, state_time=2.0) == "p_late"


def test_schedule_gap_is_explicit_error_not_silent_carry_forward():
    schedule = PiecewiseParameterSchedule(
        basis="model_time",
        bands=(ParameterBand(1.0, "p_after_year_one", None),),
    )
    with pytest.raises(TransitionConversionError, match="No schedule band"):
        schedule.parameter_id(model_time=0.5, state_time=0.0)


def test_age_specific_mortality_integrates_across_birthday():
    table = AgeSpecificMortalityTable(((50, 0.10), (51, 0.20), (52, 0.30)))
    # Six months at age 50 then six months at age 51.
    probability = table.probability(age_start=50.5, duration_years=1.0)
    expected_hazard = 0.5 * (-np.log(0.9)) + 0.5 * (-np.log(0.8))
    assert probability == pytest.approx(1 - np.exp(-expected_hazard))


def test_smr_multiplies_mortality_rate_not_probability():
    table = AgeSpecificMortalityTable(((60, 0.10), (61, 0.10)))
    probability = table.probability(
        age_start=60.0,
        duration_years=1.0,
        standardized_mortality_ratio=2.0,
    )
    assert probability == pytest.approx(1 - (1 - 0.10) ** 2)


def test_mortality_table_out_of_range_is_rejected():
    table = AgeSpecificMortalityTable(((70, 0.1), (71, 0.2)))
    with pytest.raises(TransitionConversionError, match="does not cover"):
        table.probability(age_start=69.5, duration_years=1.0)
