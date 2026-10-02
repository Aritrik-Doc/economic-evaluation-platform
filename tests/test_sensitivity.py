import pytest

from model.sensitivity import (
    ThresholdAnalysisSpec,
    TwoWaySensitivitySpec,
    threshold_analysis,
    two_way_sensitivity,
)


def test_two_way_sensitivity_full_grid():
    spec = TwoWaySensitivitySpec(
        parameter_x="cost",
        x_values=(100.0, 200.0),
        parameter_y="effect",
        y_values=(1.0, 2.0, 3.0),
    )

    result = two_way_sensitivity(
        spec,
        lambda values: values["effect"] * 1000 - values["cost"],
    )

    assert len(result) == 6
    assert result[0] == pytest.approx((100.0, 1.0, 900.0))
    assert result[-1] == pytest.approx((200.0, 3.0, 2800.0))


def test_threshold_analysis_finds_switching_value():
    spec = ThresholdAnalysisSpec(
        parameter_id="price",
        lower=0,
        upper=100,
        tolerance=1e-8,
    )

    switching_value = threshold_analysis(
        spec,
        lambda values: 50 - values["price"],
    )

    assert switching_value == pytest.approx(50.0)


def test_threshold_analysis_requires_bracket():
    spec = ThresholdAnalysisSpec(parameter_id="p", lower=0, upper=10)
    with pytest.raises(ValueError, match="bracket"):
        threshold_analysis(spec, lambda values: values["p"] + 1)
