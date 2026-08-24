"""Unit contracts for the fixed-scale shadow distillation calibration."""

from ml.alphazero_lite.run_fresh_p1_shadow_target_distillation import calibrated_weight


def test_calibrated_weight_caps_the_largest_weighted_ratio() -> None:
    weight = calibrated_weight(18.990581)

    assert weight == 0.50 / 18.990581
    assert weight * 18.990581 <= 0.500001


def test_calibrated_weight_respects_the_configured_cap() -> None:
    assert calibrated_weight(1.0) == 0.25
