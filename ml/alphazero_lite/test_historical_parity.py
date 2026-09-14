import numpy as np

from ml.alphazero_lite.historical_parity import (
    MATERIAL_PARAMETER_DIFFERENCE,
    first_differing_named_tensor,
    tensor_difference,
)


def test_tensor_difference_reports_byte_and_numerical_parity() -> None:
    left = np.array([1.0, 2.0], dtype=np.float32)
    assert tensor_difference(left, left.copy())["byte_identical"]
    right = left.copy()
    right[1] += MATERIAL_PARAMETER_DIFFERENCE * 2
    result = tensor_difference(left, right)
    assert not result["byte_identical"]
    assert result["material"]
    assert result["max_abs"] > MATERIAL_PARAMETER_DIFFERENCE


def test_first_differing_named_tensor_names_first_difference() -> None:
    left = {
        "a": np.zeros((2,), dtype=np.float32),
        "b": np.zeros((2,), dtype=np.float32),
    }
    right = {name: value.copy() for name, value in left.items()}
    right["b"][1] = 0.5
    result = first_differing_named_tensor(left, right)
    assert result is not None
    assert result["name"] == "b"
    assert result["first_max_difference_index"] == [1]
    assert result["absolute_difference"] == 0.5
