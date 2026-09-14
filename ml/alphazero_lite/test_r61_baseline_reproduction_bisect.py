import random

import numpy as np
import torch

from ml.alphazero_lite.run_r61_baseline_reproduction_bisect import (
    SEEDS,
    VARIANTS,
    array_fingerprint,
    first_divergence,
    index_hash,
    rng_hashes,
    state_hash,
    variant_configuration,
)
from ml.alphazero_lite.train import PolicyValueNet, set_seed


def test_matrix_has_only_frozen_r61_t61_t63_stages() -> None:
    assert VARIANTS == ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7")
    assert SEEDS == {"T61": 61, "T63": 63}
    assert not variant_configuration("V2")["anchor_observation"]
    assert variant_configuration("V3")["anchor_observation"]
    assert not variant_configuration("V3")["raw_gradient_telemetry"]
    assert variant_configuration("V4")["raw_gradient_telemetry"]
    assert variant_configuration("V5")["parameter_snapshots"]
    assert variant_configuration("V6")["adam_subspace_telemetry"]
    assert variant_configuration("V7")["full_pr312_live_instrumentation"]


def test_array_and_index_fingerprints_are_content_sensitive() -> None:
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    same = value.copy()
    changed = value.copy()
    changed[0, 0] = -1
    assert array_fingerprint(value) == array_fingerprint(same)
    assert array_fingerprint(value)["sha256"] != array_fingerprint(changed)["sha256"]
    assert index_hash([1, 2, 3]) != index_hash([3, 2, 1])


def test_initial_model_and_rng_hashes_are_reproducible() -> None:
    set_seed(61)
    first = PolicyValueNet((8, 1), "residual_v3", 21)
    first_hash = state_hash(first.state_dict())
    first_rng = rng_hashes()
    set_seed(61)
    second = PolicyValueNet((8, 1), "residual_v3", 21)
    assert state_hash(second.state_dict()) == first_hash
    assert rng_hashes() == first_rng
    random.random()
    np.random.random()
    torch.rand(1)
    assert rng_hashes() != first_rng


def test_first_divergence_identifies_batch_or_parameter_step() -> None:
    reference = [
        {"optimizer_step": 1, "batch_sha256": "a", "parameter_sha256": "x"},
        {"optimizer_step": 2, "batch_sha256": "b", "parameter_sha256": "y"},
    ]
    candidate = [
        {"optimizer_step": 1, "batch_sha256": "a", "parameter_sha256": "x"},
        {"optimizer_step": 2, "batch_sha256": "c", "parameter_sha256": "z"},
    ]
    assert first_divergence(reference, candidate, "batch_sha256") == 2
    assert first_divergence(reference, candidate, "parameter_sha256") == 2
