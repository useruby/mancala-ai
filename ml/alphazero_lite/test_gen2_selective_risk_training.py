from __future__ import annotations

import numpy as np
import torch

from ml.alphazero_lite.run_gen2_selective_risk_training import statewise_targets


def test_statewise_targets_replace_only_protected_rows() -> None:
    search = torch.tensor([[0.6, 0.4, 0.0], [0.2, 0.8, 0.0]])
    parent = torch.tensor([[0.4, 0.6, 0.0], [0.7, 0.3, 0.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    result = statewise_targets(search, parent, mask, np.asarray([True, False]))
    assert torch.equal(result[0], parent[0])
    assert torch.allclose(result[1], 0.05 * search[1] + 0.95 * parent[1])


def test_duplicate_state_hashes_receive_the_same_membership() -> None:
    from ml.alphazero_lite.run_gen2_selective_risk_training import state_hash

    first = {"state": [0.0] * 15}
    second = {"state": [0.0] * 15}
    protected = {state_hash(first)}
    assert (state_hash(first) in protected) == (state_hash(second) in protected)
