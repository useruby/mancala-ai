from __future__ import annotations

import torch

from ml.alphazero_lite.context_action_q_probe import CONTEXT_SIZE, ContextActionQProbe


def test_context_action_q_probe_emits_one_correction_per_edge() -> None:
    probe = ContextActionQProbe(96)
    assert probe(torch.zeros(5, 96), torch.zeros(5, CONTEXT_SIZE)).shape == (5,)
