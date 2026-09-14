import json
from pathlib import Path
import random

import numpy as np
import pytest
from ml.alphazero_lite.legal_policy_metrics import (
    legal_policy_from_logits,
    normalize_policy_over_legal_actions,
    zeroed_legal_policy,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_legal_policy_metric_parity_audit import state_modes
from ml.alphazero_lite.self_play import CheckpointEvaluator, PUCT


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/data/alphazero-lite-internal-cluster-frozen-evaluation-set.json"
G0 = ROOT / ".tmp/internal-cluster-learning-dynamics/g0-parent/checkpoint.npz"


def test_full_softmax_is_preserved_and_legacy_zeroing_does_not_renormalize() -> None:
    full = np.asarray([0.10, 0.20, 0.05, 0.25, 0.30, 0.10])
    legacy = zeroed_legal_policy(full, [0, 1, 3, 4])
    assert full.sum() == pytest.approx(1.0)
    assert legacy.sum() == pytest.approx(0.85)
    assert legacy[2] == legacy[5] == 0.0


@pytest.mark.parametrize("legal", ([0], [0, 2], [0, 1, 3, 4], [0, 1, 2, 3, 4, 5]))
def test_legal_normalization_is_a_probability_distribution(legal: list[int]) -> None:
    full = np.asarray([0.10, 0.20, 0.05, 0.25, 0.30, 0.10])
    normalized = normalize_policy_over_legal_actions(full, legal)
    assert normalized[legal].sum() == pytest.approx(1.0)
    assert all(normalized[action] == 0.0 for action in set(range(6)) - set(legal))
    assert min(legal, key=lambda action: (-full[action], action)) == min(
        legal, key=lambda action: (-normalized[action], action)
    )


@pytest.mark.parametrize("legal", ([0], [0, 2], [0, 1, 3, 4], [0, 1, 2, 3, 4, 5]))
def test_masked_logits_and_full_softmax_legal_normalization_are_identical(
    legal: list[int],
) -> None:
    logits = np.asarray([-2.0, 0.5, -0.25, 1.75, 0.75, -1.0])
    full = np.exp(logits - logits.max())
    full /= full.sum()
    assert np.allclose(
        normalize_policy_over_legal_actions(full, legal),
        legal_policy_from_logits(logits, legal),
        atol=1e-12,
    )


@pytest.mark.skipif(not G0.is_file(), reason="persisted audit checkpoint unavailable")
def test_residual_v3_checkpoint_matches_torch_and_puct_priors() -> None:
    manifest = json.loads(MANIFEST.read_text())
    by_legal_count = {
        len(entry["legal_actions"]): entry for entry in manifest["entries"]
    }
    # The immutable frozen set covers its available legal-action counts; unit cases cover 1/2/4/6.
    for entry in by_legal_count.values():
        modes = state_modes(G0, entry)
        assert modes["torch_max_abs_probability_difference"] <= 1e-5
        assert sum(modes["numpy_full_softmax"]) == pytest.approx(1.0)
        assert sum(
            modes["numpy_legal_normalized"][a] for a in modes["legal_actions"]
        ) == pytest.approx(1.0)
        game = KalahGame.from_state(entry["state"])
        search = PUCT(
            CheckpointEvaluator(G0, input_encoding="kalah_v3"),
            1,
            1.25,
            random.Random(0),
            tactical_root_bias=0.0,
        )
        _policy, root = search.run(game, dirichlet_alpha=None)
        priors = np.zeros(6)
        for action, child in root.children.items():
            priors[action] = child.prior
        assert np.allclose(priors, modes["numpy_legal_normalized"], atol=1e-5)
