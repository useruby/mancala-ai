"""Seed469 correction numerical regression tests."""

import numpy as np
import json
from pathlib import Path

import pytest

from ml.alphazero_lite.seed469_correction import OUT, realize, sha

ROOT = Path(__file__).resolve().parents[2]


def test_realize_uses_float32_parameter_writeback() -> None:
    initial = np.asarray([1.0, 1.0e8], dtype=np.float32)
    update = np.asarray([2.0**-25, 1.0], dtype=np.float32)
    realized = realize(initial, update)
    expected = (initial + update).astype(np.float32).astype(
        np.float64
    ) - initial.astype(np.float64)
    assert realized.dtype == np.float64
    assert np.array_equal(realized, expected)
    assert realized[0] == 0.0


def test_realize_preserves_unused_and_zero_displacements() -> None:
    initial = np.asarray([3.0, -7.0, 11.0], dtype=np.float32)
    update = np.asarray([0.25, 0.0, -0.5], dtype=np.float32)
    realized = realize(initial, update)
    assert realized[1] == 0.0
    assert np.array_equal(realize(initial, np.zeros_like(initial)), np.zeros(3))


def test_float32_writeback_norm_can_differ_from_proposal_norm() -> None:
    initial = np.asarray([2.0**24, 1.0], dtype=np.float32)
    proposal = np.asarray([1.0, 0.0], dtype=np.float32)
    realized = realize(initial, proposal)
    discrepancy = np.linalg.norm(realized) - np.linalg.norm(proposal.astype(np.float64))
    assert discrepancy == -1.0
    proposal[1] = 0.25
    realized = realize(initial, proposal)
    assert np.linalg.norm(realized) != np.linalg.norm(proposal.astype(np.float64))


@pytest.fixture
def rebound_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import ml.alphazero_lite.verify_seed469_correction as verifier

    source = ROOT / OUT
    root = tmp_path
    target = root / OUT
    target.mkdir(parents=True)
    for filename in (
        "registration.json",
        "diagnostic.json",
        "vectors.npz",
        "results.md",
        "receipt.json",
    ):
        (target / filename).write_bytes((source / filename).read_bytes())
    # Exercise the verifier entry point and all publication semantics against a
    # cached independent reconstruction; authentic full replay is tested separately.
    monkeypatch.setattr(verifier, "verify_seed468", lambda _root: None)
    monkeypatch.setattr(verifier, "reconstruct", lambda _root: _reconstructed)
    real_digest = verifier.digest

    def digest_with_original_inputs(path: Path) -> str:
        if not path.exists() and path.is_relative_to(root):
            path = ROOT / path.relative_to(root)
        return real_digest(path)

    monkeypatch.setattr(verifier, "digest", digest_with_original_inputs)
    old = ROOT / "docs/data/seed468-first-step-optimizer-direction-transport"
    old_target = root / "docs/data/seed468-first-step-optimizer-direction-transport"
    old_target.mkdir(parents=True)
    for filename in ("registration.json", "receipt.json", "vectors.npz"):
        (old_target / filename).write_bytes((old / filename).read_bytes())
    return root


_reconstructed: tuple[dict[str, np.ndarray], list[dict]]


@pytest.fixture(scope="module", autouse=True)
def independently_reconstruct() -> None:
    from ml.alphazero_lite.verify_seed469_correction import reconstruct

    global _reconstructed
    _reconstructed = reconstruct(ROOT)


def _rebind(target: Path) -> None:
    receipt_path = target / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for filename in (
        "registration.json",
        "diagnostic.json",
        "vectors.npz",
        "results.md",
    ):
        receipt[filename] = sha(target / filename)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("gradient", "alter"),
        ("unseen", "alter"),
        ("missing", "remove"),
        ("extra", "add"),
        ("nonfinite", "nan"),
        ("proposal", "alter"),
        ("realized", "alter"),
        ("comparison_identity", "alter"),
        ("comparison_value", "alter"),
        ("counts", "alter"),
        ("classification", "alter"),
    ],
)
def test_verifier_entry_point_rejects_receipt_rebound_semantic_tampering(
    rebound_copy: Path, kind: str, value: str
) -> None:
    import ml.alphazero_lite.verify_seed469_correction as verifier

    target = rebound_copy / OUT
    report_path = target / "diagnostic.json"
    report = json.loads(report_path.read_text())
    vector_path = target / "vectors.npz"
    with np.load(vector_path, allow_pickle=False) as archive:
        vectors = {key: archive[key] for key in archive.files}
    if kind in {
        "gradient",
        "unseen",
        "missing",
        "extra",
        "nonfinite",
        "proposal",
        "realized",
    }:
        key = {
            "gradient": "batch00_P_gradient",
            "unseen": "unseen_gradient_equal_input",
            "proposal": "batch00_P_adam_proposal",
            "realized": "batch00_P_adam_realized",
            "nonfinite": "batch00_P_adam_proposal",
        }.get(kind)
        if kind == "missing":
            vectors.pop("batch00_P_gradient")
        elif kind == "extra":
            vectors["unexpected"] = np.zeros(1, dtype=np.float32)
        else:
            assert key is not None
            vectors[key] = vectors[key].copy()
            if value == "nan":
                vectors[key].flat[0] = np.nan
            else:
                vectors[key].flat[0] += 1.0
        np.savez_compressed(vector_path, **vectors)
    elif kind == "comparison_identity":
        report["comparisons"][0]["batch"] = 15
        report["comparisons"][1]["batch"] = 0
    elif kind == "comparison_value":
        report["comparisons"][0]["dot"] += 1.0
    elif kind == "counts":
        report["classification_counts"]["P/adam/equal_input"]["descent"] += 1
    else:
        report["classification"] = "forged"
    if kind not in {
        "gradient",
        "unseen",
        "missing",
        "extra",
        "nonfinite",
        "proposal",
        "realized",
    }:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _rebind(target)
    with pytest.raises(ValueError):
        verifier.verify(rebound_copy)
