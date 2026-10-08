"""Synthetic invariants for seed440 attribution arithmetic."""

from __future__ import annotations

import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ml.alphazero_lite.seed437_gradient_alignment import equal_input_weights
from ml.alphazero_lite.seed440_recorded_update_attribution import classify

ROOT = Path(__file__).resolve().parents[2]


def test_equal_input_weights_preserve_exposure_multiplicity() -> None:
    weights = equal_input_weights(["a", "a", "b"])
    assert np.allclose(weights, [0.25, 0.25, 0.5])
    assert np.isclose(weights.sum(), 1.0)


def test_attribution_classification_branches() -> None:
    assert (
        classify({"x": {"D": 1.0, "S": 0.1, "R": 0.9}})
        == "finite_step_residual_dominant"
    )
    assert (
        classify({"x": {"D": 1.0, "S": 0.9, "R": 0.1}})
        == "first_order_direction_dominant"
    )
    assert (
        classify({"x": {"D": 1.0, "S": 0.5, "R": 0.5}})
        == "mixed_or_no_common_dominance"
    )
    assert (
        classify({"x": {"D": -1.0, "S": 0.9, "R": 0.1}})
        == "mixed_or_no_common_dominance"
    )


def test_real_seed439_cli_uses_physically_copied_dependencies() -> None:
    """Run seed439 outside the checkout and verify the copied evidence is read-only."""
    with tempfile.TemporaryDirectory(
        prefix="seed440-relocation-", dir=ROOT / ".tmp"
    ) as temporary:
        workspace = Path(temporary)
        copied = workspace / "copied-checkout"
        unrelated = workspace / "unrelated-working-directory"
        unrelated.mkdir()
        shutil.copytree(ROOT / "ml/alphazero_lite", copied / "ml/alphazero_lite")
        (copied / "ml").mkdir(exist_ok=True)
        (copied / "ml/__init__.py").write_text("")
        (copied / "ml/alphazero_lite/__init__.py").write_text("")
        data_root = ROOT / "docs/data"
        for name in (
            "seed416-policy-target-softening",
            "seed426-canonical-overlap",
            "seed429-canonical-policy-normalization",
            "seed432-policy-target-compatibility",
            "seed433-seed432-census-correction",
            "seed435-adam-direction-screen",
            "seed437-training-unseen-gradient-alignment",
            "seed438-seed437-gradient-correction",
            "seed439-gradient-publication-verification",
        ):
            shutil.copytree(data_root / name, copied / "docs/data" / name)
        evidence = (
            "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
            "docs/data/seed435-adam-direction-screen/results.json",
            "docs/data/seed437-training-unseen-gradient-alignment/gradient-vectors.npz",
            "docs/data/seed438-seed437-gradient-correction/corrected-gradient-vectors.npz",
        )
        before = {path: (copied / path).read_bytes() for path in evidence}
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(copied)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.alphazero_lite.verify_seed439_gradient_publication",
                "--root",
                str(copied),
            ],
            cwd=unrelated,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert '"status": "valid"' in result.stdout
        assert all(
            (copied / path).read_bytes() == content for path, content in before.items()
        )
        assert not any(path.is_symlink() for path in copied.rglob("*"))
