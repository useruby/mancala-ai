"""Focused invariants for seed441's append-only path evidence."""

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


def test_equal_input_weighting_retains_exposure_multiplicity() -> None:
    weights = equal_input_weights(["x", "x", "y"])
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.5])
    assert weights.sum() == 1.0


def test_fixed_primary_classification_branches_and_priority() -> None:
    both = {
        "exposure": {"D": 2.0, "S": 1.5, "R": 1.5},
        "equal": {"D": 2.0, "S": 1.5, "R": 1.5},
    }
    assert classify(both) == "finite_step_residual_dominant"
    first = {
        "exposure": {"D": 2.0, "S": 1.5, "R": 0.1},
        "equal": {"D": 2.0, "S": 1.5, "R": 0.1},
    }
    assert classify(first) == "first_order_direction_dominant"
    mixed = {
        "exposure": {"D": 2.0, "S": 1.49, "R": 1.49},
        "equal": {"D": 2.0, "S": 1.49, "R": 1.49},
    }
    assert classify(mixed) == "mixed_or_no_common_dominance"
    nonpositive = {
        "exposure": {"D": 0.0, "S": 2.0, "R": 2.0},
        "equal": {"D": 0.0, "S": 2.0, "R": 2.0},
    }
    assert classify(nonpositive) == "mixed_or_no_common_dominance"


def test_portable_cli_uses_source_only_checkout_and_separate_evidence_root() -> None:
    """Run from elsewhere with physically copied source and evidence trees."""
    with tempfile.TemporaryDirectory(
        prefix="seed441-portable-", dir=ROOT / ".tmp"
    ) as tmp:
        workspace = Path(tmp)
        source = workspace / "source-only"
        evidence = workspace / "evidence-root"
        unrelated = workspace / "unrelated-cwd"
        unrelated.mkdir()
        shutil.copytree(ROOT / "ml", source / "ml")
        (source / "__init__.py").write_text("")
        (source / "ml/__init__.py").write_text("")
        (source / "ml/alphazero_lite/__init__.py").write_text("")
        data = ROOT / "docs/data"
        for directory in (
            "seed416-policy-target-softening",
            "seed426-canonical-overlap",
            "seed429-canonical-policy-normalization",
            "seed435-adam-direction-screen",
            "seed438-seed437-gradient-correction",
            "seed440-recorded-update-attribution",
            "seed441-seed440-publication-completion",
        ):
            shutil.copytree(data / directory, evidence / "docs/data" / directory)
        (evidence / ".tmp").mkdir()
        monitored = [
            "docs/data/seed435-adam-direction-screen",
            "docs/data/seed440-recorded-update-attribution",
            "docs/data/seed441-seed440-publication-completion",
            "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        ]
        before = {
            relative: {
                path.relative_to(evidence): path.read_bytes()
                for path in (evidence / relative).rglob("*")
                if path.is_file()
            }
            for relative in monitored
        }
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.alphazero_lite.verify_seed441_seed440_publication",
                "--root",
                str(evidence),
            ],
            cwd=unrelated,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert not any(path.is_symlink() for path in workspace.rglob("*"))
        for relative, files in before.items():
            assert all(
                (evidence / path).read_bytes() == content
                for path, content in files.items()
            )
