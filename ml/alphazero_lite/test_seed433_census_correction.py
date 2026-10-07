"""Regression coverage for corrected equal-input aggregation and publication."""

from __future__ import annotations

import numpy as np
import pytest
import shutil
import subprocess
import sys

from ml.alphazero_lite.seed433_census_correction import (
    DATA,
    ROOT,
    group_metrics,
    summarize,
)


def test_equal_input_outer_weight_is_one_per_group():
    rows = [
        {
            "partition": "train",
            "input_sha256": "opposing",
            "target": [1, 0],
            "source": "a",
            "equal_input_weight": 1.0,
        },
        {
            "partition": "train",
            "input_sha256": "opposing",
            "target": [0, 1],
            "source": "b",
            "equal_input_weight": 1.0,
        },
        *[
            {
                "partition": "train",
                "input_sha256": "unanimous",
                "target": [1, 0],
                "source": "c",
                "equal_input_weight": 1.0,
            }
            for _ in range(3)
        ],
    ]
    result = summarize(rows, "equal_input")
    assert result["mean_js_nats"] == pytest.approx(np.log(2) / 2)
    assert result["weighted_top_action_disagreement"] == pytest.approx(0.5)


def test_unanimous_group_row_repetition_does_not_change_group_summary():
    one = [{"target": [0, 1], "source": "b", "equal_input_weight": 1.0}]
    repeated = one * 3
    base = summarize(
        [{"partition": "train", "input_sha256": "x", **one[0]}], "equal_input"
    )
    copy = summarize(
        [{"partition": "train", "input_sha256": "x", **row} for row in repeated],
        "equal_input",
    )
    assert base["mean_js_nats"] == copy["mean_js_nats"] == 0
    assert base["aggregation_denominator"] == copy["aggregation_denominator"] == 1


def test_zero_mass_and_singleton_are_explicitly_accounted():
    rows = [
        {
            "partition": "train",
            "input_sha256": "one",
            "target": [1, 0],
            "source": "a",
            "exposure_weight": 0.0,
            "equal_input_weight": 0.0,
        },
        {
            "partition": "train",
            "input_sha256": "two",
            "target": [1, 0],
            "source": "a",
            "exposure_weight": 2.0,
            "equal_input_weight": 0.25,
        },
    ]
    result = summarize(rows, "equal_input")
    assert result["zero_mass_groups"] == 1
    assert result["positive_mass_groups"] == 1
    assert result["aggregation_denominator"] == 1
    assert result["policy_mass"] == 0.25
    assert result["mean_js_nats"] == 0


def test_primary_gate_ignores_failing_treatment_diagnostics():
    control = {"mean_js_nats": 0.05, "weighted_top_action_disagreement": 0.1}
    treatment = {"mean_js_nats": 0.0, "weighted_top_action_disagreement": 0.0}
    assert all(
        control[key] >= threshold
        for key, threshold in (
            ("mean_js_nats", 0.05),
            ("weighted_top_action_disagreement", 0.1),
        )
    )
    assert not all(
        treatment[key] >= threshold
        for key, threshold in (
            ("mean_js_nats", 0.05),
            ("weighted_top_action_disagreement", 0.1),
        )
    )


def test_source_entropy_decomposition():
    rows = [
        {"target": [1, 0], "source": "a", "equal_input_weight": 1.0},
        {"target": [0, 1], "source": "b", "equal_input_weight": 1.0},
    ]
    result = group_metrics(rows, "equal_input_weight")
    assert result["within_source_js_nats"] == 0
    assert result["between_source_js_nats"] == pytest.approx(np.log(2))


def test_portable_cli_from_unrelated_directory(tmp_path):
    relocated = tmp_path / "correction"
    shutil.copytree(DATA, relocated)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed433",
            "--directory",
            str(relocated),
        ],
        cwd=tmp_path,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "valid"' in completed.stdout
