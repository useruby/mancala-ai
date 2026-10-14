"""Real arithmetic tests for seed478 materialized local counterfactuals."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import json
import shutil
from pathlib import Path

from ml.alphazero_lite import seed478_finite_step_evaluation as protocol
from ml.alphazero_lite import verify_seed478_finite_step_evaluation as verifier
from ml.alphazero_lite.seed478_finite_step_evaluation import _curves


def test_float32_scaled_proposals_start_from_archived_origin_and_are_independent():
    pre = torch.tensor([0.1, -3.0], dtype=torch.float32)
    post = torch.tensor([0.09, -2.99], dtype=torch.float32)
    adam = (post.double() - pre.double()).numpy()
    first = pre + torch.from_numpy((adam * 0.5).astype(np.float32))
    second = pre + torch.from_numpy((adam * 0.25).astype(np.float32))
    assert torch.equal(pre + torch.from_numpy(adam.astype(np.float32)), post)
    assert torch.equal(first, pre + torch.from_numpy((adam * 0.5).astype(np.float32)))
    assert torch.equal(second, pre + torch.from_numpy((adam * 0.25).astype(np.float32)))
    assert not torch.equal(first, second)


def test_each_endpoint_remainder_uses_its_own_observed_change():
    gradient = np.array([2.0, -1.0])
    endpoints = [np.array([0.01, 0.0]), np.array([0.0, 0.01])]
    observed = [-0.2, 0.1]
    remainder = [
        change - float(gradient @ delta)
        for change, delta in zip(observed, endpoints, strict=True)
    ]
    assert remainder == pytest.approx([-0.22, 0.11])
    assert remainder[0] != observed[0] - float(gradient @ endpoints[1])


def test_curve_counts_are_exhaustive_and_do_not_sum_across_trials():
    trials = []
    for scale, value in ((1.0, -0.2), (0.5, 0.0), (0.25, 0.2), (0.125, -0.1)):
        trials.append(
            {
                "direction": "full_negative",
                "scale": scale,
                "weightings": {
                    "exposure": {"change": value},
                    "equal_input": {"change": value},
                },
            }
        )
    curves = _curves([{"trials": trials}])
    assert curves["full_negative/1.0/exposure"] == {
        "improvement": 1,
        "regression": 0,
        "ambiguous": 0,
        "mean": -0.2,
        "median": -0.2,
    }
    assert curves["full_negative/0.5/exposure"]["ambiguous"] == 1
    assert curves["full_negative/0.25/exposure"]["regression"] == 1


@pytest.fixture
def bound_copy(tmp_path: Path):
    source = protocol.ROOT
    for relative in (*protocol.SOURCE_PATHS, *protocol.INPUT_PATHS.values()):
        src, dst = source / relative, tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for directory in (
        protocol.BASE,
        protocol.SUP,
        protocol.HIST,
        protocol.INIT.parent.parent,
    ):
        src, dst = source / directory, tmp_path / directory
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    shutil.copytree(source / protocol.OUT, tmp_path / protocol.OUT)
    return tmp_path


def _rebind(root: Path) -> None:
    out = root / protocol.OUT
    receipt = json.loads((out / "receipt.json").read_text())
    for name in receipt:
        if name != "status":
            receipt[name] = protocol.sha(out / name)
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def test_receipt_rebound_baseline_passes(bound_copy: Path):
    verifier.verify(bound_copy)


@pytest.mark.parametrize(
    "mutation",
    [
        "swapped_endpoint",
        "copied_adam_loss",
        "altered_scale",
        "altered_state",
        "altered_prediction",
        "altered_target",
        "altered_weight",
        "altered_first_order",
        "altered_remainder",
        "altered_count",
        "missing_trial",
        "duplicate_trial",
    ],
)
def test_receipt_rebound_entrypoint_rejects_trial_corruption(
    bound_copy: Path, mutation: str
):
    out = bound_copy / protocol.OUT
    evidence_path = out / "evidence.json"
    report = json.loads(evidence_path.read_text())
    first = report["steps"][0]["trials"]
    if mutation == "swapped_endpoint":
        first[0]["array_prefix"], first[1]["array_prefix"] = (
            first[1]["array_prefix"],
            first[0]["array_prefix"],
        )
    elif mutation == "copied_adam_loss":
        first[4]["weightings"]["exposure"]["change"] = first[0]["weightings"][
            "exposure"
        ]["change"]
    elif mutation == "altered_scale":
        first[0]["scale"] = 0.75
    elif mutation == "altered_state":
        first[0]["state_sha256"] = "0" * 64
    elif mutation == "altered_prediction":
        with np.load(out / "occurrence-evidence.npz", allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        values[first[0]["array_prefix"] + "_unseen_probabilities"][0, 0] += 0.01
        np.savez_compressed(out / "occurrence-evidence.npz", **values)
    elif mutation in ("altered_target", "altered_weight"):
        with np.load(out / "occurrence-evidence.npz", allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        key = (
            "unseen_targets"
            if mutation == "altered_target"
            else "unseen_equal_input_weights"
        )
        values[key].flat[0] += 0.01
        np.savez_compressed(out / "occurrence-evidence.npz", **values)
    elif mutation == "altered_first_order":
        first[0]["weightings"]["exposure"]["first_order"] += 0.1
    elif mutation == "altered_remainder":
        first[0]["weightings"]["exposure"]["remainder"] += 0.1
    elif mutation == "altered_count":
        report["curves"]["full_negative/1.0/exposure"]["improvement"] += 1
    elif mutation == "missing_trial":
        first.pop()
    elif mutation == "duplicate_trial":
        first.append(first[-1])
    if mutation not in ("altered_prediction", "altered_target", "altered_weight"):
        evidence_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _rebind(bound_copy)
    with pytest.raises(ValueError):
        verifier.verify(bound_copy)
