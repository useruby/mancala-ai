"""Unit and publication contract tests for the seed432 census."""

from __future__ import annotations

import gzip
import json
import shutil

import numpy as np
import pytest

from ml.alphazero_lite.seed432_policy_target_census import entropy, group_metrics
from ml.alphazero_lite.verify_seed432 import verify


def row(target, *, source="s", weight=1.0, multiplicity=1):
    return {
        "target": target,
        "source": source,
        "exposure_weight": weight * multiplicity,
        "equal_input_weight": weight,
    }


def test_identical_and_opposing_one_hot_targets():
    assert (
        group_metrics([row([1, 0]), row([1, 0])], "equal_input_weight")["js_nats"] == 0
    )
    assert group_metrics([row([1, 0]), row([0, 1])], "equal_input_weight")[
        "js_nats"
    ] == pytest.approx(np.log(2))


def test_soft_targets_same_top_action_and_ties():
    assert (
        group_metrics([row([0.8, 0.2]), row([0.6, 0.4])], "equal_input_weight")[
            "js_nats"
        ]
        > 0
    )
    tied = group_metrics([row([0.5, 0.5]), row([0.2, 0.8])], "equal_input_weight")
    assert tied["top_action_tie_rows"] == 1
    assert tied["top_action_disagreement"] == 1


def test_replay_multiplicity_changes_exposure_not_target_label_count():
    rows = [row([1, 0], multiplicity=4), row([0, 1])]
    result = group_metrics(rows, "exposure_weight")
    assert result["active_rows"] == 2
    assert result["policy_mass"] == 5
    assert 0 < result["js_nats"] < np.log(2)


def test_zero_coefficients_do_not_contribute_policy_mass():
    result = group_metrics([row([1, 0], weight=0), row([0, 1])], "equal_input_weight")
    assert result["policy_mass"] == 1
    assert result["js_nats"] == 0


def test_entropy_decomposition_identity():
    policies = [np.array([1.0, 0]), np.array([0, 1.0])]
    assert entropy(np.mean(policies, axis=0)) - np.mean(
        [entropy(p) for p in policies]
    ) == pytest.approx(np.log(2))


def test_source_entropy_decomposition_sums_to_total_js():
    rows = [
        {**row([1, 0], source="a"), "source": "a"},
        {**row([0, 1], source="b"), "source": "b"},
    ]
    result = group_metrics(rows, "equal_input_weight")
    assert result["within_source_js_nats"] + result[
        "between_source_js_nats"
    ] == pytest.approx(result["js_nats"])


def test_portable_read_only_verification_and_tamper_rejection(tmp_path):
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "docs/data/seed432-policy-target-compatibility"
    )
    relocated = tmp_path / "published"
    shutil.copytree(source, relocated)
    for path in relocated.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    assert verify(relocated)["status"] == "valid"
    for path in relocated.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
    with gzip.open(relocated / "row-accounting.jsonl.gz", "at") as stream:
        stream.write(json.dumps({"corrupt": True}) + "\n")
    with pytest.raises(ValueError, match="accounting_hash_mismatch"):
        verify(relocated)


@pytest.mark.parametrize("field", ["target", "replay_multiplicity", "partition"])
def test_row_evidence_mutations_are_rejected_even_with_updated_file_hash(
    tmp_path, field
):
    import hashlib
    import json
    import shutil
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "docs/data/seed432-policy-target-compatibility"
    )
    copy = tmp_path / field
    shutil.copytree(source, copy)
    accounting = copy / "row-accounting.jsonl.gz"
    with gzip.open(accounting, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    row = rows[0]
    if field == "target":
        row["target"][0] += 0.1
        row["target_sum"] += 0.1
    elif field == "replay_multiplicity":
        row["replay_multiplicity"] += 1
        row["exposure_weight"] += row["policy_coefficient"]
    else:
        row["partition"] = "validation" if row["partition"] == "train" else "train"
    with gzip.open(accounting, "wt", encoding="utf-8") as stream:
        for item in rows:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    manifest_path = copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["row_accounting_sha256"] = hashlib.sha256(
        accounting.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError):
        verify(copy)
