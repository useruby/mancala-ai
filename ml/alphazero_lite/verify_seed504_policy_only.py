"""Read-only verification of seed504 policy-only development evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import run_seed504 as base

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed504-policy-only-ablation"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _assert_metrics(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for cohort, metrics in actual.items():
        saved = expected[cohort]
        if (
            metrics["rows"] != saved["rows"]
            or metrics["exact_inputs"] != saved["exact_inputs"]
        ):
            raise ValueError(f"development_membership_mismatch:{cohort}")
        for weighting in ("exposure", "equal_exact_input"):
            for key, value in metrics[weighting].items():
                if not np.isclose(value, saved[weighting][key], rtol=0, atol=1e-7):
                    raise ValueError(
                        f"development_metric_mismatch:{cohort}:{weighting}:{key}"
                    )


def verify(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT.relative_to(ROOT)
    registration = json.loads((out / "registration.json").read_text())
    execution_freeze = json.loads((out / "execution-freeze.json").read_text())
    if (
        registration.get("status") != "frozen_before_T_training"
        or execution_freeze.get("registration_sha256") != sha(out / "registration.json")
        or execution_freeze.get("status") != "frozen_before_T_training"
    ):
        raise ValueError("prospective_freeze_binding_mismatch")
    bindings = registration["bindings"]
    manifest_path = root / "docs/data/seed504/manifest.json"
    if sha(manifest_path) != bindings["seed504_manifest_sha256"]:
        raise ValueError("seed504_manifest_binding_mismatch")
    historical_manifest = json.loads(manifest_path.read_text())
    initializer = root / "docs/data/seed508-policy-value-crossover/initializer.npz"
    if sha(initializer) != bindings["initializer_sha256"]:
        raise ValueError("initializer_binding_mismatch")
    for group in ("training", "development"):
        for entry in historical_manifest["sources"][group]:
            path = root / entry["path"]
            if sha(path) != entry["sha256"]:
                raise ValueError(f"corpus_binding_mismatch:{entry['path']}")
    if (
        sha(root / "docs/data/seed504/split.json")
        != historical_manifest["split_sha256"]
    ):
        raise ValueError("split_binding_mismatch")
    if (
        sha(root / "docs/data/seed504/permutations.json")
        != bindings["permutations_sha256"]
    ):
        raise ValueError("permutation_binding_mismatch")
    for label, entry in bindings["historical_checkpoints"].items():
        checkpoint = root / f"docs/data/seed504/seed504_epoch{label[1:]}.npz"
        if sha(checkpoint) != entry:
            raise ValueError(f"historical_checkpoint_mismatch:{label}")
    sources = bindings["sources"]
    for rel, expected in (
        ("ml/alphazero_lite/run_seed504_policy_only.py", sources["runner"]),
        ("ml/alphazero_lite/run_seed504.py", sources["seed504_runner"]),
        ("ml/alphazero_lite/train.py", sources["trainer"]),
        ("ml/alphazero_lite/verify_seed504.py", sources["seed504_verifier"]),
    ):
        actual = root / rel
        source_snapshot = out / "execution-source-snapshots" / f"{Path(rel).name}.txt"
        if (not actual.is_file() or sha(actual) != expected) and (
            not source_snapshot.is_file() or sha(source_snapshot) != expected
        ):
            raise ValueError(f"source_binding_mismatch:{rel}")

    candidate = out / "seed504_policy_only_epoch4.npz"
    init_arrays, t_arrays = _arrays(initializer), _arrays(candidate)
    if set(init_arrays) != set(t_arrays):
        raise ValueError("candidate_parameter_schema_mismatch")
    value_names = ["w_value", "b_value", "w_value_hidden", "b_value_hidden"]
    if not value_names or any(
        not np.array_equal(init_arrays[name], t_arrays[name]) for name in value_names
    ):
        raise ValueError("value_head_not_frozen")

    _, development, evidence = base.split()
    saved_split = json.loads((out / "split.json").read_text())
    if saved_split != evidence or sha(out / "split.json") != bindings["split_sha256"]:
        raise ValueError("development_membership_binding_mismatch")
    result_path = out / "development-results.json"
    results = json.loads(result_path.read_text())
    for name, checkpoint in (
        ("initializer", initializer),
        ("C_E4", root / "docs/data/seed504/seed504_epoch4.npz"),
        ("T_E4", candidate),
    ):
        digest = sha(checkpoint)
        if results[name]["checkpoint_sha256"] != digest:
            raise ValueError(f"development_artifact_binding_mismatch:{name}")
        metrics, predictions = base.evaluate(
            base.load_model(checkpoint, torch.device("cpu")), development
        )
        _assert_metrics(metrics, results[name]["metrics"])
        recorded = results[name]["predictions"]
        if len(predictions) != len(recorded):
            raise ValueError(f"prediction_count_mismatch:{name}")
        for actual, saved in zip(predictions, recorded, strict=True):
            for key in (
                "source",
                "line",
                "input_sha256",
                "target_policy",
                "target_value",
            ):
                if actual[key] != saved[key]:
                    raise ValueError(f"prediction_identity_mismatch:{name}:{key}")
            for key in ("policy_logits", "policy_probabilities", "value_prediction"):
                if not np.allclose(actual[key], saved[key], rtol=0, atol=1e-7):
                    raise ValueError(f"prediction_mismatch:{name}:{key}")

    history = json.loads((out / "epochs.json").read_text())
    if len(history) != 4:
        raise ValueError("T_epoch_history_shape")
    for row in history:
        if row["optimizer_updates"] != 27 or row["examples_sampled"] != 13724:
            raise ValueError("T_training_count_mismatch")
        if row["total_loss"] != row["policy_loss"]:
            raise ValueError("T_objective_contains_non_policy_loss")
    # No arena ledger exists yet: this is a valid training/development result,
    # but strength status remains incomplete and cannot advance.
    arena_complete = (out / "outcomes.jsonl").is_file()
    return {
        "status": "verified_development_incomplete_arena"
        if not arena_complete
        else "arena_requires_full_verification",
        "control": json.loads((out / "control-reproduction.json").read_text()),
        "T_value_head_parameter_arrays_equal_initializer": True,
        "development_rows": len(development),
        "cohorts_recomputed": ["pooled", ">32", "17-32", "<=16"],
        "arena_status": "not_run_resource_allocation_unverified"
        if not arena_complete
        else "unverified",
        "advance": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
