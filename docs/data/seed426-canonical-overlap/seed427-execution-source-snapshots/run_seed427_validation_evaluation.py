"""Freeze and execute seed427 evaluation of the two registered frozen networks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from ml.alphazero_lite import train  # noqa: E402
from ml.alphazero_lite.seed427_validation_metrics import aggregate, row_losses  # noqa: E402
from ml.alphazero_lite.seed427_validation_subsets import read_evidence  # noqa: E402
from ml.alphazero_lite.verify_seed427_publication import verify as verify_publication  # noqa: E402

DATA = ROOT / "docs/data/seed426-canonical-overlap"
REGISTRATION = ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
TRAINING_RESULTS = (
    ROOT / "docs/data/seed416-policy-target-softening/training-results.json"
)
SEED422 = ROOT / "docs/data/seed422-adam-first-moment"
CHECKPOINTS = {
    "initializer": Path(
        "/home/alex/Mancala/ai/.tmp/seed455-nextgen-s461-default-value-root16/runs/seed455-nextgen-s461-default-value-root16-iter1/parent_init_checkpoint.npz"
    ),
    "e4": Path(
        "/home/alex/Mancala/ai/.tmp/seed416-policy-target-softening/training/A/E4.npz"
    ),
}
EXPECTED_CHECKPOINTS = {
    "initializer": "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
    "e4": "836fc769c62126b649ce9691494719506b88efec0746769c9a589522d08cb2cd",
}
EXPECTED_E4 = {
    "policy_loss": 0.9556283950805664,
    "value_loss": 0.20779633522033691,
    "total_loss": 1.0179673433303833,
}
ABS_TOL = 1e-6
REL_TOL = 1e-5
CHUNK_SIZE = 512


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parameter_identity(model: torch.nn.Module) -> str:
    digestor = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digestor.update(name.encode() + b"\0")
        digestor.update(str(array.dtype).encode() + b"\0")
        digestor.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode() + b"\0"
        )
        digestor.update(array.tobytes(order="C"))
    return digestor.hexdigest()


def lineage_fingerprint(model: train.PolicyValueNet) -> str:
    """Match #416's JSON fingerprint of its serialized model parameters."""
    checkpoint = train.checkpoint_from_model(model)
    payload = {key: value.tolist() for key, value in checkpoint.items()}
    return digest(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def make_model(
    registration: dict[str, Any], checkpoint: Path
) -> tuple[train.PolicyValueNet, str]:
    architecture = registration["training"]["architecture"]
    model = train.PolicyValueNet(
        tuple(architecture["hidden_sizes"]),
        architecture["model"],
        train.input_size_for_encoding(architecture["input_encoding"]),
    )
    skipped = train.load_checkpoint_into_model(model, checkpoint)
    if skipped:
        raise ValueError(f"checkpoint_parameters_skipped:{skipped}")
    model.eval()
    return model, parameter_identity(model)


def load_compact_targets() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    registration = json.loads(REGISTRATION.read_text())
    source_data = DATA / "sources"
    compact: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    index = 0
    for spec in registration["replays"]:
        path = source_data / f"{spec['name']}.jsonl.gz"
        raw = gzip.decompress(path.read_bytes())
        for line_no, line in enumerate(raw.splitlines(), start=1):
            row = json.loads(line)
            policy = np.asarray(row["policy"], dtype=np.float32)
            train.validate_policy_target(
                policy,
                state=row["state"],
                path=path,
                row_number=line_no,
                policy_target_mode="sharpened",
                declared_mode=train.declared_policy_target_mode_for_row(row),
                row=row,
            )
            train.validate_value_target_mode(
                path=path,
                row_number=line_no,
                value_target_mode=spec["value_target_mode"],
                declared_mode=row.get("value_target_mode"),
            )
            value = float(
                np.float32(
                    train.validate_value_target(
                        row["value"], path=path, row_number=line_no
                    )
                )
            )
            weight = train.policy_loss_weight_for_row(
                row,
                exact_root_policy_loss_weight=train.DEFAULT_EXACT_ROOT_POLICY_LOSS_WEIGHT,
            )
            item = {
                "state": np.asarray(row["state"], dtype=np.float32),
                "policy": policy,
                "value": value,
                "policy_weight": weight,
            }
            compact[index] = item
            rows.append(
                {
                    "compact_row": index,
                    "source": spec["name"],
                    "raw_line": line_no,
                    **{
                        "state_sha256": hashlib.sha256(
                            item["state"].tobytes()
                        ).hexdigest(),
                        "policy": item["policy"].tolist(),
                        "value": value,
                        "policy_weight": weight,
                    },
                }
            )
            index += 1
    return rows, compact


def forward_chunks(
    model: torch.nn.Module, inputs: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic read-only inference using a fixed chunk size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size_must_be_positive")
    logits_chunks = []
    value_chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), chunk_size):
            tensor = torch.from_numpy(inputs[start : start + chunk_size])
            logits, values = model(tensor)
            logits_chunks.append(logits.cpu().numpy().astype(np.float32))
            value_chunks.append(values.reshape(-1).cpu().numpy().astype(np.float32))
    if not logits_chunks:
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(logits_chunks), np.concatenate(value_chunks)


def prepare() -> dict[str, Any]:
    publication = verify_publication(ROOT)
    membership_path = DATA / "seed427-validation-membership.jsonl.gz"
    membership_bytes = membership_path.read_bytes()
    membership = read_evidence(membership_path)
    published_counts = json.loads(
        (DATA / "seed427-validation-subsets.json").read_text()
    )
    if digest(membership_bytes) != published_counts["membership_sha256"]:
        raise ValueError("membership_hash_mismatch")
    if len(membership) != 14946:
        raise ValueError("membership_row_count_mismatch")
    counts = {
        "all": sum(
            row["subset"] in {"seen", "unseen", "canonical_unseen_input_seen"}
            for row in membership
        ),
        "seen": sum(row["subset"] == "seen" for row in membership),
        "unseen": sum(row["subset"] == "unseen" for row in membership),
    }
    if counts != {"all": 14946, "seen": 4240, "unseen": 10706}:
        raise ValueError(f"membership_counts_mismatch:{counts}")
    if any(
        row["canonical_seen_in_train"] != row["input_seen_in_train"]
        for row in membership
    ):
        raise ValueError("canonical_input_exclusion_disagreement")
    reg = json.loads(REGISTRATION.read_text())
    training = json.loads(TRAINING_RESULTS.read_text())
    seed422_registration_path = SEED422 / "registration.json"
    seed422_training_path = SEED422 / "training-results.json"
    seed422_receipt_path = SEED422 / "supplemental-verification-receipt.json"
    seed422_registration = json.loads(seed422_registration_path.read_text())
    seed422_training = json.loads(seed422_training_path.read_text())
    seed422_receipt = json.loads(seed422_receipt_path.read_text())
    if (
        seed422_registration["training"]["initializer"]["sha256"]
        != reg["seed455_initialization"]["sha256"]
        or seed422_registration["training"]["expected_A_E4_checkpoint_sha256"]
        != EXPECTED_CHECKPOINTS["e4"]
        or seed422_training["lanes"]["A"]["epochs"]["4"] != EXPECTED_CHECKPOINTS["e4"]
        or seed422_receipt["evidence_sha256"]["registration.json"]
        != digest(seed422_registration_path.read_bytes())
        or seed422_receipt["evidence_sha256"]["training-results.json"]
        != digest(seed422_training_path.read_bytes())
    ):
        raise ValueError("seed422_authoritative_lineage_receipt_mismatch")
    identities = {}
    for name, checkpoint in CHECKPOINTS.items():
        raw_hash = digest(checkpoint.read_bytes())
        if raw_hash != EXPECTED_CHECKPOINTS[name]:
            raise ValueError(f"checkpoint_hash_mismatch:{name}")
        model, loaded_identity = make_model(reg, checkpoint)
        identities[name] = {
            "path": str(checkpoint),
            "sha256": raw_hash,
            "loaded_parameter_sha256": loaded_identity,
            "lineage_serialized_parameter_sha256": lineage_fingerprint(model),
        }
        if (
            name == "initializer"
            and training["lanes"]["A"]["initialization_sha256"]
            != identities[name]["lineage_serialized_parameter_sha256"]
        ):
            raise ValueError("initializer_loaded_parameter_identity_mismatch")
    sources = [
        "ml/alphazero_lite/run_seed427_validation_evaluation.py",
        "ml/alphazero_lite/seed427_validation_metrics.py",
        "ml/alphazero_lite/seed427_validation_subsets.py",
    ]
    snapshot_dir = DATA / "seed427-execution-source-snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    source_bindings = {}
    for relative in sources:
        source = ROOT / relative
        target = snapshot_dir / source.name
        shutil.copy2(source, target)
        source_bindings[relative] = {
            "sha256": digest(source.read_bytes()),
            "snapshot_sha256": digest(target.read_bytes()),
        }
    manifest = {
        "schema": "seed427-frozen-validation-evaluation-v1",
        "status": "frozen_before_forward_passes",
        "interpretation": "retrospective subsets of historical seed416 validation; not a fresh independent holdout; no game-family independence inferred",
        "phase1": {
            "receipt_sha256": digest(
                (DATA / "seed427-supplemental-verification-receipt.json").read_bytes()
            ),
            "verification": publication,
        },
        "membership": {
            "path": "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
            "sha256": digest(membership_bytes),
            "counts": counts,
            "canonical_input_definition_differences": 0,
        },
        "lineage": {
            "seed416_registration_v3_sha256": digest(REGISTRATION.read_bytes()),
            "seed416_training_results_sha256": digest(TRAINING_RESULTS.read_bytes()),
            "initializer_registration": reg["seed455_initialization"],
            "e4_authoritative_record": "seed416 lane A epoch 4; fixed final checkpoint",
            "seed422_registration_sha256": digest(
                seed422_registration_path.read_bytes()
            ),
            "seed422_training_results_sha256": digest(
                seed422_training_path.read_bytes()
            ),
            "seed422_supplemental_verification_receipt_sha256": digest(
                seed422_receipt_path.read_bytes()
            ),
        },
        "models": identities,
        "architecture": reg["training"]["architecture"],
        "input_encoding": "kalah_v3; 27 float32 values",
        "targets": {
            "loaded_dtype": "all state, policy, and value arrays cast to float32 as in frozen train.load_jsonl_replay",
            "policy_target_mode": "sharpened",
            "value_target_modes_by_source": {
                item["name"]: item["value_target_mode"] for item in reg["replays"]
            },
            "exact_root_policy_loss_weight": train.DEFAULT_EXACT_ROOT_POLICY_LOSS_WEIGHT,
            "policy_loss_weight": "train.policy_loss_weight_for_row; row-level weights bound in prediction evidence",
            "source_target_arrays": json.loads((DATA / "results.json").read_text())[
                "loader_parity"
            ]["source_targets"],
        },
        "inference": {
            "device": "cpu",
            "model_eval": True,
            "torch_inference_mode": True,
            "chunk_size": CHUNK_SIZE,
            "row_order": "ascending compact row then original weighted-position order",
        },
        "losses": {
            "policy": "legal-masked log_softmax cross entropy; numerator=sum(row CE*policy weight), denominator=sum(policy weights)",
            "value": "smooth_l1_loss beta=1; numerator=sum(row Huber), denominator=weighted positions",
            "total": "policy + 0.3 * value",
            "equal_identity": "arithmetic mean row loss within each canonical identity, then arithmetic mean over canonical identities; row multiplicities retained within identity",
        },
        "tolerances": {
            "absolute": ABS_TOL,
            "relative": REL_TOL,
            "e4_full_validation_expected": EXPECTED_E4,
        },
        "execution_sources": source_bindings,
    }
    path = DATA / "seed427-evaluation-manifest.json"
    if path.is_file():
        archive = DATA / "seed427-evaluation-manifest-pre-seed422-receipt.json"
        if not archive.exists():
            shutil.copy2(path, archive)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def execute() -> dict[str, Any]:
    manifest_path = DATA / "seed427-evaluation-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    for relative, binding in manifest["execution_sources"].items():
        if digest((ROOT / relative).read_bytes()) != binding["sha256"]:
            raise ValueError(f"execution_source_changed:{relative}")
    membership = read_evidence(DATA / "seed427-validation-membership.jsonl.gz")
    _, compact = load_compact_targets()
    val_compacts = sorted({row["compact_row"] for row in membership})
    x = np.stack([compact[index]["state"] for index in val_compacts])
    position_lookup = {
        index: compact_index for compact_index, index in enumerate(val_compacts)
    }
    models = {}
    checkpoint_before = {}
    for name, path in CHECKPOINTS.items():
        checkpoint_before[name] = digest(path.read_bytes())
        model, loaded_identity = make_model(json.loads(REGISTRATION.read_text()), path)
        if loaded_identity != manifest["models"][name]["loaded_parameter_sha256"]:
            raise ValueError(f"loaded_parameters_changed:{name}")
        models[name] = model
    result_rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for name, model in models.items():
        logits_array, values_array = forward_chunks(model, x, CHUNK_SIZE)
        logits_list = logits_array.tolist()
        values_list = values_array.tolist()
        prediction_by_compact = {
            idx: (logits_list[position_lookup[idx]], values_list[position_lookup[idx]])
            for idx in val_compacts
        }
        outputs = []
        for member in membership:
            row = compact[member["compact_row"]]
            logits, pred_value = prediction_by_compact[member["compact_row"]]
            legal = (
                train.legal_mask_for_encoded_state(row["state"])
                .astype(np.int8)
                .tolist()
            )
            losses = row_losses(
                logits,
                pred_value,
                row["policy"].tolist(),
                row["value"],
                legal,
                row["policy_weight"],
            )
            outputs.append(
                {
                    "membership": member,
                    "logits": logits,
                    "value_prediction": pred_value,
                    "target_policy": row["policy"].tolist(),
                    "target_value": row["value"],
                    "legal_mask": legal,
                    "policy_weight": row["policy_weight"],
                    "losses": losses,
                }
            )
        result_rows_by_model[name] = outputs
    e4_full = _full_validation(result_rows_by_model["e4"])
    parity = {
        key: {
            "actual": e4_full[key],
            "expected": expected,
            "passed": math_isclose(e4_full[key], expected),
        }
        for key, expected in EXPECTED_E4.items()
    }
    if not all(item["passed"] for item in parity.values()):
        write_json(
            DATA / "seed427-evaluation-failure.json",
            {
                "schema": "seed427-evaluation-failure-v1",
                "reason": "e4_full_validation_parity_failed",
                "parity": parity,
                "manifest_sha256": digest(manifest_bytes),
            },
        )
        raise ValueError(f"e4_full_validation_parity_failed:{parity}")
    summary = {name: aggregate(rows) for name, rows in result_rows_by_model.items()}
    comparisons = {}
    for key in summary["e4"]:
        comparisons[key] = {
            aggregation: {
                metric: summary["e4"][key][aggregation][metric]
                - summary["initializer"][key][aggregation][metric]
                for metric in ("policy_loss", "value_loss", "total_loss")
            }
            for aggregation in ("exposure_weighted", "equal_canonical_identity")
        }
    output = DATA / "seed427-prediction-evidence.jsonl.gz"
    if output.is_file():
        archive = DATA / "seed427-prediction-evidence-pre-seed422-receipt.jsonl.gz"
        if not archive.exists():
            shutil.copy2(output, archive)
    results_path = DATA / "seed427-evaluation-results.json"
    if results_path.is_file():
        archive = DATA / "seed427-evaluation-results-pre-seed422-receipt.json"
        if not archive.exists():
            shutil.copy2(results_path, archive)
    with gzip.open(output, "wt", encoding="utf-8", newline="\n") as stream:
        for model_name, rows in result_rows_by_model.items():
            for row in rows:
                stream.write(
                    json.dumps(
                        {"model": model_name, **row},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    receipt = {
        "schema": "seed427-evaluation-receipt-v1",
        "status": "completed",
        "manifest_sha256": digest(manifest_bytes),
        "prediction_evidence_sha256": digest(output.read_bytes()),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": {
            name: digest(path.read_bytes()) for name, path in CHECKPOINTS.items()
        },
        "e4_full_validation_parity": parity,
        "results": summary,
        "e4_minus_initializer": comparisons,
        "primary_comparison": {
            "subset": "unseen/>32",
            "differences": comparisons["unseen/>32"],
            "interpretation": "retrospective validation-loss comparison only; no strength claim or checkpoint selection",
        },
    }
    if receipt["checkpoint_sha256_before"] != receipt["checkpoint_sha256_after"]:
        raise ValueError("checkpoint_immutability_failure")
    write_json(DATA / "seed427-evaluation-results.json", receipt)
    return receipt


def _full_validation(rows: list[dict[str, Any]]) -> dict[str, float]:
    policy_denominator = sum(row["policy_weight"] for row in rows)
    policy = (
        sum(row["losses"]["policy_loss"] * row["policy_weight"] for row in rows)
        / policy_denominator
    )
    value = sum(row["losses"]["value_loss"] for row in rows) / len(rows)
    return {
        "policy_loss": policy,
        "value_loss": value,
        "total_loss": policy + 0.3 * value,
    }


def math_isclose(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= ABS_TOL + REL_TOL * abs(expected)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute"))
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(), sort_keys=True))
    else:
        print(json.dumps(execute(), sort_keys=True))
