"""Read-only, checkpoint-free validation of seed427 prediction publication."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from ml.alphazero_lite.seed427_validation_metrics import aggregate, row_losses  # noqa: E402
from ml.alphazero_lite.seed427_validation_subsets import construct, read_evidence  # noqa: E402
from ml.alphazero_lite.verify_seed427_publication import verify as verify_publication  # noqa: E402


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify(root: Path) -> dict[str, Any]:
    publication = verify_publication(root)
    data = root / "docs/data/seed426-canonical-overlap"
    manifest_path = data / "seed427-evaluation-manifest.json"
    result_path = data / "seed427-evaluation-results.json"
    prediction_path = data / "seed427-prediction-evidence.jsonl.gz"
    for path, reason in (
        (manifest_path, "missing_evaluation_manifest"),
        (result_path, "missing_evaluation_results"),
        (prediction_path, "missing_prediction_evidence"),
    ):
        if not path.is_file():
            raise ValueError(reason)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(result_path.read_text())
    if sha256(manifest_bytes) != receipt["manifest_sha256"]:
        raise ValueError("evaluation_manifest_binding_mismatch")
    if sha256(prediction_path.read_bytes()) != receipt["prediction_evidence_sha256"]:
        raise ValueError("prediction_evidence_binding_mismatch")
    phase1_receipt = data / "seed427-supplemental-verification-receipt.json"
    if (
        not phase1_receipt.is_file()
        or sha256(phase1_receipt.read_bytes()) != manifest["phase1"]["receipt_sha256"]
    ):
        raise ValueError("phase1_receipt_binding_mismatch")
    lineage = manifest["lineage"]
    lineage_files = {
        "seed416_registration_v3_sha256": root
        / "docs/data/seed416-policy-target-softening/registration-v3.json",
        "seed416_training_results_sha256": root
        / "docs/data/seed416-policy-target-softening/training-results.json",
        "seed422_registration_sha256": root
        / "docs/data/seed422-adam-first-moment/registration.json",
        "seed422_training_results_sha256": root
        / "docs/data/seed422-adam-first-moment/training-results.json",
        "seed422_supplemental_verification_receipt_sha256": root
        / "docs/data/seed422-adam-first-moment/supplemental-verification-receipt.json",
    }
    for key, path in lineage_files.items():
        if not path.is_file() or sha256(path.read_bytes()) != lineage[key]:
            raise ValueError(f"authoritative_lineage_binding_mismatch:{key}")
    seed422_receipt = json.loads(
        lineage_files["seed422_supplemental_verification_receipt_sha256"].read_text()
    )
    if seed422_receipt["evidence_sha256"]["registration.json"] != sha256(
        lineage_files["seed422_registration_sha256"].read_bytes()
    ) or seed422_receipt["evidence_sha256"]["training-results.json"] != sha256(
        lineage_files["seed422_training_results_sha256"].read_bytes()
    ):
        raise ValueError("authoritative_lineage_receipt_mismatch")
    seed416_registration = json.loads(
        lineage_files["seed416_registration_v3_sha256"].read_text()
    )
    seed416_training = json.loads(
        lineage_files["seed416_training_results_sha256"].read_text()
    )
    seed422_registration = json.loads(
        lineage_files["seed422_registration_sha256"].read_text()
    )
    seed422_training = json.loads(
        lineage_files["seed422_training_results_sha256"].read_text()
    )
    if (
        seed416_registration["seed455_initialization"]["sha256"]
        != manifest["models"]["initializer"]["sha256"]
        or seed422_registration["training"]["initializer"]["sha256"]
        != manifest["models"]["initializer"]["sha256"]
        or seed416_training["lanes"]["A"]["epochs"]["4"]
        != manifest["models"]["e4"]["sha256"]
        or seed422_registration["training"]["expected_A_E4_checkpoint_sha256"]
        != manifest["models"]["e4"]["sha256"]
        or seed422_training["lanes"]["A"]["epochs"]["4"]
        != manifest["models"]["e4"]["sha256"]
    ):
        raise ValueError("authoritative_checkpoint_lineage_mismatch")
    for relative, binding in manifest["execution_sources"].items():
        source = root / relative
        snapshot = data / "seed427-execution-source-snapshots" / source.name
        if not source.is_file() or sha256(source.read_bytes()) != binding["sha256"]:
            raise ValueError(f"evaluation_source_binding_mismatch:{relative}")
        if (
            not snapshot.is_file()
            or sha256(snapshot.read_bytes()) != binding["snapshot_sha256"]
        ):
            raise ValueError(f"evaluation_source_snapshot_mismatch:{relative}")

    published_membership_path = data / "seed427-validation-membership.jsonl.gz"
    membership_bytes = published_membership_path.read_bytes()
    if sha256(membership_bytes) != manifest["membership"]["sha256"]:
        raise ValueError("membership_binding_mismatch")
    original = read_evidence(data / "row-accounting.jsonl.gz")
    rebuilt, counts = construct(original)
    membership = read_evidence(published_membership_path)
    if rebuilt != membership:
        raise ValueError("membership_reconstruction_mismatch")
    if (
        counts["all"]["weighted_positions"] != 14946
        or counts["seen"]["weighted_positions"] != 4240
        or counts["unseen"]["weighted_positions"] != 10706
    ):
        raise ValueError("membership_count_reconciliation_mismatch")
    if any(
        row["canonical_seen_in_train"] != row["input_seen_in_train"]
        for row in membership
    ):
        raise ValueError("canonical_input_exclusion_disagreement")
    if (
        counts["input_identity_definition_difference"][
            "validation_positions_canonical_unseen_but_input_seen"
        ]
        or counts["input_identity_definition_difference"][
            "validation_positions_canonical_seen_but_input_unseen"
        ]
    ):
        raise ValueError("identity_definition_difference_unreported")

    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    target_by_compact: dict[int, dict[str, Any]] = {}
    compact_index = 0
    for source in registration["replays"]:
        source_path = data / "sources" / f"{source['name']}.jsonl.gz"
        with gzip.open(source_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                source_row = json.loads(line)
                target_by_compact[compact_index] = {
                    "input_identity": struct.pack(
                        "<" + "f" * len(source_row["state"]), *source_row["state"]
                    ).hex(),
                    "policy": [
                        struct.unpack("<f", struct.pack("<f", float(value)))[0]
                        for value in source_row["policy"]
                    ],
                    "value": struct.unpack(
                        "<f", struct.pack("<f", float(source_row["value"]))
                    )[0],
                }
                compact_index += 1

    predictions: dict[str, list[dict[str, Any]]] = {"initializer": [], "e4": []}
    with gzip.open(prediction_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            model = item.pop("model")
            if model not in predictions:
                raise ValueError("prediction_model_identity_mismatch")
            expected_loss = row_losses(
                item["logits"],
                item["value_prediction"],
                item["target_policy"],
                item["target_value"],
                item["legal_mask"],
                item["policy_weight"],
            )
            if expected_loss != item["losses"]:
                raise ValueError("prediction_loss_reconstruction_mismatch")
            predictions[model].append(item)
    expected_rows = len(membership)
    for model, rows in predictions.items():
        if len(rows) != expected_rows:
            raise ValueError(f"prediction_row_count_mismatch:{model}")
        if [row["membership"] for row in rows] != membership:
            raise ValueError(f"prediction_membership_mismatch:{model}")
        for row in rows:
            target = target_by_compact[row["membership"]["compact_row"]]
            if (
                row["target_policy"] != target["policy"]
                or row["target_value"] != target["value"]
            ):
                raise ValueError("prediction_target_binding_mismatch")
            if row["policy_weight"] != 1.0:
                raise ValueError("prediction_weight_binding_mismatch")
            if row["policy_weight"] != row["losses"]["policy_weight"]:
                raise ValueError("prediction_weight_binding_mismatch")
            if row["membership"]["input_identity"] != target["input_identity"]:
                raise ValueError("prediction_input_identity_binding_mismatch")
            if len(row["legal_mask"]) != 6 or not any(row["legal_mask"]):
                raise ValueError("prediction_legal_mask_invalid")
            if any(
                not legal and abs(target) > 1e-6
                for legal, target in zip(row["legal_mask"], row["target_policy"])
            ):
                raise ValueError("prediction_target_legality_mismatch")
    recomputed = {name: aggregate(rows) for name, rows in predictions.items()}
    if recomputed != receipt["results"]:
        raise ValueError("published_aggregate_mismatch")
    differences = {}
    for scope, scope_metrics in recomputed["e4"].items():
        differences[scope] = {
            aggregation: {
                metric: scope_metrics[aggregation][metric]
                - recomputed["initializer"][scope][aggregation][metric]
                for metric in ("policy_loss", "value_loss", "total_loss")
            }
            for aggregation in ("exposure_weighted", "equal_canonical_identity")
        }
    if differences != receipt["e4_minus_initializer"]:
        raise ValueError("published_model_comparison_mismatch")
    if receipt["primary_comparison"]["differences"] != differences["unseen/>32"]:
        raise ValueError("primary_comparison_binding_mismatch")
    if receipt["checkpoint_sha256_before"] != receipt["checkpoint_sha256_after"]:
        raise ValueError("checkpoint_immutability_receipt_mismatch")
    verification_receipt_path = data / "seed427-evaluation-verification-receipt.json"
    if not verification_receipt_path.is_file():
        raise ValueError("missing_evaluation_verification_receipt")
    verification_receipt = json.loads(verification_receipt_path.read_text())
    expected_verifier_hash = sha256(
        (root / "ml/alphazero_lite/verify_seed427_evaluation.py").read_bytes()
    )
    if (
        verification_receipt.get("manifest_sha256") != sha256(manifest_bytes)
        or verification_receipt.get("prediction_evidence_sha256")
        != sha256(prediction_path.read_bytes())
        or verification_receipt.get("results_json_sha256")
        != sha256(result_path.read_bytes())
        or verification_receipt.get("verifier_sha256") != expected_verifier_hash
    ):
        raise ValueError("evaluation_verification_receipt_binding_mismatch")
    return {
        "status": "verified",
        "publication": publication,
        "membership_positions": expected_rows,
        "prediction_rows_per_model": expected_rows,
        "models": sorted(predictions),
        "manifest_sha256": sha256(manifest_bytes),
        "prediction_evidence_sha256": sha256(prediction_path.read_bytes()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args().root), sort_keys=True))
