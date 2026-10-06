"""Independent semantic checks layered over the immutable seed427 verifier."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.seed427_validation_metrics import aggregate, row_losses  # noqa: E402
from ml.alphazero_lite.verify_seed427_evaluation import verify as verify_seed427  # noqa: E402

MODELS = {"initializer", "e4"}
PARITY_METRICS = ("policy_loss", "value_loss", "total_loss")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(actual: float, expected: float, absolute: float, relative: float) -> bool:
    return abs(actual - expected) <= absolute + relative * abs(expected)


def _decode_kalah_v3(features: list[float]) -> dict[str, Any]:
    """Decode the invertible state prefix without importing inference code."""
    if len(features) != 27:
        raise ValueError("registered_encoded_state_shape_mismatch")

    def stones(start: int, stop: int) -> list[int]:
        result = []
        for index in range(start, stop):
            scaled = float(features[index]) * 48.0
            rounded = round(scaled)
            if abs(scaled - rounded) > 1e-5 or rounded < 0:
                raise ValueError("registered_encoded_state_invalid")
            result.append(int(rounded))
        return result

    current = int(round(float(features[14])))
    if current not in (0, 1) or abs(float(features[14]) - current) > 1e-5:
        raise ValueError("registered_encoded_state_invalid")
    return {
        "player_pits": stones(0, 6),
        "opponent_pits": stones(6, 12),
        "player_store": stones(12, 13)[0],
        "opponent_store": stones(13, 14)[0],
        "current_player": current,
    }


def verify(root: Path) -> dict[str, Any]:
    base = verify_seed427(root)
    data = root / "docs/data/seed426-canonical-overlap"
    manifest = json.loads((data / "seed427-evaluation-manifest.json").read_text())
    results = json.loads((data / "seed427-evaluation-results.json").read_text())
    authoritative = {
        "initializer": manifest["models"]["initializer"]["sha256"],
        "e4": manifest["models"]["e4"]["sha256"],
    }
    for key in ("checkpoint_sha256_before", "checkpoint_sha256_after"):
        if set(results[key]) != MODELS or results[key] != authoritative:
            raise ValueError("authoritative_checkpoint_hash_mismatch")

    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    training = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/training-results.json"
        ).read_text()
    )
    if manifest["architecture"] != registration["training"]["architecture"]:
        raise ValueError("authoritative_architecture_mismatch")
    expected_targets = manifest["targets"]
    if (
        expected_targets["policy_target_mode"] != "sharpened"
        or expected_targets["exact_root_policy_loss_weight"] != 1.0
    ):
        raise ValueError("authoritative_target_settings_mismatch")
    authoritative_value_modes = {
        replay["name"]: replay["value_target_mode"]
        for replay in registration["replays"]
    }
    if expected_targets["value_target_modes_by_source"] != authoritative_value_modes:
        raise ValueError("authoritative_target_settings_mismatch")
    if (
        manifest["losses"]
        != {
            "equal_identity": "arithmetic mean row loss within each canonical identity, then arithmetic mean over canonical identities; row multiplicities retained within identity",
            "policy": "legal-masked log_softmax cross entropy; numerator=sum(row CE*policy weight), denominator=sum(policy weights)",
            "total": "policy + 0.3 * value",
            "value": "smooth_l1_loss beta=1; numerator=sum(row Huber), denominator=weighted positions",
        }
        or registration["training"]["value_loss_weight"] != 0.3
        or registration["training"]["huber_delta"] != 1
    ):
        raise ValueError("authoritative_loss_definition_mismatch")
    tolerances = manifest["tolerances"]
    if tolerances["absolute"] != 1e-6 or tolerances["relative"] != 1e-5:
        raise ValueError("authoritative_tolerance_mismatch")
    expected_parity = {
        "policy_loss": training["lanes"]["A"]["history"][3]["validation_policy_loss"],
        "value_loss": training["lanes"]["A"]["history"][3]["validation_value_loss"],
        "total_loss": training["lanes"]["A"]["history"][3]["validation_total_loss"],
    }
    for metric, value in expected_parity.items():
        if tolerances["e4_full_validation_expected"][metric] != value:
            raise ValueError("authoritative_parity_expectation_mismatch")

    predictions: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    with gzip.open(
        data / "seed427-prediction-evidence.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        for line in stream:
            row = json.loads(line)
            model = row.pop("model")
            if model not in MODELS:
                raise ValueError("prediction_model_identity_mismatch")
            numeric = (
                row["logits"]
                + [row["value_prediction"]]
                + row["target_policy"]
                + [row["target_value"], row["policy_weight"]]
            )
            if not all(math.isfinite(float(value)) for value in numeric):
                raise ValueError("nonfinite_prediction_or_target")
            if not -1.0 <= row["value_prediction"] <= 1.0:
                raise ValueError("value_prediction_out_of_range")
            membership = row["membership"]
            encoded = list(
                struct.unpack("<27f", bytes.fromhex(membership["input_identity"]))
            )
            state = _decode_kalah_v3(encoded)
            game = KalahGame.from_state(state)
            expected_mask = [int(index in game.possible_moves()) for index in range(6)]
            if row["legal_mask"] != expected_mask:
                raise ValueError("prediction_legal_mask_rule_mismatch")
            if row["losses"] != row_losses(
                row["logits"],
                row["value_prediction"],
                row["target_policy"],
                row["target_value"],
                row["legal_mask"],
                row["policy_weight"],
            ):
                raise ValueError("prediction_loss_reconstruction_mismatch")
            if not all(math.isfinite(float(value)) for value in row["losses"].values()):
                raise ValueError("nonfinite_prediction_loss")
            predictions[model].append(row)
    for model, rows in predictions.items():
        compact_outputs: dict[int, tuple[Any, ...]] = {}
        for row in rows:
            compact = row["membership"]["compact_row"]
            output = (tuple(row["logits"]), row["value_prediction"])
            if compact in compact_outputs and compact_outputs[compact] != output:
                raise ValueError("weighted_copy_output_mismatch")
            compact_outputs[compact] = output

    actual = aggregate(predictions["e4"])["all/overall"]["exposure_weighted"]
    parity = results["e4_full_validation_parity"]
    for metric, expected in expected_parity.items():
        calculated = actual[metric]
        check = parity[metric]
        passed = _close(
            calculated, expected, tolerances["absolute"], tolerances["relative"]
        )
        if (
            check
            != {
                "actual": check.get("actual"),
                "expected": check.get("expected"),
                "passed": check.get("passed"),
            }
            or check["actual"] != calculated
            or check["expected"] != expected
            or check["passed"] is not passed
            or not passed
        ):
            raise ValueError("full_validation_parity_semantic_mismatch")
    if manifest["membership"]["counts"] != {
        "all": 14946,
        "seen": 4240,
        "unseen": 10706,
    }:
        raise ValueError("authoritative_membership_count_mismatch")
    observed_counts = {
        "all": len(predictions["e4"]),
        "seen": sum(row["membership"]["subset"] == "seen" for row in predictions["e4"]),
        "unseen": sum(
            row["membership"]["subset"] == "unseen" for row in predictions["e4"]
        ),
    }
    if manifest["membership"]["counts"] != observed_counts:
        raise ValueError("implemented_membership_count_mismatch")
    receipt_path = data / "seed428-semantic-verification-receipt.json"
    if not receipt_path.is_file():
        raise ValueError("missing_seed428_semantic_receipt")
    receipt = json.loads(receipt_path.read_text())
    bindings = {
        "manifest_sha256": _sha(data / "seed427-evaluation-manifest.json"),
        "results_sha256": _sha(data / "seed427-evaluation-results.json"),
        "prediction_evidence_sha256": _sha(
            data / "seed427-prediction-evidence.jsonl.gz"
        ),
        "verifier_sha256": _sha(
            root / "ml/alphazero_lite/verify_seed428_supplemental.py"
        ),
    }
    if (
        any(receipt.get(key) != value for key, value in bindings.items())
        or receipt.get("status") != "verified"
    ):
        raise ValueError("seed428_semantic_receipt_binding_mismatch")
    return {
        "status": "verified",
        "checks": "semantic seed427 verification passed",
        "seed427": base,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args().root), sort_keys=True))
