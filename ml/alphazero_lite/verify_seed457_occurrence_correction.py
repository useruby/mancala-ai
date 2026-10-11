"""Read-only, post-execution seed457 reconciliation preserving row occurrences.

This entry point verifies the archived execution; it never calls either arm
runner or publisher calculation routines. It writes only to a new correction
directory and assembles split archives under a temporary repo-local directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite import seed435_adam_direction as seed435
from ml.alphazero_lite import verify_seed453_fresh_policy_projection as verify453
from ml.alphazero_lite import verify_seed457_fresh_subgroup_projection as verify457
from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed457-occurrence-verification-correction")
PUBLICATION = Path("docs/data/seed457-fresh-subgroup-projection")
PARTS = Path("docs/data/seed457-fresh-subgroup-projection-parts")
T_LOGICAL_SHA256 = "d7a09d9c0d611f4921ffdce76b1967dc9393e987553d66b476a5cd4bd8365283"
T_PART_SHA256 = {
    "step-tensors-T.npz.part-00": "66f7d82eaa39790d5cdfc53987e922b3a69861ea3c74138b7017a828854e0e85",
    "step-tensors-T.npz.part-01": "9403b9c9648052d308cd73b4d2fe7779dd72ddceeac210fa68d9f38c650ebdb5",
}
REGISTRATION_SHA256 = "58034469262878a7de99658b9b09c71d59369a1960c69d2e20f83b5f31c5a8a8"
ORIGINAL_RECEIPT_SHA256 = (
    "32c968cc5bb17d3fc8e749016d586a969d0ed92427cf562dbb5546e74e31c6ef"
)
DIAGNOSIS_ARRAY_SHA256 = (
    "ba0469cf8c99cedc533513b9ff58460acb0aca2e9f6277291d57ee6f7a30fef8"
)
INITIALIZER_SHA256 = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"

_PRE_OCCURRENCE_VALUES: dict[tuple[int, ...], np.ndarray] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_occurrence_order(
    actual: np.ndarray, expected: np.ndarray, label: str
) -> None:
    """Reject reordered or deduplicated occurrence vectors."""
    if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise ValueError(f"occurrence_order_mismatch:{label}")


def require_tensor_semantics(
    actual: np.ndarray, expected: np.ndarray, label: str
) -> None:
    """Require exact reconstruction equality for an archived tensor."""
    if not np.array_equal(actual, expected):
        raise ValueError(f"semantic_tensor_mismatch:{label}")


def require_trial_measurement(
    reconstructed: float, recorded: float, name: str, scale: float
) -> None:
    """Apply the original strict serialized-measurement equality threshold."""
    if not np.isclose(reconstructed, recorded, atol=2e-12, rtol=0):
        raise ValueError(f"trial_measurement_mismatch:{name}:{scale}")


def _portable_outputs(
    model: torch.nn.Module, x: np.ndarray, rows: np.ndarray
) -> tuple[torch.Tensor, np.ndarray]:
    """Return compact policy references and cache occurrence-ordered values.

    Policy values intentionally use the original compact-row scatter semantics
    used to create the runner's pre-reference. Value outputs are separately
    retained in occurrence order, matching the runner's value movement path.
    """
    compact_logp = torch.full(
        (len(x), train.POLICY_SIZE), -torch.inf, dtype=torch.float64
    )
    value_occurrences: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), 512):
            ids = rows[start : start + 512]
            logits, predicted = model(torch.from_numpy(x[ids]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[ids])
            ).bool()
            logp = torch.log_softmax(
                logits.double().masked_fill(~legal, -torch.inf), dim=1
            )
            compact_logp[ids] = logp
            value_occurrences.append(predicted.reshape(-1).cpu().numpy().copy())
    _PRE_OCCURRENCE_VALUES[tuple(map(int, rows))] = np.concatenate(
        value_occurrences
    ).astype(np.float32, copy=False)
    value_rows = np.empty(len(x), dtype=np.float32)
    offset = 0
    for start in range(0, len(rows), 512):
        ids = rows[start : start + 512]
        values = value_occurrences[offset]
        offset += 1
        value_rows[ids] = values
    return compact_logp, value_rows


def occurrence_trial_measure(
    model: torch.nn.Module,
    x: np.ndarray,
    rows: np.ndarray,
    pre_logp_compact: torch.Tensor,
    pre_values_compact: np.ndarray,
) -> tuple[float, float]:
    """Rebuild frozen runner KL/value measurements in occurrence order.

    ``pre_logp_compact`` is deliberately gathered through compact identities,
    retaining the runner's pre-policy lookup semantics. Trial logits, masks,
    and values remain ordered by each row occurrence throughout reduction.
    """
    del pre_values_compact  # The runner used pre-update occurrence outputs.
    key = tuple(map(int, rows))
    if key not in _PRE_OCCURRENCE_VALUES:
        raise ValueError("occurrence_pre_value_reference_missing")
    before_values = _PRE_OCCURRENCE_VALUES[key]
    if before_values.shape != (len(rows),):
        raise ValueError("occurrence_pre_value_shape_invalid")
    weights = np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
    total_kl = 0.0
    after_chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 512):
            ids = rows[start : start + 512]
            logits, predicted = model(torch.from_numpy(x[ids]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[ids])
            ).bool()
            trial_logp = torch.log_softmax(
                logits.double().masked_fill(~legal, -torch.inf), dim=1
            )
            pre_log = pre_logp_compact[ids].to(torch.float64)
            pre_prob = pre_log.exp().masked_fill(~legal, 0.0)
            kl_rows = (pre_prob * (pre_log - trial_logp).masked_fill(~legal, 0.0)).sum(
                dim=1
            )
            stop = start + len(ids)
            total_kl += float(
                (kl_rows * torch.from_numpy(weights[start:stop]).double()).sum()
            )
            after_chunks.append(predicted.reshape(-1).cpu().numpy().copy())
    after_values = np.concatenate(after_chunks).astype(np.float32, copy=False)
    delta = after_values.astype(np.float64) - before_values.astype(np.float64)
    movement = float(np.sum(np.square(delta) * weights, dtype=np.float64))
    return total_kl, movement


def _patch_measurement_path():
    """Temporarily route the shared independent replay to occurrence arithmetic."""
    original_outputs, original_measure = verify457._outputs, verify457._measure_trial
    original_array_check = verify457._require_array
    verify457._outputs = _portable_outputs
    verify457._measure_trial = occurrence_trial_measure
    verify457._require_array = require_tensor_semantics

    def restore() -> None:
        verify457._outputs = original_outputs
        verify457._measure_trial = original_measure
        verify457._require_array = original_array_check

    return restore


def _snapshot(
    root: Path, receipt: dict[str, Any], registration: dict[str, Any]
) -> dict[str, tuple[int, int, str]]:
    paths = [root / rel for rel in receipt["files_sha256"]]
    paths.extend(root / PARTS / name for name in T_PART_SHA256)
    paths.extend(root / rel for rel in registration["source_sha256"])
    paths.extend(root / rel for rel in registration["input_sha256"])
    snapshot = {}
    for path in paths:
        if path.is_file():
            snapshot[path.relative_to(root).as_posix()] = (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                sha256_file(path),
            )
    return snapshot


def _verify_bindings(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    publication = root / PUBLICATION
    registration_path = publication / "registration.json"
    receipt_path = publication / "receipt.json"
    evidence_path = publication / "evidence.json"
    registration = json.loads(registration_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    if sha256_file(registration_path) != REGISTRATION_SHA256:
        raise ValueError("original_registration_hash_mismatch")
    if sha256_file(receipt_path) != ORIGINAL_RECEIPT_SHA256:
        raise ValueError("original_receipt_hash_mismatch")
    if evidence.get("status") != "real_execution":
        raise ValueError("real_evidence_status_invalid")
    if registration.get("status") != "frozen_before_either_arm":
        raise ValueError("registration_status_invalid")
    for section in ("source_sha256", "input_sha256"):
        identities = registration.get(section)
        if not isinstance(identities, dict) or not identities:
            raise ValueError(f"registration_bindings_missing:{section}")
        for rel, digest in identities.items():
            path = root / rel
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"frozen_binding_mismatch:{rel}")
            if section == "source_sha256":
                frozen = publication / "source-snapshots" / rel
                if not frozen.is_file() or sha256_file(frozen) != digest:
                    raise ValueError(f"frozen_source_snapshot_mismatch:{rel}")
    expected_files = set(receipt.get("files_sha256", {}))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in publication.rglob("*")
        if path.is_file() and path.name != "receipt.json"
    }
    logical_rel = (PUBLICATION / "step-tensors-T.npz").as_posix()
    if expected_files - {logical_rel} != actual_files:
        raise ValueError("original_receipt_inventory_incomplete")
    for rel, digest in receipt["files_sha256"].items():
        if rel == logical_rel:
            if digest != T_LOGICAL_SHA256:
                raise ValueError("registered_logical_T_hash_invalid")
            continue
        path = root / rel
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"original_receipt_hash_mismatch:{rel}")
    if receipt.get("input_sha256") != registration.get("input_sha256"):
        raise ValueError("receipt_input_bindings_mismatch")
    if receipt.get("source_sha256") != registration.get("source_sha256"):
        raise ValueError("receipt_source_bindings_mismatch")
    return registration, receipt, evidence


def _assemble_t(root: Path, temporary: Path) -> tuple[Path, dict[str, Any]]:
    parts_dir = root / PARTS
    target = temporary / "step-tensors-T.logical.npz"
    part_receipt = {}
    digest = hashlib.sha256()
    length = 0
    with target.open("wb") as destination:
        for name, expected in T_PART_SHA256.items():
            part = parts_dir / name
            if not part.is_file() or sha256_file(part) != expected:
                raise ValueError(f"archive_part_hash_mismatch:{name}")
            part_receipt[name] = {"bytes": part.stat().st_size, "sha256": expected}
            with part.open("rb") as source:
                while block := source.read(1024 * 1024):
                    destination.write(block)
                    digest.update(block)
                    length += len(block)
    if length != 107_888_076 or digest.hexdigest() != T_LOGICAL_SHA256:
        raise ValueError("logical_T_archive_identity_mismatch")
    return target, {
        "bytes": length,
        "sha256": digest.hexdigest(),
        "parts": part_receipt,
    }


def _reconstruct(
    root: Path, registration: dict[str, Any], evidence: dict[str, Any], t_path: Path
) -> dict[str, Any]:
    seed442.configure_root(root)
    from ml.alphazero_lite import seed457_fresh_subgroup_projection as seed457_frozen

    if not set(seed457_frozen.REQUIRED_INPUTS).issubset(registration["input_sha256"]):
        raise ValueError("required_input_binding_inventory_incomplete")
    if not set(seed457_frozen.REQUIRED_SOURCES).issubset(registration["source_sha256"]):
        raise ValueError("required_source_binding_inventory_incomplete")
    protocol = {
        "steps": 16,
        "batch_size": 512,
        "arms": {
            "C": "seed453-B aggregate gradient projection",
            "T": "nearest joint eight-subgroup projection",
        },
        "kl_cap": 0.005,
        "kl_tolerance": 1e-10,
        "value_cap": 0.0001,
        "value_tolerance": 1e-12,
        "dot_allowance": 1e-7,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "initial_moments": "zero",
        "gradient_clip_norm": 1.0,
        "projection_arithmetic": "float64",
        "parameter_writeback": "float32",
        "rejection": "restore_parameters_retain_advanced_adam_state",
    }
    if registration.get("protocol") != protocol:
        raise ValueError("production_protocol_mismatch")
    if registration.get("model") != {
        "architecture": "residual_v3",
        "shape": [96, 3],
        "input_size": 27,
    }:
        raise ValueError("production_model_mismatch")
    if registration.get("steps") != 16 or registration.get("scales") != list(
        verify453.SCALES
    ):
        raise ValueError("production_trajectory_shape_mismatch")
    arrays = verify453._portable_inputs(root)
    x, _policy, _value, replay, _coefficients, train_positions, *_rest, sources = arrays
    full_guard = json.loads((root / PUBLICATION / "full-guard.json").read_text())
    fresh_guard = json.loads((root / PUBLICATION / "fresh-guard.json").read_text())
    if len(full_guard) != 2048 or len(fresh_guard) != 1399:
        raise ValueError("guard_population_invalid")
    shards = verify457.partition(fresh_guard)
    if (
        registration.get("shards") != shards
        or registration.get("fresh_guard") != fresh_guard
        or registration.get("full_guard") != full_guard
    ):
        raise ValueError("frozen_partition_or_guard_mismatch")
    training_rows = set(map(int, replay[train_positions]))
    membership = seed435.membership_rows()
    unseen = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    exact_eval = {r["input_identity"] for r in unseen}
    canonical_eval = {r["canonical_identity"] for r in unseen}
    for row in full_guard:
        compact = int(row["compact_row"])
        identity = x[compact].astype("<f4", copy=False).tobytes().hex()
        if compact not in training_rows or identity != row["exact_input_identity"]:
            raise ValueError(f"guard_training_or_identity_invalid:{compact}")
        if identity in exact_eval or row["canonical_identity"] in canonical_eval:
            raise ValueError(f"guard_evaluation_overlap:{compact}")
    for shard in shards:
        for row in shard:
            if sources[int(row["compact_row"])].get("source") != "fresh":
                raise ValueError("fresh_guard_provenance_invalid")

    batches = np.asarray(registration["seed442_first_16_positions"], dtype=np.int64)
    if batches.shape != (8192,) or len(set(map(int, batches))) != 8192:
        raise ValueError("frozen_minibatches_invalid")
    seed442_reg = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/registration.json").read_text()
    )
    if batches.tolist() != seed442_reg["first_16_minibatch_positions"]:
        raise ValueError("seed442_minibatch_protocol_mismatch")
    historical_evidence = json.loads(
        (root / "docs/data/seed453-fresh-policy-projection/evidence.json").read_text()
    )
    historical_receipt = json.loads(
        (root / "docs/data/seed453-fresh-policy-projection/receipt.json").read_text()
    )
    historical_evidence_path = (
        root / "docs/data/seed453-fresh-policy-projection/evidence.json"
    )
    if historical_receipt.get("files_sha256", {}).get(
        "docs/data/seed453-fresh-policy-projection/evidence.json"
    ) != sha256_file(historical_evidence_path):
        raise ValueError("seed453_historical_evidence_receipt_mismatch")
    historical_archive = np.load(
        root / "docs/data/seed453-fresh-policy-projection/step-tensors-B.npz",
        allow_pickle=False,
    )
    archives = {
        "C": np.load(root / PUBLICATION / "step-tensors-C.npz", allow_pickle=False),
        "T": np.load(t_path, allow_pickle=False),
    }
    predictions = np.load(
        root / PUBLICATION / "endpoint-predictions.npz", allow_pickle=False
    )
    initializer_path = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    if sha256_file(initializer_path) != INITIALIZER_SHA256:
        raise ValueError("initializer_hash_invalid")
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, initializer_path)
    initializer = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    full_rows = np.asarray([int(r["compact_row"]) for r in full_guard], dtype=np.int64)
    fresh_rows = np.asarray(
        [int(r["compact_row"]) for r in fresh_guard], dtype=np.int64
    )
    subgroup_rows = [
        np.asarray([int(r["compact_row"]) for r in s], dtype=np.int64) for s in shards
    ]
    restore = _patch_measurement_path()
    try:
        expected_occurrences = replay[train_positions][batches]
        for arm in ("C", "T"):
            records = evidence["arms"][arm]["steps"]
            if len(records) != 16:
                raise ValueError(f"trajectory_record_count_invalid:{arm}")
            for index, record in enumerate(records):
                expected_rows = expected_occurrences[index * 512 : (index + 1) * 512]
                actual_rows = np.asarray(record.get("compact_rows"), dtype=np.int64)
                require_occurrence_order(
                    actual_rows, expected_rows, f"{arm}:{index + 1}"
                )
        trajectories = {
            arm: verify457.reconstruct_trajectory(
                model=model,
                initializer=initializer,
                arrays=arrays,
                arm=arm,
                records=evidence["arms"][arm]["steps"],
                archive=archives[arm],
                batches=batches,
                full_guard_rows=full_rows,
                fresh_guard_rows=fresh_rows,
                subgroup_rows=subgroup_rows,
            )
            for arm in ("C", "T")
        }
        control = verify457.compare_c_reference(
            evidence["arms"]["C"]["steps"],
            archives["C"],
            historical_evidence["arms"]["B"]["steps"],
            historical_archive,
        )
        if not control["matches"]:
            raise ValueError(
                f"c_control_reproduction_mismatch:{control['mismatches'][:8]}"
            )
        if control != evidence.get("control_reproduction"):
            raise ValueError("control_reproduction_evidence_mismatch")
        recorded_metrics = evidence["metrics"]
        endpoint = verify457.reconstruct_endpoint_evidence(
            model=model,
            arrays=arrays,
            membership=membership,
            initializer=initializer,
            trajectories=trajectories,
            prediction_archive=predictions,
            recorded_metrics=recorded_metrics,
        )
        activation = verify457.reconstruct_activation(
            evidence["arms"]["T"]["steps"], trajectories["T"]["verified_steps"]
        )
        if activation != evidence.get("activation"):
            raise ValueError("activation_semantic_mismatch")
        historical_metrics = json.loads(
            (root / "docs/data/seed442-kl-capped-adam-screen/evidence.json").read_text()
        )["metrics"]["A"]
        if historical_metrics != recorded_metrics["seed442_A"]:
            raise ValueError("seed442_comparator_historical_mismatch")
        if registration.get("historical_comparator") != historical_metrics:
            raise ValueError("historical_comparator_registration_mismatch")
        gates, classification = verify457.finalize_reconstruction(
            endpoint=endpoint,
            evidence=evidence,
            activation=activation,
            historical_metrics=historical_metrics,
            verified_steps=trajectories["T"]["verified_steps"],
        )
        if gates != evidence.get("gates"):
            raise ValueError("decision_gates_semantic_mismatch")
        return {
            "trajectories": trajectories,
            "control_reproduction": control,
            "endpoint": endpoint,
            "activation": activation,
            "gates": gates,
            "classification": classification,
            "historical_metrics": historical_metrics,
        }
    finally:
        restore()
        for archive in archives.values():
            archive.close()
        historical_archive.close()
        predictions.close()


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def validate_reconciliation_semantics(report: dict[str, Any]) -> None:
    """Reject semantically altered summaries independently of their hashes."""
    if report.get("verification_timing") != "post_execution_correction":
        raise ValueError("correction_timing_invalid")
    if report.get("coverage") != {
        "steps_per_arm": {"C": 16, "T": 16},
        "trials_per_arm": {"C": 128, "T": 128},
        "trials_total": 256,
    }:
        raise ValueError("correction_coverage_invalid")
    control = report.get("control_reproduction", {})
    if control.get("matches") is not True or control.get("mismatches") != []:
        raise ValueError("correction_control_reproduction_invalid")
    metrics = report.get("metrics", {})
    effects = report.get("effects_T_minus_C", {})
    for weighting in ("exposure_weighted", "equal_input"):
        expected = {
            metric: metrics["T"][weighting][metric] - metrics["C"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
        if effects.get(weighting) != expected:
            raise ValueError(f"correction_effect_mismatch:{weighting}")
        expected_fresh = {
            metric: metrics["T"]["fresh"][weighting][metric]
            - metrics["C"]["fresh"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
        if effects.get(f"fresh_{weighting}") != expected_fresh:
            raise ValueError(f"correction_effect_mismatch:fresh_{weighting}")
    expected_objective = {
        "T_minus_C": metrics["T"]["full_training_objective"]
        - metrics["C"]["full_training_objective"]
    }
    if effects.get("full_training_objective") != expected_objective:
        raise ValueError("correction_effect_mismatch:full_training_objective")
    gates = report.get("decision", {}).get("gates", {})
    if set(gates) != {
        "pooled_ce_vs_seed442_A_exposure_weighted",
        "pooled_ce_vs_seed442_A_equal_input",
        "pooled_ce_vs_initializer_exposure_weighted",
        "pooled_ce_vs_initializer_equal_input",
        "pooled_mse_exposure_weighted",
        "pooled_mse_equal_input",
        "fresh_ce_exposure_weighted",
        "fresh_ce_equal_input",
        "training_objective_decreases",
        "no_nonzero_rejection",
        "additional_subgroup_protection",
    } or any(type(value) is not bool for value in gates.values()):
        raise ValueError("correction_gate_inventory_invalid")
    active = report.get("activation", {}).get("activated") is True
    if gates["additional_subgroup_protection"] != active:
        raise ValueError("correction_activation_gate_mismatch")
    expected_gates: dict[str, bool] = {}
    for weighting in ("exposure_weighted", "equal_input"):
        expected_gates[f"pooled_ce_vs_seed442_A_{weighting}"] = (
            metrics["T"][weighting]["policy_ce"]
            <= metrics["seed442_A"][weighting]["policy_ce"] - 0.01
        )
        expected_gates[f"pooled_ce_vs_initializer_{weighting}"] = (
            metrics["T"][weighting]["policy_ce"]
            <= metrics["initializer"][weighting]["policy_ce"] - 0.005
        )
        expected_gates[f"pooled_mse_{weighting}"] = (
            metrics["T"][weighting]["value_mse"]
            <= metrics["initializer"][weighting]["value_mse"] + 0.002
        )
        expected_gates[f"fresh_ce_{weighting}"] = (
            metrics["T"]["fresh"][weighting]["policy_ce"]
            <= metrics["initializer"]["fresh"][weighting]["policy_ce"] + 2e-6
        )
    expected_gates["training_objective_decreases"] = (
        metrics["T"]["full_training_objective"]
        < metrics["initializer"]["full_training_objective"]
    )
    expected_gates["no_nonzero_rejection"] = not any(
        step["raw_proposal_norm"] > 0 and step["rejected"]
        for step in report["arm_trajectories"]["T"]["steps"]
    )
    expected_gates["additional_subgroup_protection"] = active
    if gates != expected_gates:
        raise ValueError("correction_gate_semantic_mismatch")
    expected_classification = (
        "subgroup_projection_inactive_no_followup"
        if not active
        else "advance_to_separately_preregistered_strength_experiment"
        if all(gates.values())
        else "close_fresh_subgroup_projection_branch"
    )
    if report["decision"].get("classification") != expected_classification:
        raise ValueError("correction_classification_semantic_mismatch")
    for arm in ("C", "T"):
        trajectory = report.get("arm_trajectories", {}).get(arm, {})
        steps = trajectory.get("steps", [])
        if (
            len(steps) != 16
            or trajectory.get("verified_trials") != 128
            or any(step.get("verified_trials") != 8 for step in steps)
        ):
            raise ValueError(f"correction_trajectory_summary_invalid:{arm}")


def verify_correction_receipt(output_dir: Path) -> dict[str, Any]:
    """Validate ordinary receipt hashes and independently re-evaluate meaning."""
    receipt = json.loads((output_dir / "receipt.json").read_text())
    for name, expected in receipt.get("correction_files_sha256", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"correction_file_hash_mismatch:{name}")
    report = json.loads((output_dir / "reconciliation.json").read_text())
    validate_reconciliation_semantics(report)
    if receipt.get("classification") != report["decision"]["classification"]:
        raise ValueError("correction_receipt_classification_mismatch")
    if "original_registration_sha256" in receipt:
        root = output_dir.resolve().parents[2]
        validate_correction_receipt_bindings(root, receipt)
        if receipt.get("archive_parts") != report["logical_T_archive"]["parts"]:
            raise ValueError("correction_receipt_archive_parts_mismatch")
    return report


def verify(
    root: Path, *, output_dir: Path | None = None, publish: bool = True
) -> dict[str, Any]:
    """Verify archived execution and optionally publish append-only correction."""
    root = root.resolve()
    publication = root / PUBLICATION
    registration, receipt, evidence = _verify_bindings(root)
    before = _snapshot(root, receipt, registration)
    scratch_root = root / ".tmp"
    scratch_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="seed457-occurrence-", dir=scratch_root
    ) as scratch:
        t_path, logical_identity = _assemble_t(root, Path(scratch))
        reconstruction = _reconstruct(root, registration, evidence, t_path)
        if _snapshot(root, receipt, registration) != before:
            raise ValueError("original_publication_mutated_during_verification")

    trajectories = reconstruction["trajectories"]
    trial_count = sum(
        facts["verified_trials"]
        for trajectory in trajectories.values()
        for facts in trajectory["verified_steps"]
    )
    if trial_count != 256 or any(
        result["steps"] != 16 for result in trajectories.values()
    ):
        raise ValueError("complete_trajectory_coverage_invalid")
    reconciliation = {
        "schema": "seed457-occurrence-reconciliation-v1",
        "verification_timing": "post_execution_correction",
        "original_status": json.loads(
            (publication / "verification-status.json").read_text()
        ),
        "diagnosis": {
            "archived_runner_kl_C_step6_scale1": 0.032772165412068455,
            "original_verifier_kl": 0.03277216548941382,
            "source_diagnosis_arrays_sha256": DIAGNOSIS_ARRAY_SHA256,
            "cause": "occurrence 343 at compact row 72594 has a distinct trial output; original verifier compact-row scatter/gather substitutes occurrence 360 output",
        },
        "coverage": {
            "steps_per_arm": {arm: 16 for arm in ("C", "T")},
            "trials_per_arm": {arm: 128 for arm in ("C", "T")},
            "trials_total": trial_count,
        },
        "control_reproduction": reconstruction["control_reproduction"],
        "activation": reconstruction["activation"],
        "metrics": {
            **_plain(reconstruction["endpoint"]["metrics"]),
            "seed442_A": _plain(reconstruction["historical_metrics"]),
        },
        "effects_T_minus_C": _plain(reconstruction["endpoint"]["effects_T_minus_C"]),
        "decision": {
            "classification": reconstruction["classification"],
            "gates": reconstruction["gates"],
        },
        "arm_trajectories": {
            arm: {
                "selected_scales": result["selected_scales"],
                "rejected_count": result["rejected_count_verified"],
                "verified_trials": sum(
                    s["verified_trials"] for s in result["verified_steps"]
                ),
                "steps": [
                    {
                        key: step[key]
                        for key in (
                            "step",
                            "selected_scale",
                            "rejected",
                            "verified_trials",
                            "raw_proposal_norm",
                            "accepted_displacement_norm",
                        )
                    }
                    for step in result["verified_steps"]
                ],
            }
            for arm, result in trajectories.items()
        },
        "logical_T_archive": logical_identity,
        "original_bindings": {
            "registration_sha256": sha256_file(publication / "registration.json"),
            "receipt_sha256": sha256_file(publication / "receipt.json"),
            "evidence_sha256": sha256_file(publication / "evidence.json"),
            "status_sha256": sha256_file(publication / "verification-status.json"),
            "source_sha256": registration["source_sha256"],
            "input_sha256": registration["input_sha256"],
            "publication_files_sha256": receipt["files_sha256"],
            "archive_parts": logical_identity["parts"],
        },
        "preservation": {
            "unchanged": _snapshot(root, receipt, registration) == before,
            "snapshot": {key: list(value) for key, value in before.items()},
        },
    }
    if not reconciliation["preservation"]["unchanged"]:
        raise ValueError("original_publication_changed_after_verification")
    validate_reconciliation_semantics(reconciliation)
    if not publish:
        return reconciliation
    destination = (output_dir or root / OUT).resolve()
    if destination.exists():
        raise ValueError("correction_output_already_exists")
    destination.mkdir(parents=True)
    diagnosis = (
        "# Seed457 occurrence-preserving verification correction\n\n"
        "This is a post-execution verification correction; it does not alter the frozen execution, original evidence, or #460 status.\n\n"
        "At C step 6, scale 1.0, archived runner KL is `0.032772165412068455`; the original verifier reconstructed `0.03277216548941382`. Compact row `72594` appears at occurrences 343 and 360 with identical inputs but different trial logits (maximum absolute difference `4.76837158203125e-07`). The compact-row scatter/gather path reused occurrence 360's output at occurrence 343, changing one KL term by `3.96008305453277e-08`. The aggregate discrepancy is thereby explained to within `5.94e-18`; reduction-only difference with identical terms is zero.\n\n"
        "The correction preserves compact-row pre-policy reference lookup while keeping trial policy and value predictions, masks, terms, and weights occurrence ordered as in the frozen runner. See `reconciliation.json` for the complete reconstruction, fixed gates, classification, and source bindings.\n"
    )
    (destination / "diagnosis.md").write_text(diagnosis)
    report_path = destination / "reconciliation.json"
    report_path.write_text(
        json.dumps(_plain(reconciliation), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    source_files = [Path(__file__).resolve()]
    test_path = Path(__file__).with_name("test_seed457_occurrence_correction.py")
    if test_path.is_file():
        source_files.append(test_path)
    correction_sources = {
        str(path.relative_to(root)): sha256_file(path) for path in source_files
    }
    correction_files = {
        path.name: sha256_file(path)
        for path in (destination / "diagnosis.md", report_path)
    }
    correction_receipt = {
        "schema": "seed457-occurrence-correction-receipt-v1",
        "verification_timing": "post_execution_correction",
        "original_registration_sha256": reconciliation["original_bindings"][
            "registration_sha256"
        ],
        "original_receipt_sha256": reconciliation["original_bindings"][
            "receipt_sha256"
        ],
        "original_evidence_sha256": reconciliation["original_bindings"][
            "evidence_sha256"
        ],
        "original_source_sha256": registration["source_sha256"],
        "original_input_sha256": registration["input_sha256"],
        "original_publication_files_sha256": receipt["files_sha256"],
        "archive_parts": logical_identity["parts"],
        "correction_source_sha256": correction_sources,
        "correction_files_sha256": correction_files,
        "classification": reconstruction["classification"],
    }
    (destination / "receipt.json").write_text(
        json.dumps(correction_receipt, indent=2, sort_keys=True) + "\n"
    )
    validate_correction_receipt_bindings(root, correction_receipt)
    if _snapshot(root, receipt, registration) != before:
        raise ValueError("original_publication_changed_during_correction_publication")
    return reconciliation


def validate_correction_receipt_bindings(
    root: Path, correction_receipt: dict[str, Any]
) -> None:
    """Rebind the correction receipt to complete original and correction sources."""
    registration_path = root / PUBLICATION / "registration.json"
    original_receipt_path = root / PUBLICATION / "receipt.json"
    evidence_path = root / PUBLICATION / "evidence.json"
    if (
        sha256_file(registration_path) != REGISTRATION_SHA256
        or sha256_file(original_receipt_path) != ORIGINAL_RECEIPT_SHA256
    ):
        raise ValueError("correction_original_binding_mismatch")
    original_receipt = json.loads(original_receipt_path.read_text())
    registration = json.loads(registration_path.read_text())
    expected = {
        "original_registration_sha256": REGISTRATION_SHA256,
        "original_receipt_sha256": ORIGINAL_RECEIPT_SHA256,
        "original_evidence_sha256": sha256_file(evidence_path),
        "original_source_sha256": registration["source_sha256"],
        "original_input_sha256": registration["input_sha256"],
        "original_publication_files_sha256": original_receipt["files_sha256"],
    }
    for key, value in expected.items():
        if correction_receipt.get(key) != value:
            raise ValueError(f"correction_receipt_binding_mismatch:{key}")
    for relative, digest in correction_receipt.get(
        "correction_source_sha256", {}
    ).items():
        source = root / relative
        if not source.is_file() or sha256_file(source) != digest:
            raise ValueError(f"correction_source_hash_mismatch:{relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = verify(args.root, output_dir=args.output_dir)
    print(json.dumps(_plain(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
