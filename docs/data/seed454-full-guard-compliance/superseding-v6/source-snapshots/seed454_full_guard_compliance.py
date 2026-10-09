"""Publish a retrospective full-guard compliance audit of archived seed453."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite import seed447_joint_output_cap as seed447
from ml.alphazero_lite import seed454_analysis as analysis
from ml.alphazero_lite import train
from ml.alphazero_lite.seed453_fresh_policy_projection import portable_inputs

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed454-full-guard-compliance/superseding-v6")
SEED453 = Path("docs/data/seed453-fresh-policy-projection")
ORIGINAL_GUARD_SHA = "d25c03960d1220c58f5401d9536862ce647a02a5f8d9abcf6ad7174da0c53297"
REGISTRATION_SHA = "9805cba7e7808217cbb381a11816ba26c2ca5c7f4c9e539a4dd9e2daeb3529fd"
EVIDENCE_SHA = "b87cfc46d4a9b3e338404fe91691c6565cd616016a28b94c5042bbe8ae163d6f"


def sha(path: Path) -> str:
    """Hash file content using SHA256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registration(root: Path) -> dict[str, Any]:
    """Freeze replacement sources and archived inputs before audit forwards."""
    root = root.resolve()
    seed442.configure_root(root)
    files = [
        SEED453 / name
        for name in (
            "registration.json",
            "evidence.json",
            "fresh-guard.json",
            "step-tensors-B.npz",
            "step-tensors-C.npz",
            "B-final.npz",
            "C-final.npz",
            "verifier-correction.json",
            "verifier-correction-receipt.json",
            "receipt.json",
        )
    ] + [
        Path("docs/data/seed442-kl-capped-adam-screen/guard-cohort.json"),
        Path("docs/data/seed416-policy-target-softening/registration-v3.json"),
        Path(
            "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz"
        ),
        Path(
            "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz"
        ),
        Path(
            "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
        ),
    ]
    seed416 = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    files += [
        Path(f"docs/data/seed426-canonical-overlap/sources/{row['name']}.jsonl.gz")
        for row in seed416["replays"]
    ]
    files += [
        Path("ml/alphazero_lite/verify_seed450_seed449_correction.py"),
        Path("ml/alphazero_lite/seed416_policy_target_softening.py"),
        Path("ml/alphazero_lite/seed435_adam_direction.py"),
        Path("ml/alphazero_lite/seed442_kl_capped_adam.py"),
        Path("ml/alphazero_lite/seed429_policy_normalization.py"),
        Path("ml/alphazero_lite/train.py"),
        Path("ml/alphazero_lite/verify_seed434_census_source.py"),
        Path("ml/alphazero_lite/exact_root_policy_targets.py"),
        Path("ml/alphazero_lite/seed432_policy_target_census.py"),
        Path("ml/alphazero_lite/seed433_census_correction.py"),
        Path("ml/alphazero_lite/checkpoint_phase_selection.py"),
        Path("ml/alphazero_lite/input_encodings.py"),
        Path("ml/alphazero_lite/kalah_rules.py"),
        Path(
            "docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json"
        ),
        Path(
            "docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json"
        ),
    ]
    inputs = {str(path): sha(root / path) for path in files}
    seed453_registration = json.loads(
        (root / SEED453 / "registration.json").read_text()
    )
    for relative in seed453_registration["input_sha256"]:
        inputs[relative] = sha(root / relative)
    if inputs[str(SEED453 / "registration.json")] != REGISTRATION_SHA:
        raise ValueError("seed453_registration_hash_mismatch")
    if inputs[str(SEED453 / "evidence.json")] != EVIDENCE_SHA:
        raise ValueError("seed453_evidence_hash_mismatch")
    if (
        inputs[str(Path("docs/data/seed442-kl-capped-adam-screen/guard-cohort.json"))]
        != ORIGINAL_GUARD_SHA
    ):
        raise ValueError("seed442_guard_hash_mismatch")
    sources = [
        "ml/alphazero_lite/seed454_analysis.py",
        "ml/alphazero_lite/seed454_full_guard_compliance.py",
        "ml/alphazero_lite/verify_seed454_full_guard_compliance.py",
        "ml/alphazero_lite/test_seed454_full_guard_compliance.py",
        "ml/alphazero_lite/seed453_fresh_policy_projection.py",
        "ml/alphazero_lite/seed447_joint_output_cap.py",
    ]
    return {
        "schema": "seed454-audit-registration-v1",
        "scope": "retrospective audit of archived seed453 evidence; no training rerun or corrected trajectory",
        "frozen_after_seed453_execution_before_seed454_diagnostic_forward_passes": True,
        "temporal_scope": "superseding retrospective audit of the same archived seed453 experiment; not a prospective registration",
        "input_sha256": inputs,
        "source_sha256": {source: sha(root / source) for source in sources},
        "seed453_registration_sha256": REGISTRATION_SHA,
        "seed453_evidence_sha256": EVIDENCE_SHA,
        "seed442_guard_sha256": ORIGINAL_GUARD_SHA,
        "conventions": {
            "policy_kl": "seed442 convention: KL(pre policy || trial policy), legal actions only; float32 logits promoted to float64; float64 log_softmax; row-uniform weighting; 512-row batching",
            "value_movement": "seed447 convention: float32 predictions, float64 subtraction and weighted squared mean; uniform row weighting; 512-row batching",
            "trial_state": "scale 1 uses archived projected proposal directly; others float32(pre + (proposal - pre) * scale), matching seed442/447 operation order",
            "caps": {
                "kl": analysis.KL_LIMIT,
                "value_movement": analysis.VALUE_LIMIT,
                "fresh_gradient_dot": analysis.DOT_LIMIT,
            },
            "selection": "largest feasible scale among all eight tested scales; no monotonicity assumption",
        },
    }


def publish(root: Path) -> dict[str, Any]:
    """Run the frozen retrospective comparison and publish append-only evidence."""
    root = root.resolve()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = out / "audit-registration.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
    else:
        raise ValueError("replacement_audit_must_be_frozen_before_publish")
    for rel, digest in frozen["input_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_input_changed:{rel}")
    audit_sources = {rel: sha(root / rel) for rel in frozen["source_sha256"]}
    if audit_sources != frozen["source_sha256"]:
        raise ValueError("audit_source_changed_after_freeze")

    seed442.configure_root(root)
    arrays = portable_inputs(root)
    x, _policy, _value, replay, _coeff, train_positions, _valid, _source = arrays
    training_rows = set(map(int, replay[train_positions]))
    full_guard = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json").read_text()
    )
    fresh_guard = json.loads((root / SEED453 / "fresh-guard.json").read_text())
    original_ids = [int(row["compact_row"]) for row in full_guard]
    fresh_ids = [int(row["compact_row"]) for row in fresh_guard]
    if (
        len(full_guard) != 2048
        or len(set(row["exact_input_identity"] for row in full_guard)) != 2048
    ):
        raise ValueError("full_guard_membership_invalid")
    if (
        len(fresh_guard) != 1399
        or len(set(row["exact_input_identity"] for row in fresh_guard)) != 1399
    ):
        raise ValueError("fresh_guard_membership_invalid")
    identities = {row["exact_input_identity"] for row in fresh_guard}
    historical = [
        row for row in full_guard if row["exact_input_identity"] not in identities
    ]
    if len(historical) != 649:
        raise ValueError("historical_guard_count_invalid")
    for row in full_guard:
        actual = x[int(row["compact_row"])].astype("<f4", copy=False).tobytes().hex()
        if (
            actual != row["exact_input_identity"]
            or int(row["compact_row"]) not in training_rows
        ):
            raise ValueError("full_guard_identity_or_training_membership_invalid")

    evidence = json.loads((root / SEED453 / "evidence.json").read_text())
    archive_context = {
        arm: np.load(root / SEED453 / f"step-tensors-{arm}.npz", allow_pickle=False)
        for arm in ("B", "C")
    }
    membership = seed442.seed435.membership_rows()
    unseen_exact = {
        row["input_identity"]
        for row in membership
        if row["subset"] == "unseen" and row["active_stones"] > 32
    }
    for row in full_guard:
        if row["exact_input_identity"] in unseen_exact:
            raise ValueError("guard_overlaps_strict_unseen")

    trial_rows: list[dict[str, Any]] = []
    for arm in ("C", "B"):
        archive = archive_context[arm]
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        steps = evidence["arms"][arm]["steps"]
        for step, record in enumerate(steps):
            pre = [
                archive[f"{arm}_{step:02d}_{i:02d}_pre"].copy()
                for i in range(len(parameters))
            ]
            proposal = [
                archive[f"{arm}_{step:02d}_{i:02d}_projected_proposal"].copy()
                for i in range(len(parameters))
            ]
            analysis.set_state(parameters, pre)
            pre_log_policy = seed442._pre_policy(
                model, x, np.asarray(original_ids, dtype=np.int64)
            )
            pre_values = seed447._predict_values(
                model, x, np.asarray(original_ids, dtype=np.int64)
            )
            pre_fresh_log = seed442._pre_policy(
                model, x, np.asarray(fresh_ids, dtype=np.int64)
            )
            pre_fresh_values = seed447._predict_values(
                model, x, np.asarray(fresh_ids, dtype=np.int64)
            )
            pre_batch_log = seed442._pre_policy(
                model, x, np.asarray(record["compact_rows"], dtype=np.int64)
            )
            pre_batch_values = seed447._predict_values(
                model, x, np.asarray(record["compact_rows"], dtype=np.int64)
            )
            fresh_gradient = [
                archive[f"{arm}_{step:02d}_{i:02d}_fresh_gradient"].astype(np.float64)
                for i in range(len(parameters))
            ]
            for scale, prior in zip(analysis.SCALES, record["trials"], strict=True):
                trial = analysis.trial_state(pre, proposal, scale)
                analysis.set_state(parameters, trial)
                batch_ids = np.asarray(record["compact_rows"], dtype=np.int64)
                batch_kl, batch_move = analysis.measurements(
                    model, x, batch_ids, pre_batch_log, pre_batch_values
                )
                executed_kl, executed_move = analysis.measurements(
                    model, x, np.asarray(fresh_ids), pre_fresh_log, pre_fresh_values
                )
                full_kl, full_move = analysis.measurements(
                    model, x, np.asarray(original_ids), pre_log_policy, pre_values
                )
                delta = [
                    b.astype(np.float64) - a.astype(np.float64)
                    for a, b in zip(pre, trial, strict=True)
                ]
                dot = (
                    float(
                        sum(
                            np.sum(d * g, dtype=np.float64)
                            for d, g in zip(delta, fresh_gradient, strict=True)
                        )
                    )
                    if arm == "B"
                    else 0.0
                )
                row = {
                    "arm": arm,
                    "step": step + 1,
                    "scale": scale,
                    "batch_kl": batch_kl,
                    "batch_value_movement": batch_move,
                    "executed_guard_kl": executed_kl,
                    "executed_guard_value_movement": executed_move,
                    "full_guard_kl": full_kl,
                    "full_guard_value_movement": full_move,
                    "realized_fresh_dot": dot,
                    "recorded_fresh_guard_kl": prior["guard_kl"],
                    "recorded_fresh_guard_value_movement": prior[
                        "guard_value_movement"
                    ],
                    "recorded_batch_kl": prior["batch_kl"],
                    "recorded_batch_value_movement": prior["batch_value_movement"],
                }
                for metric, recorded in (
                    ("batch_kl", prior["batch_kl"]),
                    ("batch_value_movement", prior["batch_value_movement"]),
                    ("executed_guard_kl", prior["guard_kl"]),
                    ("executed_guard_value_movement", prior["guard_value_movement"]),
                ):
                    if not np.isclose(row[metric], recorded, atol=2e-12, rtol=0):
                        raise ValueError(
                            f"seed453_metric_reconciliation_failed:{arm}:{step + 1}:{scale}:{metric}"
                        )
                row["executed_feasible"] = analysis.feasibility(
                    row, arm, requested=False
                )
                row["requested_feasible"] = analysis.feasibility(
                    row, arm, requested=True
                )
                trial_rows.append(row)
            analysis.set_state(parameters, pre)
    for archive in archive_context.values():
        archive.close()
    by_step: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in trial_rows:
        by_step.setdefault((row["arm"], row["step"]), []).append(row)
    selected_comparisons = []
    accepted_violations = selection_mismatches = 0
    for (arm, step), rows in sorted(by_step.items()):
        record = evidence["arms"][arm]["steps"][step - 1]
        requested_selected = analysis.largest_feasible(rows, arm, requested=True)
        executed_selected = analysis.largest_feasible(rows, arm, requested=False)
        accepted = record["selected_scale"]
        accepted_row = next((row for row in rows if row["scale"] == accepted), None)
        violation = bool(accepted_row and not accepted_row["requested_feasible"])
        mismatch = accepted != requested_selected
        accepted_violations += int(violation)
        selection_mismatches += int(mismatch)
        selected_comparisons.append(
            {
                "arm": arm,
                "step": step,
                "recorded_selected_scale": accepted,
                "executed_rule_selected_scale": executed_selected,
                "requested_rule_selected_scale": requested_selected,
                "accepted_full_guard_violation": violation,
                "selection_mismatch": mismatch,
            }
        )
    classification = analysis.classify(accepted_violations, selection_mismatches)
    result = {
        "schema": "seed454-full-guard-audit-v1",
        "audit_type": "retrospective",
        "classification": classification,
        "preserved_seed453_classification": "close_fresh_policy_projection_branch",
        "guard_population": {
            "full_guard_unique_training_inputs": len(full_guard),
            "fresh_subset_unique_training_inputs": len(fresh_guard),
            "historical_inputs": len(historical),
        },
        "trial_count": len(trial_rows),
        "trial_measurements": trial_rows,
        "maxima": {
            key: max(float(row[key]) for row in trial_rows)
            for key in (
                "batch_kl",
                "batch_value_movement",
                "executed_guard_kl",
                "executed_guard_value_movement",
                "full_guard_kl",
                "full_guard_value_movement",
                "realized_fresh_dot",
            )
        },
        "selected_scale_comparisons": selected_comparisons,
        "accepted_full_guard_violation_count": accepted_violations,
        "selection_mismatch_count": selection_mismatches,
        "first_divergence": next(
            (
                row
                for row in selected_comparisons
                if row["selection_mismatch"] or row["accepted_full_guard_violation"]
            ),
            None,
        ),
        "no_corrected_trajectory_inference": True,
        "prospective_guard_note": "Seed453 prospectively used the fresh subset; this retrospective compatibility audit does not change that fact.",
    }
    report_path = out / "audit.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "analysis.md").write_text(render(result))
    (out / "supersession-receipt.json").write_text(
        json.dumps(
            {
                "schema": "seed454-supersession-receipt-v1",
                "supersedes": "../audit-registration.json and ../audit.json (unverified draft attempt)",
                "scope": "same archived seed453 experiment; retrospective only",
                "seed453_training_rerun": False,
                "constraints_or_classification_rules_changed": False,
                "draft_attempt_receipt": "../draft-attempt-receipt.json",
                "superseding_registration_sha256": sha(frozen_path),
                "audit_sha256": sha(report_path),
                "classification": classification,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    receipt_files = {
        str(path.relative_to(root)): sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }
    (out / "receipt.json").write_text(
        json.dumps(
            {"schema": "seed454-superseding-receipt-v1", "files_sha256": receipt_files},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return result


def render(result: dict[str, Any]) -> str:
    """Render a concise audit report from published numerical evidence."""
    return (
        "# Seed454 — retrospective full-guard compliance audit\n\n"
        "This is a retrospective audit of archived seed453 evidence. It does not retrain, construct a corrected trajectory, or alter seed453 artifacts.\n\n"
        f"**Audit label:** `{result['classification']}`  \n"
        "**Preserved seed453 branch:** `close_fresh_policy_projection_branch`\n\n"
        f"Guard: {result['guard_population']['full_guard_unique_training_inputs']} original unique training inputs; "
        f"{result['guard_population']['fresh_subset_unique_training_inputs']} fresh; {result['guard_population']['historical_inputs']} historical.\n\n"
        f"Trials: {result['trial_count']}; accepted full-guard violations: {result['accepted_full_guard_violation_count']}; "
        f"selection mismatches: {result['selection_mismatch_count']}.\n\n"
        "Per-trial measurements, feasibility, and selected-scale comparisons are in `audit.json`. A selection mismatch does not imply subsequent states of a corrected run.\n"
    )


def main() -> None:
    """Freeze sources and publish the retrospective audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "publish"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    out = args.root.resolve() / OUT
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "freeze":
        result = registration(args.root)
        (out / "audit-registration.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        snapshot_dir = out / "source-snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for relative, expected in result["source_sha256"].items():
            destination = snapshot_dir / Path(relative).name
            shutil.copyfile(args.root / relative, destination)
            if sha(destination) != expected:
                raise ValueError(f"snapshot_copy_mismatch:{relative}")
    else:
        result = publish(args.root)
    print(
        json.dumps(
            result
            if args.command == "freeze"
            else {k: v for k, v in result.items() if k != "trial_measurements"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
