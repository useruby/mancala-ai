"""Independent, read-only verifier for the superseding seed454 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite import train
from ml.alphazero_lite.seed453_fresh_policy_projection import portable_inputs

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed454-full-guard-compliance/superseding-v2")
SCALES = tuple(2.0**-i for i in range(8))
KL_LIMIT = 0.005 + 1e-10
VALUE_LIMIT = 0.0001 + 1e-12


def sha(path: Path) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independent_reference(
    model: torch.nn.Module, inputs: np.ndarray, rows: np.ndarray
) -> tuple[torch.Tensor, np.ndarray]:
    """Compute reference log probabilities and values without publisher helpers."""
    logp = torch.full((len(inputs), train.POLICY_SIZE), -torch.inf, dtype=torch.float64)
    outputs = np.empty(len(rows), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), 512):
            selected = rows[offset : offset + 512]
            logits, values = model(torch.from_numpy(inputs[selected]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(inputs[selected])
            ).bool()
            logp[selected] = torch.log_softmax(
                logits.double().masked_fill(~legal, -torch.inf), dim=1
            )
            outputs[offset : offset + len(selected)] = values.reshape(-1).cpu().numpy()
    return logp, outputs


def _independent_measure(
    model: torch.nn.Module,
    inputs: np.ndarray,
    rows: np.ndarray,
    reference_logp: torch.Tensor,
    reference_values: np.ndarray,
) -> tuple[float, float]:
    """Recompute row-uniform KL and value movement using seed442/447 reductions."""
    kl_sum = np.float64(0)
    predictions = np.empty(len(rows), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), 512):
            selected = rows[offset : offset + 512]
            logits, values = model(torch.from_numpy(inputs[selected]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(inputs[selected])
            ).bool()
            logq = torch.log_softmax(
                logits.double().masked_fill(~legal, -torch.inf), dim=1
            )
            p_log = reference_logp[selected]
            p = p_log.exp().masked_fill(~legal, 0.0)
            each_kl = (p * (p_log - logq).masked_fill(~legal, 0.0)).sum(dim=1)
            kl_sum += each_kl.sum().cpu().numpy().astype(np.float64)
            predictions[offset : offset + len(selected)] = (
                values.reshape(-1).cpu().numpy()
            )
    delta = predictions.astype(np.float64) - reference_values.astype(np.float64)
    return float(kl_sum / len(rows)), float(np.mean(delta * delta, dtype=np.float64))


def _fresh_gradient(
    model: torch.nn.Module,
    inputs: np.ndarray,
    policy: np.ndarray,
    rows: np.ndarray,
) -> list[np.ndarray]:
    """Independently reconstruct B's mean fresh policy gradient at a pre-state."""
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    accum = [np.zeros(tuple(p.shape), dtype=np.float64) for p in parameters]
    for start in range(0, len(rows), 512):
        selected = rows[start : start + 512]
        logits, _ = model(torch.from_numpy(inputs[selected]))
        legal = torch.from_numpy(
            train.legal_mask_matrix_for_encoded_states(inputs[selected])
        )
        losses = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[selected])
        )
        grads = torch.autograd.grad(
            losses.sum() / len(rows), parameters, allow_unused=True
        )
        for index, grad in enumerate(grads):
            if grad is not None:
                accum[index] += grad.detach().cpu().numpy().astype(np.float64)
    return [value.astype(np.float32) for value in accum]


def _close(actual: float, recorded: float, label: str) -> None:
    if not np.isclose(actual, recorded, atol=2e-12, rtol=0):
        raise ValueError(f"semantic_measurement_mismatch:{label}")


def verify(root: Path) -> dict[str, Any]:
    """Rebuild all 256 trials and classify them independently of publisher code."""
    root = root.resolve()
    out = root / OUT
    freeze = json.loads((out / "audit-registration.json").read_text())
    if (
        freeze.get("temporal_scope")
        != "superseding retrospective audit of the same archived seed453 experiment; not a prospective registration"
    ):
        raise ValueError("audit_temporal_scope_invalid")
    for relative, expected in freeze["input_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"frozen_input_hash_invalid:{relative}")
    seed453 = root / "docs/data/seed453-fresh-policy-projection"
    if (
        sha(seed453 / "registration.json")
        != "9805cba7e7808217cbb381a11816ba26c2ca5c7f4c9e539a4dd9e2daeb3529fd"
    ):
        raise ValueError("seed453_registration_identity_invalid")
    if (
        sha(seed453 / "evidence.json")
        != "b87cfc46d4a9b3e338404fe91691c6565cd616016a28b94c5042bbe8ae163d6f"
    ):
        raise ValueError("seed453_evidence_identity_invalid")
    seed453_receipt = json.loads((seed453 / "receipt.json").read_text())
    for relative, expected in seed453_receipt["files_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"seed453_publication_receipt_invalid:{relative}")
    for relative, expected in freeze["source_sha256"].items():
        if (
            sha(root / relative) != expected
            or sha(out / "source-snapshots" / Path(relative).name) != expected
        ):
            raise ValueError(f"frozen_source_binding_invalid:{relative}")
    receipt = json.loads((out / "receipt.json").read_text())
    for relative, expected in receipt["files_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"publication_receipt_invalid:{relative}")

    seed442.configure_root(root)
    arrays = portable_inputs(root)
    x, policy, _value, replay, _coeff, train_positions, _validation, sources = arrays
    training = set(map(int, replay[train_positions]))
    full = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json").read_text()
    )
    fresh = json.loads(
        (
            root / "docs/data/seed453-fresh-policy-projection/fresh-guard.json"
        ).read_text()
    )
    if (
        sha(root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json")
        != "d25c03960d1220c58f5401d9536862ce647a02a5f8d9abcf6ad7174da0c53297"
    ):
        raise ValueError("original_guard_hash_invalid")
    if len(full) != 2048 or len({r["exact_input_identity"] for r in full}) != 2048:
        raise ValueError("guard_membership_invalid:full_count_or_uniqueness")
    if len(fresh) != 1399 or len({r["exact_input_identity"] for r in fresh}) != 1399:
        raise ValueError("guard_membership_invalid:fresh_count_or_uniqueness")
    if [r["exact_input_identity"] for r in full if r["source"] == "fresh"] != [
        r["exact_input_identity"] for r in fresh
    ]:
        raise ValueError("guard_membership_invalid:fresh_order")
    if len([r for r in full if r["source"] != "fresh"]) != 649:
        raise ValueError("guard_membership_invalid:historical_count")
    for row in full:
        rowid = int(row["compact_row"])
        if (
            rowid not in training
            or x[rowid].astype("<f4", copy=False).tobytes().hex()
            != row["exact_input_identity"]
        ):
            raise ValueError("guard_membership_invalid:training_or_identity")
        if sources[rowid].get("source") != row["source"]:
            raise ValueError("guard_membership_invalid:source_identity")
    membership = seed442.seed435.membership_rows()
    unseen = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    unseen_exact = {r["input_identity"] for r in unseen}
    unseen_canonical = {r["canonical_identity"] for r in unseen}
    if any(
        r["exact_input_identity"] in unseen_exact
        or r["canonical_identity"] in unseen_canonical
        for r in full
    ):
        raise ValueError("guard_membership_invalid:strict_unseen_overlap")

    evidence = json.loads(
        (root / "docs/data/seed453-fresh-policy-projection/evidence.json").read_text()
    )
    seed453_registration = json.loads(
        (
            root / "docs/data/seed453-fresh-policy-projection/registration.json"
        ).read_text()
    )
    expected_positions = np.asarray(
        seed453_registration["seed442_first_16_positions"], dtype=np.int64
    )
    training_order = replay[train_positions]
    report = json.loads((out / "audit.json").read_text())
    report_rows = report["trial_measurements"]
    if len(report_rows) != 256:
        raise ValueError("trial_inventory_invalid")
    full_rows = np.asarray([int(r["compact_row"]) for r in full], dtype=np.int64)
    fresh_rows = np.asarray([int(r["compact_row"]) for r in fresh], dtype=np.int64)
    archives = {
        arm: np.load(
            root / f"docs/data/seed453-fresh-policy-projection/step-tensors-{arm}.npz",
            allow_pickle=False,
        )
        for arm in ("B", "C")
    }
    report_by_trial = {
        (row["arm"], int(row["step"]), float(row["scale"])): row for row in report_rows
    }
    comparison_by_step = {
        (row["arm"], int(row["step"])): row
        for row in report["selected_scale_comparisons"]
    }
    derived: dict[tuple[str, int], list[dict[str, Any]]] = {}
    try:
        for arm in ("C", "B"):
            archive = archives[arm]
            model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            params = [p for p in model.parameters() if p.requires_grad]
            previous: list[np.ndarray] | None = None
            for step, record in enumerate(evidence["arms"][arm]["steps"]):
                pre = [
                    archive[f"{arm}_{step:02d}_{i:02d}_pre"].copy()
                    for i in range(len(params))
                ]
                proposal = [
                    archive[f"{arm}_{step:02d}_{i:02d}_projected_proposal"].copy()
                    for i in range(len(params))
                ]
                if previous is not None and any(
                    not np.array_equal(a, b) for a, b in zip(pre, previous, strict=True)
                ):
                    raise ValueError("parameter_continuity_invalid")
                for parameter, value in zip(params, pre, strict=True):
                    parameter.data.copy_(torch.from_numpy(value.copy()))
                batch = np.asarray(record["compact_rows"], dtype=np.int64)
                expected_batch = training_order[
                    expected_positions[step * 512 : (step + 1) * 512]
                ]
                if not np.array_equal(batch, expected_batch) or any(
                    row not in training for row in batch
                ):
                    raise ValueError("minibatch_order_or_training_membership_invalid")
                refs = {
                    "batch": _independent_reference(model, x, batch),
                    "fresh": _independent_reference(model, x, fresh_rows),
                    "full": _independent_reference(model, x, full_rows),
                }
                gradients = (
                    _fresh_gradient(model, x, policy, fresh_rows)
                    if arm == "B"
                    else [np.zeros_like(p) for p in pre]
                )
                if arm == "B":
                    for idx, calculated in enumerate(gradients):
                        if not np.array_equal(
                            calculated,
                            archive[f"{arm}_{step:02d}_{idx:02d}_fresh_gradient"],
                        ):
                            raise ValueError("archived_fresh_gradient_invalid")
                trial_rows = []
                for scale_idx, scale in enumerate(SCALES):
                    state = [
                        b.copy()
                        if scale == 1
                        else np.asarray(a + (b - a) * scale, dtype=np.float32)
                        for a, b in zip(pre, proposal, strict=True)
                    ]
                    for parameter, value in zip(params, state, strict=True):
                        parameter.data.copy_(torch.from_numpy(value.copy()))
                    for idx, value in enumerate(state):
                        expected_tensor = archive[
                            f"{arm}_{step:02d}_{idx:02d}_trial_{scale:g}"
                        ]
                        if not np.array_equal(value, expected_tensor):
                            raise ValueError("trial_tensor_invalid")
                    measured = {}
                    for name, selected in (
                        ("batch", batch),
                        ("fresh", fresh_rows),
                        ("full", full_rows),
                    ):
                        measured[name] = _independent_measure(
                            model, x, selected, *refs[name]
                        )
                    differences = [
                        b.astype(np.float64) - a.astype(np.float64)
                        for a, b in zip(pre, state, strict=True)
                    ]
                    dot = (
                        float(
                            sum(
                                np.sum(d * g.astype(np.float64), dtype=np.float64)
                                for d, g in zip(differences, gradients, strict=True)
                            )
                        )
                        if arm == "B"
                        else 0.0
                    )
                    row = report_by_trial[(arm, step + 1, scale)]
                    if (row["arm"], row["step"], row["scale"]) != (
                        arm,
                        step + 1,
                        scale,
                    ):
                        raise ValueError("trial_order_or_identity_invalid")
                    for metric, value in (
                        ("batch_kl", measured["batch"][0]),
                        ("batch_value_movement", measured["batch"][1]),
                        ("executed_guard_kl", measured["fresh"][0]),
                        ("executed_guard_value_movement", measured["fresh"][1]),
                        ("full_guard_kl", measured["full"][0]),
                        ("full_guard_value_movement", measured["full"][1]),
                        ("realized_fresh_dot", dot),
                    ):
                        _close(
                            value,
                            float(row[metric]),
                            f"{arm}:{step + 1}:{scale}:{metric}",
                        )
                    recorded = record["trials"][scale_idx]
                    for actual, key in (
                        (measured["batch"][0], "batch_kl"),
                        (measured["batch"][1], "batch_value_movement"),
                        (measured["fresh"][0], "guard_kl"),
                        (measured["fresh"][1], "guard_value_movement"),
                    ):
                        _close(
                            actual,
                            float(recorded[key]),
                            f"seed453:{arm}:{step + 1}:{scale}:{key}",
                        )

                    def feasible(guard: str) -> bool:
                        return bool(
                            measured["batch"][0] <= KL_LIMIT
                            and measured[guard][0] <= KL_LIMIT
                            and measured["batch"][1] <= VALUE_LIMIT
                            and measured[guard][1] <= VALUE_LIMIT
                            and (arm == "C" or dot <= 1e-7)
                        )

                    row_exec, row_req = feasible("fresh"), feasible("full")
                    if (
                        row["executed_feasible"] != row_exec
                        or row["requested_feasible"] != row_req
                    ):
                        raise ValueError("feasibility_invalid")
                    trial_rows.append(
                        {"scale": scale, "executed": row_exec, "requested": row_req}
                    )
                selected_recorded = record["selected_scale"]
                accepted = next(
                    (t for t in trial_rows if t["scale"] == selected_recorded), None
                )
                violation = accepted is not None and not accepted["requested"]
                requested_scales = [t["scale"] for t in trial_rows if t["requested"]]
                executed_scales = [t["scale"] for t in trial_rows if t["executed"]]
                req_selected = max(requested_scales) if requested_scales else None
                exec_selected = max(executed_scales) if executed_scales else None
                mismatch = selected_recorded != req_selected
                comparison = comparison_by_step[(arm, step + 1)]
                expected_comparison = (
                    arm,
                    step + 1,
                    selected_recorded,
                    exec_selected,
                    req_selected,
                    bool(violation),
                    bool(mismatch),
                )
                got_comparison = (
                    comparison["arm"],
                    comparison["step"],
                    comparison["recorded_selected_scale"],
                    comparison["executed_rule_selected_scale"],
                    comparison["requested_rule_selected_scale"],
                    comparison["accepted_full_guard_violation"],
                    comparison["selection_mismatch"],
                )
                if got_comparison != expected_comparison:
                    raise ValueError("selected_scale_comparison_invalid")
                derived[(arm, step + 1)] = trial_rows
                selected_post = (
                    pre
                    if selected_recorded is None
                    else [
                        b.copy()
                        if selected_recorded == 1
                        else np.asarray(
                            a + (b - a) * selected_recorded, dtype=np.float32
                        )
                        for a, b in zip(pre, proposal, strict=True)
                    ]
                )
                previous = [value.copy() for value in selected_post]
    finally:
        for archive in archives.values():
            archive.close()

    violations = sum(
        int(
            any(
                c["arm"] == arm
                and c["step"] == step
                and c["accepted_full_guard_violation"]
                for c in report["selected_scale_comparisons"]
            )
        )
        for arm in ("B", "C")
        for step in range(1, 17)
    )
    mismatches = sum(
        int(item["selection_mismatch"]) for item in report["selected_scale_comparisons"]
    )
    label = (
        "full_guard_constraint_violation"
        if violations
        else "full_guard_selection_mismatch"
        if mismatches
        else "archived_trajectory_matches_full_guard_rule"
    )
    if (
        report["accepted_full_guard_violation_count"] != violations
        or report["selection_mismatch_count"] != mismatches
        or report["classification"] != label
    ):
        raise ValueError("audit_classification_or_counts_invalid")
    if (
        report["preserved_seed453_classification"]
        != "close_fresh_policy_projection_branch"
    ):
        raise ValueError("seed453_classification_not_preserved")
    maxima = {
        key: max(float(row[key]) for row in report_rows)
        for key in (
            "batch_kl",
            "batch_value_movement",
            "executed_guard_kl",
            "executed_guard_value_movement",
            "full_guard_kl",
            "full_guard_value_movement",
            "realized_fresh_dot",
        )
    }
    if maxima != report["maxima"]:
        raise ValueError("reported_maxima_invalid")
    expected_first = next(
        (
            row
            for row in report["selected_scale_comparisons"]
            if row["accepted_full_guard_violation"] or row["selection_mismatch"]
        ),
        None,
    )
    if report["first_divergence"] != expected_first:
        raise ValueError("first_divergence_invalid")
    return {
        "status": "valid",
        "trials": len(report_rows),
        "classification": label,
        "full_guard_rows": len(full),
        "fresh_rows": len(fresh),
    }


def main() -> None:
    """Run the read-only verifier from an arbitrary working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
