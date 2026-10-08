"""Read-only structural, hash, KL, metric, and decision verification for seed442."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict[str, Any]:
    seed442.configure_root(root)
    out = seed442.OUT
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("status") != "frozen_before_either_arm":
        raise ValueError("registration_status_invalid")
    if sha(seed442.seed435.INIT) != seed442.INIT_SHA:
        raise ValueError("initializer_sha256_mismatch")
    for relative, expected in registration["frozen_input_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"frozen_input_hash_mismatch:{relative}")
    for relative, expected in registration["source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"execution_source_hash_mismatch:{relative}")
    cohort_path = out / "guard-cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if (
        len(cohort) != 2048
        or sha(cohort_path) != registration["guard"]["cohort_sha256"]
    ):
        raise ValueError("guard_binding_invalid")
    evidence_path = out / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence["registration_sha256"] != sha(registration_path):
        raise ValueError("evidence_registration_binding_invalid")
    arrays = seed442.load_inputs()
    x, _p, _v, replay, _coeff, train_positions, _val, _sources = arrays
    train_replay = replay[train_positions]
    expected_batches = registration["first_16_minibatch_positions"]
    if len(expected_batches) != 8192:
        raise ValueError("batch_prefix_length_invalid")
    membership = seed442.seed435.membership_rows()
    guard_rows = np.asarray([int(row["compact_row"]) for row in cohort], dtype=np.int64)
    if len(set(guard_rows.tolist())) != 2048:
        raise ValueError("guard_rows_not_unique")

    with np.load(out / "step-tensors.npz", allow_pickle=False) as archive:
        tensor_keys = set(archive.files)
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        for arm in ("A", "B"):
            steps = evidence["arms"][arm]["steps"]
            if len(steps) != 16:
                raise ValueError(f"step_count_invalid:{arm}")
            for index, record in enumerate(steps):
                if (
                    record["expanded_positions"]
                    != expected_batches[index * 512 : (index + 1) * 512]
                ):
                    raise ValueError(f"batch_order_invalid:{arm}:{index + 1}")
                rows = train_replay[
                    np.asarray(record["expanded_positions"], dtype=np.int64)
                ]
                if not np.array_equal(
                    rows, np.asarray(record["compact_rows"], dtype=np.int64)
                ):
                    raise ValueError(f"batch_rows_invalid:{arm}:{index + 1}")
                parameters = [p for p in model.parameters() if p.requires_grad]
                before = [
                    archive[f"{arm}_{index:02d}_pre_{j:02d}"]
                    for j in range(len(parameters))
                ]
                proposal = [
                    archive[f"{arm}_{index:02d}_proposal_{j:02d}"]
                    for j in range(len(parameters))
                ]
                post = [
                    archive[f"{arm}_{index:02d}_post_{j:02d}"]
                    for j in range(len(parameters))
                ]
                if record["optimizer_step"] != index + 1:
                    raise ValueError(f"optimizer_counter_invalid:{arm}:{index + 1}")
                for j in range(len(parameters)):
                    grad = archive[f"{arm}_{index:02d}_gradient_{j:02d}"]
                    m0 = archive[f"{arm}_{index:02d}_exp_avg_pre_{j:02d}"]
                    v0 = archive[f"{arm}_{index:02d}_exp_avg_sq_pre_{j:02d}"]
                    m1 = archive[f"{arm}_{index:02d}_exp_avg_{j:02d}"]
                    v1 = archive[f"{arm}_{index:02d}_exp_avg_sq_{j:02d}"]
                    if not np.allclose(m1, 0.9 * m0 + 0.1 * grad, atol=2e-7, rtol=2e-6):
                        raise ValueError(
                            f"first_moment_transition_invalid:{arm}:{index + 1}"
                        )
                    if not np.allclose(
                        v1, 0.999 * v0 + 0.001 * grad**2, atol=2e-7, rtol=2e-6
                    ):
                        raise ValueError(
                            f"second_moment_transition_invalid:{arm}:{index + 1}"
                        )
                if record["pre_hash"] != seed442.vector_hash(
                    [torch.from_numpy(a) for a in before]
                ):
                    raise ValueError(f"pre_hash_invalid:{arm}:{index + 1}")
                if record["proposal_hash"] != seed442.vector_hash(
                    [torch.from_numpy(a) for a in proposal]
                ):
                    raise ValueError(f"proposal_hash_invalid:{arm}:{index + 1}")
                selected = record["selected_scale"]
                expected_post = (
                    [np.asarray(p, dtype=np.float32) for p in before]
                    if selected is None
                    else [
                        np.asarray(p, dtype=np.float32)
                        if selected == 1.0
                        else np.asarray(p + (q - p) * selected, dtype=np.float32)
                        for p, q in zip(before, proposal, strict=True)
                    ]
                )
                if any(
                    not np.array_equal(a, b)
                    for a, b in zip(post, expected_post, strict=True)
                ):
                    raise ValueError(
                        f"accepted_parameter_restore_invalid:{arm}:{index + 1}"
                    )
                for scale_trial in record["trials"]:
                    scale = float(scale_trial["scale"])
                    trial = [
                        np.asarray(proposed, dtype=np.float32)
                        if scale == 1.0
                        else np.asarray(
                            pre + (proposed - pre) * scale, dtype=np.float32
                        )
                        for pre, proposed in zip(before, proposal, strict=True)
                    ]
                    with torch.no_grad():
                        for parameter, value in zip(parameters, trial, strict=True):
                            parameter.copy_(torch.from_numpy(value))
                    if arm == "B":
                        pre_state = train.PolicyValueNet(
                            (96, 3), "residual_v3", x.shape[1]
                        )
                        # Reconstruct the pre-policy from archived pre-parameters.
                        with torch.no_grad():
                            for parameter, value in zip(
                                pre_state.parameters(), before, strict=True
                            ):
                                parameter.copy_(
                                    torch.from_numpy(
                                        np.asarray(value, dtype=np.float32)
                                    )
                                )
                        batch_ids = rows
                        batch_pre = seed442._pre_policy(pre_state, x, batch_ids)
                        guard_pre = seed442._pre_policy(pre_state, x, guard_rows)
                        batch_kl = seed442._kl(
                            model,
                            x,
                            batch_ids,
                            np.full(len(batch_ids), 1 / len(batch_ids)),
                            batch_pre,
                        )
                        guard_kl = seed442._kl(
                            model,
                            x,
                            guard_rows,
                            np.full(len(guard_rows), 1 / len(guard_rows)),
                            guard_pre,
                        )
                        tolerance = registration["kl"]["acceptance_tolerance"]
                        if (
                            abs(batch_kl - scale_trial["batch_kl"]) > tolerance
                            or abs(guard_kl - scale_trial["guard_kl"]) > tolerance
                        ):
                            raise ValueError(
                                f"trial_kl_mismatch:{arm}:{index + 1}:{scale}"
                            )
                        should_accept = (
                            batch_kl <= registration["kl"]["cap"] + tolerance
                            and guard_kl <= registration["kl"]["cap"] + tolerance
                        )
                        if bool(scale_trial["accepted"]) != should_accept:
                            raise ValueError(
                                f"trial_acceptance_mismatch:{arm}:{index + 1}:{scale}"
                            )
                if len(tensor_keys) == 0:
                    raise ValueError("step_tensor_archive_empty")

    for arm in ("A", "B"):
        checkpoint = out / f"{arm}-final.npz"
        if sha(checkpoint) != evidence["arms"][arm]["checkpoint_sha256"]:
            raise ValueError(f"checkpoint_hash_mismatch:{arm}")
        prediction_path = out / f"{arm}-predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as prediction:
            if (
                not np.isfinite(prediction["policy_logits"]).all()
                or not np.isfinite(prediction["value_predictions"]).all()
            ):
                raise ValueError(f"nonfinite_prediction:{arm}")
            expected_rows = [
                r
                for r in membership
                if r["subset"] == "unseen" and r["active_stones"] > 32
            ]
            if not np.array_equal(
                prediction["compact_rows"],
                np.asarray([r["compact_row"] for r in expected_rows]),
            ):
                raise ValueError(f"prediction_rows_invalid:{arm}")
            if not np.array_equal(
                prediction["input_identity"],
                np.asarray([r["input_identity"] for r in expected_rows]),
            ):
                raise ValueError(f"prediction_identities_invalid:{arm}")
    decision = seed442.decide(evidence)
    if decision != evidence["decision"]:
        raise ValueError("decision_reproduction_mismatch")
    return {
        "status": "valid",
        "classification": decision["classification"],
        "proposals_per_arm": 16,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
