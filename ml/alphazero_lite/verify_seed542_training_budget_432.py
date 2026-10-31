"""Independent read-only verifier for the seed542 432-update experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import (
    publish_seed542_training_budget_432 as publication,
    run_seed542_fixed_update_cd as fixed_update,
    seed542_training_budget_432 as experiment,
    train,
    verify_seed542_fixed_update_cd_v2 as prior_verifier,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0")
        h.update(str(array.dtype).encode() + b"\0")
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def control_mismatches(
    reference: dict[str, Any], observed: dict[str, Any]
) -> list[str]:
    fields = (
        "update",
        "batch",
        "forward",
        "policy_loss",
        "value_loss",
        "weighted_total_objective",
        "gradient_norm_preclip",
        "gradient_norm_clip_function",
        "raw_gradients",
        "clipped_gradients",
        "optimizer_before",
        "optimizer_after",
        "parameter_state_before",
        "parameter_state_after",
    )
    return [
        field
        for field in fields
        if prior_verifier.canonical(reference.get(field))
        != prior_verifier.canonical(observed.get(field))
    ]


def frozen_inputs(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registration = json.loads((output / "registration.json").read_text())
    if registration["status"] != "prospectively frozen; no training executed":
        raise ValueError("432_registration_not_prospective")
    for name, expected in registration["execution_source_sha256"].items():
        if digest(experiment.ROOT / name) != expected:
            raise ValueError(f"432_source_tampered:{name}")
    for name, expected in registration["input_sha256"].items():
        if digest(experiment.ROOT / name) != expected:
            raise ValueError(f"432_input_tampered:{name}")
    schedule_path = output / "schedule.json"
    if digest(schedule_path) != registration["schedule_sha256"]:
        raise ValueError("432_schedule_tampered")
    schedule = json.loads(schedule_path.read_text())
    if len(schedule) != 432 or [item["update"] for item in schedule] != list(
        range(1, 433)
    ):
        raise ValueError("432_schedule_shape_invalid")
    if experiment.accounting(schedule) != json.loads(
        (output / "schedule-accounting.json").read_text()
    ):
        raise ValueError("432_schedule_accounting_mismatch")
    return registration, schedule


def independently_replay(output: Path, schedule: list[dict[str, Any]]) -> None:
    x, policy, value, weights, identities = fixed_update.load_sources("D")
    positions = {identity: index for index, identity in enumerate(identities)}
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, experiment.CORPUS / "initializer.npz")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    records = [
        json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()
    ]
    reference = [
        json.loads(line)
        for line in (experiment.REFERENCE / "D-updates.jsonl").read_text().splitlines()
    ]
    if len(records) != 432:
        raise ValueError("432_trace_length_mismatch")
    for spec, recorded in zip(schedule, records, strict=True):
        if (
            recorded.get("update") != spec["update"]
            or recorded.get("phase") != spec["phase"]
            or recorded.get("row_ids") != spec["rows"]
            or recorded.get("batch_size") != spec["batch_size"]
        ):
            raise ValueError(f"432_trace_schedule_identity_mismatch:{spec['update']}")
        idx = np.asarray([positions[row] for row in spec["rows"]], dtype=np.int64)
        bx, bp, bv, bw = (
            torch.from_numpy(arr[idx]) for arr in (x, policy, value, weights)
        )
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[idx]))
        model.train()
        before = parameter_digest(model)
        logits, predicted_value = model(bx)
        policy_loss = train.weighted_policy_loss(
            train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9), bp
            ),
            bw,
        )
        value_loss = train.compute_value_loss_vector(
            predicted_value, bv, value_loss="huber", huber_delta=1.0
        ).mean()
        objective = policy_loss + 0.5 * value_loss
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        after = parameter_digest(model)
        if (
            recorded.get("parameter_state_before") != before
            or recorded.get("parameter_state_after") != after
            or recorded.get("parameter_state_sha256") != after
            or recorded.get("policy_loss") != float(policy_loss.detach())
            or recorded.get("value_loss") != float(value_loss.detach())
            or recorded.get("gradient_norm_preclip") != clip_norm
            or int(optimizer.state[next(model.parameters())]["step"].item())
            != spec["update"]
        ):
            raise ValueError(f"432_independent_replay_mismatch:{spec['update']}")
        if spec["update"] <= 108:
            different = control_mismatches(reference[spec["update"] - 1], recorded)
            if different:
                raise ValueError(
                    f"432_registered_control_mismatch:{spec['update']}:{','.join(different)}"
                )
        if spec["update"] in experiment.CHECKPOINTS:
            checkpoint = output / f"D-update-{spec['update']:03d}.npz"
            with np.load(checkpoint) as saved:
                current = train.checkpoint_from_model(model)
                if set(saved.files) != set(current) or any(
                    not np.array_equal(saved[name], current[name]) for name in current
                ):
                    raise ValueError(f"432_checkpoint_replay_mismatch:{spec['update']}")
    with (
        np.load(output / "D-update-108.npz") as actual,
        np.load(experiment.REFERENCE / "D-update-108.npz") as registered,
    ):
        if set(actual.files) != set(registered.files) or any(
            not np.array_equal(actual[name], registered[name]) for name in actual.files
        ):
            raise ValueError("432_update108_registered_checkpoint_mismatch")


def verify(output: Path) -> dict[str, Any]:
    _registration, schedule = frozen_inputs(output)
    status = json.loads((output / "run-status.json").read_text())
    if status.get("status") != "completed" or status.get("control_passed") is not True:
        raise ValueError("432_experiment_not_complete")
    independently_replay(output, schedule)
    actual = json.loads((output / "development-metrics.json").read_text())
    predictions = [
        json.loads(line)
        for line in (output / "development-predictions.jsonl").read_text().splitlines()
    ]
    names, x, p, value = publication.base.retained_development()
    expected_predictions = []
    expected_metrics = {}
    expected_identity = {}
    for endpoint, checkpoint in (
        ("initializer", experiment.CORPUS / "initializer.npz"),
        *(
            (f"update-{step}", output / f"D-update-{step:03d}.npz")
            for step in experiment.CHECKPOINTS
        ),
    ):
        metrics, rows = publication.base.evaluate(checkpoint, names, x, p, value)
        expected_metrics[endpoint] = metrics
        expected_identity[endpoint] = {
            "sha256": digest(checkpoint),
            "rows": len(names),
            "input_identities": len({state.tobytes() for state in x}),
        }
        for row in rows:
            row["endpoint"] = endpoint
            expected_predictions.append(row)
    if (
        actual.get("metrics") != expected_metrics
        or actual.get("checkpoint_identity") != expected_identity
        or actual.get("sole_candidate_checkpoint") != 432
        or predictions != expected_predictions
    ):
        raise ValueError("432_development_prediction_or_metric_reconciliation_failed")
    return {
        "status": "verified_complete",
        "updates": 432,
        "training_replay_matches": True,
        "registered_first_108_match": True,
        "development_reconciles": True,
        "candidate_update": 432,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps(verify(args.out.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
