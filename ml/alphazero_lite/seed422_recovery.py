"""Post-A bookkeeping correction and resumable B lane for seed422."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite.run_seed422_adam_memory import (
    DATA,
    EXPECTED_A_E4,
    HISTORIC,
    INIT_HASH,
    WORK,
    load_training_data,
    read_registration,
    sha256,
    write_json_new,
)
from ml.alphazero_lite.seed422_adam_memory_analysis import make_adam
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    load_checkpoint_into_model,
    set_seed,
    train_one_epoch,
)

ROOT = Path(__file__).resolve().parents[2]
HIST_RESULTS = HISTORIC / "training-results.json"
AMENDMENT = DATA / "execution-accounting-amendment.json"


def _model_drift_l2(initializer: Path, checkpoint: Path) -> float:
    initial = np.load(initializer)
    final = np.load(checkpoint)
    if set(initial.files) != set(final.files):
        raise ValueError("checkpoint_parameter_key_set_mismatch")
    squared = sum(
        float(np.square(final[name] - initial[name]).sum()) for name in initial.files
    )
    return float(np.sqrt(squared))


def reconcile_resume_progress(
    progress: dict[str, Any], saved_progress: dict[str, Any]
) -> dict[str, Any]:
    previous_epoch = int(progress["completed_epoch"])
    saved_epoch = int(saved_progress["completed_epoch"])
    if saved_epoch not in {previous_epoch, previous_epoch + 1}:
        raise ValueError("B_resume_epoch_state_conflict")
    if (
        saved_epoch == previous_epoch + 1
        and len(saved_progress["history"]) != saved_epoch
    ):
        raise ValueError("B_resume_history_count_conflict")
    return saved_progress


def _prepare_amendment() -> dict[str, Any]:
    registration = read_registration()
    if AMENDMENT.exists():
        raise FileExistsError(f"immutable_execution_amendment_exists:{AMENDMENT}")
    historical = json.loads(HIST_RESULTS.read_text())
    old_lane = historical["lanes"]["A"]
    lane_dir = WORK / "training/A"
    observed = {str(epoch): sha256(lane_dir / f"E{epoch}.npz") for epoch in range(1, 5)}
    if observed != old_lane["epochs"] or observed["4"] != EXPECTED_A_E4:
        raise ValueError("A_complete_epoch_identity_does_not_reproduce_416")
    source = Path(__file__)
    snapshot = DATA / "execution-source-snapshots/seed422_recovery.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("xb") as stream:
        stream.write(source.read_bytes())
    amendment = {
        "schema": "seed422-post-A-bookkeeping-amendment-v1",
        "timing": "after complete A E4; before B execution",
        "registration_sha256": sha256(DATA / "registration.json"),
        "reason": "The frozen driver completed A's 1,052 updates and wrote E1–E4, then its post-training parameter-drift report looked up model.state_dict keys using the exported checkpoint key namespace and raised KeyError('w_policy'). No optimizer step or checkpoint write was affected.",
        "original_frozen_driver_sha256": registration["source_hashes"][
            "ml/alphazero_lite/run_seed422_adam_memory.py"
        ],
        "failed_post_training_lookup": "state_dict['w_policy']; checkpoint_from_model exports that tensor as 'w_policy' while state_dict uses module parameter names",
        "A_epoch_checkpoint_hashes": observed,
        "A_identity_reproduced": True,
        "A_history_source": {
            "path": str(HIST_RESULTS.relative_to(ROOT)),
            "sha256": sha256(HIST_RESULTS),
            "justification": "All four A epoch checkpoint hashes and the four frozen permutation hashes match #416 exactly; the optimizer/target/split/initialization settings are also the frozen #416 A settings.",
        },
        "supplemental_execution_source": {
            "path": "ml/alphazero_lite/seed422_recovery.py",
            "sha256": sha256(source),
            "snapshot": str(snapshot.relative_to(DATA)),
            "snapshot_sha256": sha256(snapshot),
        },
        "protocol_changes": [],
        "no_extra_training_samples_or_arena_games": True,
    }
    write_json_new(AMENDMENT, amendment)
    return amendment


def _a_record(
    registration: dict[str, Any], amendment: dict[str, Any]
) -> dict[str, Any]:
    historical = json.loads(HIST_RESULTS.read_text())
    old = historical["lanes"]["A"]
    initializer = Path(registration["training"]["initializer"]["path"])
    checkpoints = {
        str(epoch): sha256(WORK / f"training/A/E{epoch}.npz") for epoch in range(1, 5)
    }
    if checkpoints != old["epochs"]:
        raise ValueError("A_epoch_checkpoint_changed_after_reproduction")
    training_data = load_training_data(registration)
    replay = training_data[3]
    if hashlib.sha256(replay.tobytes()).hexdigest() != old["multiplicity_sha256"]:
        raise ValueError("A_multiplicity_hash_mismatch")
    return {
        "epochs": checkpoints,
        "initialization_sha256": INIT_HASH,
        "multiplicity_sha256": hashlib.sha256(replay.tobytes()).hexdigest(),
        "permutations": old["permutations"],
        "optimizer_parameter_group": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "optimizer_updates": 1052,
        "history": old["history"],
        "history_provenance": {
            "path": str(HIST_RESULTS.relative_to(ROOT)),
            "sha256": sha256(HIST_RESULTS),
            "matched_epoch_checkpoints": True,
            "matched_epoch_permutations": True,
            "execution_amendment_sha256": sha256(AMENDMENT),
        },
        "parameter_drift_l2": _model_drift_l2(initializer, WORK / "training/A/E4.npz"),
    }


def _train_b(registration: dict[str, Any]) -> dict[str, Any]:
    x, policy, value, replay, loss_weights, train_positions, _ = load_training_data(
        registration
    )
    train_indexes = replay[train_positions]
    freeze_path = HISTORIC / "training-freeze-v3/epoch-permutations.json.gz"
    import gzip

    with gzip.open(freeze_path, "rt", encoding="utf-8") as stream:
        expected_permutations = json.load(stream)
    lane_dir = WORK / "training/B"
    lane_dir.mkdir(parents=True, exist_ok=True)
    state_path = lane_dir / "resume-state.pt"
    progress_path = lane_dir / "resume-progress.json"
    if progress_path.exists() != state_path.exists():
        raise ValueError("B_resume_artifact_pair_incomplete")

    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["registration_sha256"] != sha256(DATA / "registration.json"):
            raise ValueError("B_resume_registration_mismatch")
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        progress = reconcile_resume_progress(progress, resume["progress"])
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        model.load_state_dict(resume["model"])
        optimizer = make_adam(model, "B")
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(progress["completed_epoch"]) + 1
        rng_state = resume["generator_state"]
        histories = list(progress["history"])
        epoch_hashes = dict(progress["epoch_checkpoints"])
        permutation_hashes = dict(progress["permutations"])
        updates = int(progress["optimizer_updates"])
    else:
        set_seed(416)
        model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        load_checkpoint_into_model(
            model, Path(registration["training"]["initializer"]["path"])
        )
        optimizer = make_adam(model, "B")
        start_epoch, rng_state, histories = 1, None, []
        epoch_hashes, permutation_hashes, updates = {}, {}, 0

    if start_epoch > 4:
        raise ValueError("B_resume_epoch_out_of_range")
    for epoch in range(start_epoch, 5):
        generator = torch.Generator(device="cpu")
        if rng_state is None:
            generator.manual_seed(416)
        else:
            generator.set_state(rng_state)
        callback_rows: list[list[int]] = []

        def permutation_callback(_epoch: int | None, permutation: list[int]) -> None:
            callback_rows.append(permutation)

        metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=x,
            compact_p=policy,
            compact_v=value,
            compact_policy_loss_weights=loss_weights,
            replay_indexes=train_indexes,
            batch_size=512,
            device=torch.device("cpu"),
            value_loss_weight=0.3,
            value_loss="huber",
            huber_delta=1.0,
            grad_clip=1.0,
            permutation_callback=permutation_callback,
            primary_order_generator=generator,
            epoch=epoch,
        )
        if (
            len(callback_rows) != 1
            or callback_rows[0] != expected_permutations[epoch - 1]
        ):
            raise ValueError(f"B_frozen_permutation_mismatch:E{epoch}")
        rng_state = generator.get_state()
        permutation_hashes[str(epoch)] = hashlib.sha256(
            json.dumps(callback_rows[0]).encode()
        ).hexdigest()
        updates += int(metrics["optimizer_updates"] or 0)
        histories.append({"epoch": epoch, **metrics})
        checkpoint = lane_dir / f"E{epoch}.npz"
        checkpoint_payload = checkpoint_from_model(model)
        if checkpoint.exists():
            with np.load(checkpoint) as existing:
                if any(
                    not np.array_equal(existing[name], checkpoint_payload[name])
                    for name in checkpoint_payload
                ):
                    raise ValueError(f"B_resume_checkpoint_conflict:E{epoch}")
        else:
            with checkpoint.open("xb") as stream:
                np.savez(stream, **checkpoint_payload)
        epoch_hashes[str(epoch)] = sha256(checkpoint)
        progress = {
            "registration_sha256": sha256(DATA / "registration.json"),
            "completed_epoch": epoch,
            "history": histories,
            "epoch_checkpoints": epoch_hashes,
            "permutations": permutation_hashes,
            "optimizer_updates": updates,
        }
        with state_path.open("wb") as stream:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "generator_state": rng_state,
                    "progress": progress,
                },
                stream,
            )
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    if (
        updates != 1052
        or permutation_hashes
        != registration["training"]["epoch_permutations"]["epoch_sha256"]
    ):
        raise ValueError("B_final_training_accounting_mismatch")
    initial_path = Path(registration["training"]["initializer"]["path"])
    final_checkpoint = lane_dir / "E4.npz"
    return {
        "epochs": epoch_hashes,
        "initialization_sha256": INIT_HASH,
        "multiplicity_sha256": hashlib.sha256(replay.tobytes()).hexdigest(),
        "permutations": permutation_hashes,
        "optimizer_parameter_group": {
            "lr": float(optimizer.param_groups[0]["lr"]),
            "betas": list(optimizer.param_groups[0]["betas"]),
            "eps": float(optimizer.param_groups[0]["eps"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        },
        "optimizer_updates": updates,
        "history": histories,
        "parameter_drift_l2": _model_drift_l2(initial_path, final_checkpoint),
    }


def complete_training() -> None:
    registration = read_registration()
    if not AMENDMENT.is_file():
        raise FileNotFoundError("post-A accounting amendment must be frozen first")
    amendment = json.loads(AMENDMENT.read_text())
    if amendment["registration_sha256"] != sha256(DATA / "registration.json"):
        raise ValueError("execution_amendment_registration_mismatch")
    if amendment["supplemental_execution_source"]["sha256"] != sha256(Path(__file__)):
        raise ValueError("supplemental_execution_source_changed")
    snapshot = DATA / amendment["supplemental_execution_source"]["snapshot"]
    if sha256(snapshot) != amendment["supplemental_execution_source"]["sha256"]:
        raise ValueError("supplemental_source_snapshot_hash_mismatch")
    lanes = {"A": _a_record(registration, amendment), "B": _train_b(registration)}
    if lanes["A"]["permutations"] != lanes["B"]["permutations"]:
        raise ValueError("paired_batch_plan_mismatch")
    if lanes["A"]["multiplicity_sha256"] != lanes["B"]["multiplicity_sha256"]:
        raise ValueError("paired_exposure_mismatch")
    if any(lane["optimizer_updates"] != 1052 for lane in lanes.values()):
        raise ValueError("paired_optimizer_update_mismatch")
    payload = {
        "registration_sha256": sha256(DATA / "registration.json"),
        "execution_amendment_sha256": sha256(AMENDMENT),
        "lanes": lanes,
        "fixed_checkpoint": "E4",
        "optimizer_updates_per_lane": 1052,
    }
    result_path = DATA / "training-results.json"
    if result_path.exists():
        if json.loads(result_path.read_text()) != payload:
            raise ValueError("completed_training_results_conflict")
    else:
        write_json_new(result_path, payload)


def prepare_amendment() -> None:
    _prepare_amendment()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare-amendment", "complete-training"))
    stage = parser.parse_args().stage
    (prepare_amendment if stage == "prepare-amendment" else complete_training)()
