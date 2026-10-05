#!/usr/bin/env python3
"""Run the preregistered seed422 Adam first-moment memory ablation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.seed416_policy_target_softening import audit_jsonl_derivative
from ml.alphazero_lite.seed422_adam_memory_analysis import analyze, make_adam
from ml.alphazero_lite.seed422_exclusions import ROOT, build_union, make_suite, sha256
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    split_replay_positions_by_source_row,
    train_one_epoch,
)

DATA = ROOT / "docs/data/seed422-adam-first-moment"
WORK = ROOT / ".tmp/seed422-adam-first-moment"
HISTORIC = ROOT / "docs/data/seed416-policy-target-softening"
HIST_REG = HISTORIC / "registration-v3.json"
HIST_BINDING = HISTORIC / "evaluation-binding.json"
INIT_HASH = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
EXPECTED_A_E4 = "836fc769c62126b649ce9691494719506b88efec0746769c9a589522d08cb2cd"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
NATIVE_PROBE = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
RUNTIME_POLICY = ROOT / "model-artifact/current/search_policy.json"
SOURCE_FILES = (
    "ml/alphazero_lite/run_seed422_adam_memory.py",
    "ml/alphazero_lite/seed422_adam_memory_analysis.py",
    "ml/alphazero_lite/seed422_exclusions.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/build_opening_suite.py",
    "ml/alphazero_lite/evaluation_seed_contract.py",
    "ml/alphazero_lite/export_artifact.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/seed416_policy_target_softening.py",
    "ml/alphazero_lite/test_seed422_adam_memory.py",
    "ml/alphazero_lite/verify_seed422_adam_memory.py",
)


def write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def record_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_overwrite(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable_seed422_output_exists:{path}")


def register() -> None:
    registration_path = DATA / "registration.json"
    no_overwrite(registration_path)
    no_overwrite(DATA / "opening-exclusion-proof.json")
    no_overwrite(DATA / "openings.jsonl")
    if (
        sha256(HIST_REG)
        != "4839ed63d6a48945ea11ea4995570f11118adbd65c027f49c3c5f04066bbd066"
    ):
        raise ValueError("#416 registration v3 hash changed")
    historical = json.loads(HIST_REG.read_text())
    if [int(item["weight"]) for item in historical["replays"]] != [1, 4, 1, 8, 4]:
        raise ValueError("#416 replay weights mismatch")
    if sha256(Path(historical["seed455_initialization"]["path"])) != INIT_HASH:
        raise ValueError(f"seed455 initializer hash mismatch:{INIT_HASH}")

    exclusions, proof = build_union()
    suite_rows = make_suite(exclusions)
    write_json_new(DATA / "opening-exclusion-proof.json", proof)
    write_jsonl_new(DATA / "openings.jsonl", suite_rows)
    suite_ids = suites.validate_arena_entries(suite_rows)
    if len(suite_ids) != 512 or set(suite_ids) & exclusions:
        raise ValueError("prospective_suite_validation_failed")

    replay_bindings = []
    for record in historical["replays"]:
        derivative = Path(historical["derivatives"][record["name"]]["A"]["derivative"])
        derivative_audit = audit_jsonl_derivative(Path(record["path"]), derivative, "A")
        expected = historical["derivatives"][record["name"]]["A"]["derivative_sha256"]
        if derivative_audit["derivative_sha256"] != expected:
            raise ValueError(f"#416 original A target mismatch:{record['name']}")
        replay_bindings.append(
            {
                "name": record["name"],
                "source_path": record["path"],
                "source_sha256": record["sha256"],
                "A_target_path": str(derivative),
                "A_target_sha256": expected,
                "weight": int(record["weight"]),
                "value_target_mode": record["value_target_mode"],
            }
        )
    hist_train = historical["training"]
    split = hist_train["source_row_split"]
    permutations = hist_train["epoch_permutations"]
    train_freeze = HISTORIC / "training-freeze-v3"
    split_path = train_freeze / "source-row-split.json.gz"
    permutation_path = train_freeze / "epoch-permutations.json.gz"
    input_hashes = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    snapshot_root = DATA / "execution-source-snapshots"
    for name in SOURCE_FILES:
        destination = snapshot_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
    hist_runtime = json.loads(HIST_BINDING.read_text())
    opponent_hashes = {
        name: sha256(OPPONENT / name)
        for name in ("metadata.json", "search_policy.json", "weights.json")
    }
    for name, digest in hist_runtime["opponent_files"].items():
        if opponent_hashes[name] != digest:
            raise ValueError(f"frozen_seed455_opponent_hash_mismatch:{name}")
    runtime_hashes = {
        "native_probe": sha256(NATIVE_PROBE),
        "tablebase": sha256(TABLEBASE),
        "runtime_policy": sha256(RUNTIME_POLICY),
    }
    expected_runtime = {
        "native_probe": hist_runtime["native_probe_sha256"],
        "tablebase": hist_runtime["tablebase_sha256"],
        "runtime_policy": hist_runtime["runtime_policy_sha256"],
    }
    if runtime_hashes != expected_runtime:
        raise ValueError("frozen_runtime_identity_mismatch")

    registration = {
        "schema": "seed422-adam-first-moment-registration-v1",
        "status": "registered_before_training_and_model_probing",
        "hypothesis": "Adam accumulated first-moment state may contribute to harmful training dynamics despite aggregate replay-gradient alignment; unconfirmed.",
        "intervention": {
            "A": {"optimizer": "Adam", "betas": [0.9, 0.999]},
            "B": {"optimizer": "Adam", "betas": [0.0, 0.999]},
            "common": {
                "lr": 0.001,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "scheduler": "none",
            },
        },
        "training": {
            "replays": replay_bindings,
            "weights": [1, 4, 1, 8, 4],
            "fresh_value_target_mode": "default",
            "historical_value_target_mode": "sharpened",
            "initializer": {
                "path": historical["seed455_initialization"]["path"],
                "sha256": INIT_HASH,
            },
            "initializer_seed": hist_train["seed"],
            "architecture": hist_train["architecture"],
            "source_row_split": {
                "path": str(split_path.relative_to(ROOT)),
                "sha256": sha256(split_path),
                **split,
            },
            "epoch_permutations": {
                "path": str(permutation_path.relative_to(ROOT)),
                "sha256": sha256(permutation_path),
                **permutations,
            },
            "batch_size": 512,
            "epochs": 4,
            "expected_updates_per_lane": 1052,
            "final_checkpoint": "fixed E4",
            "value_loss_weight": 0.3,
            "value_loss": "huber",
            "huber_delta": 1,
            "gradient_clip_global_norm": 1,
            "expected_A_E4_checkpoint_sha256": EXPECTED_A_E4,
            "expected_A_identity_source": "published #416 evaluation-binding candidate A checkpoint_sha256 and model.npz identity",
        },
        "exclusion_proof": {
            "path": "opening-exclusion-proof.json",
            "sha256": sha256(DATA / "opening-exclusion-proof.json"),
            "identity_set_sha256": proof["identity_set_sha256"],
            "identity_count": proof["identity_count"],
            "suite_overlap": 0,
        },
        "suite": {
            "path": "openings.jsonl",
            "sha256": sha256(DATA / "openings.jsonl"),
            "count": len(suite_rows),
            "seed": 422,
            "strictly_replayed": True,
            "unique_nonterminal_gt32": True,
        },
        "evaluation": {
            "candidate_lanes": ["A", "B"],
            "opponent": str(OPPONENT.relative_to(ROOT)),
            "opponent_files": opponent_hashes,
            "runtime_hashes": runtime_hashes,
            "games": 2048,
            "games_per_lane": 1024,
            "both_seats": True,
            "simulations_per_side": 384,
            "c_puct": 1.25,
            "seed": 422,
            "seed_contract": "azlite_eval_seed_v2",
            "native_root_solve_threshold": 16,
            "exact_leaf_solve": "disabled",
            "adaptive_extension": False,
            "new_outcomes_only": True,
        },
        "analysis": {
            "primary": "paired B-minus-A opening-average score",
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 422,
            "cluster": "opening",
            "confidence": 0.95,
            "advance": "B-A >= 0.03 AND paired lower bound > 0 AND B score >= 0.53 AND B opening-cluster lower bound > 0.5",
            "otherwise": "retain baseline and close fixed beta1 intervention",
            "checkpoint_selection": "none; fixed final E4",
            "beta_sweep": False,
            "promotion": False,
        },
        "source_hashes": input_hashes,
        "source_snapshots": "execution-source-snapshots/",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
    }
    registration["registration_sha256"] = hashlib.sha256(
        json.dumps(registration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json_new(registration_path, registration)


def read_registration() -> dict[str, Any]:
    path = DATA / "registration.json"
    if not path.is_file():
        raise FileNotFoundError("seed422 registration is not frozen")
    registration = json.loads(path.read_text())
    for name, digest in registration["source_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"frozen_source_hash_mismatch:{name}")
        snapshot = DATA / registration["source_snapshots"] / name
        if sha256(snapshot) != digest:
            raise ValueError(f"source_snapshot_hash_mismatch:{name}")
    return registration


def load_training_data(registration: dict[str, Any]) -> tuple[Any, ...]:
    sources = registration["training"]["replays"]
    paths = [Path(item["A_target_path"]) for item in sources]
    modes = [item["value_target_mode"] for item in sources]
    set_seed(416)
    data = load_jsonl_replay(
        paths,
        [int(item["weight"]) for item in sources],
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=modes,
        include_policy_loss_weights=True,
    )
    x, policy, value, replay, loss_weights = data
    frozen = gzip.open(HISTORIC / "training-freeze-v3/source-row-split.json.gz", "rt")
    split = json.load(frozen)
    train_positions, validation_positions = split_replay_positions_by_source_row(
        replay, val_split=0.1
    )
    if not np.array_equal(train_positions, np.asarray(split["train_positions"])):
        raise ValueError("#416 source-row train split mismatch")
    if not np.array_equal(
        validation_positions, np.asarray(split["validation_positions"])
    ):
        raise ValueError("#416 source-row validation split mismatch")
    if (
        hashlib.sha256(replay.tobytes()).hexdigest()
        != "529cca2a76e2230c66a1ee8c757d6eac8cd7b30ad638bd698cde16cf4ae558ef"
    ):
        raise ValueError("#416 multiplicity identity mismatch")
    return x, policy, value, replay, loss_weights, train_positions, validation_positions


def train_lanes() -> None:
    registration = read_registration()
    progress_path = WORK / "training-progress.json"
    WORK.mkdir(parents=True, exist_ok=True)
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"registration_sha256": sha256(DATA / "registration.json"), "lanes": {}}
    )
    if progress["registration_sha256"] != sha256(DATA / "registration.json"):
        raise ValueError("resume_training_registration_mismatch")
    x, policy, value, replay, loss_weights, train_positions, _ = load_training_data(
        registration
    )
    train_indexes = replay[train_positions]
    split_file = gzip.open(
        HISTORIC / "training-freeze-v3/epoch-permutations.json.gz", "rt"
    )
    expected_permutations = json.load(split_file)
    results: dict[str, Any] = dict(progress.get("lanes", {}))
    for lane in ("A", "B"):
        lane_dir = WORK / "training" / lane
        lane_dir.mkdir(parents=True, exist_ok=True)
        completed = results.get(lane)
        if completed:
            for epoch, digest in completed["epoch_checkpoints"].items():
                checkpoint = lane_dir / f"E{epoch}.npz"
                if not checkpoint.is_file() or sha256(checkpoint) != digest:
                    raise ValueError(
                        f"resume_checkpoint_identity_mismatch:{lane}:E{epoch}"
                    )
            if len(completed["epoch_checkpoints"]) == 4:
                continue
            raise ValueError(f"partial_lane_resume_not_permitted:{lane}")
        set_seed(416)
        model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        load_checkpoint_into_model(
            model, Path(registration["training"]["initializer"]["path"])
        )
        before = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        optimizer = make_adam(model, lane)
        group = optimizer.param_groups[0]
        actual_options = {
            key: group[key] for key in ("lr", "betas", "eps", "weight_decay")
        }
        expected = registration["intervention"]["A" if lane == "A" else "B"]
        if tuple(actual_options["betas"]) != tuple(expected["betas"]):
            raise ValueError("optimizer_beta_configuration_mismatch")
        history = []
        hashes = {}
        permutation_hashes = {}
        updates = 0
        rng_state = None
        for epoch in range(1, 5):
            generator = torch.Generator(device="cpu")
            if epoch == 1:
                generator.manual_seed(416)
            else:
                assert rng_state is not None
                generator.set_state(rng_state)
            observed_permutations: list[list[int]] = []

            def permutation_callback(_epoch: int | None, values: list[int]) -> None:
                observed_permutations.append(values)

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
            rng_state = generator.get_state()
            if len(observed_permutations) != 1:
                raise ValueError("epoch_permutation_callback_count_mismatch")
            permutation = observed_permutations[0]
            permutation_hash = hashlib.sha256(
                json.dumps(permutation).encode()
            ).hexdigest()
            permutation_hashes[str(epoch)] = permutation_hash
            if permutation != expected_permutations[epoch - 1]:
                raise ValueError(f"frozen_epoch_permutation_mismatch:{lane}:{epoch}")
            updates += int(metrics["optimizer_updates"] or 0)
            history.append({"epoch": epoch, **metrics})
            checkpoint = lane_dir / f"E{epoch}.npz"
            with checkpoint.open("xb") as stream:
                np.savez(stream, **checkpoint_from_model(model))
            hashes[str(epoch)] = sha256(checkpoint)
            if epoch < 4:
                continue
        if updates != 1052:
            raise ValueError(f"optimizer_update_count_mismatch:{lane}:{updates}")
        if (
            permutation_hashes
            != registration["training"]["epoch_permutations"]["epoch_sha256"]
        ):
            raise ValueError(f"frozen_permutation_identity_mismatch:{lane}")
        if lane == "A" and hashes["4"] != EXPECTED_A_E4:
            raise ValueError(
                f"#416 A reproduction failed: expected {EXPECTED_A_E4}, observed {hashes['4']}"
            )
        after = checkpoint_from_model(model)
        squared = sum(
            float(np.square(after[name] - before[name].numpy()).sum()) for name in after
        )
        results[lane] = {
            "epochs": hashes,
            "initialization_sha256": INIT_HASH,
            "multiplicity_sha256": hashlib.sha256(replay.tobytes()).hexdigest(),
            "permutations": permutation_hashes,
            "optimizer_parameter_group": {
                "lr": float(group["lr"]),
                "betas": list(group["betas"]),
                "eps": float(group["eps"]),
                "weight_decay": float(group["weight_decay"]),
            },
            "optimizer_updates": updates,
            "history": history,
            "parameter_drift_l2": float(np.sqrt(squared)),
        }
        progress["lanes"] = results
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    if results["A"]["permutations"] != results["B"]["permutations"]:
        raise ValueError("paired_batch_plan_mismatch")
    public = {
        "registration_sha256": sha256(DATA / "registration.json"),
        "lanes": results,
        "fixed_checkpoint": "E4",
        "optimizer_updates_per_lane": 1052,
    }
    write_json_new(DATA / "training-results.json", public)


def bind_artifacts() -> None:
    registration = read_registration()
    training_path = DATA / "training-results.json"
    training = json.loads(training_path.read_text())
    if training["registration_sha256"] != sha256(DATA / "registration.json"):
        raise ValueError("training_result_registration_binding_mismatch")
    binding = {
        "schema": "seed422-runtime-binding-v1",
        "registration_sha256": sha256(DATA / "registration.json"),
        "training_sha256": sha256(training_path),
        "suite_sha256": sha256(DATA / "openings.jsonl"),
        "candidates": {},
    }
    for lane in ("A", "B"):
        checkpoint = WORK / "training" / lane / "E4.npz"
        checkpoint_hash = sha256(checkpoint)
        expected_hash = training["lanes"][lane]["epochs"]["4"]
        if checkpoint_hash != expected_hash:
            raise ValueError(f"E4_checkpoint_binding_mismatch:{lane}")
        if lane == "A" and checkpoint_hash != EXPECTED_A_E4:
            raise ValueError("A_model_identity_does_not_reproduce_416")
        artifact = WORK / "artifacts" / lane
        if artifact.exists():
            raise FileExistsError(f"candidate_artifact_already_exists:{artifact}")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                "--checkpoint",
                str(checkpoint),
                "--out-dir",
                str(artifact),
                "--version",
                f"seed422-adam-{lane}-e4",
                "--model-type",
                "residual_v3",
                "--rules-version",
                "kalah_v1",
                "--input-encoding",
                "kalah_v3",
            ],
            cwd=ROOT,
            check=True,
        )
        shutil.copy2(RUNTIME_POLICY, artifact / "search_policy.json")
        if sha256(artifact / "model.npz") != checkpoint_hash:
            raise ValueError(f"exported_model_identity_mismatch:{lane}")
        if sha256(artifact / "search_policy.json") != sha256(RUNTIME_POLICY):
            raise ValueError(f"exported_search_policy_mismatch:{lane}")
        contract = resolve_strength_comparison_runtime_contract(
            current_artifact=OPPONENT,
            challenger_artifact=artifact,
            exact_root_threshold=16,
            native_probe=NATIVE_PROBE,
            tablebase=TABLEBASE,
        )
        if contract is None:
            raise ValueError("frozen_seed455_runtime_contract_missing")
        if lane == "A":
            binding["runtime_contract"] = contract
        elif binding["runtime_contract"] != contract:
            raise ValueError("candidate_runtime_contract_mismatch")
        artifact_hashes = {
            path.name: sha256(path)
            for path in sorted(artifact.iterdir())
            if path.is_file()
        }
        if not {"metadata.json", "weights.json", "search_policy.json"}.issubset(
            artifact_hashes
        ):
            raise ValueError("candidate_sidecar_incomplete")
        binding["candidates"][lane] = {
            "checkpoint_sha256": checkpoint_hash,
            "artifact_path": str(artifact.relative_to(ROOT)),
            "artifact_files": artifact_hashes,
            "runtime_contract": contract,
        }
    binding["opponent"] = {
        "path": str(OPPONENT.relative_to(ROOT)),
        "files": registration["evaluation"]["opponent_files"],
    }
    binding["native_probe_sha256"] = registration["evaluation"]["runtime_hashes"][
        "native_probe"
    ]
    binding["tablebase_sha256"] = registration["evaluation"]["runtime_hashes"][
        "tablebase"
    ]
    binding["runtime_policy_sha256"] = registration["evaluation"]["runtime_hashes"][
        "runtime_policy"
    ]
    write_json_new(DATA / "runtime-binding.json", binding)


def evaluate() -> None:
    registration = read_registration()
    binding_path = DATA / "runtime-binding.json"
    binding = json.loads(binding_path.read_text())
    if binding["registration_sha256"] != sha256(DATA / "registration.json"):
        raise ValueError("runtime_binding_registration_mismatch")
    if binding["training_sha256"] != sha256(DATA / "training-results.json"):
        raise ValueError("runtime_binding_training_mismatch")
    for lane, candidate in binding["candidates"].items():
        artifact = ROOT / candidate["artifact_path"]
        for name, digest in candidate["artifact_files"].items():
            if sha256(artifact / name) != digest:
                raise ValueError(f"candidate_artifact_changed:{lane}:{name}")
    for name, digest in registration["evaluation"]["opponent_files"].items():
        if sha256(OPPONENT / name) != digest:
            raise ValueError(f"opponent_artifact_changed:{name}")
    for path, key in (
        (NATIVE_PROBE, "native_probe_sha256"),
        (TABLEBASE, "tablebase_sha256"),
    ):
        if sha256(path) != binding[key]:
            raise ValueError(f"native_runtime_identity_changed:{path}")
    if sha256(RUNTIME_POLICY) != binding["runtime_policy_sha256"]:
        raise ValueError("runtime_policy_identity_changed")
    WORK.mkdir(parents=True, exist_ok=True)
    outcomes_path = DATA / "outcome-binding.json"
    state = (
        json.loads(outcomes_path.read_text())
        if outcomes_path.exists()
        else {"runtime_binding_sha256": sha256(binding_path), "lanes": {}}
    )
    if state["runtime_binding_sha256"] != sha256(binding_path):
        raise ValueError("resume_outcome_binding_mismatch")
    result_lanes = dict(state["lanes"])
    for lane in ("A", "B"):
        game_file = WORK / f"{lane}-games.jsonl"
        report_file = WORK / f"{lane}-arena.json"
        if lane in result_lanes:
            expected = result_lanes[lane]
            if (
                sha256(game_file) != expected["games_sha256"]
                or sha256(report_file) != expected["report_sha256"]
            ):
                raise ValueError(f"resume_arena_identity_mismatch:{lane}")
            continue
        if game_file.exists() or report_file.exists():
            raise ValueError(f"unbound_partial_arena_output:{lane}")
        artifact = ROOT / binding["candidates"][lane]["artifact_path"]
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "ml/alphazero_lite/arena.py"),
                "--challenger",
                str(artifact),
                "--current",
                str(OPPONENT),
                "--games",
                "1024",
                "--games-per-opening",
                "2",
                "--opening-prefixes-jsonl",
                str(DATA / "openings.jsonl"),
                "--suite-sha256",
                sha256(DATA / "openings.jsonl"),
                "--challenger-simulations",
                "384",
                "--current-simulations",
                "384",
                "--seed",
                "422",
                "--workers",
                "24",
                "--c-puct",
                "1.25",
                "--seed-contract",
                "azlite_eval_seed_v2",
                "--exact-root-solve-threshold",
                "16",
                "--exact-root-native-probe",
                str(NATIVE_PROBE),
                "--exact-root-tablebase",
                str(TABLEBASE),
                "--game-jsonl",
                str(game_file),
                "--out",
                str(report_file),
            ],
            cwd=ROOT,
            check=True,
        )
        games = jsonl_rows(game_file)
        report = json.loads(report_file.read_text())
        if len(games) != 1024 or int(report["games_played"]) != 1024:
            raise ValueError(f"arena_game_count_invalid:{lane}")
        result_lanes[lane] = {
            "games_sha256": sha256(game_file),
            "report_sha256": sha256(report_file),
            "games": 1024,
            "report": report,
        }
        state["lanes"] = result_lanes
        state["runtime_binding_sha256"] = sha256(binding_path)
        if lane == "A":
            write_json_new(outcomes_path, state)
        else:
            outcomes_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    combined = []
    for lane in ("A", "B"):
        for row in jsonl_rows(WORK / f"{lane}-games.jsonl"):
            combined.append(
                {
                    "lane": lane,
                    "opening_id": str(row["opening_index"]),
                    "opponent_score": 1.0
                    if row["winner"] == "challenger"
                    else 0.5
                    if row["winner"] == "draw"
                    else 0.0,
                    "game": row,
                }
            )
    suite = jsonl_rows(DATA / "openings.jsonl")
    from ml.alphazero_lite.seed422_adam_memory_analysis import validate_ledger

    validate_ledger(combined, suite)
    write_jsonl_new(DATA / "outcome-ledger.jsonl", combined)


def publish_analysis() -> None:
    rows = jsonl_rows(DATA / "outcome-ledger.jsonl")
    result = analyze(rows)
    result["registration_sha256"] = sha256(DATA / "registration.json")
    result["runtime_binding_sha256"] = sha256(DATA / "runtime-binding.json")
    result["outcome_ledger_sha256"] = sha256(DATA / "outcome-ledger.jsonl")
    write_json_new(DATA / "analysis.json", result)
    write_json_new(DATA / "opening-matrix.json", result["opening_matrix"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("register", "train", "bind", "evaluate", "analyze")
    )
    args = parser.parse_args()
    {
        "register": register,
        "train": train_lanes,
        "bind": bind_artifacts,
        "evaluate": evaluate,
        "analyze": publish_analysis,
    }[args.stage]()


if __name__ == "__main__":
    main()
