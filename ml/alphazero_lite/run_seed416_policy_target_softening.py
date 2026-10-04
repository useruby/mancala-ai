#!/usr/bin/env python3
"""Register, train, evaluate, and publish the seed416 policy-softening ablation."""

from __future__ import annotations

import argparse
import gzip
import os
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import run_seed461_order_exploratory as training_helpers
from ml.alphazero_lite.arena import apply_opening_moves
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed416_policy_target_softening import (
    audit_jsonl_derivative,
    bootstrap_paired,
    resume_action,
    sha256,
    transform_jsonl,
)
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    train as train_model,
)
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed416-policy-target-softening"
WORK = ROOT / ".tmp/seed416-policy-target-softening"
GENERATION = (
    ROOT
    / "docs/data/alphazero-lite-generations/seed455-nextgen-s461-default-value-root16/generation.json"
)
CORRECTED_415 = (
    ROOT
    / "docs/data/seed414-root-budget-confirmation/post-execution-corrected-exclusion-proof.json"
)
REG_414 = ROOT / "docs/data/seed414-root-budget-confirmation/registration.json"
CONT_414 = ROOT / "docs/data/seed414-root-budget-confirmation/continuations.jsonl"
SUITE_414 = ROOT / "docs/data/seed414-root-budget-confirmation/suite.jsonl"
REGISTRATION = DATA / "registration-v3.json"
SUITE = DATA / "openings-v3.jsonl"
DERIVATIVES = WORK / "derivatives"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
NATIVE_PROBE = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
RUNTIME_POLICY = ROOT / "model-artifact/current/search_policy.json"
TRAIN_SOURCE = Path(__file__)
ANALYSIS_SOURCE = Path(__file__).with_name("seed416_policy_target_softening.py")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_identity(state: dict[str, Any]) -> str:
    return suites.canonical_key(state)


def strict_414_consumed() -> set[str]:
    registration = json.loads(REG_414.read_text(encoding="utf-8"))
    suite = [json.loads(line) for line in SUITE_414.read_text().splitlines() if line]
    consumed = {canonical_identity(row["state"]) for row in suite}
    consumed.update(canonical_identity(row["state"]) for row in registration["states"])
    for row in (json.loads(line) for line in CONT_414.read_text().splitlines() if line):
        index = int(row["opening_index"])
        opening = suite[index]
        game = KalahGame.from_state(suites.INITIAL_STATE)
        prefix = [int(move) for move in opening["prefix_moves"]]
        if apply_opening_moves(game, prefix) != len(prefix):
            raise ValueError(f"414_prefix_replay_failed:{index}")
        if canonical_identity(game.to_state()) != canonical_identity(opening["state"]):
            raise ValueError(f"414_opening_state_mismatch:{index}")
        consumed.add(canonical_identity(game.to_state()))
        trajectory = row["outcome"]["trajectory"]
        if not trajectory:
            raise ValueError(f"414_empty_continuation:{index}")
        for ply, item in enumerate(trajectory):
            action = int(item["action_absolute"])
            if int(item["actor"]) != game.current_player:
                raise ValueError(f"414_continuation_actor_mismatch:{index}:{ply}")
            if action != game.pit_index(int(item["action_relative"])):
                raise ValueError(
                    f"414_continuation_relative_action_mismatch:{index}:{ply}"
                )
            if (
                game.over()
                or game.pit_owner(action) != game.current_player
                or not game.move(action)
            ):
                raise ValueError(f"414_continuation_action_invalid:{index}:{ply}")
            if canonical_identity(game.to_state()) != canonical_identity(item["state"]):
                raise ValueError(f"414_continuation_state_mismatch:{index}:{ply}")
            consumed.add(canonical_identity(game.to_state()))
        if not game.over():
            raise ValueError(f"414_continuation_not_terminal:{index}")
        outcome = row["outcome"]
        root_player = int(outcome["root_player"])
        stores = [int(value) for value in outcome["stores"]]
        if stores != [int(value) for value in game.captured_seeds]:
            raise ValueError(f"414_continuation_store_mismatch:{index}")
        margin = stores[root_player] - stores[1 - root_player]
        if margin != int(outcome["store_margin_root_perspective"]):
            raise ValueError(f"414_continuation_outcome_mismatch:{index}")
    if registration["schema"] != "seed414-root-budget-confirmation-registration-v1":
        raise ValueError("414_registration_schema_mismatch")
    return consumed


def replay_records() -> tuple[list[dict[str, Any]], Path]:
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    sources = generation["replay"]["sources"]
    records = []
    for item in sources:
        path = ROOT / item["artifact"]["path"]
        records.append(
            {
                "name": item["name"],
                "path": str(path),
                "sha256": item["artifact"]["sha256"],
                "weight": int(item["weight"]),
                "value_target_mode": "default"
                if item["name"] == "fresh"
                else "sharpened",
            }
        )
    return records, ROOT / generation["training"]["config"]["init_checkpoint"]


def register() -> None:
    if REGISTRATION.exists():
        raise RuntimeError("seed416_registration_is_immutable")
    records, init = replay_records()
    if [row["weight"] for row in records] != [1, 4, 1, 8, 4]:
        raise ValueError("registered_replay_weights_mismatch")
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"registered_source_hash_mismatch:{path}")
    if not init.is_file():
        raise FileNotFoundError(init)
    corrected = json.loads(CORRECTED_415.read_text(encoding="utf-8"))
    excluded = set(corrected["excluded_state_identities"])
    declared_and_replayed = strict_414_consumed()
    excluded |= declared_and_replayed
    replay_training_identities: set[str] = set()
    for source in records:
        with Path(source["path"]).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                state = row["state"]
                if len(state) < 15:
                    raise ValueError(
                        f"malformed_replay_state:{source['name']}:{line_number}"
                    )
                game = KalahGame.from_state(
                    {
                        "player_pits": [int(round(float(v) * 48)) for v in state[:6]],
                        "opponent_pits": [
                            int(round(float(v) * 48)) for v in state[6:12]
                        ],
                        "player_store": int(round(float(state[12]) * 48)),
                        "opponent_store": int(round(float(state[13]) * 48)),
                        "current_player": int(round(float(state[14]))),
                    }
                )
                replay_training_identities.add(canonical_identity(game.to_state()))
    excluded |= replay_training_identities
    population_rows = suites.deduplicate_openings(suites.enumerate_legal_prefixes(8))[0]
    eligible = [
        row
        for row in population_rows
        if int(row["pit_sum"]) > 32
        and not KalahGame.from_state(row["state"]).over()
        and canonical_identity(row["state"]) not in excluded
    ]
    selected = [
        suites.export_arena_entry(row)
        for row in suites.select_diverse(suites.stratify_openings(eligible), 512, 416)
    ]
    ids = [canonical_identity(row["state"]) for row in selected]
    if len(set(ids)) != 512 or set(ids) & excluded:
        raise ValueError("opening_suite_identity_or_exclusion_failure")
    if SUITE.exists():
        existing_suite = suites.load_suite_jsonl(str(SUITE))
        if [canonical_identity(item["state"]) for item in existing_suite] != ids:
            raise ValueError("unregistered_existing_suite_mismatch")
    else:
        suites.write_suite_jsonl(selected, str(SUITE))
    suites.validate_arena_entries(suites.load_suite_jsonl(str(SUITE)))
    derivatives = {}
    for record in records:
        name = record["name"]
        derivatives[name] = {}
        for lane in ("A", "B"):
            target = DERIVATIVES / lane / f"{name}.jsonl"
            if target.exists():
                derivatives[name][lane] = audit_jsonl_derivative(
                    Path(record["path"]), target, lane
                )
            else:
                derivatives[name][lane] = transform_jsonl(
                    Path(record["path"]), target, lane
                )
    set_seed(416)
    # Freeze exact source-row split and minibatch permutations before either fit.
    _, _, _, replay_indices, _ = load_jsonl_replay(
        [Path(row["path"]) for row in records],
        [row["weight"] for row in records],
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=[row["value_target_mode"] for row in records],
        include_policy_loss_weights=True,
    )
    train_positions, val_positions = (
        training_helpers.train.split_replay_positions_by_source_row(
            replay_indices, val_split=0.1
        )
    )
    n_train = len(train_positions)
    gen = torch.Generator(device="cpu").manual_seed(416)
    perms = [torch.randperm(n_train, generator=gen).tolist() for _ in range(4)]
    freeze_dir = DATA / "training-freeze-v3"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "train_positions": train_positions.tolist(),
        "validation_positions": val_positions.tolist(),
        "train_source_rows": sorted(
            set(int(x) for x in replay_indices[train_positions])
        ),
        "validation_source_rows": sorted(
            set(int(x) for x in replay_indices[val_positions])
        ),
    }
    split_bytes = gzip.compress(
        json.dumps(split_payload, separators=(",", ":")).encode(), mtime=0
    )
    split_path = freeze_dir / "source-row-split.json.gz"
    if split_path.exists():
        if split_path.read_bytes() != split_bytes:
            raise ValueError("frozen_split_artifact_conflict")
    else:
        split_path.write_bytes(split_bytes)
    permutation_path = freeze_dir / "epoch-permutations.json.gz"
    permutation_bytes = gzip.compress(
        json.dumps(perms, separators=(",", ":")).encode(), mtime=0
    )
    if permutation_path.exists():
        if permutation_path.read_bytes() != permutation_bytes:
            raise ValueError("frozen_permutation_artifact_conflict")
    else:
        permutation_path.write_bytes(permutation_bytes)
    source_files = [
        TRAIN_SOURCE,
        ANALYSIS_SOURCE,
        ROOT / "ml/alphazero_lite/train.py",
        ROOT / "ml/alphazero_lite/self_play.py",
        ROOT / "ml/alphazero_lite/arena.py",
        ROOT / "ml/alphazero_lite/kalah_rules.py",
        ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py",
        ROOT / "ml/alphazero_lite/runtime_search_policy.py",
        ROOT / "ml/alphazero_lite/export_artifact.py",
        ROOT / "ml/alphazero_lite/build_opening_suite.py",
        ROOT / "ml/alphazero_lite/evaluation_seed_contract.py",
        ROOT / "ml/alphazero_lite/seed414_post_execution_audit.py",
        ROOT / "ml/alphazero_lite/verify_seed416_policy_target_softening.py",
    ]
    proof = {
        "schema": "seed416-exclusion-proof-v1",
        "corrected_415_proof": {
            "path": str(CORRECTED_415),
            "sha256": sha256(CORRECTED_415),
        },
        "corrected_415_union_count": corrected["corrected_union_count"],
        "414_declared_records": {
            "registration_sha256": sha256(REG_414),
            "suite_sha256": sha256(SUITE_414),
            "completed_continuations_sha256": sha256(CONT_414),
        },
        "414_declared_and_strictly_replayed_identity_count": len(declared_and_replayed),
        "seed455_training_replay_identity_count": len(replay_training_identities),
        "combined_union_count": len(excluded),
        "identity_sha256": hashlib.sha256(
            "\n".join(sorted(excluded)).encode()
        ).hexdigest(),
        "suite_overlap": len(set(ids) & excluded),
        "excluded_identities": sorted(excluded),
    }
    proof_path = DATA / "opening-exclusion-proof-v3.json"
    if proof_path.exists():
        if json.loads(proof_path.read_text(encoding="utf-8")) != proof:
            raise ValueError("frozen_exclusion_proof_conflict")
    else:
        write_json(proof_path, proof)
    write_json(
        REGISTRATION,
        {
            "schema": "seed416-policy-target-softening-registration-v1",
            "status": "registered_before_training_and_model_probing",
            "hypothesis": "Less concentrated policy supervision produces a stronger successor at the same search budget.",
            "derivative_semantics": "B applies sqrt to the stored sharpened policy distribution; it does not recover original visit counts.",
            "seed455_initialization": {"path": str(init), "sha256": sha256(init)},
            "frozen_evaluation_runtime": {
                "opponent_artifact": str(OPPONENT),
                "opponent_files": {
                    name: sha256(OPPONENT / name)
                    for name in ("weights.json", "metadata.json", "search_policy.json")
                },
                "native_probe": {
                    "path": str(NATIVE_PROBE),
                    "sha256": sha256(NATIVE_PROBE),
                },
                "tablebase": {"path": str(TABLEBASE), "sha256": sha256(TABLEBASE)},
                "runtime_policy": {
                    "path": str(RUNTIME_POLICY),
                    "sha256": sha256(RUNTIME_POLICY),
                },
            },
            "replays": records,
            "derivatives": derivatives,
            "training": {
                "seed": 416,
                "order_seed": 416,
                "architecture": {
                    "model": "residual_v3",
                    "hidden_sizes": [96, 3],
                    "input_encoding": "kalah_v3",
                },
                "epochs": 4,
                "batch_size": 512,
                "optimizer": "Adam",
                "lr": 0.001,
                "weight_decay": 0,
                "scheduler": "none",
                "value_loss": "huber",
                "huber_delta": 1,
                "value_loss_weight": 0.3,
                "grad_clip": 1,
                "validation_fraction": 0.1,
                "final_checkpoint": "E4; fixed; validation loss does not select",
                "source_row_split": {
                    "path": str(split_path.relative_to(ROOT)),
                    "sha256": sha256(split_path),
                    "train_positions_sha256": hashlib.sha256(
                        train_positions.tobytes()
                    ).hexdigest(),
                    "validation_positions_sha256": hashlib.sha256(
                        val_positions.tobytes()
                    ).hexdigest(),
                    "train_count": len(train_positions),
                    "validation_count": len(val_positions),
                },
                "epoch_permutations": {
                    "path": str(permutation_path.relative_to(ROOT)),
                    "sha256": sha256(permutation_path),
                    "epoch_sha256": {
                        str(index + 1): hashlib.sha256(
                            json.dumps(permutation).encode()
                        ).hexdigest()
                        for index, permutation in enumerate(perms)
                    },
                },
            },
            "suite": {
                "path": str(SUITE),
                "sha256": sha256(SUITE),
                "seed": 416,
                "openings": 512,
                "unique": True,
                "nonterminal": True,
                "active_stones_gt": 32,
                "overlap": 0,
            },
            "exclusion_proof": "opening-exclusion-proof-v3.json",
            "exclusion_proof_sha256": sha256(proof_path),
            "evaluation": {
                "games": 2048,
                "games_per_lane": 1024,
                "both_seats": True,
                "simulations_per_side": 384,
                "c_puct": 1.25,
                "seed": 416,
                "seed_contract": "azlite_eval_seed_v2",
                "opponent": "frozen seed455",
                "adaptive_extension": False,
                "primary": "paired mean B-minus-A score against seed455 across opening clusters",
            },
            "analysis": {
                "bootstrap_samples": 10000,
                "bootstrap_seed": 416,
                "confidence": 0.95,
                "cluster": "opening; matched across lanes",
            },
            "source_hashes": {
                str(path.relative_to(ROOT)): sha256(path) for path in source_files
            },
        },
    )
    write_json(
        DATA / "registration-supersession-v3.json",
        {
            "schema": "seed416-registration-supersession-v1",
            "superseded_registration": "registration-v2.json",
            "superseded_reason": "The pretraining runtime parity check compared unmasked model priors to the runtime's legal-masked policy. Runtime parameter hashes matched exactly; this corrected check masks legal actions before function comparison. No optimizer update or candidate arena probe occurred.",
            "replacement_registration": "registration-v3.json",
            "replacement_registration_sha256": sha256(REGISTRATION),
            "training_or_model_probes_preceded_supersession": True,
            "failed_initialization_preflight_was_candidate_strength_evidence": False,
            "optimizer_updates_preceded_supersession": False,
            "candidate_arena_probes_preceded_supersession": False,
        },
    )


def train_lanes() -> None:
    reg = json.loads(REGISTRATION.read_text())
    if reg["status"] != "registered_before_training_and_model_probing":
        raise ValueError("registration_status_invalid")
    source_records = reg["replays"]
    init_path = Path(reg["seed455_initialization"]["path"])
    if sha256(init_path) != reg["seed455_initialization"]["sha256"]:
        raise ValueError("seed455_initialization_hash_mismatch")
    init_artifact = WORK / "seed455-init-runtime-check"
    if not init_artifact.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                "--checkpoint",
                str(init_path),
                "--out-dir",
                str(init_artifact),
                "--version",
                "seed416-verified-seed455-initialization",
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
    if sha256(init_artifact / "weights.json") != sha256(OPPONENT / "weights.json"):
        raise ValueError("seed455_initialization_runtime_parameter_hash_mismatch")
    if (DATA / "training-results.json").exists():
        record = json.loads((DATA / "training-results.json").read_text())
        if record.get("registration_sha256") != sha256(REGISTRATION):
            raise ValueError("published_training_record_binding_mismatch")
        for lane, item in record["lanes"].items():
            for epoch, expected in item["epochs"].items():
                if sha256(WORK / "training" / lane / f"E{epoch}.npz") != expected:
                    raise ValueError(
                        f"published_training_checkpoint_mismatch:{lane}:{epoch}"
                    )
        return
    progress_path = WORK / "training-progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"registration_sha256": sha256(REGISTRATION), "lanes": {}}
    )
    if progress.get("registration_sha256") != sha256(REGISTRATION):
        raise ValueError("training_progress_registration_mismatch")
    trajectories = dict(progress.get("lanes", {}))
    for lane in ("A", "B"):
        if lane in trajectories:
            observed_epochs = {}
            for epoch, expected in trajectories[lane]["epochs"].items():
                checkpoint = WORK / "training" / lane / f"E{epoch}.npz"
                if not checkpoint.is_file() or sha256(checkpoint) != expected:
                    raise ValueError(
                        f"resumed_training_checkpoint_mismatch:{lane}:{epoch}"
                    )
                observed_epochs[epoch] = sha256(checkpoint)
            if (
                resume_action(
                    {"epochs": trajectories[lane]["epochs"]},
                    {"epochs": observed_epochs},
                )
                == "skip"
            ):
                continue
        derivative_paths = [
            Path(reg["derivatives"][item["name"]][lane]["derivative"])
            for item in source_records
        ]
        for item, path in zip(source_records, derivative_paths, strict=True):
            if (
                sha256(path)
                != reg["derivatives"][item["name"]][lane]["derivative_sha256"]
            ):
                raise ValueError(f"derivative_hash_mismatch:{lane}:{path}")
        set_seed(416)
        x, policy, value, replay, loss_weights = load_jsonl_replay(
            derivative_paths,
            [item["weight"] for item in source_records],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[
                item["value_target_mode"] for item in source_records
            ],
            include_policy_loss_weights=True,
        )
        model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        load_checkpoint_into_model(model, init_path)
        # The frozen initialization must be the exact policy/value function in
        # the immutable seed455 runtime, not merely a checkpoint with a known name.
        from ml.alphazero_lite.arena import ArtifactEvaluator

        runtime = ArtifactEvaluator(OPPONENT)
        check_rows = [
            json.loads(line)
            for line in Path(source_records[0]["path"]).open()
            if line.strip()
        ][:64]
        model.eval()
        with torch.no_grad():
            for check_row in check_rows:
                encoded = torch.tensor([check_row["state"]], dtype=torch.float32)
                logits, value_prediction = model(encoded)
                logits = logits[0].numpy().copy()
                logits -= logits.max()
                probs = np.exp(logits) / np.exp(logits).sum()
                probe_game = KalahGame.from_state(
                    {
                        "player_pits": [
                            int(round(v * 48)) for v in check_row["state"][:6]
                        ],
                        "opponent_pits": [
                            int(round(v * 48)) for v in check_row["state"][6:12]
                        ],
                        "player_store": int(round(check_row["state"][12] * 48)),
                        "opponent_store": int(round(check_row["state"][13] * 48)),
                        "current_player": int(round(check_row["state"][14])),
                    }
                )
                legal_mask = np.zeros(6, dtype=bool)
                legal_mask[probe_game.possible_moves()] = True
                probs[~legal_mask] = 0.0
                probs /= probs.sum()
                runtime_policy, runtime_value = runtime.evaluate(probe_game)
                if not np.allclose(probs, runtime_policy, atol=2e-6) or not np.isclose(
                    float(value_prediction.item()), runtime_value, atol=2e-6
                ):
                    raise ValueError(
                        "seed455_initialization_runtime_parameter_mismatch"
                    )
        init_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    key: val.tolist()
                    for key, val in checkpoint_from_model(model).items()
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        out = WORK / "training" / lane
        out.mkdir(parents=True, exist_ok=True)
        epoch_hashes = {}
        permutations = {}
        history = []

        def on_epoch(epoch, _optimizer, current):
            checkpoint = out / f"E{epoch}.npz"
            np.savez(checkpoint, **checkpoint_from_model(current))
            epoch_hashes[str(epoch)] = sha256(checkpoint)

        train_model(
            model,
            x,
            policy,
            value,
            replay,
            policy_loss_weights=loss_weights,
            epochs=4,
            batch_size=512,
            lr=0.001,
            device=torch.device("cpu"),
            value_loss_weight=0.3,
            value_loss="huber",
            huber_delta=1.0,
            val_split=0.1,
            grad_clip=1.0,
            save_top_k=0,
            lr_scheduler="none",
            weight_decay=0.0,
            final_checkpoint="final",
            primary_order_seed=416,
            epoch_history=history,
            epoch_callback=on_epoch,
            permutation_callback=lambda epoch, values: permutations.__setitem__(
                str(epoch), hashlib.sha256(json.dumps(values).encode()).hexdigest()
            ),
        )
        trajectories[lane] = {
            "epochs": epoch_hashes,
            "initialization_sha256": init_fingerprint,
            "history": history,
            "permutations": permutations,
            "metrics": model.last_train_metrics,
            "multiplicity_sha256": hashlib.sha256(replay.tobytes()).hexdigest(),
        }
        registered_permutations = reg["training"]["epoch_permutations"]["epoch_sha256"]
        if permutations != registered_permutations:
            raise ValueError(f"frozen_epoch_permutation_mismatch:{lane}")
        progress["lanes"] = trajectories
        write_json(progress_path, progress)
    if (
        trajectories["A"]["initialization_sha256"]
        != trajectories["B"]["initialization_sha256"]
        or trajectories["A"]["permutations"] != trajectories["B"]["permutations"]
        or trajectories["A"]["multiplicity_sha256"]
        != trajectories["B"]["multiplicity_sha256"]
    ):
        raise ValueError("paired_training_invariant_mismatch")
    for field in (
        "train_split_count",
        "validation_count",
        "train_split_sha256",
        "validation_split_sha256",
        "optimizer_updates",
        "full_epochs_completed",
    ):
        if trajectories["A"]["metrics"][field] != trajectories["B"]["metrics"][field]:
            raise ValueError(f"paired_training_metric_mismatch:{field}")
    if any(
        trajectories[lane]["metrics"]["full_epochs_completed"] != 4
        for lane in ("A", "B")
    ):
        raise ValueError("training_did_not_complete_four_epochs")
    write_json(
        DATA / "training-results.json",
        {"registration_sha256": sha256(REGISTRATION), "lanes": trajectories},
    )


def bind_candidates() -> dict[str, Any]:
    reg = json.loads(REGISTRATION.read_text())
    frozen_runtime = reg["frozen_evaluation_runtime"]
    for filename, expected in frozen_runtime["opponent_files"].items():
        if sha256(OPPONENT / filename) != expected:
            raise ValueError(f"frozen_opponent_identity_mismatch:{filename}")
    for name, path in (
        ("native_probe", NATIVE_PROBE),
        ("tablebase", TABLEBASE),
        ("runtime_policy", RUNTIME_POLICY),
    ):
        if sha256(path) != frozen_runtime[name]["sha256"]:
            raise ValueError(f"frozen_runtime_identity_mismatch:{name}")
    training_results = json.loads((DATA / "training-results.json").read_text())
    if training_results.get("registration_sha256") != sha256(REGISTRATION):
        raise ValueError("training_record_registration_mismatch")
    binding_path = DATA / "evaluation-binding.json"
    if binding_path.exists():
        binding = json.loads(binding_path.read_text())
        if binding["registration_sha256"] != sha256(REGISTRATION):
            raise ValueError("existing_evaluation_binding_registration_mismatch")
        return binding
    candidates = {}
    for lane in ("A", "B"):
        checkpoint = WORK / "training" / lane / "E4.npz"
        artifact = WORK / "artifacts" / lane
        staging = WORK / "artifacts" / f"{lane}.staging"
        if not artifact.exists():
            if staging.exists():
                shutil.rmtree(staging)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--out-dir",
                    str(staging),
                    "--version",
                    f"seed416-policy-softening-{lane}-E4",
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
            shutil.copy2(RUNTIME_POLICY, staging / "search_policy.json")
            if sha256(staging / "model.npz") != sha256(checkpoint):
                raise ValueError(f"exported_checkpoint_mismatch:{lane}")
            os.replace(staging, artifact)
        elif sha256(artifact / "model.npz") != sha256(checkpoint):
            raise ValueError(f"existing_candidate_checkpoint_mismatch:{lane}")
        if sha256(artifact / "search_policy.json") != sha256(RUNTIME_POLICY):
            raise ValueError(f"candidate_runtime_sidecar_mismatch:{lane}")
        contract = resolve_strength_comparison_runtime_contract(
            current_artifact=OPPONENT,
            challenger_artifact=artifact,
            native_probe=NATIVE_PROBE,
            tablebase=TABLEBASE,
        )
        if contract is None:
            raise ValueError("frozen_seed455_runtime_contract_missing")
        candidates[lane] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "artifact": str(artifact),
            "artifact_sha256": {
                name: sha256(artifact / name)
                for name in (
                    "model.npz",
                    "weights.json",
                    "metadata.json",
                    "search_policy.json",
                )
            },
        }
        candidates[lane]["runtime_contract"] = contract
    binding = {
        "schema": "seed416-evaluation-binding-v1",
        "registration_sha256": sha256(REGISTRATION),
        "training_sha256": sha256(DATA / "training-results.json"),
        "suite_sha256": sha256(SUITE),
        "opponent": str(OPPONENT),
        "opponent_files": {
            name: sha256(OPPONENT / name)
            for name in ("weights.json", "metadata.json", "search_policy.json")
        },
        "native_probe_sha256": sha256(NATIVE_PROBE),
        "tablebase_sha256": sha256(TABLEBASE),
        "runtime_policy_sha256": sha256(RUNTIME_POLICY),
        "runtime_contract": candidates["A"]["runtime_contract"],
        "candidates": candidates,
    }
    if candidates["A"]["runtime_contract"] != candidates["B"]["runtime_contract"]:
        raise ValueError("candidate_runtime_contract_mismatch")
    write_json(binding_path, binding)
    return binding


def validate_lane_arena(
    lane: str, report_path: Path, game_path: Path
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    games = [json.loads(line) for line in game_path.read_text().splitlines() if line]
    suite_rows = suites.load_suite_jsonl(str(SUITE))
    if len(games) != 1024 or int(report.get("games_played", -1)) != 1024:
        raise ValueError(f"arena_game_count_invalid:{lane}")
    if (
        report.get("notes", {}).get("base_seed") != 416
        or report.get("notes", {}).get("seed_contract") != "azlite_eval_seed_v2"
    ):
        raise ValueError(f"arena_seed_contract_invalid:{lane}")
    if report.get("notes", {}).get("suite_sha256") != sha256(SUITE):
        raise ValueError(f"arena_suite_binding_invalid:{lane}")
    profile = report.get("notes", {}).get("search_profile", {})
    if (
        int(profile.get("challenger_simulations", -1)) != 384
        or int(profile.get("current_simulations", -1)) != 384
        or float(profile.get("c_puct", -1)) != 1.25
    ):
        raise ValueError(f"arena_search_budget_invalid:{lane}")
    per_opening: dict[int, list[dict[str, Any]]] = {}
    for row in games:
        opening = int(row["opening_index"])
        game_index = int(row["game_index"])
        if not 0 <= opening < 512 or game_index != opening * 2 + int(
            row["game_within_opening"]
        ):
            raise ValueError(f"arena_game_index_invalid:{lane}:{game_index}")
        if row.get("opening_contract") != "arena_player_relative_v2":
            raise ValueError(f"arena_opening_contract_invalid:{lane}:{game_index}")
        if row.get("opening_state_hash") != suite_rows[opening].get("state_hash"):
            raise ValueError(f"arena_opening_identity_invalid:{lane}:{game_index}")
        game_state = KalahGame.from_state(suites.INITIAL_STATE)
        prefix = [int(move) for move in suite_rows[opening]["prefix_moves"]]
        if apply_opening_moves(game_state, prefix) != len(prefix):
            raise ValueError(f"arena_opening_replay_invalid:{lane}:{game_index}")
        trajectory = [int(move) for move in row["trajectory"].split(",") if move != ""]
        for ply, move in enumerate(trajectory):
            if (
                game_state.over()
                or not 0 <= move < 12
                or game_state.pit_owner(move) != game_state.current_player
                or move % 6 not in game_state.possible_moves()
                or not game_state.move(move)
            ):
                raise ValueError(f"arena_trajectory_invalid:{lane}:{game_index}:{ply}")
        if not game_state.over() or len(trajectory) != int(row["game_length"]):
            raise ValueError(f"arena_trajectory_incomplete:{lane}:{game_index}")
        challenger = int(row["challenger_player"])
        margin = (
            game_state.captured_seeds[challenger]
            - game_state.captured_seeds[1 - challenger]
        )
        winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
        if margin != int(row["margin"]) or winner != row["winner"]:
            raise ValueError(f"arena_terminal_accounting_invalid:{lane}:{game_index}")
        per_opening.setdefault(opening, []).append(row)
    if set(per_opening) != set(range(512)) or any(
        len(pair) != 2 or {int(row["challenger_player"]) for row in pair} != {0, 1}
        for pair in per_opening.values()
    ):
        raise ValueError(f"arena_seat_pairing_invalid:{lane}")
    return {
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "games": str(game_path),
        "games_sha256": sha256(game_path),
        "game_count": len(games),
        "opening_count": len(per_opening),
        "seed_identity_ledger_sha256": report["notes"]["seed_identity_ledger_sha256"],
    }


def record_accounting_amendment(report_path: Path, game_path: Path) -> None:
    registration = json.loads(REGISTRATION.read_text())
    frozen_sources = registration["source_hashes"]
    names = (
        "ml/alphazero_lite/run_seed416_policy_target_softening.py",
        "ml/alphazero_lite/verify_seed416_policy_target_softening.py",
    )
    replacements = {}
    for name in names:
        actual = sha256(ROOT / name)
        expected = frozen_sources[name]
        if actual != expected:
            replacements[name] = {
                "preregistered_sha256": expected,
                "post_run_sha256": actual,
            }
    if not replacements:
        return
    report = json.loads(report_path.read_text())
    amendment = {
        "schema": "seed416-post-first-lane-accounting-amendment-v1",
        "registration_sha256": sha256(REGISTRATION),
        "evaluation_binding_sha256": sha256(DATA / "evaluation-binding.json"),
        "suite_sha256": sha256(SUITE),
        "source_replacements": replacements,
        "reason": "The first-lane verifier treated absolute pit indices in the arena trajectory ledger as relative moves. The arena output is valid; this amendment corrects only strict ledger replay and evidence validation, not game generation, seeds, candidate selection, search budget, runtime, or analysis thresholds.",
        "timing": "after fixed A lane completed and before B lane launch",
        "A_lane": {
            "report_sha256": sha256(report_path),
            "games_sha256": sha256(game_path),
            "games_played": int(report["games_played"]),
            "wins": int(report["wins"]),
            "losses": int(report["losses"]),
            "draws": int(report["draws"]),
            "score": float(report["score"]),
        },
        "no_adaptive_extension": True,
    }
    path = DATA / "post-first-lane-accounting-amendment.json"
    if path.exists():
        if json.loads(path.read_text()) != amendment:
            raise ValueError("accounting_amendment_identity_conflict")
    else:
        write_json(path, amendment)


def evaluate() -> None:
    binding = json.loads((DATA / "evaluation-binding.json").read_text())
    if binding["registration_sha256"] != sha256(REGISTRATION) or binding[
        "suite_sha256"
    ] != sha256(SUITE):
        raise ValueError("evaluation_binding_input_mismatch")
    for lane, candidate in binding["candidates"].items():
        for name, digest in candidate["artifact_sha256"].items():
            if sha256(Path(candidate["artifact"]) / name) != digest:
                raise ValueError(f"candidate_artifact_identity_mismatch:{lane}:{name}")
    outcomes_path = DATA / "outcome-binding.json"
    previous = (
        json.loads(outcomes_path.read_text())
        if outcomes_path.exists()
        else {
            "evaluation_binding_sha256": sha256(DATA / "evaluation-binding.json"),
            "lanes": {},
        }
    )
    if previous.get("evaluation_binding_sha256") != sha256(
        DATA / "evaluation-binding.json"
    ):
        raise ValueError("resume_outcome_binding_identity_mismatch")
    results = dict(previous.get("lanes", {}))
    for lane, candidate in binding["candidates"].items():
        games = WORK / f"{lane}-games.jsonl"
        report = WORK / f"{lane}-arena.json"
        seed_ledger = WORK / f"{lane}-seed-identities.jsonl"
        config_ledger = WORK / f"{lane}-search-configurations.jsonl"
        search_ledger = WORK / f"{lane}-search-outcomes.jsonl"
        present = [path.exists() for path in (games, report)]
        if any(present) and not all(present):
            raise ValueError(f"interrupted_arena_output_incomplete:{lane}")
        if not all(present):
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "ml/alphazero_lite/arena.py"),
                    "--challenger",
                    candidate["artifact"],
                    "--current",
                    str(OPPONENT),
                    "--games",
                    "1024",
                    "--games-per-opening",
                    "2",
                    "--opening-prefixes-jsonl",
                    str(SUITE),
                    "--suite-sha256",
                    sha256(SUITE),
                    "--challenger-simulations",
                    "384",
                    "--current-simulations",
                    "384",
                    "--seed",
                    "416",
                    "--workers",
                    "24",
                    "--c-puct",
                    "1.25",
                    "--seed-contract",
                    "azlite_eval_seed_v2",
                    "--game-jsonl",
                    str(games),
                    "--out",
                    str(report),
                    "--seed-ledger-output",
                    str(seed_ledger),
                    "--search-configuration-ledger-output",
                    str(config_ledger),
                    "--search-outcome-ledger-output",
                    str(search_ledger),
                ],
                cwd=ROOT,
                check=True,
            )
        validated = validate_lane_arena(lane, report, games)
        if lane == "A":
            record_accounting_amendment(report, games)
        for kind, path in (
            ("seed_identity_ledger", seed_ledger),
            ("search_configuration_ledger", config_ledger),
            ("search_outcome_ledger", search_ledger),
        ):
            if not path.is_file():
                raise ValueError(f"arena_provenance_ledger_missing:{lane}:{kind}")
            validated[f"{kind}_path"] = str(path)
            validated[f"{kind}_sha256"] = sha256(path)
        old = results.get(lane)
        resume_action(old, validated)
        results[lane] = validated
        write_json(
            outcomes_path,
            {
                "schema": "seed416-outcome-binding-v1",
                "status": "in_progress",
                "evaluation_binding_sha256": sha256(DATA / "evaluation-binding.json"),
                "lanes": results,
            },
        )
    a_contexts = {
        row["seed_context_hash"]
        for row in read_jsonl(Path(results["A"]["seed_identity_ledger_path"]))
        if int(row["ply"]) == 0
    }
    b_contexts = {
        row["seed_context_hash"]
        for row in read_jsonl(Path(results["B"]["seed_identity_ledger_path"]))
        if int(row["ply"]) == 0
    }
    if a_contexts != b_contexts or len(a_contexts) != 1024:
        raise ValueError("paired_lane_seed_context_mismatch")
    write_json(
        outcomes_path,
        {
            "schema": "seed416-outcome-binding-v1",
            "status": "completed_2048_games",
            "evaluation_binding_sha256": sha256(DATA / "evaluation-binding.json"),
            "matched_initial_seed_contexts": len(a_contexts),
            "lanes": results,
        },
    )


def publish_analysis() -> None:
    binding = json.loads((DATA / "outcome-binding.json").read_text())
    rows = []
    for lane in ("A", "B"):
        ledger = Path(binding["lanes"][lane]["games"])
        if sha256(ledger) != binding["lanes"][lane]["games_sha256"]:
            raise ValueError(f"outcome_ledger_hash_mismatch:{lane}")
        game_rows = [
            json.loads(line) for line in ledger.read_text().splitlines() if line
        ]
        if len(game_rows) != 1024:
            raise ValueError(f"lane_game_count_mismatch:{lane}:{len(game_rows)}")
        for item in game_rows:
            game_index = int(item.get("game_index", item.get("index", -1)))
            opening_index = int(item.get("opening_index", game_index // 2))
            if not 0 <= opening_index < 512:
                raise ValueError(f"opening_index_invalid:{lane}:{opening_index}")
            # Arena game records report score from challenger perspective.
            winner = str(item.get("winner", ""))
            score = (
                1.0
                if winner == "challenger"
                else 0.5
                if winner == "draw"
                else 0.0
                if winner == "current"
                else -1.0
            )
            if score < 0:
                raise ValueError("arena_winner_missing_or_invalid")
            rows.append(
                {
                    "lane": lane,
                    "opening_id": str(opening_index),
                    "opponent_score": score,
                    "game": item,
                }
            )
    if len(rows) != 2048:
        raise ValueError(f"complete_outcome_count_mismatch:{len(rows)}")
    analysis = bootstrap_paired(rows)
    ledger_path = DATA / "outcome-ledger.jsonl"
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    if ledger_path.exists():
        if ledger_path.read_text(encoding="utf-8") != payload:
            raise ValueError("published_outcome_ledger_conflict")
    else:
        ledger_path.write_text(payload, encoding="utf-8")
    matrix_path = DATA / "per-opening-matrix.json"
    if matrix_path.exists():
        if json.loads(matrix_path.read_text()) != analysis["per_opening"]:
            raise ValueError("published_opening_matrix_conflict")
    else:
        write_json(matrix_path, analysis["per_opening"])
    analysis["outcome_ledger_sha256"] = sha256(ledger_path)
    analysis["outcome_binding_sha256"] = sha256(DATA / "outcome-binding.json")
    analysis["outcome_counts"] = {
        lane: {
            result: sum(
                1
                for row in rows
                if row["lane"] == lane and row["game"]["winner"] == result
            )
            for result in ("challenger", "current", "draw")
        }
        for lane in ("A", "B")
    }
    analysis_path = DATA / "analysis.json"
    if analysis_path.exists():
        if json.loads(analysis_path.read_text()) != analysis:
            raise ValueError("published_analysis_conflict")
    else:
        write_json(analysis_path, analysis)
    report = (
        "# Seed416 policy-target softening ablation\n\n"
        "Inference is conditional on this dataset and training seed. The A and B lanes "
        "were trained from the same frozen seed455 initialization; B applies the "
        "square-root transform to stored ordinary sharpened policy distributions. "
        "This is not recovery of original visit counts.\n\n"
        f"- Games: 2,048 (1,024 per lane), 512 opening clusters, both seats.\n"
        f"- Paired B-minus-A score: {analysis['primary_mean_B_minus_A']:.6f} "
        f"(95% percentile interval {analysis['primary_95_percentile_interval'][0]:.6f}, "
        f"{analysis['primary_95_percentile_interval'][1]:.6f}).\n"
        f"- B score against frozen seed455: {analysis['B_opponent_score_mean']:.6f} "
        f"(95% percentile interval {analysis['B_opponent_score_95_percentile_interval'][0]:.6f}, "
        f"{analysis['B_opponent_score_95_percentile_interval'][1]:.6f}).\n"
        f"- Decision: **{analysis['decision']}**.\n\n"
        "The registered thresholds nominate only a separate confirmation; they do not "
        "authorize promotion. The complete ledger, per-opening matrix, analysis, "
        "registrations, bindings, and portable verifier accompany this report.\n"
    )
    report_path = DATA / "results.md"
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != report:
            raise ValueError("published_report_conflict")
    else:
        report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("register", "train", "bind", "evaluate", "analyze")
    )
    args = parser.parse_args()
    {
        "register": register,
        "train": train_lanes,
        "bind": bind_candidates,
        "evaluate": evaluate,
        "analyze": publish_analysis,
    }[args.stage]()


if __name__ == "__main__":
    main()
