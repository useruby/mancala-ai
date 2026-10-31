"""Prepare and execute the frozen 432-update seed542 budget experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import (
    run_seed542_fixed_update_cd as fixed_update,
    run_seed542_fixed_update_cd_v2 as v2,
    train,
    verify_seed542_amended_corpus,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
REFERENCE = ROOT / "docs/data/seed542-fixed-update-cd-v2"
OUT = ROOT / "docs/data/seed542-training-budget-432"
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
SOURCE_FILES = (
    "ml/alphazero_lite/seed542_training_budget_432.py",
    "ml/alphazero_lite/verify_seed542_training_budget_432.py",
    "ml/alphazero_lite/publish_seed542_training_budget_432.py",
    "ml/alphazero_lite/test_seed542_training_budget_432.py",
    "ml/alphazero_lite/run_seed542_fixed_update_cd.py",
    "ml/alphazero_lite/run_seed542_fixed_update_cd_v2.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/verify_seed542_amended_corpus.py",
    "ml/alphazero_lite/publish_seed542_fixed_update_cd_v2.py",
    "ml/alphazero_lite/verify_seed542_fixed_update_cd_v2.py",
)
INPUT_FILES = (
    "docs/data/seed542-selfplay-4096/continuation-amendment.json",
    "docs/data/seed542-selfplay-4096/amended-accounting.json",
    "docs/data/seed542-selfplay-4096/source-manifest.json",
    "docs/data/seed542-selfplay-4096/development-membership.json",
    "docs/data/seed542-selfplay-4096/training-protocol.json",
    "docs/data/seed542-selfplay-4096/initializer.npz",
    "docs/data/seed542-selfplay-4096/recovered-seed542001-complete.jsonl",
    "docs/data/seed542-selfplay-4096/recovered-seed542001-provenance.jsonl",
    "docs/data/seed542-fixed-update-cd-v2/D-updates.jsonl",
    "docs/data/seed542-fixed-update-cd-v2/D-update-108.npz",
    "docs/data/seed542-fixed-update-cd-v2/current-runtime-control-validity.json",
    "docs/data/seed542-fixed-update-cd-v2/registration.json",
)
CHECKPOINTS = (108, 216, 432)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def expanded_rows() -> list[str]:
    _x, _p, _v, _w, ids = fixed_update.load_sources("D")
    return ids


def build_schedule(
    protocol: dict[str, Any], identities: list[str]
) -> list[dict[str, Any]]:
    del protocol
    first = fixed_update.load_schedule()["updates"]
    original = [row for update in first for row in update["D"]]
    if (
        len(identities) != 109504
        or len(original) != 54896
        or len(set(original)) != 54896
    ):
        raise ValueError("432_initial_schedule_population_mismatch")
    generator = torch.Generator(device="cpu").manual_seed(542)
    permutation = torch.randperm(len(identities), generator=generator).tolist()
    seeded = [identities[index] for index in permutation]
    if seeded[:54896] != original:
        raise ValueError("432_original_seed542_schedule_reproduction_failed")

    schedule: list[dict[str, Any]] = []
    for row in first:
        schedule.append(
            {
                "update": row["update"],
                "batch_size": row["batch_size"],
                "phase": "registered_initial_schedule",
                "rows": row["D"],
            }
        )
    cursor = 54896
    update = 109
    while cursor < len(seeded):
        size = min(512, len(seeded) - cursor)
        schedule.append(
            {
                "update": update,
                "batch_size": size,
                "phase": "finish_first_permutation",
                "rows": seeded[cursor : cursor + size],
            }
        )
        cursor += size
        update += 1

    continuation_rows: list[str] = []
    continuation_seeds: list[int] = []
    pass_seed = 543
    while len(continuation_rows) < (432 - update + 1) * 512:
        pass_generator = torch.Generator(device="cpu").manual_seed(pass_seed)
        shuffled = torch.randperm(len(identities), generator=pass_generator).tolist()
        continuation_rows.extend(identities[index] for index in shuffled)
        continuation_seeds.extend([pass_seed] * len(shuffled))
        pass_seed += 1
    cursor = 0
    while update <= 432:
        size = 512
        schedule.append(
            {
                "update": update,
                "batch_size": size,
                "phase": "deterministic_shuffled_passes",
                "shuffle_seeds": sorted(
                    set(continuation_seeds[cursor : cursor + size])
                ),
                "rows": continuation_rows[cursor : cursor + size],
            }
        )
        cursor += size
        update += 1
    if (
        len(schedule) != 432
        or schedule[214]["batch_size"] != 336
        or schedule[214]["update"] != 215
    ):
        raise ValueError("432_schedule_shape_mismatch")
    if any(len(item["rows"]) != item["batch_size"] for item in schedule):
        raise ValueError("432_schedule_batch_accounting_mismatch")
    if set(row for item in schedule[108:215] for row in item["rows"]) != set(
        seeded[54896:]
    ):
        raise ValueError("432_unused_first_pass_rows_not_consumed_exactly_once")
    if set(row for item in schedule[108:215] for row in item["rows"]) & set(original):
        raise ValueError("432_unused_tail_overlaps_initial_schedule")
    return schedule


def prepare(out: Path, tablebase: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"432_output_not_empty:{out}")
    verify_seed542_amended_corpus.verify(root=ROOT, tablebase_path=tablebase)
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((CORPUS / "training-protocol.json").read_text())
    schedule = build_schedule(protocol, expanded_rows())
    (out / "schedule.json").write_text(
        json.dumps(schedule, separators=(",", ":")) + "\n"
    )
    source_hashes = {name: sha(ROOT / name) for name in SOURCE_FILES}
    manifest = json.loads((CORPUS / "source-manifest.json").read_text())
    input_paths = {
        *INPUT_FILES,
        *(source["path"] for source in manifest["sources"]),
    }
    input_hashes = {name: sha(ROOT / name) for name in sorted(input_paths)}
    registration = {
        "status": "prospectively frozen; no training executed",
        "experiment": "seed542 expanded-data 432-update controlled training-budget experiment",
        "sole_candidate_checkpoint": 432,
        "diagnostic_checkpoints": [108, 216],
        "schedule_sha256": sha(out / "schedule.json"),
        "schedule_accounting": {
            "total_updates": 432,
            "initial_registered_updates": 108,
            "initial_exposures": 54896,
            "finish_first_permutation_updates": 107,
            "finish_first_permutation_exposures": 54608,
            "finish_first_permutation_batch_sizes": [512] * 106 + [336],
            "second_pass_seed": 543,
            "third_pass_seed": 544,
            "second_pass_updates": 217,
            "second_pass_exposures": 111104,
            "total_exposures": 220608,
            "available_training_rows": 109504,
        },
        "training_contract": {
            "only_changed_factor": "update budget (108 to 432)",
            "initializer_sha256": sha(CORPUS / "initializer.npz"),
            "corpus_sha256": sha(CORPUS / "source-manifest.json"),
            "architecture": "residual_v3/kalah_v3 hidden=(96,3), 27 inputs",
            "loss": "legal policy CE + 0.5 * Huber(value, delta=1)",
            "optimizer": "Adam lr=0.0001 betas=(0.9,0.999) eps=1e-8 weight_decay=0 scheduler=none",
            "gradient_clip_norm": 1.0,
            "optimizer_state": "one fresh Adam optimizer; preserve in-memory moments across all 432 updates",
            "first_108_gate": "compare every registered D update record and post-update parameter digest; compare update108 checkpoint bitwise; stop before update109 on any mismatch",
        },
        "development_contract": "unchanged development-membership.json; report CE and actual-value MSE by stone bucket with exposure and equal-input weighting",
        "execution_source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "tablebase_sha256": sha(tablebase),
        "runtime_contract": "reuse seed542 v2 runtime and resource settings; no numerical environment changes",
        "runtime_and_resource_settings": v2.environment_settings(),
    }
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    (out / "preflight.json").write_text(
        json.dumps(
            {
                "status": "frozen_before_training",
                "registration_sha256": sha(out / "registration.json"),
                "schedule_updates": len(schedule),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def verify_freeze(out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registration = json.loads((out / "registration.json").read_text())
    if registration["status"] != "prospectively frozen; no training executed":
        raise ValueError("432_registration_not_prospective")
    for name, digest in registration["execution_source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"432_frozen_source_changed:{name}")
    for name, digest in registration["input_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"432_frozen_input_changed:{name}")
    validate_schedule_integrity(
        (out / "schedule.json").read_bytes(), registration["schedule_sha256"]
    )
    frozen_runtime = dict(registration["runtime_and_resource_settings"])
    current_runtime = v2.environment_settings()
    frozen_runtime.pop("cgroup", None)
    current_runtime.pop("cgroup", None)
    if canonical(frozen_runtime) != canonical(current_runtime):
        raise ValueError("432_runtime_changed_after_freeze")
    schedule = json.loads((out / "schedule.json").read_text())
    if len(schedule) != 432 or [row["update"] for row in schedule] != list(
        range(1, 433)
    ):
        raise ValueError("432_frozen_schedule_invalid")
    return registration, schedule


def validate_schedule_integrity(contents: bytes, expected_sha256: str) -> None:
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise ValueError("432_frozen_schedule_changed")


def apply_schedule_item(
    recorder: v2.DirectRecorder,
    optimizer: torch.optim.Optimizer,
    item: dict[str, Any],
    positions: dict[str, int],
) -> dict[str, Any]:
    record = recorder.update(optimizer, item["rows"], positions)
    record["update"] = item["update"]
    record["phase"] = item["phase"]
    return record


def execute(out: Path, tablebase: Path) -> dict[str, Any]:
    registration, schedule = verify_freeze(out)
    if (out / "execution-binding.json").exists():
        raise FileExistsError("432_execution_already_started; preserve attempt")
    verify_seed542_amended_corpus.verify(root=ROOT, tablebase_path=tablebase)
    if sha(tablebase) != registration["tablebase_sha256"]:
        raise ValueError("432_tablebase_changed_after_freeze")
    (out / "execution-binding.json").write_text(
        json.dumps(
            {
                "registration_sha256": sha(out / "registration.json"),
                "actual_command": sys.argv,
                "runtime": v2.environment_settings(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    x, p, value, weights, identities = fixed_update.load_sources("D")
    positions = {identity: index for index, identity in enumerate(identities)}
    reference = [
        json.loads(line)
        for line in (REFERENCE / "D-updates.jsonl").read_text().splitlines()
    ]
    if len(reference) != 108:
        raise ValueError("432_registered_D_trace_count_invalid")
    model = fixed_update.load_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    recorder = v2.DirectRecorder(model, x, p, value, weights, identities)
    checkpoints: dict[str, str] = {}
    first_mismatch = None
    trace_path = out / "updates.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for item in schedule:
            record = apply_schedule_item(recorder, optimizer, item, positions)
            if item["update"] <= 108:
                mismatch = v2.compare_records(reference[item["update"] - 1], record)
                if mismatch:
                    first_mismatch = {
                        "update": item["update"],
                        "mismatched_fields": mismatch,
                    }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            if item["update"] in CHECKPOINTS:
                checkpoint_path = out / f"D-update-{item['update']:03d}.npz"
                arrays = train.checkpoint_from_model(model)
                np.savez(checkpoint_path, **arrays)
                checkpoints[str(item["update"])] = sha(checkpoint_path)
                if item["update"] == 108:
                    with np.load(REFERENCE / "D-update-108.npz") as expected:
                        checkpoint_match = set(expected.files) == set(arrays) and all(
                            np.array_equal(arrays[name], expected[name])
                            for name in arrays
                        )
                    if not checkpoint_match and first_mismatch is None:
                        first_mismatch = {
                            "update": 108,
                            "mismatched_fields": ["checkpoint_arrays"],
                        }
            if first_mismatch is not None:
                break
    recorder.close()
    count = len(trace_path.read_text().splitlines())
    if first_mismatch is not None:
        result = {
            "status": "control_failed; continuation_not_run",
            "first_mismatch": first_mismatch,
            "updates_completed": count,
            "continuation_started": False,
            "checkpoints": checkpoints,
            "trace_sha256": sha(trace_path),
        }
        (out / "run-status.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result
    if count != 432:
        raise RuntimeError(f"432_unexpected_update_count:{count}")
    (out / "schedule-accounting.json").write_text(
        json.dumps(accounting(schedule), indent=2, sort_keys=True) + "\n"
    )
    from ml.alphazero_lite import publish_seed542_training_budget_432 as publication

    publication.publish(out)
    result = {
        "status": "completed",
        "control_passed": True,
        "updates_completed": 432,
        "checkpoints": checkpoints,
        "trace_sha256": sha(trace_path),
        "sole_candidate_checkpoint": 432,
    }
    (out / "run-status.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def accounting(schedule: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        phase: {
            "updates": sum(item["phase"] == phase for item in schedule),
            "exposures": sum(
                item["batch_size"] for item in schedule if item["phase"] == phase
            ),
            "batch_sizes": [
                item["batch_size"] for item in schedule if item["phase"] == phase
            ],
        }
        for phase in sorted({item["phase"] for item in schedule})
    } | {
        "total_updates": len(schedule),
        "total_exposures": sum(item["batch_size"] for item in schedule),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tablebase", type=Path, default=TABLEBASE)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if tuple(os.environ.get(key) for key in v2.ENVIRONMENT_KEYS) != ("1",) * len(
        v2.ENVIRONMENT_KEYS
    ):
        raise ValueError("432_thread_environment_must_be_explicit")
    if args.prepare_only:
        prepare(args.out.resolve(), args.tablebase.resolve())
    else:
        print(
            json.dumps(
                execute(args.out.resolve(), args.tablebase.resolve()),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
