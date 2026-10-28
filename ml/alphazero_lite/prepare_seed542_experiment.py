"""Read-only corpus validation and frozen fixed-update schedules for seed542."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train, validate_seed500_pilot

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
SEED504 = ROOT / "docs/data/seed504"
SEED500 = ROOT / "docs/data/seed500-selfplay-512"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_key(row: dict[str, Any]) -> tuple[int, ...]:
    state = np.asarray(row["state"], dtype=np.float32)
    pits_stores = np.rint(state[:14] * np.float32(48)).astype(np.int64)
    return (*map(int, pits_stores), int(round(float(state[14]))))


def input_key(row: dict[str, Any]) -> bytes:
    return np.asarray(row["state"], dtype=np.float32).tobytes()


def row_identity(path: Path, line: int) -> str:
    return f"{path.relative_to(ROOT).as_posix()}:{line}"


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        contents = path.read_bytes()
        source_hash = sha_bytes(contents)
        for line_no, raw_line in enumerate(contents.splitlines(), 1):
            text = raw_line.decode("utf-8")
            row = json.loads(text)
            row["_identity"] = row_identity(path, line_no)
            row["_path"] = path
            row["_line"] = line_no
            row["_source_sha256"] = source_hash
            row["_line_sha256"] = sha_bytes(raw_line)
            result.append(row)
    return result


def active_stones(row: dict[str, Any]) -> int:
    state = np.asarray(row["state"], dtype=np.float32)
    return int(round(float(state[:12].sum(dtype=np.float32)) * 48))


def replay_and_loader(paths: list[Path]) -> dict[str, Any]:
    raw_rows = read_rows(paths)
    report = validate_seed500_pilot.validate_shards(paths)
    x, policies, values = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
    )[:3]
    if len(x) != len(raw_rows):
        raise ValueError(f"loader_row_multiplicity_mismatch:{len(x)}:{len(raw_rows)}")
    if (
        x.shape != (len(raw_rows), 27)
        or policies.shape != (len(raw_rows), 6)
        or values.shape != (len(raw_rows), 1)
    ):
        raise ValueError("production_loader_shape_mismatch")
    report["loader_accepted_rows"] = len(x)
    report["loader_rejected_rows"] = 0
    report["loader_rejection_reasons"] = {}
    report["row_identity_count"] = len(raw_rows)
    report["row_identity_sha256"] = sha_bytes(
        "\n".join(r["_identity"] for r in raw_rows).encode()
    )
    report["target_mode_counts"] = {
        key: dict(Counter(str(row.get(key, "missing")) for row in raw_rows))
        for key in (
            "policy_target_mode",
            "policy_target_actual_mode",
            "value_target_mode",
        )
    }
    report["source_rows_preserved_with_multiplicity"] = True
    return {"rows": raw_rows, "report": report}


def shard_paths() -> tuple[list[Path], list[Path]]:
    training = [CORPUS / f"shard-{seed}.jsonl" for seed in range(542000, 542048)]
    development = [CORPUS / f"shard-{seed}.jsonl" for seed in range(542048, 542064)]
    return training, development


def build_membership(
    dev_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    old_paths = [SEED500 / f"shard-{seed}.jsonl" for seed in range(503000, 503006)]
    old_rows = read_rows(old_paths)
    canonical_train = {canonical_key(row) for row in old_rows + train_rows}
    input_train = {input_key(row) for row in old_rows + train_rows}
    membership = []
    included = []
    for row in dev_rows:
        c_overlap = canonical_key(row) in canonical_train
        f_overlap = input_key(row) in input_train
        reasons = []
        if c_overlap:
            reasons.append("canonical_state_overlap")
        if f_overlap:
            reasons.append("production_float32_input_overlap")
        record = {
            "row_id": row["_identity"],
            "source_sha256": row["_source_sha256"],
            "line_sha256": row["_line_sha256"],
            "canonical_sha256": sha_bytes(
                json.dumps(canonical_key(row), separators=(",", ":")).encode()
            ),
            "production_float32_input_sha256": sha_bytes(input_key(row)),
            "included": not reasons,
            "exclusion_reasons": reasons,
        }
        membership.append(record)
        if not reasons:
            included.append(row)
    return {
        "scope": "separation from seed504 training and the new 48 training-source shards only",
        "membership": membership,
        "membership_sha256": sha_bytes(
            json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()
        ),
        "counts": {
            "development_source_rows": len(dev_rows),
            "included_rows": len(included),
            "excluded_rows": len(dev_rows) - len(included),
            "excluded_canonical": sum(
                "canonical_state_overlap" in r["exclusion_reasons"] for r in membership
            ),
            "excluded_float32": sum(
                "production_float32_input_overlap" in r["exclusion_reasons"]
                for r in membership
            ),
            "remaining_canonical_overlap": len(
                {canonical_key(r) for r in included} & canonical_train
            ),
            "remaining_float32_overlap": len(
                {input_key(r) for r in included} & input_train
            ),
            "included_canonical_unique": len({canonical_key(r) for r in included}),
            "included_float32_unique": len({input_key(r) for r in included}),
        },
    }


def batch_lengths() -> list[int]:
    lengths = []
    for _ in range(4):
        lengths.extend([512] * 26)
        lengths.append(412)
    if len(lengths) != 108:
        raise ValueError("seed504_update_count_mismatch")
    return lengths


def schedules(
    c_rows: list[dict[str, Any]], d_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    lengths = batch_lengths()
    original_permutations = json.loads((SEED504 / "permutations.json").read_text())
    if len(original_permutations) != 4 or any(
        len(p) != 13724 for p in original_permutations
    ):
        raise ValueError("seed504_frozen_permutation_mismatch")
    control_flat = [
        index for permutation in original_permutations for index in permutation
    ]
    c_ids = [row["_identity"] for row in c_rows]
    if len(c_ids) != 13724:
        raise ValueError(f"control_population_mismatch:{len(c_ids)}")
    c_schedule = []
    d_schedule = []
    offset = 0
    d_generator = torch.Generator(device="cpu").manual_seed(542)
    d_permutation = torch.randperm(len(d_rows), generator=d_generator).tolist()
    required = sum(lengths)
    if len(d_permutation) < required:
        raise ValueError(
            f"expanded_population_cannot_support_no_replacement_schedule:{len(d_rows)}:{required}"
        )
    d_ids = [row["_identity"] for row in d_rows]
    for update, length in enumerate(lengths, 1):
        c_indices = control_flat[offset : offset + length]
        d_indices = d_permutation[offset : offset + length]
        c_schedule.append(
            {
                "update": update,
                "batch_size": length,
                "rows": [c_ids[i] for i in c_indices],
            }
        )
        d_schedule.append(
            {
                "update": update,
                "batch_size": length,
                "rows": [d_ids[i] for i in d_indices],
            }
        )
        offset += length
    if offset != required or len(d_schedule) != 108:
        raise ValueError("schedule_length_mismatch")
    return {
        "protocol": {
            "status": "frozen prospective; NOT EXECUTED",
            "comparison": {
                "C": "original seed504 training data",
                "D": "new 3,072-game training-source data",
            },
            "updates_per_arm": 108,
            "batch_size_sequence": lengths,
            "sampling": "deterministic permutation without replacement; no replacement sampling",
            "expanded_schedule_seed": 542,
            "schedule_runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "initializer_sha256": "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
            "architecture": "residual_v3/kalah_v3 (96, 3), 27 inputs",
            "objective": "legal policy CE + 0.5 * Huber(value, delta=1); joint policy/value",
            "optimizer": "Adam lr=0.0001 betas=(0.9,0.999) epsilon=1e-8 weight_decay=0; gradient clip norm=1; no scheduler",
            "control_permutations": "original frozen seed504 permutations",
            "expanded_data_description": "fixed-update experiment, not four complete passes over the expanded dataset",
            "candidate_endpoint": "final update only",
            "development_metrics": "descriptive only",
            "later_strength_comparison": "candidate policy plus frozen initializer value versus current P",
        },
        "control": schedule_summary(c_schedule),
        "expanded": schedule_summary(d_schedule),
        "updates": [
            {
                "update": i,
                "batch_size": lengths[i],
                "C": c_schedule[i]["rows"],
                "D": d_schedule[i]["rows"],
            }
            for i in range(108)
        ],
    }


def schedule_summary(schedule: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row for update in schedule for row in update["rows"])
    data = json.dumps(schedule, separators=(",", ":")).encode()
    return {
        "updates": len(schedule),
        "exposures": sum(counts.values()),
        "unique_rows_covered": len(counts),
        "maximum_row_exposure": max(counts.values()),
        "exposure_counts": dict(sorted(counts.items())),
        "schedule": schedule,
        "schedule_sha256": sha_bytes(data),
    }


def build_artifacts() -> dict[str, dict[str, Any]]:
    train_paths, dev_paths = shard_paths()
    missing = [str(path) for path in train_paths + dev_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"corpus_incomplete:{len(missing)} missing shards")
    training = replay_and_loader(train_paths)
    development = replay_and_loader(dev_paths)
    membership = build_membership(development["rows"], training["rows"])
    control_rows = read_rows(
        [SEED500 / f"shard-{seed}.jsonl" for seed in range(503000, 503006)]
    )
    schedules_report = schedules(control_rows, training["rows"])
    report = {
        "requested_games": 4096,
        "game_status": {
            "complete": training["report"]["games"] + development["report"]["games"],
            "capped": 0,
            "incomplete": 0,
        },
        "training_corpus": training["report"],
        "development_source": development["report"],
        "development_membership": membership,
        "training_protocol": schedules_report,
    }
    return {
        "accounting.json": report,
        "development-membership.json": membership,
        "training-protocol.json": schedules_report,
    }


def run(output: Path | None = None) -> None:
    output = output or CORPUS
    artifacts = build_artifacts()
    targets = [output / name for name in artifacts]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite published evidence: {existing}")
    for name, payload in artifacts.items():
        target = output / name
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
