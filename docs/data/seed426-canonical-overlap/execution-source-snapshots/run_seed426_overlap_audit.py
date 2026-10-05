"""Build the data-only seed426 audit from the hash-registered #416 inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.alphazero_lite.fresh_p1_adapter_teacher_audit import (  # noqa: E402
    decode_kalah_v3_base_state,
)
from ml.alphazero_lite.seed426_overlap_analysis import (  # noqa: E402
    ACCOUNTING_DEFINITIONS,
    DECISION_RULES,
    IDENTITY_DEFINITIONS,
    audit,
    float32_identity,
    sha256,
    state_identity,
)
from ml.alphazero_lite import train  # noqa: E402
from ml.alphazero_lite.self_play import encode_state  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, publish: bool = False) -> None:
    data = root / "docs/data/seed426-canonical-overlap"
    data.mkdir(parents=True, exist_ok=True)
    registration_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    registration = json.loads(registration_path.read_text())
    sources = registration["replays"]
    if [int(source["weight"]) for source in sources] != [1, 4, 1, 8, 4]:
        raise ValueError("registered_weights_mismatch")
    if any(
        source["value_target_mode"] not in {"default", "sharpened"}
        for source in sources
    ):
        raise ValueError("registered_target_mode_mismatch")
    if (
        digest(Path(train.__file__))
        != registration["source_hashes"]["ml/alphazero_lite/train.py"]
    ):
        raise ValueError("frozen_loader_registration_hash_mismatch")
    if (
        digest(Path(train.__file__).with_name("self_play.py"))
        != registration["source_hashes"]["ml/alphazero_lite/self_play.py"]
    ):
        raise ValueError("encoder_registration_hash_mismatch")
    source_paths = [Path(source["path"]) for source in sources]
    # Original sources may live outside a relocated checkout; published snapshots are portable.
    snapshots = data / "sources"
    snapshots.mkdir(exist_ok=True)
    records = []
    for index, source in enumerate(sources):
        original = source_paths[index]
        if original.exists():
            raw = original.read_bytes()
        else:
            with gzip.open(snapshots / f"{source['name']}.jsonl.gz", "rb") as handle:
                raw = handle.read()
        if sha256(raw) != source["sha256"]:
            raise ValueError(f"registered_source_hash_mismatch:{source['name']}")
        if publish and original.exists():
            with gzip.open(
                snapshots / f"{source['name']}.jsonl.gz", "wb", compresslevel=9
            ) as handle:
                handle.write(raw)
        records.append((source, raw))

    split_path = root / registration["training"]["source_row_split"]["path"]
    split_raw = split_path.read_bytes()
    if sha256(split_raw) != registration["training"]["source_row_split"]["sha256"]:
        raise ValueError("source_split_hash_mismatch")
    split = json.loads(gzip.decompress(split_raw))
    train_positions = split["train_positions"]
    validation_positions = split["validation_positions"]
    if len(train_positions) != 134502 or len(validation_positions) != 14946:
        raise ValueError("published_position_count_mismatch")
    if set(train_positions) & set(validation_positions):
        raise ValueError("position_partition_overlap")
    if sorted(train_positions + validation_positions) != list(range(149448)):
        raise ValueError("position_partition_incomplete")

    position_partition = {position: "train" for position in train_positions}
    position_partition.update(
        {position: "validation" for position in validation_positions}
    )
    compact_row = 0
    evidence = []
    paths = []
    source_replays = []
    loader_tmp_context = tempfile.TemporaryDirectory(
        prefix="seed426-loader-", dir=root / ".tmp"
    )
    loader_tmp = Path(loader_tmp_context.name)
    for source, raw in records:
        source_path = loader_tmp / f"{source['name']}.jsonl"
        source_path.write_bytes(raw)
        paths.append(source_path)
        parsed = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            row = json.loads(line)
            encoded = np.asarray(row["state"], dtype=np.float32)
            state = decode_kalah_v3_base_state(encoded.tolist())
            if len(encoded) != 27:
                raise ValueError("kalah_v3_feature_count_mismatch")
            if (
                np.asarray(
                    encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
                ).tobytes()
                != encoded.tobytes()
            ):
                raise ValueError("kalah_v3_encoder_identity_mismatch")
            # Frozen loader accepts every source row under its registered target modes.
            canonical = state_identity(state).decode("ascii")
            input_id = float32_identity(encoded.tolist()).hex()
            active = sum(state["player_pits"]) + sum(state["opponent_pits"])
            parsed.append((line_number, encoded, row, state, active))
            evidence.append(
                {
                    "source": source["name"],
                    "raw_line": line_number,
                    "compact_row": compact_row,
                    "active_stones": active,
                    "canonical_identity": canonical,
                    "input_identity": input_id,
                }
            )
            compact_row += 1
        source_replays.append((source, parsed))
    if compact_row != len(set(row["compact_row"] for row in evidence)):
        raise ValueError("compact_row_accounting_mismatch")
    expanded = []
    weighted_position = 0
    for source, _parsed in source_replays:
        source_rows = [row for row in evidence if row["source"] == source["name"]]
        for copy in range(int(source["weight"])):
            for row in source_rows:
                expanded.append(
                    {
                        **row,
                        "copy": copy,
                        "weighted_position": weighted_position,
                        "partition": position_partition[weighted_position],
                        "weight": 1,
                    }
                )
                weighted_position += 1
    evidence = expanded
    if compact_row != 87625 or weighted_position != 149448:
        raise ValueError("source_row_or_weighted_count_mismatch")
    if (
        sha256(np.asarray(train_positions, dtype=np.int64).tobytes())
        != registration["training"]["source_row_split"]["train_positions_sha256"]
    ):
        raise ValueError("train_position_identity_mismatch")
    if (
        sha256(np.asarray(validation_positions, dtype=np.int64).tobytes())
        != registration["training"]["source_row_split"]["validation_positions_sha256"]
    ):
        raise ValueError("validation_position_identity_mismatch")

    # Execute the historically registered loader and split implementation as a
    # parity control; this stage is the only one requiring its torch runtime.
    modes = [source["value_target_mode"] for source, _ in source_replays]
    train.set_seed(416)
    loaded_x, loaded_p, loaded_v, replay_indexes = train.load_jsonl_replay(
        paths,
        [int(source["weight"]) for source, _ in source_replays],
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=modes,
        include_policy_loss_weights=False,
    )
    expected_x = np.asarray(
        [row[1] for _, rows in source_replays for row in rows], dtype=np.float32
    )
    if not np.array_equal(loaded_x, expected_x):
        raise ValueError("frozen_loader_state_parity_mismatch")
    train_positions_actual, validation_positions_actual = (
        train.split_replay_positions_by_source_row(replay_indexes, val_split=0.1)
    )
    if not np.array_equal(
        train_positions_actual, np.asarray(train_positions, dtype=np.int64)
    ):
        raise ValueError("frozen_train_split_identity_mismatch")
    if not np.array_equal(
        validation_positions_actual, np.asarray(validation_positions, dtype=np.int64)
    ):
        raise ValueError("frozen_validation_split_identity_mismatch")
    if not np.array_equal(
        replay_indexes,
        np.asarray([row["compact_row"] for row in evidence], dtype=np.int64),
    ):
        raise ValueError("frozen_replay_multiplicity_mismatch")

    # Per-source loader calls bind compact-row membership to raw JSONL lines and
    # expose exact loaded policy/value arrays without treating copies as rows.
    target_checks = {}
    for source, _raw in records:
        loaded = train.load_jsonl_replay(
            [paths[len(target_checks)]],
            [1],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[source["value_target_mode"]],
        )
        x, policy, value, _indexes = loaded
        compact_states = np.asarray(
            [row[1] for row in source_replays[len(target_checks)][1]], dtype=np.float32
        )
        if not np.array_equal(x, compact_states):
            raise ValueError(f"frozen_source_row_mapping_mismatch:{source['name']}")
        target_checks[source["name"]] = {
            "eligible_rows": int(x.shape[0]),
            "policy_float32_sha256": sha256(policy.tobytes()),
            "value_float32_sha256": sha256(value.tobytes()),
        }
    loader_tmp_context.cleanup()

    train_source_rows = set(split["train_source_rows"])
    validation_source_rows = set(split["validation_source_rows"])
    if train_source_rows & validation_source_rows:
        raise ValueError("frozen_source_row_partition_overlap")
    if train_source_rows | validation_source_rows != set(range(compact_row)):
        raise ValueError("frozen_source_row_partition_incomplete")
    if set(replay_indexes[train_positions_actual]) != train_source_rows:
        raise ValueError("frozen_train_source_rows_mismatch")
    if set(replay_indexes[validation_positions_actual]) != validation_source_rows:
        raise ValueError("frozen_validation_source_rows_mismatch")

    # Bind every verified input and execution source before aggregate overlap
    # analysis is invoked below.
    snapshot_dir = data / "execution-source-snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    execution_paths = {
        "runner": Path(__file__),
        "analysis": Path(__file__).with_name("seed426_overlap_analysis.py"),
        "frozen_train": Path(train.__file__),
        "kalah_v3_encoder": Path(train.__file__).with_name("self_play.py"),
        "state_decoder": Path(__file__).with_name("fresh_p1_adapter_teacher_audit.py"),
        "verifier": Path(__file__).with_name("verify_seed426_overlap_audit.py"),
    }
    execution_snapshot_hashes = {}
    for name, path in execution_paths.items():
        destination = snapshot_dir / path.name
        shutil.copyfile(path, destination)
        execution_snapshot_hashes[destination.relative_to(data).as_posix()] = digest(
            destination
        )
    manifest = {
        "schema": "seed426-canonical-overlap-manifest-v1",
        "registrations": {
            "seed416-v3": digest(registration_path),
            "source_split": sha256(split_raw),
            "frozen_train": registration["source_hashes"]["ml/alphazero_lite/train.py"],
            "kalah_v3_encoder": registration["source_hashes"][
                "ml/alphazero_lite/self_play.py"
            ],
        },
        "sources": [
            {
                "name": source["name"],
                "sha256": source["sha256"],
                "weight": source["weight"],
                "policy_target_mode": "sharpened",
                "value_target_mode": source["value_target_mode"],
                "eligibility": "no exclusion buckets",
            }
            for source, _ in records
        ],
        "identity_definitions": IDENTITY_DEFINITIONS,
        "accounting": ACCOUNTING_DEFINITIONS,
        "chronology": {
            "classification": "retrospective data-only audit; not part of or prospective to the #416 historical training freeze",
            "preliminary_observations": [
                "87,625 eligible rows",
                "149,448 weighted positions",
                "6,974 shared canonical identities",
                ">32 overlap 5,109/5,973",
            ],
            "amendment": "These values were observed before exact frozen np.tile weighted-position order was reconstructed. Preliminary expansion repeated each row consecutively; the loader repeats the complete source compact-row sequence by weight. The provisional overlap values are superseded by this hash-bound reconstruction and are retained here only to document chronology.",
        },
        "execution_sources": {
            name: digest(path) for name, path in execution_paths.items()
        },
        "execution_source_snapshots": execution_snapshot_hashes,
        "decision_rules": DECISION_RULES,
    }
    (data / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    result = {
        "schema": "seed426-canonical-overlap-results-v1",
        "input_hashes": {source["name"]: source["sha256"] for source, _ in records},
        "weighted_positions": weighted_position,
        "eligible_source_rows": compact_row,
        "eligibility": {
            "frozen_call_exclude_buckets": None,
            "raw_jsonl_lines": compact_row,
            "eligible_compact_rows": compact_row,
            "skipped_raw_lines": [],
            "rule": "the frozen load_jsonl_replay call supplies no exclusion buckets; every parsed raw row must validate and load",
        },
        "split_positions": {
            "train": len(train_positions),
            "validation": len(validation_positions),
        },
        "source_accounting": {
            source["name"]: {
                "weight": int(source["weight"]),
                "eligible_source_rows": sum(
                    row["source"] == source["name"] for row in evidence
                )
                // int(source["weight"]),
                "weighted_positions": sum(
                    row["source"] == source["name"] for row in evidence
                ),
                "train_source_rows": len(
                    {
                        row["compact_row"]
                        for row in evidence
                        if row["source"] == source["name"]
                        and row["partition"] == "train"
                    }
                ),
                "validation_source_rows": len(
                    {
                        row["compact_row"]
                        for row in evidence
                        if row["source"] == source["name"]
                        and row["partition"] == "validation"
                    }
                ),
                "train_weighted_positions": sum(
                    row["source"] == source["name"] and row["partition"] == "train"
                    for row in evidence
                ),
                "validation_weighted_positions": sum(
                    row["source"] == source["name"] and row["partition"] == "validation"
                    for row in evidence
                ),
            }
            for source, _ in records
        },
        "loader_parity": {
            "frozen_train_sha256": digest(Path(train.__file__)),
            "loaded_x_shape": list(loaded_x.shape),
            "loaded_x_float32_sha256": sha256(loaded_x.tobytes()),
            "loaded_policy_float32_sha256": sha256(loaded_p.tobytes()),
            "loaded_value_float32_sha256": sha256(loaded_v.tobytes()),
            "source_targets": target_checks,
        },
        "identity_census": audit(evidence),
        "decision": "canonical_state_overlap_detected"
        if audit(evidence)["canonical"]["intersection_unique"]
        else "canonical_state_disjoint",
        "recommendation": "separately_authorized_validation_metric_correction_using_identity_disjoint_holdout"
        if audit(evidence)["canonical"]["buckets"][">32"][
            "validation_overlap_denominators"
        ]["weighted_positions"]["fraction"]
        >= 0.05
        else "close_diagnostic_without_training_or_replay_changes",
        "provenance": "retrospective data-only audit; not part of historical training freeze",
    }
    with gzip.open(
        data / "row-accounting.jsonl.gz", "wt", encoding="utf-8", compresslevel=9
    ) as handle:
        handle.writelines(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in evidence
        )
    (data / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--publish-snapshots", action="store_true")
    arguments = parser.parse_args()
    run(arguments.root, arguments.publish_snapshots)
