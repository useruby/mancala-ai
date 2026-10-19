from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite.validate_seed500_pilot import (
    EXPECTED_SHA256,
    _validate_and_account,
    partition_overlap,
    validate,
    validate_shards,
)


CORPUS = Path(__file__).parents[2] / "docs/data/seed500-selfplay-64.jsonl"


def test_archived_corpus_replays_and_passes_production_loader() -> None:
    report = validate(CORPUS)
    assert report["corpus_sha256"] == EXPECTED_SHA256
    assert (report["games"], report["rows"], report["actions"]) == (64, 2376, 2376)
    assert report["production_loader"] == "accepted"
    assert report["canonical_unique_states"] == 1954
    assert report["encoded_input_unique_states"] == 1954
    table = report["subgroups_by_teacher_and_active_stones"]
    assert table["PUCT"][">32"]["rows"] == 800
    assert table["PUCT"]["17–32"]["rows"] == 726
    assert table["exact-root"]["≤16"]["rows"] == 850
    assert (
        sum(
            cell["usable_rows"]
            for by_bucket in table.values()
            for cell in by_bucket.values()
        )
        == 2376
    )


def test_subgroups_separate_puct_and_exact_root_and_count_duplicates() -> None:
    report = _validate_and_account(
        [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]
    )
    assert (
        report["subgroups_by_teacher_and_active_stones"]["exact-root"]["≤16"]["rows"]
        == 850
    )
    assert (
        report["subgroups_by_teacher_and_active_stones"]["exact-root"]["≤16"][
            "one_hot_fraction"
        ]
        == 1
    )


def test_replay_rejects_value_target_corruption() -> None:
    rows = [
        json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
    ]
    game = [row for row in rows if row["game_index"] == 0]
    game[0]["value"] = -1.0
    with pytest.raises(ValueError, match="value_target_mismatch"):
        _validate_and_account(rows)


def test_eight_shards_aggregate_and_partition_overlap() -> None:
    root = CORPUS.parent / "seed500-selfplay-512"
    shards = [root / f"shard-{seed}.jsonl" for seed in range(503000, 503008)]
    report = validate_shards(shards)
    assert (report["games"], report["rows"], report["actions"]) == (512, 18281, 18281)
    assert report["production_loader"] == "accepted"
    assert report["game_status"] == {"complete": 512, "capped": 0, "incomplete": 0}
    assert report["diversity_growth_by_shard"][-1]["cumulative_unique_inputs"] == 11896
    overlap = partition_overlap(shards, set(range(503000, 503006)))
    assert overlap["training_source_rows"] == 13724
    assert overlap["development_original"]["rows"] == 4557
    assert overlap["development_after_training_identity_exclusion"]["rows"] == 2671
