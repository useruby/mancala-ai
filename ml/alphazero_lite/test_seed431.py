"""Semantic rejection tests for seed431's archived-trajectory verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.evaluation_seed_contract import stable_hash
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import verify_seed431 as verifier
from ml.alphazero_lite.verify_seed430 import rows, sha

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(verifier.DATA_REL)


def _publication_lane(
    lane: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[Any],
    list[Any],
    list[Any],
    list[Any],
]:
    data = ROOT / DATA
    registration = json.loads((data / "registration.json").read_text())
    binding = json.loads((data / "runtime-binding.json").read_text())
    outcome = json.loads((data / "outcome-binding.json").read_text())["lanes"][lane]
    return (
        registration,
        binding,
        outcome,
        rows(ROOT / outcome["games_path"]),
        rows(ROOT / outcome["seed_identity_ledger_path"]),
        rows(ROOT / outcome["search_configuration_ledger_path"]),
        rows(ROOT / outcome["search_outcome_ledger_path"]),
    )


def _first_game_records(lane: str = "A") -> tuple[Any, ...]:
    registration, binding, _outcome, games, seeds, configs, searches = (
        _publication_lane(lane)
    )
    game = games[0]
    count = int(game["game_length"])
    return (
        registration,
        binding,
        [game],
        seeds[:count],
        configs[:count],
        searches[:count],
    )


def _canonical_entries_digest(entries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_complete_publication_passes_and_derives_expected_coverage() -> None:
    result = verifier.verify(ROOT)
    assert result["replayed_per_ply_records"] == {"A": 33438, "B": 33059}
    assert result["reconciliation"]["chunks"] == 47
    assert result["reconciliation"]["pre_repair_games"] == 1504
    assert (
        result["reconciliation"]["result"]
        == "published game entries match the recorded digests"
    )
    assert result["decision"] == "stop_normalization_branch"
    assert result["original_results"] == {
        "games": 2048,
        "games_per_lane": 1024,
        "matched_openings": 512,
        "paired_B_minus_A_mean": 0.0166015625,
        "paired_B_minus_A_95_interval": [-0.00537109375, 0.03857421875],
        "B_score_mean": 0.5146484375,
        "B_score_95_interval": [0.49462890625, 0.53515625],
        "decision": "stop_normalization_branch",
    }


def test_player_one_absolute_pit_eight_is_relative_two() -> None:
    game = KalahGame.from_state(suites.INITIAL_STATE)
    game.current_player = 1
    game.pits[6 + 2] = 1
    assert (
        verifier.relative_action_for_absolute(game, 8, lane="T", game_index=0, ply=0)
        == 2
    )


@pytest.mark.parametrize(
    ("absolute", "message"),
    [(8, "trajectory_wrong_side"), (1, "trajectory_empty_or_illegal")],
)
def test_wrong_side_and_empty_absolute_pits_rejected(
    absolute: int, message: str
) -> None:
    game = KalahGame.from_state(suites.INITIAL_STATE)
    if absolute == 1:
        game.pits[1] = 0
    with pytest.raises(ValueError, match=message):
        verifier.relative_action_for_absolute(
            game, absolute, lane="T", game_index=0, ply=0
        )


def test_self_consistent_forged_context_and_derived_seed_rejected_by_replay() -> None:
    registration, binding, games, seeds, configs, outcomes = _first_game_records()
    seeds[0]["canonical_current_state_hash"] = "f" * 64
    forged_identity = {
        key: seeds[0][key]
        for key in (
            "contract_version",
            "base_seed",
            "suite_sha256",
            "opening_index",
            "opening_state_hash",
            "challenger_player",
            "game_within_opening",
            "ply",
            "canonical_current_state_hash",
            "acting_role",
            "rng_stream_name",
        )
    }
    seeds[0]["seed_context_hash"] = stable_hash(forged_identity)
    from ml.alphazero_lite.evaluation_seed_contract import stable_seed

    seeds[0]["derived_search_seed"] = stable_seed(forged_identity)
    # Keep linked rows and their own hashes internally consistent. Replay must be the rejection.
    context = seeds[0]["seed_context_hash"]
    configs[0]["seed_context_hash"] = context
    config_body = {
        key: value
        for key, value in configs[0].items()
        if key != "search_configuration_hash"
    }
    configs[0]["search_configuration_hash"] = stable_hash(config_body)
    outcomes[0]["seed_context_hash"] = context
    with pytest.raises(ValueError, match="replayed_seed_identity_mismatch"):
        verifier.replay_lane(
            ROOT, registration, binding, "A", games, seeds, configs, outcomes
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_current_state_hash", "e" * 64),
        ("acting_role", "challenger"),
        ("ply", 77),
    ],
)
def test_refreshed_forged_state_role_and_ply_contexts_rejected(
    field: str, value: Any
) -> None:
    registration, binding, games, seeds, configs, outcomes = _first_game_records()
    seeds, configs, outcomes = copy.deepcopy((seeds, configs, outcomes))
    seeds[0][field] = value
    identity_fields = (
        "contract_version",
        "base_seed",
        "suite_sha256",
        "opening_index",
        "opening_state_hash",
        "challenger_player",
        "game_within_opening",
        "ply",
        "canonical_current_state_hash",
        "acting_role",
        "rng_stream_name",
    )
    identity = {key: seeds[0][key] for key in identity_fields}
    seeds[0]["seed_context_hash"] = stable_hash(identity)
    from ml.alphazero_lite.evaluation_seed_contract import stable_seed

    seeds[0]["derived_search_seed"] = stable_seed(identity)
    configs[0]["seed_context_hash"] = seeds[0]["seed_context_hash"]
    config_body = {
        key: value
        for key, value in configs[0].items()
        if key != "search_configuration_hash"
    }
    configs[0]["search_configuration_hash"] = stable_hash(config_body)
    outcomes[0]["seed_context_hash"] = seeds[0]["seed_context_hash"]
    outcomes[0]["ply"] = int(seeds[0]["ply"])
    outcomes[0]["acting_role"] = seeds[0]["acting_role"]
    with pytest.raises(ValueError, match="replayed_seed_identity_mismatch"):
        verifier.replay_lane(
            ROOT, registration, binding, "A", games, seeds, configs, outcomes
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "extra",
        "seed",
        "artifact",
        "runtime",
        "selected",
        "illegal_selected",
    ],
)
def test_replay_rejects_per_ply_semantic_mutations(mutation: str) -> None:
    registration, binding, games, seeds, configs, outcomes = _first_game_records()
    seeds, configs, outcomes = copy.deepcopy((seeds, configs, outcomes))
    if mutation == "missing":
        seeds.pop()
        configs.pop()
        outcomes.pop()
    elif mutation == "duplicate":
        seeds.append(copy.deepcopy(seeds[0]))
        configs.append(copy.deepcopy(configs[0]))
        outcomes.append(copy.deepcopy(outcomes[0]))
    elif mutation == "extra":
        seeds.append(copy.deepcopy(seeds[-1]))
        configs.append(copy.deepcopy(configs[-1]))
        outcomes.append(copy.deepcopy(outcomes[-1]))
    elif mutation == "seed":
        seeds[0]["derived_search_seed"] ^= 1
    elif mutation in {"artifact", "runtime"}:
        configs[0][
            "artifact_hash" if mutation == "artifact" else "runtime_profile_hash"
        ] = "0" * 64
        config_body = {
            key: value
            for key, value in configs[0].items()
            if key != "search_configuration_hash"
        }
        configs[0]["search_configuration_hash"] = stable_hash(config_body)
    elif mutation == "selected":
        outcomes[0]["selected_move"] = (int(outcomes[0]["selected_move"]) + 1) % 6
    else:
        outcomes[0]["selected_move"] = 99
    with pytest.raises(ValueError):
        verifier.replay_lane(
            ROOT, registration, binding, "A", games, seeds, configs, outcomes
        )


def test_exact_root_and_extra_turn_games_are_included_in_replay() -> None:
    _registration, _binding, _outcome, games, *_ = _publication_lane("A")
    assert any(int(game["exact_root_handoff_count"]) > 0 for game in games)
    # Explicitly observe rules-engine actor retention after a sowing extra turn.
    game = KalahGame.from_state(suites.INITIAL_STATE)
    found_extra_turn = False
    for action in game.possible_moves():
        trial = game.clone()
        actor = trial.current_player
        if trial.move(trial.pit_index(action)) and trial.current_player == actor:
            found_extra_turn = True
            break
    assert found_extra_turn


def test_reconciliation_digest_alteration_rejected() -> None:
    data = ROOT / DATA
    receipt = json.loads((data / "arena-resume-reconciliation.json").read_text())
    outcome = json.loads((data / "outcome-binding.json").read_text())
    entries = {
        lane: rows(ROOT / outcome["lanes"][lane]["games_path"]) for lane in ("A", "B")
    }
    first = receipt["chunks"][0]
    selected = [
        row
        for row in entries[first["lane"]]
        if 0 <= int(row["game_index"]) < int(first["game_count"])
    ]
    first["game_entries_sha256"] = _canonical_entries_digest(selected)
    verifier.verify_chunk_game_entries(first, entries[first["lane"]])
    first["game_entries_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest_mismatch"):
        verifier.verify_chunk_game_entries(first, entries[first["lane"]])


def test_relocated_entrypoint_is_read_only(tmp_path: Path) -> None:
    data = ROOT / DATA
    registration = json.loads((data / "registration.json").read_text())
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for relative in (
        "docs/data/seed429-canonical-policy-normalization",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed422-adam-first-moment/opening-exclusion-proof.json",
        "docs/data/seed422-adam-first-moment/openings.jsonl",
        "docs/data/seed422-adam-first-moment/outcome-ledger.jsonl",
        *(
            f"docs/data/seed426-canonical-overlap/sources/{name}.jsonl.gz"
            for name in (
                "fresh",
                "generic_bootstrap",
                "random_teacher",
                "opening_disagreement",
                "stability",
            )
        ),
    ):
        source, destination = ROOT / relative, relocated / relative
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    proof = json.loads((data / "opening-exclusion-proof.json").read_text())
    for component in proof["components"]:
        source, destination = ROOT / component["path"], relocated / component["path"]
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    correction = json.loads((data / "seed430-verification-correction.json").read_text())
    for source in correction["supplemental_sources"]:
        src, dst = ROOT / source, relocated / source
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for source in registration["execution_source_inventory"]:
        src, dst = ROOT / source, relocated / source
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    seed431 = json.loads((data / "seed431-verification-correction.json").read_text())
    for source in (*seed431["supplemental_sources"], *seed431["frozen_source_sha256"]):
        src, dst = ROOT / source, relocated / source
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    publication = relocated / DATA
    before = {
        path.relative_to(publication): sha(path)
        for path in publication.rglob("*")
        if path.is_file()
    }
    other_cwd = tmp_path / "unrelated"
    other_cwd.mkdir()
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "ml.alphazero_lite.verify_seed431",
            "--root",
            str(relocated),
        ],
        cwd=other_cwd,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["seed431_semantic_verification"] == "valid"
    after = {
        path.relative_to(publication): sha(path)
        for path in publication.rglob("*")
        if path.is_file()
    }
    assert before == after
