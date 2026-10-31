"""Prospective seed482 generation, provenance, and independent verification.

Importing the module and using ``fixture-run`` never runs inference. The separate
``generate`` command invokes the explicitly hash-bound seed455 behavior model
only after complete historical exclusion coverage is available. Selection is
based only on replayed states.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    build_policy_target,
    encode_state,
    trajectory_hash_for_encoded_states,
)
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint

SCHEMA = "seed482-prospective-confirmation-v1"
TARGET = 512
GAME_CAP = 2048
BASE_SEED = 482_000_000
SEED_STRIDE = 1
MIN_ACTIVE_STONES = 33
MAX_MOVES = 200

PROTOCOL: dict[str, Any] = {
    "schema": SCHEMA,
    "distribution": "new prospective seed482 confirmation; not a reproduction of historical fresh sampling",
    "historical_recipe_status": "incomplete: generation source snapshot, exact historical runtime invocation and search implementation closure are not bound by the generation receipt",
    "behavior_model": {
        "identity": "seed455 parent behavior model; not seed479 M or F",
        "weights_sha256": "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c",
        "metadata_sha256": "8b9f4d02395271accd5accb5cde20ec6ddc4ecd63ded192b0b84d9ecf6c649ac",
        "search_policy_sha256": "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
        "native_probe_sha256": "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
        "tablebase_sha256": "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
    },
    "runtime_environment": {
        "python_implementation": "CPython",
        "python_version": "3.14.7",
        "numpy_version": "2.5.2",
        "platform": "Linux-7.2.6-arch2-1-x86_64-with-glibc2.44",
        "inference_precision": "float32 model arithmetic",
        "action_prng": "Python random.Random with seed=(game_seed*1_000_003)+global_index+(worker_id*9_973); each one-game invocation uses global_index=0 and worker_id=0",
    },
    "prospective_generator_source_closure": {
        "ml/alphazero_lite/self_play.py": "76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c",
        "ml/alphazero_lite/pipeline.py": "ad1c9919e13418c4af53be9856406272bdb4f8ed9a53e7424cc9364364772dfa",
        "ml/alphazero_lite/kalah_rules.py": "6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036",
        "ml/alphazero_lite/input_encodings.py": "0a4f56ed60f0c2ed6b93a673dbc0850a966a907912dc26edd092bfd1ac209dbe",
        "ml/alphazero_lite/runtime_search_policy.py": "ccf7cba630a1f852fc603c74cded1ee406b34ed2224e026fedc4cb627f9b16f9",
        "ml/alphazero_lite/search_ablation.py": "6c9a6b9bf85eb4826c1ba33964d5f8675f1402bd6cbd6008a06b6c15d05d4193",
        "ml/alphazero_lite/exact_root_policy_targets.py": "2aca4c1e2e9e9e199fb1d951b143c506785c23121531e26628a59c9afaf79898",
        "ml/alphazero_lite/eval_cache.py": "7f9da64a0ec2b3c00418727d8d5d0de5257037d47dc42cd2b8e72071fb70b8f6",
        "ml/alphazero_lite/endgame_tablebase.py": "fe00e7b48277e2f8a5795232f78b25df4d29449b7b5dc819a09771999b2350f5",
        "ml/alphazero_lite/exact_root_decision.py": "c806a7c3c3a8fd7d7be800f59fc568c670889164a46c7acbde8ff92473776905",
        "ml/alphazero_lite/classic_mcts.py": "9643d859e22fad8ef83211fb38256a30aa15f245775408310c147d7db65e61b9",
        "ml/alphazero_lite/native_exact_root_tablebase.py": "49fe76e8dc0e827db3181397dee7bfbcf3b1b35234757be1ac441ea09eefab6a",
        "ml/alphazero_lite/opening_cache.py": "8ba33240d25c4cad54011b1f2712d0326ad09952c63515a4b794c585de5195ed",
        "ml/alphazero_lite/opponent_pool.py": "8852d67f26d2805c9123b2506d2e0b9229aee4638e0044b9ec482941c4bd299f",
        "ml/alphazero_lite/shadow_root_q.py": "93563da902a4020a312fa02c175e6115ac98e9a03b0679ea0f6109434aab8b13",
        "ml/alphazero_lite/value_transforms.py": "d63bf9772389614cae22e3a599d2151bd0b4f7d1dfbb7f8e5079bb01c6e5a21b",
        "ml/alphazero_lite/train.py": "2dcb42f8dff790d5033e2c9a865f3c24eb1fee6b9b37302200dd176d109dad32",
        "ml/alphazero_lite/report_validation.py": "5d1c02e8b1ac11b1224153769b6d843a046381ef0d64206d33f6b70c30399699",
        "ml/alphazero_lite/run_manifest.py": "ee2453c27ab7a896f145d26d4d6a5727b57de0a64f1128d6702533315a0dcf99",
        "ml/alphazero_lite/generation_record.py": "31a03a28918b1ed2e42ee5749a98d0fde801dfadae1080802335d5d08e8af78e",
        "ml/alphazero_lite/worker_config.py": "b9919e30119c8ff3f99e6a1ff6f7a3eca8aff550815cbc36da8d1e32eabff174",
        "ml/alphazero_lite/export_artifact.py": "f720f34c8b77b32a0f20a00ddcd4f428167235bf0f5a52c29ebbde31663ae481",
    },
    "preserved_recoverable_historical_settings": {
        "simulations_per_root": 1200,
        "input_encoding": "kalah_v3",
        "temperature": 1.1,
        "temperature_late": 0.15,
        "temperature_threshold_plies": 12,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.3,
        "policy_target_mode": "sharpened",
        "policy_target_noise_mode": "denoised",
        "value_target_mode": "default",
        "tree_reuse_enabled": True,
        "runtime_exact_root_threshold": 16,
        "runtime_exact_leaf_solve": "disabled",
    },
    "prospective_settings_not_recovered_historical_facts": {
        "start_state": "standard_4x6",
        "games_per_process": 1,
        "workers": 1,
        "global_game_order": "game_index 0..2047; process independently and serially",
        "seed_schedule": f"game_seed(i) = {BASE_SEED} + {SEED_STRIDE}*i, for i in [0,2047]",
        "schedule_progress_mode": "linear; iteration=1; total_iterations=1",
        "c_puct": 1.25,
        "fpu_mode": "zero",
        "normalize_values": False,
        "root_policy_mode": "visit_count",
        "tactical_root_bias": 0.0,
        "root_temperature": 0.0,
        "action_sampling_noise": "historical denoised policy targets; gameplay Dirichlet alpha=0.3, epsilon=0.3 before ply 12 and disabled from ply 12 onward",
        "new_search_choices": "The receipt does not bind these flags; seed482 explicitly freezes them prospectively and does not call them historical settings.",
        "selection": "scan replayed positions in ascending ply; select the first state with >32 active stones whose canonical and float32 identities are both absent from all exclusions and prior selections; at most one per game",
        "stop": "512 selected states or after complete game index 2047; if fewer, publish shortfall without changing rules",
        "target_distinct_inputs": 512,
        "maximum_complete_games": 2048,
        "eligibility_active_stones_strictly_greater_than": 32,
        "max_moves_per_game": 200,
    },
    "labels": {
        "policy": "prospective current-source semantics: denoised target is from an independent second 1200-simulation PUCT search with no Dirichlet noise and a Random seed drawn from the gameplay RNG; z is its legal visit distribution raised to reciprocal configured temperature and q(a)=z(a)^2/sum_legal(z^2); historical mode name recovered, exact historical generator semantics are not",
        "value": "prospective current-source semantics: default self-play terminal outcome from selected state's player perspective: +1 win, 0 draw, -1 loss; no search-value shaping; historical mode name recovered, exact historical generator semantics are not",
    },
    "exclusion_basis": {
        "seed466_manifest_sha256": "db80f3bdfcc0a126032c1f5a16e78cda74b160d4fa2d80330cec8bed18085b34",
        "seed466_training_sha256": "2e3b503b4e5d64c40d2bca0b44560bfd9f380c2f57606ad8c2db949a1a9c93ae",
        "seed466_evaluation_sha256": "72950845b76cb0cb842d9ba8065591978469e74c9ac51cd1c0f76c120158b364",
        "seed427_validation_membership_sha256": "554f3bd016a0f920ef253538d741a91cb27c9882d2dba94cf7509030af8ca385",
        "available_reconstructed_manifest_sha256": "38e97a88041afd3a7f2e580f30155d17fae93f77ddf9267a6e663ca644b758b8",
        "available_reconstructed_manifest_file_sha256": "129391fc88118a99552029faafb1880f1daed62476b2d1eafd7e931f621dc60d",
        "historical_evaluation_cohort": "all seed466 evaluation identities, including all 851 fresh identities",
        "other_historical_probes": "must be supplied as identity records in the exclusion manifest, reconstructed from their source evidence; no filename/count-only trust",
        "historical_probe_ledger_status": "incomplete: consolidated seed427-480 probe identity reconstruction is unavailable; real generation is blocked until represented and hash-bound",
        "historical_probe_ledger_path": "docs/data/seed482-historical-probe-ledger.jsonl",
        "historical_probe_ledger_sha256": None,
        "required_historical_probe_seed_ids": list(range(427, 481)),
    },
    "endpoints": {
        "initializer_sha256": "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
        "seed479_trajectory_sha256": "2b78bb032d4bd2c384661bd63391abcd8eb66544a1dee8201aabadaff693aa32",
        "M_endpoint_parameters_sha256": "c283db6c4c33a01de5991d8f2a1f56463a9d047c2b7523f4280512f35d8d5ae2",
        "F_endpoint_parameters_sha256": "29f9a687dcdd8f39ca09bb7cf8fe6ba7ea268f4fdabc6b2af04f4d93a06402f3",
        "restriction": "M/F endpoints are inference-only evaluation arms; never used for generation, selection, or labels",
    },
    "confirmation": {
        "contrasts": ["F_vs_initializer", "F_vs_M"],
        "fresh_policy_ce_improvement_minimum": 0.001,
        "pooled_policy_ce_regression_maximum": 0.005,
        "pooled_value_head_mse_regression_maximum": 0.01,
        "full_fresh_training_policy_ce": "must decrease",
        "weightings": ["exposure", "equal_input"],
        "bootstrap_unit": "independent game; one selected position at most per game",
        "bootstrap_groups": "resample game IDs with replacement; all (at most one) selected record per sampled game travels together",
        "bootstrap_settings": {
            "status": "prospective_seed482_choice; not recovered seed481 settings",
            "replicates": 10000,
            "rng_seed": 482,
            "confidence_level": 0.95,
            "interval": "two-sided percentile interval",
            "unit": "game-cluster resampling with replacement",
            "role": "report uncertainty intervals; retained point-estimate gates decide",
        },
        "decision": "advance only if all retained seed479/480 gates pass under both weightings; otherwise close the full-population direction branch",
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.pending")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def canonical_identity(state: dict[str, Any]) -> str:
    return sha_bytes(_canonical_json(state))


def float32_identity(state: dict[str, Any]) -> str:
    encoded = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return sha_bytes(encoded.tobytes())


def state_from_encoded(encoded: Any) -> dict[str, Any]:
    values = np.asarray(encoded, dtype=np.float32)
    if values.shape != (27,) or not np.isfinite(values).all():
        raise ValueError("encoded_state_invalid")
    counts = np.rint(values[:14] * 48).astype(np.int64)
    if np.any(counts < 0) or np.any(np.abs(values[:14] * 48 - counts) > 1e-5):
        raise ValueError("encoded_state_not_invertible")
    current_player = int(round(float(values[14])))
    if current_player not in (0, 1) or abs(float(values[14]) - current_player) > 1e-6:
        raise ValueError("encoded_player_invalid")
    return {
        "player_pits": counts[:6].tolist(),
        "opponent_pits": counts[6:12].tolist(),
        "player_store": int(counts[12]),
        "opponent_store": int(counts[13]),
        "current_player": current_player,
    }


def build_exclusion_manifest(
    training_npz: Path,
    evaluation_npz: Path,
    output: Path,
    *,
    additional_sources: list[Path] | None = None,
) -> dict[str, Any]:
    """Reconstruct exclusion identities from arrays and explicitly supplied evidence.

    Additional JSONL/JSON evidence must contain either state dicts or encoded
    27-feature states. Raw source bytes are hashed into the resulting manifest.
    """
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    sources: dict[str, str] = {}
    historical_seed_ids: set[int] = set()

    def add_state(state: dict[str, Any], encoded: Any | None = None) -> None:
        vector = np.asarray(
            encode_state(state, input_encoding="kalah_v3")
            if encoded is None
            else encoded,
            dtype="<f4",
        )
        record = {
            "state": state,
            "canonical_identity": canonical_identity(state),
            "float32_identity": sha_bytes(vector.tobytes()),
        }
        identities[(record["canonical_identity"], record["float32_identity"])] = record

    sources["seed466_training"] = sha_bytes(training_npz.read_bytes())
    with np.load(training_npz, allow_pickle=False) as archive:
        if "inputs" not in archive:
            raise ValueError("exclusion_source_missing_inputs:seed466_training")
        training_inputs = archive["inputs"]
        for vector in training_inputs:
            state = state_from_encoded(vector)
            add_state(state, vector)
    sources["seed466_evaluation"] = sha_bytes(evaluation_npz.read_bytes())
    with np.load(evaluation_npz, allow_pickle=False) as archive:
        for compact_row, input_identity in zip(
            archive["compact_rows"], archive["input_identity"], strict=True
        ):
            vector = training_inputs[int(compact_row)]
            if (
                bytes.fromhex(str(input_identity))
                != np.asarray(vector, dtype="<f4").tobytes()
            ):
                raise ValueError("seed466_evaluation_input_identity_mismatch")
            add_state(state_from_encoded(vector), vector)
    for source_path in additional_sources or []:
        raw = source_path.read_bytes()
        sources[str(source_path)] = sha_bytes(raw)
        if source_path.suffix == ".npz":
            with np.load(source_path, allow_pickle=False) as archive:
                member = next(
                    (name for name in ("inputs", "states") if name in archive),
                    None,
                )
                if member is None:
                    raise ValueError(f"exclusion_npz_missing_inputs:{source_path}")
                for vector in archive[member]:
                    add_state(state_from_encoded(vector), vector)
            continue
        decoded = gzip.decompress(raw) if source_path.suffix == ".gz" else raw
        if source_path.name.endswith((".jsonl", ".jsonl.gz")):
            rows = [json.loads(line) for line in decoded.splitlines() if line]
        else:
            parsed = json.loads(decoded)
            rows = parsed if isinstance(parsed, list) else parsed.get("identities", [])
        for row in rows:
            if row.get("historical_seed") is not None:
                historical_seed_ids.add(int(row["historical_seed"]))
            state = row.get("state")
            if isinstance(state, list):
                state = state_from_encoded(state)
                add_state(state, row["state"])
            elif isinstance(state, dict):
                add_state(state)
            elif isinstance(row.get("input"), list):
                state = state_from_encoded(row["input"])
                add_state(state, row["input"])
            elif row.get("canonical_identity") and row.get("input_identity"):
                raw_input = bytes.fromhex(str(row["input_identity"]))
                vector = np.frombuffer(raw_input, dtype="<f4")
                state = state_from_encoded(vector)
                source_key = json.loads(row["canonical_identity"])
                if isinstance(source_key, list) and len(source_key) == 5:
                    source_state = {
                        "player_pits": source_key[0],
                        "opponent_pits": source_key[1],
                        "player_store": source_key[2],
                        "opponent_store": source_key[3],
                        "current_player": source_key[4],
                    }
                elif isinstance(source_key, dict):
                    source_state = source_key
                else:
                    raise ValueError(
                        f"historical_canonical_identity_invalid:{source_path}"
                    )
                if canonical_identity(source_state) != canonical_identity(state):
                    raise ValueError(f"historical_identity_disagreement:{source_path}")
                add_state(state, vector)
            else:
                raise ValueError(f"exclusion_source_state_missing:{source_path}")
    manifest: dict[str, Any] = {
        "schema": "seed482-exclusions-v1",
        "sources_sha256": sources,
        "identity_definition": {
            "canonical": "SHA-256 of normalized canonical state JSON",
            "float32": "SHA-256 of exact little-endian float32 kalah_v3 input bytes",
        },
        "historical_probe_seed_ids_reconstructed": sorted(historical_seed_ids),
        "historical_probe_coverage": (
            "complete_seed427_to_seed480"
            if set(range(427, 481)) <= historical_seed_ids
            else "incomplete"
        ),
        "identities": sorted(
            identities.values(),
            key=lambda item: (item["canonical_identity"], item["float32_identity"]),
        ),
    }
    manifest["manifest_sha256"] = sha_bytes(_canonical_json(manifest))
    serialized = (
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    if output.suffix == ".gz":
        output.write_bytes(gzip.compress(serialized, mtime=0))
    else:
        output.write_bytes(serialized)
    return manifest


def protocol_digest(protocol: dict[str, Any] = PROTOCOL) -> str:
    bound = {
        "protocol": protocol,
        "preparation_implementation_sha256": sha_bytes(Path(__file__).read_bytes()),
    }
    return sha_bytes(_canonical_json(bound))


def verify_runtime_files(root: Path) -> dict[str, str]:
    """Verify all pinned teacher/runtime/source bytes without loading a model."""
    expected = {
        "model-artifact/current/weights.json": PROTOCOL["behavior_model"][
            "weights_sha256"
        ],
        "model-artifact/current/metadata.json": PROTOCOL["behavior_model"][
            "metadata_sha256"
        ],
        "model-artifact/current/search_policy.json": PROTOCOL["behavior_model"][
            "search_policy_sha256"
        ],
        "model-artifact/runtime/kalah_v1_tablebase": PROTOCOL["behavior_model"][
            "native_probe_sha256"
        ],
        "model-artifact/runtime/kalah_v1_21.kvtb": PROTOCOL["behavior_model"][
            "tablebase_sha256"
        ],
        **PROTOCOL["prospective_generator_source_closure"],
    }
    actual: dict[str, str] = {}
    for relative, digest in expected.items():
        actual[relative] = sha_bytes((root / relative).read_bytes())
        if actual[relative] != digest:
            raise ValueError(f"runtime_identity_mismatch:{relative}")
    return actual


def expected_seed(game_index: int) -> int:
    if not 0 <= game_index < GAME_CAP:
        raise ValueError("game_index_out_of_range")
    return BASE_SEED + SEED_STRIDE * game_index


def _active_stones(state: dict[str, Any]) -> int:
    return sum(int(item) for item in state["player_pits"] + state["opponent_pits"])


def _replay(game_record: dict[str, Any]) -> list[dict[str, Any]]:
    index = int(game_record["game_index"])
    if int(game_record["seed"]) != expected_seed(index):
        raise ValueError("game_seed_mismatch")
    expected_start = {
        "player_pits": [4] * 6,
        "opponent_pits": [4] * 6,
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    if game_record["start_state"] != expected_start:
        raise ValueError("start_state_mismatch")
    game = KalahGame.from_state(game_record["start_state"])
    rows = game_record["positions"]
    moves = game_record["moves"]
    if len(moves) > MAX_MOVES:
        raise ValueError("game_move_cap_exceeded")
    if len(rows) != len(moves):
        raise ValueError("incomplete_game_record")
    for ply, (row, move) in enumerate(zip(rows, moves, strict=True)):
        if row.get("ply") != ply or row.get("state") != game.to_state():
            raise ValueError("replay_state_mismatch")
        if int(move) not in game.possible_moves():
            raise ValueError("illegal_replay_move")
        if not game.move(game.pit_index(int(move))):
            raise ValueError("illegal_replay_move")
    if not game.over():
        raise ValueError("game_not_complete")
    if game_record.get("winner") != game.winner:
        raise ValueError("game_winner_mismatch")
    return rows


def _row_label_valid(row: dict[str, Any], winner: int | None) -> None:
    state = row["state"]
    if row.get("value") != (
        0 if winner is None else (1 if winner == state["current_player"] else -1)
    ):
        raise ValueError("value_label_mismatch")
    policy = np.asarray(row["policy"], dtype=np.float32)
    if policy.shape != (6,) or not np.isfinite(policy).all() or np.any(policy < 0):
        raise ValueError("policy_label_invalid")
    legal = KalahGame.from_state(state).possible_moves()
    if any(float(policy[action]) != 0.0 for action in range(6) if action not in legal):
        raise ValueError("policy_label_illegal_mass")
    if not np.isclose(float(policy.sum(dtype=np.float64)), 1.0, atol=1e-6):
        raise ValueError("policy_label_not_normalized")
    if (
        row.get("policy_target_mode") != "sharpened"
        or row.get("policy_target_noise_mode") != "denoised"
    ):
        raise ValueError("target_semantics_mismatch")


def select_game_position(
    game_record: dict[str, Any],
    excluded_canonical: set[str],
    excluded_float32: set[str],
    already_selected_canonical: set[str],
    already_selected_float32: set[str],
) -> dict[str, Any] | None:
    """Select first eligible replayed position; target labels are never consulted."""
    rows = _replay(game_record)
    for row in rows:
        state = row["state"]
        if _active_stones(state) < MIN_ACTIVE_STONES:
            continue
        c_id = canonical_identity(state)
        f_id = float32_identity(state)
        if c_id in excluded_canonical or f_id in excluded_float32:
            continue
        if c_id in already_selected_canonical or f_id in already_selected_float32:
            continue
        return {
            "game_index": int(game_record["game_index"]),
            "seed": int(game_record["seed"]),
            "ply": int(row["ply"]),
            "state": state,
            "policy": row["policy"],
            "value": row["value"],
            "canonical_identity": c_id,
            "float32_identity": f_id,
            "active_stones": _active_stones(state),
            "label_source": "same frozen seed455 behavior search; stored policy target and terminal-outcome value",
            "game_provenance": game_record.get(
                "provenance", {"fixture": "synthetic_test_game"}
            ),
        }
    return None


def load_exclusions(path: Path) -> tuple[dict[str, Any], set[str], set[str]]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    manifest = json.loads(raw)
    claimed = manifest.pop("manifest_sha256", None)
    actual = sha_bytes(_canonical_json(manifest))
    if claimed != actual:
        raise ValueError("exclusion_manifest_receipt_mismatch")
    manifest["manifest_sha256"] = claimed
    canonical: set[str] = set()
    floats: set[str] = set()
    for record in manifest["identities"]:
        state = record["state"]
        if canonical_identity(state) != record["canonical_identity"]:
            raise ValueError("exclusion_canonical_identity_mismatch")
        if float32_identity(state) != record["float32_identity"]:
            raise ValueError("exclusion_float32_identity_mismatch")
        canonical.add(record["canonical_identity"])
        floats.add(record["float32_identity"])
    return manifest, canonical, floats


def run_fixture_games(
    game_path: Path,
    exclusions_path: Path,
    output_dir: Path,
    protocol: dict[str, Any] = PROTOCOL,
    *,
    max_games: int = GAME_CAP,
) -> dict[str, Any]:
    """Resume in fixed index order over complete game records supplied by caller."""
    if protocol != PROTOCOL:
        raise ValueError("protocol_identity_mismatch")
    exclusions, ex_c, ex_f = load_exclusions(exclusions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    selected_path = output_dir / "selected.jsonl"
    seen_games: dict[int, str] = {}
    selected: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint_body = dict(checkpoint)
        claimed_checkpoint_sha = checkpoint_body.pop("checkpoint_sha256", None)
        if claimed_checkpoint_sha != sha_bytes(_canonical_json(checkpoint_body)):
            raise ValueError("checkpoint_receipt_mismatch")
        if checkpoint["protocol_sha256"] != protocol_digest(protocol):
            raise ValueError("checkpoint_protocol_mismatch")
        if checkpoint["exclusion_manifest_sha256"] != exclusions["manifest_sha256"]:
            raise ValueError("checkpoint_exclusion_mismatch")
        for line in selected_path.read_text().splitlines():
            item = json.loads(line)
            if str(item["game_index"]) in checkpoint["completed_games"]:
                selected.append(item)
        seen_games = {int(k): v for k, v in checkpoint["completed_games"].items()}
        recovered_selection = b"".join(
            (json.dumps(item, sort_keys=True) + "\n").encode() for item in selected
        )
        if (
            len(selected) != checkpoint["selected_count"]
            or sha_bytes(recovered_selection) != checkpoint["selected_sha256"]
        ):
            raise ValueError("checkpoint_selection_reconstruction_mismatch")
        _atomic_write(selected_path, recovered_selection)
    else:
        _atomic_write(selected_path, b"")
    records = [json.loads(line) for line in game_path.read_text().splitlines() if line]
    records.sort(key=lambda item: int(item["game_index"]))
    indexes = [int(item["game_index"]) for item in records]
    if indexes and indexes != list(range(len(indexes))):
        raise ValueError("game_order_or_cap_violation")
    if len(indexes) > GAME_CAP:
        raise ValueError("game_order_or_cap_violation")
    indices = [int(item["game_index"]) for item in records]
    if max_games > GAME_CAP:
        raise ValueError("game_cap_exceeds_frozen_limit")
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate_game_index")
    if indices and indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("game_order_gap")
    if indices and indices[0] != 0:
        raise ValueError("game_order_must_start_at_zero")
    if not set(seen_games) <= set(indices):
        raise ValueError("checkpoint_game_inventory_mismatch")
    done_c = {item["canonical_identity"] for item in selected}
    done_f = {item["float32_identity"] for item in selected}
    for record in records:
        if len(selected) >= TARGET:
            break
        index = int(record["game_index"])
        if index >= max_games:
            break
        fingerprint = sha_bytes(_canonical_json(record))
        if index in seen_games:
            if seen_games[index] != fingerprint:
                raise ValueError("resumed_game_receipt_mismatch")
            continue
        _replay(record)
        for row in record["positions"]:
            if row.get("winner", record.get("winner")) != record.get("winner"):
                raise ValueError("row_winner_mismatch")
            _row_label_valid(row, record.get("winner"))
        choice = select_game_position(record, ex_c, ex_f, done_c, done_f)
        if choice is not None:
            selected.append(choice)
            done_c.add(choice["canonical_identity"])
            done_f.add(choice["float32_identity"])
        seen_games[index] = fingerprint
        selected_bytes = b"".join(
            (json.dumps(item, sort_keys=True) + "\n").encode() for item in selected
        )
        _atomic_write(selected_path, selected_bytes)
        checkpoint = {
            "schema": "seed482-generation-checkpoint-v1",
            "protocol_sha256": protocol_digest(protocol),
            "exclusion_manifest_sha256": exclusions["manifest_sha256"],
            "completed_games": {str(k): v for k, v in sorted(seen_games.items())},
            "selected_sha256": sha_bytes(selected_bytes),
            "selected_count": len(selected),
        }
        checkpoint["checkpoint_sha256"] = sha_bytes(_canonical_json(checkpoint))
        _atomic_write(
            checkpoint_path,
            (json.dumps(checkpoint, indent=2, sort_keys=True) + "\n").encode(),
        )
        if len(selected) >= TARGET:
            break
    completed_count = min(max_games, len(seen_games))
    cohort = {
        "schema": "seed482-cohort-receipt-v1",
        "status": "complete" if len(selected) == TARGET else "shortfall",
        "protocol_sha256": protocol_digest(protocol),
        "exclusion_manifest_sha256": exclusions["manifest_sha256"],
        "selected_count": len(selected),
        "target_count": TARGET,
        "games_examined": completed_count,
        "game_cap": max_games,
        "selected_sha256": sha_bytes(selected_path.read_bytes()),
        "selection_unit": "game",
        "decision": "selection fixed before inference; shortfall does not authorize rule changes",
    }
    cohort["receipt_sha256"] = sha_bytes(_canonical_json(cohort))
    (output_dir / "cohort.json").write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n"
    )
    return cohort


def _game_record_from_selfplay(
    rows: list[dict[str, Any]], index: int
) -> dict[str, Any]:
    if not rows or not all(row.get("game_completed") is True for row in rows):
        raise ValueError("selfplay_game_incomplete")
    first = rows[0]
    moves = first.get("game_moves")
    if not isinstance(moves, list) or len(moves) != len(rows):
        raise ValueError("selfplay_move_receipt_missing")
    positions = [
        {
            "ply": int(row["move_index"]),
            "state": state_from_encoded(row["state"]),
            "policy": row["policy"],
            "value": row["value"],
            "policy_target_mode": row["policy_target_mode"],
            "policy_target_noise_mode": row["policy_target_noise_mode"],
        }
        for row in rows
    ]
    if any(
        row.get("game_moves") is not None and row["game_moves"] != moves for row in rows
    ):
        raise ValueError("selfplay_move_receipt_disagreement")
    if any(
        row.get("winner") != first.get("winner")
        or row.get("game_seed") != first.get("game_seed")
        or row.get("game_index") != 0
        or row.get("game_length") != len(rows)
        or row.get("trajectory_hash") != first.get("trajectory_hash")
        or (
            row.get("game_start_state") is not None
            and row["game_start_state"] != first.get("game_start_state")
        )
        for row in rows
    ):
        raise ValueError("selfplay_game_receipt_disagreement")
    expected_trajectory_hash = trajectory_hash_for_encoded_states(
        [list(row["state"]) for row in rows], winner=first.get("winner")
    )
    if first.get("trajectory_hash") != expected_trajectory_hash:
        raise ValueError("selfplay_trajectory_hash_mismatch")
    for raw_row in rows:
        _validate_raw_selfplay_row(raw_row)
    return {
        "game_index": index,
        "seed": int(first["game_seed"]),
        "start_state": first["game_start_state"],
        "moves": moves,
        "winner": first["winner"],
        "positions": positions,
    }


def _validate_raw_selfplay_row(row: dict[str, Any]) -> None:
    if (
        row.get("policy_target_mode") != "sharpened"
        or row.get("policy_target_noise_mode") != "denoised"
        or row.get("value_target_mode") != "default"
    ):
        raise ValueError("selfplay_target_mode_mismatch")
    profile = dict(row["search_profile"])
    claimed_profile_hash = profile.pop("hash", None)
    profile_bytes = json.dumps(
        profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    if (
        claimed_profile_hash != sha_bytes(profile_bytes)
        or row.get("search_profile_hash") != claimed_profile_hash
    ):
        raise ValueError("selfplay_search_profile_hash_mismatch")
    options = profile.get("search_options", {})
    expected_options = {
        "fpu_mode": "zero",
        "reuse_subtree": True,
        "normalize_values": False,
        "root_policy_mode": "visit_count",
        "tactical_root_bias": 0.0,
        "root_temperature": 0.0,
    }
    if (
        profile.get("kind") != "self_play"
        or profile.get("player_mode") != "puct"
        or profile.get("simulations") != 1200
        or profile.get("c_puct") != 1.25
        or any(options.get(key) != value for key, value in expected_options.items())
    ):
        raise ValueError("selfplay_search_settings_mismatch")
    expected_runtime = {
        "runtime_search_policy_mode": "puct_exact_root_hybrid",
        "runtime_search_policy_native_probe_sha256": PROTOCOL["behavior_model"][
            "native_probe_sha256"
        ],
        "runtime_search_policy_tablebase_sha256": PROTOCOL["behavior_model"][
            "tablebase_sha256"
        ],
        "exact_root_solve_threshold": 16,
        "exact_root_solver": "native_kvtb_root_action_probe_v1",
        "exact_root_solve_enabled": True,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
    }
    if any(profile.get(key) != value for key, value in expected_runtime.items()):
        raise ValueError("selfplay_runtime_search_settings_mismatch")
    state = state_from_encoded(row["state"])
    if _active_stones(state) > 32:
        sampling_epsilon = 0.3 if int(row["move_index"]) < 12 else 0.0
        if (
            row.get("policy_target_actual_mode") != "sharpened"
            or row.get("teacher_source") != "puct"
            or int(row.get("simulations", -1)) != 1200
            or row.get("dirichlet_alpha") != sampling_epsilon
            or row.get("sampling_dirichlet_epsilon") != sampling_epsilon
            or row.get("dirichlet_epsilon_for_sampling") != sampling_epsilon
            or row.get("target_dirichlet_epsilon") != 0.0
            or row.get("dirichlet_epsilon_for_target") != 0.0
            or "root_visit_counts" not in row
        ):
            raise ValueError("high_stone_teacher_search_provenance_mismatch")
        temperature = 1.1 if int(row["move_index"]) < 12 else 0.15
        reconstructed = np.asarray(
            build_policy_target(
                np.asarray(row["root_visit_counts"], dtype=np.int64),
                legal_moves=KalahGame.from_state(state).possible_moves(),
                temperature=temperature,
                mode="sharpened",
            ),
            dtype=np.float32,
        )
        stored = np.asarray(row["policy"], dtype=np.float32)
        if not np.array_equal(reconstructed, stored):
            raise ValueError("policy_target_reconstruction_mismatch")


def generate_frozen_games(
    root: Path, exclusions_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Run the frozen historical behavior model through prospective seed482 settings.

    This function is an execution entry point and is intentionally not called by
    preparation tests. It performs real model searches when explicitly invoked.
    """
    root = root.resolve()
    exclusions_path = exclusions_path.resolve()
    output_dir = output_dir.resolve()
    runtime_files = verify_runtime_files(root)
    exclusions, _, _ = load_exclusions(exclusions_path)
    ledger_sha256 = PROTOCOL["exclusion_basis"].get("historical_probe_ledger_sha256")
    ledger_path = PROTOCOL["exclusion_basis"].get("historical_probe_ledger_path")
    if (
        not isinstance(ledger_sha256, str)
        or len(ledger_sha256) != 64
        or not isinstance(ledger_path, str)
        or exclusions.get("sources_sha256", {}).get(ledger_path) != ledger_sha256
    ):
        raise ValueError("historical_probe_identity_ledger_unavailable")
    if exclusions.get("historical_probe_coverage") != "complete_seed427_to_seed480":
        raise ValueError("historical_probe_identity_ledger_unavailable")
    required_source_hashes = {
        "seed466_training": PROTOCOL["exclusion_basis"]["seed466_training_sha256"],
        "seed466_evaluation": PROTOCOL["exclusion_basis"]["seed466_evaluation_sha256"],
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz": PROTOCOL[
            "exclusion_basis"
        ]["seed427_validation_membership_sha256"],
    }
    for source, digest in required_source_hashes.items():
        if exclusions.get("sources_sha256", {}).get(source) != digest:
            raise ValueError(f"required_exclusion_source_mismatch:{source}")
    if not set(
        PROTOCOL["exclusion_basis"]["required_historical_probe_seed_ids"]
    ) <= set(exclusions.get("historical_probe_seed_ids_reconstructed", [])):
        raise ValueError("historical_probe_identity_ledger_incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    games_path = output_dir / "complete-games.jsonl"
    if games_path.exists():
        existing = [
            json.loads(line) for line in games_path.read_text().splitlines() if line
        ]
    else:
        existing = []
    if existing:
        run_fixture_games(games_path, exclusions_path, output_dir)
        current = json.loads((output_dir / "cohort.json").read_text())
        if current["selected_count"] == TARGET:
            return current
    start_index = len(existing)
    checkpoint_npz = output_dir / "seed455-behavior-checkpoint.npz"
    materialize_weights_json_checkpoint(
        weights_path=root / "model-artifact/current/weights.json",
        out_path=checkpoint_npz,
    )
    checkpoint_sha256 = sha_bytes(checkpoint_npz.read_bytes())
    for game in existing:
        if (
            game.get("provenance", {}).get("behavior_checkpoint_npz_sha256")
            != checkpoint_sha256
        ):
            raise ValueError("resumed_behavior_checkpoint_mismatch")
    self_play_script = root / "ml/alphazero_lite/self_play.py"
    for index in range(start_index, GAME_CAP):
        game_seed = expected_seed(index)
        raw_path = output_dir / f"self-play-game-{index:04d}.jsonl"
        command = [
            sys.executable,
            str(self_play_script),
            "--out",
            str(raw_path),
            "--games",
            "1",
            "--workers",
            "1",
            "--seed",
            str(game_seed),
            "--checkpoint",
            str(checkpoint_npz),
            "--runtime-search-policy-artifact",
            str(root / "model-artifact/current"),
            "--input-encoding",
            "kalah_v3",
            "--simulations",
            "1200",
            "--c-puct",
            "1.25",
            "--temperature",
            "1.1",
            "--temperature-late",
            "0.15",
            "--temperature-threshold",
            "12",
            "--dirichlet-alpha",
            "0.3",
            "--dirichlet-epsilon",
            "0.3",
            "--schedule-progress-mode",
            "linear",
            "--iteration",
            "1",
            "--total-iterations",
            "1",
            "--max-moves",
            "200",
            "--tree-reuse-enabled",
            "--policy-target-mode",
            "sharpened",
            "--policy-target-noise-mode",
            "denoised",
            "--value-target-mode",
            "default",
            "--write-game-metadata",
            "--write-root-target-telemetry",
            "--start-state-mode",
            "standard_4x6",
            "--fpu-mode",
            "zero",
            "--root-policy-mode",
            "visit_count",
            "--tactical-root-bias",
            "0",
            "--root-temperature",
            "0",
        ]
        subprocess.run(command, cwd=root, check=True)
        raw_rows = [
            json.loads(line) for line in raw_path.read_text().splitlines() if line
        ]
        game = _game_record_from_selfplay(raw_rows, index)
        game["provenance"] = {
            "protocol_sha256": protocol_digest(),
            "behavior_weights_sha256": PROTOCOL["behavior_model"]["weights_sha256"],
            "behavior_checkpoint_npz_path": checkpoint_npz.name,
            "behavior_checkpoint_npz_sha256": checkpoint_sha256,
            "runtime_search_policy_sha256": PROTOCOL["behavior_model"][
                "search_policy_sha256"
            ],
            "runtime_source_closure_sha256": sha_bytes(_canonical_json(runtime_files)),
            "raw_selfplay_path": raw_path.relative_to(output_dir).as_posix(),
            "raw_selfplay_sha256": sha_bytes(raw_path.read_bytes()),
        }
        with games_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(game, sort_keys=True) + "\n")
            stream.flush()
        receipt = run_fixture_games(games_path, exclusions_path, output_dir)
        if receipt["selected_count"] == TARGET:
            return receipt
    return json.loads((output_dir / "cohort.json").read_text())


def verify_cohort(
    game_path: Path, exclusions_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Independently reconstruct selection and validate receipts, without writes."""
    exclusions, ex_c, ex_f = load_exclusions(exclusions_path)
    selected_path = output_dir / "selected.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    cohort_path = output_dir / "cohort.json"
    selected = [
        json.loads(line) for line in selected_path.read_text().splitlines() if line
    ]
    checkpoint = json.loads(checkpoint_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    checkpoint_body = dict(checkpoint)
    claimed_checkpoint_sha = checkpoint_body.pop("checkpoint_sha256", None)
    if claimed_checkpoint_sha != sha_bytes(_canonical_json(checkpoint_body)):
        raise ValueError("checkpoint_receipt_mismatch")
    if (
        checkpoint["protocol_sha256"] != protocol_digest()
        or cohort["protocol_sha256"] != protocol_digest()
    ):
        raise ValueError("protocol_receipt_mismatch")
    if checkpoint["exclusion_manifest_sha256"] != exclusions["manifest_sha256"]:
        raise ValueError("checkpoint_exclusion_mismatch")
    if (
        cohort["selected_sha256"] != sha_bytes(selected_path.read_bytes())
        or checkpoint["selected_sha256"] != cohort["selected_sha256"]
    ):
        raise ValueError("selected_receipt_mismatch")
    receipt = dict(cohort)
    claimed = receipt.pop("receipt_sha256")
    if claimed != sha_bytes(_canonical_json(receipt)):
        raise ValueError("cohort_receipt_mismatch")
    records = [json.loads(line) for line in game_path.read_text().splitlines() if line]
    records.sort(key=lambda item: int(item["game_index"]))
    verify_indexes = [int(item["game_index"]) for item in records]
    if verify_indexes and verify_indexes != list(range(len(verify_indexes))):
        raise ValueError("game_order_or_cap_violation")
    if len(verify_indexes) > GAME_CAP:
        raise ValueError("game_order_or_cap_violation")
    checkpoint_games = {
        str(key): value for key, value in checkpoint["completed_games"].items()
    }
    replayed_games: dict[str, str] = {}
    expected: list[dict[str, Any]] = []
    seen_c: set[str] = set()
    seen_f: set[str] = set()
    runtime_sources: dict[str, str] | None = None
    for record in records:
        if len(expected) == TARGET:
            break
        _replay(record)
        provenance = record.get("provenance")
        if provenance is not None:
            if provenance.get("protocol_sha256") != protocol_digest():
                raise ValueError("game_protocol_provenance_mismatch")
            if (
                provenance.get("behavior_weights_sha256")
                != PROTOCOL["behavior_model"]["weights_sha256"]
            ):
                raise ValueError("game_behavior_model_mismatch")
            checkpoint_file = output_dir / provenance["behavior_checkpoint_npz_path"]
            if sha_bytes(checkpoint_file.read_bytes()) != provenance.get(
                "behavior_checkpoint_npz_sha256"
            ):
                raise ValueError("game_behavior_checkpoint_mismatch")
            if (
                provenance.get("runtime_search_policy_sha256")
                != PROTOCOL["behavior_model"]["search_policy_sha256"]
            ):
                raise ValueError("game_search_policy_mismatch")
            if runtime_sources is None:
                runtime_sources = verify_runtime_files(
                    Path(__file__).resolve().parents[2]
                )
            if provenance.get("runtime_source_closure_sha256") != sha_bytes(
                _canonical_json(runtime_sources)
            ):
                raise ValueError("game_runtime_closure_mismatch")
            raw_path = output_dir / provenance["raw_selfplay_path"]
            raw_bytes = raw_path.read_bytes()
            if sha_bytes(raw_bytes) != provenance.get("raw_selfplay_sha256"):
                raise ValueError("raw_selfplay_receipt_mismatch")
            raw_rows = [json.loads(line) for line in raw_bytes.splitlines() if line]
            raw_game = _game_record_from_selfplay(raw_rows, int(record["game_index"]))
            if any(record[key] != raw_game[key] for key in raw_game):
                raise ValueError("raw_selfplay_reconstruction_mismatch")
        replayed_games[str(record["game_index"])] = sha_bytes(_canonical_json(record))
        for row in record["positions"]:
            if row.get("winner", record.get("winner")) != record.get("winner"):
                raise ValueError("row_winner_mismatch")
            _row_label_valid(row, record.get("winner"))
        choice = select_game_position(record, ex_c, ex_f, seen_c, seen_f)
        if choice is not None:
            expected.append(choice)
            seen_c.add(choice["canonical_identity"])
            seen_f.add(choice["float32_identity"])
    if replayed_games != checkpoint_games:
        raise ValueError("completed_game_receipt_mismatch")
    if len(expected) != len(selected) or any(
        _canonical_json(a) != _canonical_json(b)
        for a, b in zip(expected, selected, strict=True)
    ):
        raise ValueError("selection_reconstruction_mismatch")
    if len(selected) > TARGET or len({x["game_index"] for x in selected}) != len(
        selected
    ):
        raise ValueError("cohort_uniqueness_violation")
    if (
        cohort["target_count"] != TARGET
        or cohort["game_cap"] != GAME_CAP
        or cohort["selected_count"] != len(selected)
        or cohort["games_examined"] != len(checkpoint_games)
        or cohort["status"] != ("complete" if len(selected) == TARGET else "shortfall")
    ):
        raise ValueError("cohort_completion_fields_mismatch")
    if any(
        x["canonical_identity"] in ex_c or x["float32_identity"] in ex_f
        for x in selected
    ):
        raise ValueError("cohort_exclusion_overlap")
    return {
        "status": "verified",
        "selected_count": len(selected),
        "games": len(records),
    }


def require_frozen_cohort(receipt_path: Path, expected_sha256: str) -> dict[str, Any]:
    """Evaluation gate: only a complete independently verified receipt is accepted."""
    raw = receipt_path.read_bytes()
    if sha_bytes(raw) != expected_sha256:
        raise ValueError("frozen_cohort_identity_mismatch")
    receipt = json.loads(raw)
    receipt_body = dict(receipt)
    claimed_receipt = receipt_body.pop("frozen_receipt_sha256", None)
    if claimed_receipt != sha_bytes(_canonical_json(receipt_body)):
        raise ValueError("frozen_cohort_receipt_mismatch")
    if (
        receipt.get("schema") != "seed482-frozen-cohort-receipt-v1"
        or receipt.get("status") != "complete"
        or receipt.get("selected_count") != TARGET
        or receipt.get("independent_verification") != "passed"
    ):
        raise ValueError("cohort_not_complete_and_frozen")
    return receipt


def authorize_evaluation(
    receipt_path: Path,
    expected_sha256: str,
    *,
    initializer_sha256: str,
    m_endpoint_sha256: str,
    f_endpoint_sha256: str,
) -> dict[str, Any]:
    """Require a frozen verified cohort and separately bound endpoints."""
    receipt = require_frozen_cohort(receipt_path, expected_sha256)
    expected_endpoints = PROTOCOL["endpoints"]
    actual = {
        "initializer_sha256": initializer_sha256,
        "M_endpoint_parameters_sha256": m_endpoint_sha256,
        "F_endpoint_parameters_sha256": f_endpoint_sha256,
    }
    for key, value in actual.items():
        if expected_endpoints[key] != value:
            raise ValueError(f"evaluation_endpoint_identity_mismatch:{key}")
    return {"status": "authorized_for_separate_evaluation", "cohort": receipt}


def freeze_cohort(
    game_path: Path, exclusions_path: Path, output_dir: Path, frozen_path: Path
) -> dict[str, Any]:
    """Publish a separate, verified cohort receipt; generated files remain untouched."""
    verification = verify_cohort(game_path, exclusions_path, output_dir)
    cohort_path = output_dir / "cohort.json"
    cohort = json.loads(cohort_path.read_text())
    if cohort.get("status") != "complete" or verification["selected_count"] != TARGET:
        raise ValueError("cohort_not_complete")
    receipt: dict[str, Any] = {
        "schema": "seed482-frozen-cohort-receipt-v1",
        "status": "complete",
        "protocol_sha256": protocol_digest(),
        "cohort_receipt_file_sha256": sha_bytes(cohort_path.read_bytes()),
        "selected_sha256": cohort["selected_sha256"],
        "exclusion_manifest_sha256": cohort["exclusion_manifest_sha256"],
        "selected_count": TARGET,
        "independent_verification": "passed",
        "verification": verification,
    }
    receipt["frozen_receipt_sha256"] = sha_bytes(_canonical_json(receipt))
    frozen_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("fixture-run")
    run.add_argument("--games", type=Path, required=True)
    run.add_argument("--exclusions", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--root", type=Path, default=Path.cwd())
    generate.add_argument("--exclusions", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--games", type=Path, required=True)
    verify.add_argument("--exclusions", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--games", type=Path, required=True)
    freeze.add_argument("--exclusions", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--frozen-receipt", type=Path, required=True)
    gate = sub.add_parser("evaluation-gate")
    gate.add_argument("--cohort", type=Path, required=True)
    gate.add_argument("--expected-sha256", required=True)
    gate.add_argument("--initializer-sha256", required=True)
    gate.add_argument("--m-endpoint-sha256", required=True)
    gate.add_argument("--f-endpoint-sha256", required=True)
    preflight = sub.add_parser("runtime-preflight")
    preflight.add_argument("--root", type=Path, default=Path.cwd())
    exclusions_command = sub.add_parser("build-exclusions")
    exclusions_command.add_argument("--training", type=Path, required=True)
    exclusions_command.add_argument("--evaluation", type=Path, required=True)
    exclusions_command.add_argument(
        "--historical-input", type=Path, action="append", default=[]
    )
    exclusions_command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fixture-run":
        result = run_fixture_games(args.games, args.exclusions, args.output)
    elif args.command == "generate":
        result = generate_frozen_games(args.root, args.exclusions, args.output)
    elif args.command == "freeze":
        result = freeze_cohort(
            args.games, args.exclusions, args.output, args.frozen_receipt
        )
    elif args.command == "verify":
        result = verify_cohort(args.games, args.exclusions, args.output)
    elif args.command == "evaluation-gate":
        result = authorize_evaluation(
            args.cohort,
            args.expected_sha256,
            initializer_sha256=args.initializer_sha256,
            m_endpoint_sha256=args.m_endpoint_sha256,
            f_endpoint_sha256=args.f_endpoint_sha256,
        )
    elif args.command == "runtime-preflight":
        result = verify_runtime_files(args.root)
    else:
        exclusion_manifest = build_exclusion_manifest(
            args.training,
            args.evaluation,
            args.output,
            additional_sources=args.historical_input,
        )
        result = {
            "manifest_sha256": exclusion_manifest["manifest_sha256"],
            "identity_count": len(exclusion_manifest["identities"]),
            "source_count": len(exclusion_manifest["sources_sha256"]),
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
