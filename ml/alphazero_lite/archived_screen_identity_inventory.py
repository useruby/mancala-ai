"""Authenticate archived screens, replay their rules traces, and census state identities.

This entry point intentionally imports only the standard library and the pure
Python Kalah rules engine. It never loads model artifacts, NumPy, Torch, or a
tablebase. The first supported source is the final published #545 archive.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_RELATIVE = Path("docs/data/seed542-expanded-policy-strength-screen")
AUDIT_RELATIVE = Path("docs/data/seed564-archived-screen-identity-audit")
PUBLICATION_COMMIT = "56c5ae4e24ee73500f4f4ad027974cd186bba1ab"
CORRECTED_VERIFIER_SHA256 = (
    "7af938402a096fc9c81d747efc99f42a2d58abc8339b40e385710e0d3df43b6c"
)
CORRECTION_RESULTS_SHA256 = (
    "0beb84282868470b9d9a8e8a1ebcf913a879bcf88d70ccfa9cfdd164202ede9a"
)
REGISTRATION_SHA256 = "b39247e1d28c130da682df654cb7856a56985c79b0df7bb6297c9c65930e862e"
EXECUTION_FREEZE_SHA256 = (
    "e2d30aa8b1df6658bc3636b487b1bde4e7969e923557c19937a10f46c5a18311"
)
CONTINUATION_REGISTRATION_SHA256 = (
    "c749bc760019fddc3a53755c41b4ea1bf28e87251cfc44c9f0cb71ac1088cb6b"
)
CONTINUATION_FREEZE_SHA256 = (
    "9f44fdf3a9d4a5c87a29aa85aa4bb3677c1aa1ac7b8dd204affb61292440b4d4"
)
SUITE_SHA256 = "27a003322e5469507a508010a9743b40062928140e6a4b9d5628b0290e32c21d"
LEDGER_SHA256 = "e3c54882fec44163fe1d8166241883728a491372f7c98021d2faf75c52f15e13"
BASELINE_IDENTITIES = 147_146
OPENINGS = 1024
GAMES_PER_OPENING = 2


class ArchiveVerificationError(ValueError):
    """Raised when an archive binding or replay invariant fails."""


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    """Serialize state values using the production canonical JSON convention."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_identity(state: dict[str, Any]) -> str:
    """Return the canonical rule-state identity."""
    return sha256_bytes(canonical_json(state))


def _has_tactical_action(state: dict[str, Any], side: int, *, capture: bool) -> bool:
    game = KalahGame(
        pits=[int(value) for value in state["player_pits"]]
        + [int(value) for value in state["opponent_pits"]],
        captured_seeds=[int(state["player_store"]), int(state["opponent_store"])],
        current_player=side,
    )
    for relative_move in game.possible_moves():
        absolute_move = game.pit_index(relative_move)
        seeds = game.pits[absolute_move]
        store_before = game.captured_seeds[side]
        simulated = game.clone()
        simulated.move(absolute_move)
        if capture:
            distance_to_store = 6 - relative_move
            own_store_passes = (
                0
                if seeds < distance_to_store
                else 1 + ((seeds - distance_to_store) // 13)
            )
            if simulated.captured_seeds[side] - store_before > own_store_passes:
                return True
        elif simulated.current_player == side and not simulated.over():
            return True
    return False


def kalah_v3_identity(state: dict[str, Any]) -> str:
    """Return production `kalah_v3` identity without NumPy or model imports."""
    features: list[float] = []
    features.extend(float(value) / 48.0 for value in state["player_pits"])
    features.extend(float(value) / 48.0 for value in state["opponent_pits"])
    features.extend(
        (
            float(state["player_store"]) / 48.0,
            float(state["opponent_store"]) / 48.0,
            float(state["current_player"]),
        )
    )
    for side, pits_key in ((0, "player_pits"), (1, "opponent_pits")):
        pits = [int(value) for value in state[pits_key]]
        features.extend(
            (
                float(_has_tactical_action(state, side, capture=False)),
                float(_has_tactical_action(state, side, capture=True)),
                pits.count(0) / 6.0,
                sum(value >= 7 for value in pits) / 6.0,
                sum(value == 6 - index for index, value in enumerate(pits)) / 6.0,
                sum(pits) / 48.0,
            )
        )
    encoded = b"".join(struct.pack("<f", feature) for feature in features)
    return sha256_bytes(encoded)


def _state_from_opening(opening: dict[str, Any]) -> dict[str, Any]:
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    prefix = opening.get("prefix_moves")
    if not isinstance(prefix, list):
        raise ArchiveVerificationError("opening_prefix_missing")
    for ply, move in enumerate(prefix):
        if game.over() or int(move) not in game.possible_moves():
            raise ArchiveVerificationError(f"opening_prefix_illegal:{ply}")
        if not game.move(game.pit_index(int(move))):
            raise ArchiveVerificationError(f"opening_prefix_move_failed:{ply}")
    state = game.to_state()
    if state != opening.get("state"):
        raise ArchiveVerificationError("opening_prefix_state_mismatch")
    if canonical_identity(state) != opening.get("state_hash"):
        raise ArchiveVerificationError("opening_state_hash_mismatch")
    return state


def _state_game(state: dict[str, Any]) -> KalahGame:
    return KalahGame.from_state(state)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveVerificationError(f"invalid_jsonl:{path.name}") from exc


def _pin_manifest_path(root: Path) -> Path:
    return root / AUDIT_RELATIVE / "seed545-source-manifest.json"


def _snapshot(paths: set[Path]) -> dict[Path, tuple[int, int, str]]:
    """Capture evidence sizes, mtimes, and hashes for a read-only assertion."""
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path))
        for path in sorted(paths)
    }


def bootstrap_source_manifest(root: Path) -> dict[str, Any]:
    """Build the one-time static #545 manifest from the merged publication."""
    archive = root / ARCHIVE_RELATIVE
    pinned = {
        "POSTRUN-VERIFICATION-RESULTS.md": CORRECTION_RESULTS_SHA256,
        "registration.json": REGISTRATION_SHA256,
        "execution-freeze.json": EXECUTION_FREEZE_SHA256,
        "continuation-registration-v3.json": CONTINUATION_REGISTRATION_SHA256,
        "continuation-execution-freeze-v3.json": CONTINUATION_FREEZE_SHA256,
        "openings.jsonl": SUITE_SHA256,
        "outcomes.jsonl": LEDGER_SHA256,
        "opening0-recovery-binding.json": "8cc56b54db9b93cb0638fada4e93eceefdc38b1ba288915c65871a58a8d0dc15",
        "opening1-recovery-binding-v1.json": "04076cc725e721ceafa6e6430b14376764fe9a69b08b7ce017d997e39e0ba58c",
    }
    for relative, expected in pinned.items():
        path = archive / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ArchiveVerificationError(f"published_binding_mismatch:{relative}")
    if (
        sha256_file(
            root
            / "ml/alphazero_lite/verify_seed542_expanded_policy_strength_completeness.py"
        )
        != CORRECTED_VERIFIER_SHA256
    ):
        raise ArchiveVerificationError("published_corrected_verifier_mismatch")

    games = {
        f"batches/games-{index:04d}.jsonl": sha256_file(
            archive / f"batches/games-{index:04d}.jsonl"
        )
        for index in range(OPENINGS)
    }
    recovery = {
        relative: sha256_file(archive / relative)
        for relative in (
            "batches/completion-0000.json",
            "batches/completion-0001.json",
            "batches/report-0000.json",
            "batches/report-0001.json",
        )
    }
    return {
        "schema": "seed564_seed545_source_manifest_v1",
        "publication": {
            "pull_request": 545,
            "merge_commit": PUBLICATION_COMMIT,
            "final_corrected_verifier_sha256": CORRECTED_VERIFIER_SHA256,
            "final_correction_results_sha256": CORRECTION_RESULTS_SHA256,
        },
        "bindings": pinned,
        "recovery_receipts": recovery,
        "game_batches": games,
        "source_archive": str(ARCHIVE_RELATIVE),
        "prelaunch_failure": {
            "path": "PRELAUNCH-FAILURE-0000.json",
            "sha256": "03ed05abe358a823bf5a784e9665389a492772b5081cecb5b42b9d572d9d120c",
            "classification": "pre-game runtime asset contract failure; no committed game",
        },
    }


def _expected_manifest_hash(root: Path) -> str:
    """Read the fixed manifest digest embedded in the verifier module."""
    del root
    return "5664eb3ee58665de9370f105b2fd31d60448e5d0dd3174868ad7aead1f1a5c09"


def authenticate_source(
    root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check final publication bindings and authenticate all exact replay inputs."""
    if manifest.get("schema") != "seed564_seed545_source_manifest_v1":
        raise ArchiveVerificationError("source_manifest_schema_mismatch")
    archive = root / ARCHIVE_RELATIVE
    if manifest.get("publication", {}).get("merge_commit") != PUBLICATION_COMMIT:
        raise ArchiveVerificationError("publication_commit_mismatch")
    for relative, expected in manifest.get("bindings", {}).items():
        if sha256_file(archive / relative) != expected:
            raise ArchiveVerificationError(f"pinned_binding_mismatch:{relative}")
    if (
        sha256_file(
            root
            / "ml/alphazero_lite/verify_seed542_expanded_policy_strength_completeness.py"
        )
        != CORRECTED_VERIFIER_SHA256
    ):
        raise ArchiveVerificationError("corrected_verifier_source_mismatch")
    for relative, expected in manifest.get("recovery_receipts", {}).items():
        if sha256_file(archive / relative) != expected:
            raise ArchiveVerificationError(f"recovery_receipt_mismatch:{relative}")
    prelaunch = manifest["prelaunch_failure"]
    if sha256_file(archive / prelaunch["path"]) != prelaunch["sha256"]:
        raise ArchiveVerificationError("prelaunch_failure_evidence_mismatch")
    if len(manifest.get("game_batches", {})) != OPENINGS:
        raise ArchiveVerificationError("game_batch_manifest_coverage_mismatch")
    for relative, expected in manifest["game_batches"].items():
        path = archive / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ArchiveVerificationError(
                f"authenticated_game_bytes_mismatch:{relative}"
            )
    openings = _load_rows(archive / "openings.jsonl")
    ledger = _load_rows(archive / "outcomes.jsonl")
    if len(openings) != OPENINGS or len(ledger) != OPENINGS * GAMES_PER_OPENING:
        raise ArchiveVerificationError("source_population_size_mismatch")
    return openings, ledger


def _validate_result(game: KalahGame, row: dict[str, Any], seat: int) -> None:
    if not game.over():
        raise ArchiveVerificationError("trajectory_not_terminal")
    margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    if row.get("winner") != winner or int(row.get("margin", 10**9)) != margin:
        raise ArchiveVerificationError("trajectory_result_mismatch")
    if int(row.get("game_length", -1)) != int(row["_replayed_plies"]):
        raise ArchiveVerificationError("trajectory_length_mismatch")


def replay_game_trace(
    opening: dict[str, Any],
    row: dict[str, Any],
    seat: int,
    *,
    source: str,
    run: str,
    opening_index: int,
    game_position: int,
    batch_path: str,
) -> tuple[list[dict[str, Any]], int]:
    """Replay one absolute-action trace and return every post-action state."""
    root_state = _state_from_opening(opening)
    if (
        int(opening.get("opening_index", -1)) != opening_index
        or int(row.get("game_within_opening", -1)) != game_position
        or row.get("opening_prefix_moves") != opening.get("prefix_moves")
        or row.get("opening_state_hash") != opening.get("state_hash")
        or int(row.get("opening_applied_prefix_length", -1))
        != len(opening["prefix_moves"])
        or row.get("opening_contract") != opening.get("opening_contract")
    ):
        raise ArchiveVerificationError("game_opening_or_position_association_mismatch")
    if seat not in (0, 1):
        raise ArchiveVerificationError("invalid_game_seat")
    game = _state_game(root_state)
    provenance = {
        "source": source,
        "run": run,
        "opening_index": opening_index,
        "seat": seat,
        "game_position": game_position,
        "batch_path": batch_path,
    }
    occurrences = [
        {
            **provenance,
            "ply": -1,
            "action_absolute": None,
            "state": game.to_state(),
        }
    ]
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, str) or not trajectory:
        raise ArchiveVerificationError("trajectory_missing")
    moves = [int(token) for token in trajectory.split(",") if token]
    if len(moves) > 200 or len(moves) != int(row.get("game_length", -1)):
        raise ArchiveVerificationError("trajectory_length_or_cap")
    for ply, absolute_move in enumerate(moves):
        if (
            game.over()
            or not 0 <= absolute_move < 12
            or absolute_move // 6 != game.current_player
            or absolute_move - game.current_player * 6 not in game.possible_moves()
        ):
            raise ArchiveVerificationError(f"illegal_trajectory_move:{ply}")
        if not game.move(absolute_move):
            raise ArchiveVerificationError(f"trajectory_move_failed:{ply}")
        occurrences.append(
            {
                **provenance,
                "ply": ply,
                "action_absolute": absolute_move,
                "state": game.to_state(),
            }
        )
    result_row = dict(row)
    result_row["_replayed_plies"] = len(moves)
    _validate_result(game, result_row, seat)
    return occurrences, game.captured_seeds[seat] - game.captured_seeds[1 - seat]


def replay_source(
    root: Path, openings: list[dict[str, Any]], ledger: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay all opening pairs once and return deterministic state occurrences."""
    archive = root / ARCHIVE_RELATIVE
    ledger_map: dict[tuple[int, int], dict[str, Any]] = {}
    for record in ledger:
        key = (int(record["opening_index"]), int(record["game_position"]))
        if key in ledger_map or key[1] not in (0, 1):
            raise ArchiveVerificationError("duplicate_or_invalid_committed_game")
        ledger_map[key] = record
    occurrences: list[dict[str, Any]] = []
    verified_pairs = 0
    for opening_index, opening in enumerate(openings):
        if int(opening.get("opening_index", -1)) != opening_index:
            raise ArchiveVerificationError(f"opening_index_mismatch:{opening_index}")
        game_rows = _load_rows(archive / f"batches/games-{opening_index:04d}.jsonl")
        if len(game_rows) != GAMES_PER_OPENING:
            raise ArchiveVerificationError(f"pair_game_count_mismatch:{opening_index}")
        report_path = archive / f"batches/report-{opening_index:04d}.json"
        report_sha = sha256_file(report_path)
        report = json.loads(report_path.read_text())
        if int(report.get("games_played", -1)) != GAMES_PER_OPENING:
            raise ArchiveVerificationError(
                f"pair_report_game_count_mismatch:{opening_index}"
            )
        if opening_index in (0, 1):
            completion_path = archive / f"batches/completion-{opening_index:04d}.json"
            completion = json.loads(completion_path.read_text())
            if (
                int(completion.get("opening_index", -1)) != opening_index
                or completion.get("games_sha256")
                != sha256_file(archive / f"batches/games-{opening_index:04d}.jsonl")
                or completion.get("report_sha256") != report_sha
                or completion.get("registration_sha256") != REGISTRATION_SHA256
                or completion.get("suite_sha256") != SUITE_SHA256
            ):
                raise ArchiveVerificationError(
                    f"recovered_completion_receipt_mismatch:{opening_index}"
                )
        pair_wins = pair_losses = pair_draws = 0
        for position, row in enumerate(game_rows):
            record = ledger_map.get((opening_index, position))
            if record is None:
                raise ArchiveVerificationError("committed_game_missing")
            game_file = archive / f"batches/games-{opening_index:04d}.jsonl"
            if record.get("game_sha256") != sha256_file(game_file):
                raise ArchiveVerificationError(
                    f"ledger_game_digest_mismatch:{opening_index}"
                )
            if record.get("report_sha256") != report_sha:
                raise ArchiveVerificationError(
                    f"ledger_report_digest_mismatch:{opening_index}"
                )
            if (
                int(record["challenger_player"]) != int(row["challenger_player"])
                or int(record["game_position"]) != position
                or int(record["opening_index"]) != opening_index
            ):
                raise ArchiveVerificationError(
                    f"game_seat_ledger_mismatch:{opening_index}:{position}"
                )
            seat = int(row["challenger_player"])
            trace, margin = replay_game_trace(
                opening,
                row,
                seat,
                source="PR #545",
                run="seed542-expanded-policy-strength-screen",
                opening_index=opening_index,
                game_position=position,
                batch_path=f"batches/games-{opening_index:04d}.jsonl",
            )
            occurrences.extend(trace)
            if int(record["challenger_margin"]) != margin or record["winner"] != (
                "challenger" if margin > 0 else "opponent" if margin < 0 else "draw"
            ):
                raise ArchiveVerificationError(
                    f"ledger_result_mismatch:{opening_index}:{position}"
                )
            pair_wins += int(row["winner"] == "challenger")
            pair_losses += int(row["winner"] == "current")
            pair_draws += int(row["winner"] == "draw")
        if (
            int(report.get("wins", -1)) != pair_wins
            or int(report.get("losses", -1)) != pair_losses
            or int(report.get("draws", -1)) != pair_draws
        ):
            raise ArchiveVerificationError(
                f"pair_report_result_mismatch:{opening_index}"
            )
        verified_pairs += 1
    if len(ledger_map) != OPENINGS * GAMES_PER_OPENING or verified_pairs != OPENINGS:
        raise ArchiveVerificationError("committed_game_accounting_mismatch")
    return occurrences, {
        "openings": OPENINGS,
        "pairs": verified_pairs,
        "games": len(ledger_map),
    }


def identity_indexes(
    occurrences: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build both deterministic occurrence-provenance indexes."""
    canonical_rows: list[dict[str, Any]] = []
    float_rows: list[dict[str, Any]] = []
    opening_roots: dict[str, set[str]] = {"canonical": set(), "kalah_v3_float32": set()}
    continuation: dict[str, set[str]] = {"canonical": set(), "kalah_v3_float32": set()}
    for occurrence in occurrences:
        state = occurrence["state"]
        canonical = canonical_identity(state)
        float32 = kalah_v3_identity(state)
        provenance = {key: value for key, value in occurrence.items() if key != "state"}
        canonical_rows.append({"identity": canonical, **provenance})
        float_rows.append({"identity": float32, **provenance})
        target = opening_roots if occurrence["ply"] == -1 else continuation
        target["canonical"].add(canonical)
        target["kalah_v3_float32"].add(float32)
    key_order = (
        "source",
        "run",
        "opening_index",
        "seat",
        "game_position",
        "ply",
        "action_absolute",
        "identity",
    )
    canonical_rows.sort(
        key=lambda row: tuple((row.get(key) is None, row.get(key)) for key in key_order)
    )
    float_rows.sort(
        key=lambda row: tuple((row.get(key) is None, row.get(key)) for key in key_order)
    )
    counts = {
        "canonical": {
            "distinct": len({row["identity"] for row in canonical_rows}),
            "opening_root_distinct": len(opening_roots["canonical"]),
            "continuation_distinct": len(continuation["canonical"]),
            "root_continuation_overlap": len(
                opening_roots["canonical"] & continuation["canonical"]
            ),
        },
        "kalah_v3_float32": {
            "distinct": len({row["identity"] for row in float_rows}),
            "opening_root_distinct": len(opening_roots["kalah_v3_float32"]),
            "continuation_distinct": len(continuation["kalah_v3_float32"]),
            "root_continuation_overlap": len(
                opening_roots["kalah_v3_float32"] & continuation["kalah_v3_float32"]
            ),
        },
        "occurrences": len(occurrences),
    }
    return canonical_rows, float_rows, counts


def verify(root: Path, out: Path, *, write_manifest: bool = False) -> dict[str, Any]:
    """Public verification operation; source evidence is read-only."""
    root, out = root.resolve(), out.resolve()
    manifest_path = _pin_manifest_path(root)
    if write_manifest:
        manifest = bootstrap_source_manifest(root)
        out.mkdir(parents=True, exist_ok=True)
        (out / manifest_path.name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return {
            "status": "source_manifest_bootstrapped",
            "manifest_sha256": sha256_file(out / manifest_path.name),
        }
    if not manifest_path.is_file():
        raise ArchiveVerificationError("pinned_source_manifest_missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    expected_sha = _expected_manifest_hash(root)
    if manifest_sha != expected_sha:
        raise ArchiveVerificationError("pinned_source_manifest_sha256_mismatch")
    manifest = json.loads(manifest_bytes)
    archive = root / ARCHIVE_RELATIVE
    evidence_paths = {
        manifest_path,
        root
        / "ml/alphazero_lite/verify_seed542_expanded_policy_strength_completeness.py",
        *(archive / relative for relative in manifest["bindings"]),
        *(archive / relative for relative in manifest["recovery_receipts"]),
        *(archive / relative for relative in manifest["game_batches"]),
        *(archive / f"batches/report-{index:04d}.json" for index in range(OPENINGS)),
        archive / manifest["prelaunch_failure"]["path"],
    }
    before = _snapshot(evidence_paths)
    openings, ledger = authenticate_source(root, manifest)
    occurrences, replay = replay_source(root, openings, ledger)
    canonical_rows, float_rows, counts = identity_indexes(occurrences)
    unchanged = before == _snapshot(evidence_paths)
    if not unchanged:
        raise ArchiveVerificationError("verifier_changed_evidence_bytes_or_mtimes")
    out.mkdir(parents=True, exist_ok=True)
    canonical_bytes = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in canonical_rows
    ).encode("utf-8")
    float_bytes = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in float_rows
    ).encode("utf-8")
    (out / "seed545-canonical-index.jsonl.gz").write_bytes(
        gzip.compress(canonical_bytes, mtime=0)
    )
    (out / "seed545-kalah-v3-float32-index.jsonl.gz").write_bytes(
        gzip.compress(float_bytes, mtime=0)
    )
    report = {
        "schema": "seed564_seed545_replay_report_v1",
        "publication_commit": PUBLICATION_COMMIT,
        "corrected_publication_verifier_sha256": CORRECTED_VERIFIER_SHA256,
        "recorded_runtime_reproduction": "unverified",
        "different_runtime_cross_check": "supporting_evidence_only",
        "archive_authenticated": True,
        "archive_replayed": True,
        "evidence_bytes_and_mtimes_unchanged": unchanged,
        "prelaunch_failure_separate_from_committed_games": manifest[
            "prelaunch_failure"
        ],
        "recovered_openings_counted_once": [0, 1],
        "replay": replay,
        "identities": counts,
        "tablebase_required": False,
    }
    (out / "seed545-replay-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / AUDIT_RELATIVE)
    parser.add_argument("--write-source-manifest", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.root, args.out, write_manifest=args.write_source_manifest)
    except (ArchiveVerificationError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"verification failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
