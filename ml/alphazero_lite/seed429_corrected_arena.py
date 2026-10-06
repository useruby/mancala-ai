"""Separately bound arena launcher for the frozen seed429 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from ml.alphazero_lite import arena
from ml.alphazero_lite import run_seed429_canonical_policy_normalization as frozen

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed429-canonical-policy-normalization"
WORK = ROOT / ".tmp/seed429-canonical-policy-normalization"
CHUNK_SIZE = 32
LANES = ("A", "B")


def digest(path: Path) -> str:
    return frozen.sha256_file(path)


def deterministic_output_paths(lane: str, start_index: int) -> tuple[Path, Path]:
    """Return unique machine-report and outcome chunk paths for a game-index range."""
    directory = WORK / "arena-chunks" / lane
    stem = f"chunk-{start_index:04d}"
    return directory / f"{stem}.json", directory / f"{stem}-report.json"


def construct_parser_args(
    lane: str,
    start_index: int,
    count: int,
    candidate: Path,
    opponent: Path,
    suite_path: Path,
    report_path: Path,
) -> argparse.Namespace:
    """Parse the unchanged arena invocation, including its required --out path."""
    argv = [
        "--challenger",
        str(candidate),
        "--current",
        str(opponent),
        "--games",
        str(count),
        "--games-per-opening",
        "2",
        "--opening-prefixes-jsonl",
        str(suite_path),
        "--suite-sha256",
        digest(suite_path),
        "--challenger-simulations",
        "384",
        "--current-simulations",
        "384",
        "--seed",
        "429",
        "--workers",
        "1",
        "--c-puct",
        "1.25",
        "--seed-contract",
        "azlite_eval_seed_v2",
        "--out",
        str(report_path),
    ]
    saved_argv = sys.argv
    try:
        sys.argv = ["arena.py", *argv]
        parsed = arena.parse_args()
    finally:
        sys.argv = saved_argv
    if parsed.out != str(report_path):
        raise ValueError(
            f"seed429_corrected_out_path_parse_mismatch:{lane}:{start_index}"
        )
    return parsed


def verify_correction_bindings(
    receipt: dict[str, Any],
    *,
    registration_sha256: str,
    runtime_binding_sha256: str,
    suite_sha256: str,
    source_hashes: dict[str, str],
) -> None:
    if receipt.get("registration_sha256") != registration_sha256:
        raise ValueError("seed429_correction_registration_mismatch")
    if receipt.get("runtime_binding_sha256") != runtime_binding_sha256:
        raise ValueError("seed429_correction_runtime_binding_mismatch")
    if receipt.get("suite_sha256") != suite_sha256:
        raise ValueError("seed429_correction_suite_mismatch")
    if receipt.get("corrected_source_hashes") != source_hashes:
        raise ValueError("seed429_correction_source_binding_mismatch")
    if receipt.get("games_completed_before_correction") != 0:
        raise ValueError("seed429_correction_not_prospective_to_arena")


def _read_index(lane: str, identities: dict[str, str]) -> dict[str, Any]:
    path = WORK / "arena-chunks" / lane / "correction-resume-index.json"
    if path.exists():
        index = json.loads(path.read_text(encoding="utf-8"))
        if index.get("identities") != identities:
            raise ValueError(f"seed429_correction_resume_binding_mismatch:{lane}")
        return index
    return {"identities": identities, "chunks": {}}


def _write_index(lane: str, index: dict[str, Any]) -> None:
    path = WORK / "arena-chunks" / lane / "correction-resume-index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.staging")
    staging.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, path)


def _worker_result(
    lane: str,
    start: int,
    count: int,
    args: argparse.Namespace,
    runtime_contract: dict[str, Any],
    suite_sha256: str,
) -> dict[str, Any]:
    options = arena.build_eval_search_options(**arena.search_options_from_args(args))
    challenger_options = arena.parse_search_options_override(
        args.challenger_search_options_json, base=options
    )
    current_options = arena.parse_search_options_override(
        args.current_search_options_json, base=options
    )
    return arena.run_arena_worker(
        worker_id=start // CHUNK_SIZE,
        start_index=start,
        games=count,
        challenger_path=args.challenger,
        current_path=args.current,
        challenger_simulations=384,
        current_simulations=384,
        seed=429,
        c_puct=1.25,
        max_moves=args.max_moves,
        fpu_mode=str(options["fpu_mode"]),
        reuse_subtree=bool(options["reuse_subtree"]),
        normalize_values=bool(options["normalize_values"]),
        root_policy_mode=str(options["root_policy_mode"]),
        tactical_root_bias=float(options["tactical_root_bias"]),
        root_temperature=float(options.get("root_temperature", 0.0)),
        challenger_search_options=challenger_options,
        current_search_options=current_options,
        exact_root_solve_threshold=runtime_contract["exact_root_solve_threshold"],
        exact_root_native_probe=Path(runtime_contract["exact_root_native_probe"]),
        exact_root_tablebase_path=Path(runtime_contract["exact_root_tablebase"]),
        runtime_search_contract=runtime_contract,
        opening_prefixes_jsonl=args.opening_prefixes_jsonl,
        games_per_opening=2,
        seed_contract="azlite_eval_seed_v2",
        suite_sha256_override=suite_sha256,
    )


def _load_chunk(
    path: Path,
    *,
    expected: dict[str, Any],
    source_hash: str,
) -> dict[str, Any]:
    if not path.is_file() or digest(path) != source_hash:
        raise ValueError("seed429_corrected_chunk_hash_mismatch")
    chunk = json.loads(path.read_text(encoding="utf-8"))
    if any(chunk.get(key) != value for key, value in expected.items()):
        raise ValueError("seed429_corrected_chunk_identity_mismatch")
    frozen._validate_chunk_rows(
        chunk["worker_result"]["game_entries"], expected["start"], expected["count"]
    )
    return chunk


def _execute_lane(
    lane: str,
    binding: dict[str, Any],
    receipt: dict[str, Any],
    correction_source_hashes: dict[str, str],
) -> tuple[Path, Path, Path, Path, Path]:
    registration_hash = digest(DATA / "registration.json")
    runtime_hash = digest(DATA / "runtime-binding.json")
    suite_path = DATA / "openings.jsonl"
    suite_hash = digest(suite_path)
    verify_correction_bindings(
        receipt,
        registration_sha256=registration_hash,
        runtime_binding_sha256=runtime_hash,
        suite_sha256=suite_hash,
        source_hashes=correction_source_hashes,
    )
    identities = {
        "registration_sha256": registration_hash,
        "runtime_binding_sha256": runtime_hash,
        "suite_sha256": suite_hash,
        "correction_receipt_sha256": digest(DATA / "arena-correction-receipt.json"),
        "correction_source_hashes_sha256": hashlib.sha256(
            json.dumps(correction_source_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "seed": 429,
        "seed_contract": "azlite_eval_seed_v2",
        "challenger_simulations": 384,
        "current_simulations": 384,
        "c_puct": 1.25,
        "chunk_size": CHUNK_SIZE,
    }
    index = _read_index(lane, identities)
    candidate = ROOT / binding["candidates"][lane]["artifact"]
    opponent = ROOT / binding["opponent"]
    local_opponent = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
    local_candidate = (
        ROOT / ".tmp/seed429-canonical-policy-normalization/artifacts" / lane
    )
    runtime_contract = frozen.resolve_strength_comparison_runtime_contract(
        current_artifact=local_opponent,
        challenger_artifact=local_candidate,
        native_probe=ROOT / "model-artifact/runtime/kalah_v1_tablebase",
        tablebase=ROOT / "model-artifact/runtime/kalah_v1_21.kvtb",
    )
    if runtime_contract != binding["candidates"][lane]["runtime_contract"]:
        raise ValueError(f"seed429_correction_runtime_contract_mismatch:{lane}")
    for start in range(0, 1024, CHUNK_SIZE):
        count = min(CHUNK_SIZE, 1024 - start)
        key = str(start)
        chunk_path, output_path = deterministic_output_paths(lane, start)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        expected = {"lane": lane, "start": start, "count": count, **identities}
        prior = index["chunks"].get(key)
        if prior is not None:
            chunk = _load_chunk(
                chunk_path, expected=expected, source_hash=prior["sha256"]
            )
            output = chunk.get("output_sha256")
            if not output_path.is_file() or digest(output_path) != output:
                raise ValueError("seed429_corrected_chunk_report_altered")
            continue
        if chunk_path.exists():
            # Recover the atomic chunk if interruption occurred before the resume
            # index was advanced. The report is a deterministic derivative of it.
            orphan = json.loads(chunk_path.read_text(encoding="utf-8"))
            if any(orphan.get(key) != value for key, value in expected.items()):
                raise ValueError("seed429_corrected_orphan_chunk_identity_mismatch")
            frozen._validate_chunk_rows(
                orphan["worker_result"]["game_entries"], start, count
            )
            report = {
                "lane": lane,
                "start_index": start,
                "game_count": count,
                "games_played": len(orphan["worker_result"]["game_entries"]),
                "seed_contract": "azlite_eval_seed_v2",
                "base_seed": 429,
                "suite_sha256": suite_hash,
                "search_profile": orphan["worker_result"]["search_profile"],
            }
            output_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            index["chunks"][key] = {
                "sha256": digest(chunk_path),
                "output_sha256": digest(output_path),
                "game_count": count,
            }
            _write_index(lane, index)
            continue
        if output_path.exists():
            # No outcomes were committed, so this report cannot authenticate a
            # completed chunk; discard it and regenerate the deterministic range.
            output_path.unlink()
        parsed = construct_parser_args(
            lane,
            start,
            count,
            candidate,
            opponent,
            suite_path,
            output_path,
        )
        worker = _worker_result(
            lane, start, count, parsed, runtime_contract, suite_hash
        )
        frozen._validate_chunk_rows(worker["game_entries"], start, count)
        chunk = {
            **expected,
            "output_path": str(output_path.relative_to(ROOT)),
            "worker_result": worker,
        }
        chunk_staging = chunk_path.with_suffix(".json.staging")
        chunk_staging.write_text(
            json.dumps(chunk, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        report = {
            "lane": lane,
            "start_index": start,
            "game_count": count,
            "games_played": len(worker["game_entries"]),
            "seed_contract": "azlite_eval_seed_v2",
            "base_seed": 429,
            "suite_sha256": suite_hash,
            "search_profile": worker["search_profile"],
        }
        output_staging = output_path.with_suffix(".json.staging")
        output_staging.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(chunk_staging, chunk_path)
        os.replace(output_staging, output_path)
        index["chunks"][key] = {
            "sha256": digest(chunk_path),
            "output_sha256": digest(output_path),
            "game_count": count,
        }
        _write_index(lane, index)
    chunks = [
        _load_chunk(
            deterministic_output_paths(lane, start)[0],
            expected={
                "lane": lane,
                "start": start,
                "count": min(CHUNK_SIZE, 1024 - start),
                **identities,
            },
            source_hash=index["chunks"][str(start)]["sha256"],
        )
        for start in range(0, 1024, CHUNK_SIZE)
    ]
    return _publish_lane(lane, chunks, suite_hash)


def _publish_lane(
    lane: str, chunks: list[dict[str, Any]], suite_hash: str
) -> tuple[Path, Path, Path, Path, Path]:
    work = WORK
    results = [chunk["worker_result"] for chunk in chunks]
    games = [row for result in results for row in result["game_entries"]]
    games.sort(key=lambda row: int(row["game_index"]))
    if len(games) != 1024 or len({int(row["game_index"]) for row in games}) != 1024:
        raise ValueError("seed429_corrected_lane_games_incomplete")
    game_path = work / f"{lane}-games.jsonl"
    seed_path = work / f"{lane}-seed-identities.jsonl"
    config_path = work / f"{lane}-search-configurations.jsonl"
    search_path = work / f"{lane}-search-outcomes.jsonl"
    game_path.write_text(
        "".join(json.dumps(row) + "\n" for row in games), encoding="utf-8"
    )
    for path, key in (
        (seed_path, "seed_identity_ledger"),
        (config_path, "search_configuration_ledger"),
        (search_path, "search_outcome_ledger"),
    ):
        rows = [row for result in results for row in result[key]]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    wins = sum(row["winner"] == "challenger" for row in games)
    losses = sum(row["winner"] == "current" for row in games)
    draws = sum(row["winner"] == "draw" for row in games)
    report = {
        "games_played": 1024,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": (wins + 0.5 * draws) / 1024,
        "notes": {
            "base_seed": 429,
            "seed_contract": "azlite_eval_seed_v2",
            "suite_sha256": suite_hash,
            "search_profile": results[0]["search_profile"],
        },
    }
    report_path = work / f"{lane}-arena.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path, game_path, seed_path, config_path, search_path


def execute() -> dict[str, Any]:
    """Run the fixed arena only after receipt and both runtime bindings validate."""
    frozen.execution_readiness(require_candidates=True)
    binding_path = DATA / "runtime-binding.json"
    receipt_path = DATA / "arena-correction-receipt.json"
    if not receipt_path.is_file():
        raise ValueError("seed429_arena_correction_receipt_missing")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    correction_hashes = {
        relative: digest(ROOT / relative)
        for relative in receipt["corrected_source_hashes"]
    }
    verify_correction_bindings(
        receipt,
        registration_sha256=digest(DATA / "registration.json"),
        runtime_binding_sha256=digest(binding_path),
        suite_sha256=digest(DATA / "openings.jsonl"),
        source_hashes=correction_hashes,
    )
    outcome_path = DATA / "outcome-binding.json"
    previous = (
        json.loads(outcome_path.read_text())
        if outcome_path.is_file()
        else {
            "registration_sha256": digest(DATA / "registration.json"),
            "runtime_binding_sha256": digest(binding_path),
            "correction_receipt_sha256": digest(receipt_path),
            "lanes": {},
        }
    )
    if (
        previous["registration_sha256"] != digest(DATA / "registration.json")
        or previous["runtime_binding_sha256"] != digest(binding_path)
        or previous["correction_receipt_sha256"] != digest(receipt_path)
    ):
        raise ValueError("seed429_corrected_execution_resume_binding_mismatch")
    lanes = dict(previous["lanes"])
    for lane in LANES:
        existing = lanes.get(lane)
        if existing:
            for key, hash_key in (
                ("games_path", "games_sha256"),
                ("report_path", "report_sha256"),
            ):
                path = ROOT / existing[key]
                if digest(path) != existing[hash_key]:
                    raise ValueError("seed429_completed_lane_outcome_altered")
        report, games, seed, config, search = _execute_lane(
            lane, binding, receipt, correction_hashes
        )
        validation = frozen._validate_lane_arena(lane, report, games)
        evidence_paths = {
            "games_path": games,
            "report_path": report,
            "seed_identity_ledger_path": seed,
            "search_configuration_ledger_path": config,
            "search_outcome_ledger_path": search,
        }
        lane_publication = DATA / "arena-evidence" / lane
        lane_publication.mkdir(parents=True, exist_ok=True)
        for field, source in evidence_paths.items():
            destination = lane_publication / source.name
            if not destination.exists():
                import shutil

                shutil.copy2(source, destination)
            if digest(destination) != digest(source):
                raise ValueError("seed429_corrected_publication_copy_mismatch")
            validation[field] = str(destination.relative_to(ROOT))
            if field.endswith("_ledger_path"):
                validation[field.replace("_path", "_sha256")] = digest(destination)
        if existing is not None and existing != validation:
            raise ValueError("seed429_corrected_completed_lane_conflict")
        lanes[lane] = validation
        frozen.write_progress(
            outcome_path,
            {
                "schema": "seed429-outcome-binding-v1",
                "status": "in_progress",
                "registration_sha256": digest(DATA / "registration.json"),
                "runtime_binding_sha256": digest(binding_path),
                "correction_receipt_sha256": digest(receipt_path),
                "lanes": lanes,
            },
        )
    if any(lanes[lane]["game_count"] != 1024 for lane in LANES):
        raise ValueError("seed429_corrected_final_game_count_invalid")
    final = {
        "schema": "seed429-outcome-binding-v1",
        "status": "completed_2048_games",
        "registration_sha256": digest(DATA / "registration.json"),
        "runtime_binding_sha256": digest(binding_path),
        "correction_receipt_sha256": digest(receipt_path),
        "lanes": lanes,
    }
    frozen.write_progress(outcome_path, final)
    analysis = frozen.analyze()
    diagnostics = frozen.retrospective_diagnostics()
    return {"outcomes": final, "analysis": analysis, "diagnostics": diagnostics}


def register_correction() -> dict[str, Any]:
    """Append an immutable correction receipt; must complete before arena launch."""
    frozen.execution_readiness(require_candidates=True)
    receipt_path = DATA / "arena-correction-receipt.json"
    if receipt_path.exists():
        raise ValueError("seed429_correction_receipt_is_append_only")
    if (DATA / "outcome-binding.json").exists():
        raise ValueError("seed429_correction_not_before_arena_progress")
    chunk_root = WORK / "arena-chunks"
    if chunk_root.exists() and any(path.is_file() for path in chunk_root.rglob("*")):
        raise ValueError("seed429_correction_not_before_arena_chunks")
    registration_path = DATA / "registration.json"
    runtime_path = DATA / "runtime-binding.json"
    training_path = DATA / "training-results.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    runtime_binding = json.loads(runtime_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    for lane in ("A", "B"):
        lane_result = training["lanes"][lane]
        for epoch, expected in lane_result["epoch_hashes"].items():
            checkpoint = ROOT / lane_result["checkpoint_dir"] / f"E{epoch}.npz"
            if digest(checkpoint) != expected:
                raise ValueError(
                    f"seed429_correction_checkpoint_changed:{lane}:E{epoch}"
                )
        for filename, expected in runtime_binding["candidates"][lane][
            "artifact_files"
        ].items():
            artifact_file = (
                ROOT / runtime_binding["candidates"][lane]["artifact"] / filename
            )
            if digest(artifact_file) != expected:
                raise ValueError(
                    f"seed429_correction_candidate_changed:{lane}:{filename}"
                )
    corrected_sources = (
        "ml/alphazero_lite/seed429_corrected_arena.py",
        "ml/alphazero_lite/verify_seed429_correction.py",
    )
    snapshots_dir = DATA / "correction-source-snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    snapshot_paths: dict[str, str] = {}
    for relative in corrected_sources:
        source = ROOT / relative
        snapshot = snapshots_dir / source.name
        payload = source.read_bytes()
        if snapshot.exists() and snapshot.read_bytes() != payload:
            if receipt_path.exists():
                raise ValueError(f"seed429_correction_snapshot_conflict:{relative}")
        if not snapshot.exists() or snapshot.read_bytes() != payload:
            snapshot.write_bytes(payload)
        hashes[relative] = digest(source)
        snapshot_paths[relative] = str(snapshot.relative_to(ROOT))
    correction = {
        "schema": "seed429-pre-arena-output-launch-correction-v1",
        "status": "registered_before_any_arena_game",
        "timing": "after A/B training and runtime binding; before arena games",
        "games_completed_before_correction": 0,
        "failure": {
            "frozen_runner": "ml/alphazero_lite/run_seed429_canonical_policy_normalization.py",
            "failure": "arena.parse_args() rejected constructed argv because --out is required",
            "parser_evidence": "usage: arena.py ... --out OUT; error: the following arguments are required: --out",
            "games_or_chunks_completed": 0,
        },
        "registration_sha256": digest(registration_path),
        "original_execution_source_snapshots": registration[
            "execution_source_snapshots"
        ],
        "runtime_binding_sha256": digest(runtime_path),
        "suite_sha256": digest(DATA / "openings.jsonl"),
        "training_results_sha256": digest(training_path),
        "training_lanes": {
            lane: {
                "epoch_sha256": training["lanes"][lane]["epoch_hashes"],
                "optimizer_updates": training["lanes"][lane]["optimizer_updates"],
            }
            for lane in LANES
        },
        "A_E4_sha256": training["lanes"]["A"]["epoch_hashes"]["4"],
        "candidate_artifacts": {
            lane: runtime_binding["candidates"][lane] for lane in LANES
        },
        "frozen_opponent_files": runtime_binding["opponent_files"],
        "corrected_source_hashes": hashes,
        "corrected_source_snapshots": snapshot_paths,
        "corrected_launcher": {
            "register_correction_command": "python -m ml.alphazero_lite.seed429_corrected_arena register-correction",
            "readiness_command": "python -m ml.alphazero_lite.seed429_corrected_arena readiness",
            "command": "python -m ml.alphazero_lite.seed429_corrected_arena execute",
            "verification_command": "python -m ml.alphazero_lite.verify_seed429_correction --require-complete",
            "parser_contract": "construct args through arena.parse_args with required --out set",
            "output_rule": ".tmp/seed429-canonical-policy-normalization/arena-chunks/{lane}/chunk-{start:04d}-report.json; start is the global game index of each registered 32-game chunk",
            "outcome_rule": "the parser output file contains chunk report metadata; game and search ledgers are atomically bound in the sibling chunk JSON and final arena-evidence files",
        },
        "protocol_identity": registration["evaluation"],
        "protocol_unchanged": True,
        "changed_behavior": "only required parser output path construction and output-path recording; run_arena_worker receives the same candidate/opponent, game indexes, suite, search options, seeds, budgets, runtime contract, and game count",
    }
    verify_correction_bindings(
        correction,
        registration_sha256=digest(registration_path),
        runtime_binding_sha256=digest(runtime_path),
        suite_sha256=digest(DATA / "openings.jsonl"),
        source_hashes=hashes,
    )
    staging = receipt_path.with_suffix(".json.staging")
    staging.write_text(
        json.dumps(correction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, receipt_path)
    return correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("register-correction", "readiness", "execute")
    )
    args = parser.parse_args()
    if args.action == "register-correction":
        print(json.dumps(register_correction(), indent=2, sort_keys=True))
    elif args.action == "readiness":
        receipt = json.loads((DATA / "arena-correction-receipt.json").read_text())
        frozen.execution_readiness(require_candidates=True)
        verify_correction_bindings(
            receipt,
            registration_sha256=digest(DATA / "registration.json"),
            runtime_binding_sha256=digest(DATA / "runtime-binding.json"),
            suite_sha256=digest(DATA / "openings.jsonl"),
            source_hashes={
                relative: digest(ROOT / relative)
                for relative in receipt["corrected_source_hashes"]
            },
        )
        print(json.dumps({"ready": True, "games_started": 0}, sort_keys=True))
    else:
        print(json.dumps(execute(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
