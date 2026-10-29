"""Seed542 continuation source accounting and traceable partial extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_complete_games(
    staging_path: Path,
    output_path: Path,
    provenance_path: Path,
    *,
    source_path_label: str | None = None,
) -> dict[str, Any]:
    """Copy only complete games verbatim, recording original row provenance."""
    original = staging_path.read_bytes()
    source_hash = sha256(original)
    lines = original.splitlines(keepends=True)
    rows = [json.loads(line) for line in original.splitlines()]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["game_index"]), []).append(row)
    if sorted(grouped) != list(range(len(grouped))) or len(grouped) > 64:
        raise ValueError("interrupted_shard_game_indices_invalid")
    completed: list[int] = []
    capped: list[int] = []
    incomplete: list[int] = []
    for game_index, game_rows in sorted(grouped.items()):
        ordered = sorted(game_rows, key=lambda row: int(row["move_index"]))
        expected = len(ordered[0].get("game_moves", []))
        if any(row.get("game_capped") is True for row in ordered):
            capped.append(game_index)
        elif (
            len(ordered) == expected
            and [int(row["move_index"]) for row in ordered] == list(range(expected))
            and all(row.get("game_completed") is True for row in ordered)
        ):
            completed.append(game_index)
        else:
            incomplete.append(game_index)
    completion = {
        "completed_indices": completed,
        "capped_indices": capped,
        "incomplete_indices": incomplete,
        "unstarted_indices": list(range(len(grouped), 64)),
    }
    complete = set(completion["completed_indices"])
    selected: list[tuple[int, bytes, dict[str, Any]]] = []
    for line_number, (line, row) in enumerate(zip(lines, rows, strict=True), 1):
        if int(row["game_index"]) in complete:
            selected.append((line_number, line, row))
    if len(complete) != 51:
        raise ValueError(f"recovered_complete_game_count_mismatch:{len(complete)}")
    if completion["incomplete_indices"] != [51] or completion[
        "unstarted_indices"
    ] != list(range(52, 64)):
        raise ValueError("interrupted_shard_accounting_changed")
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError("refusing_to_replace_seed542_recovered_derivative")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    derivative = b"".join(line for _, line, _ in selected)
    output_path.write_bytes(derivative)
    derivative_hash = sha256(derivative)
    provenance = [
        {
            "derivative_line": index,
            "original_path": source_path_label or staging_path.as_posix(),
            "original_line": original_line,
            "source_sha256": source_hash,
            "source_line_sha256": sha256(line.rstrip(b"\r\n")),
            "derivative_line_sha256": sha256(line.rstrip(b"\r\n")),
        }
        for index, (original_line, line, _) in enumerate(selected, 1)
    ]
    provenance_path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in provenance
        ),
        encoding="utf-8",
    )
    reread = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(reread) != len(selected) or any(
        reread[index - 1] != row for index, (_, _, row) in enumerate(selected, 1)
    ):
        raise ValueError("recovered_derivative_row_mismatch")
    return {
        "source_path": staging_path.as_posix(),
        "source_sha256": source_hash,
        "source_bytes": len(original),
        "complete_games": len(complete),
        "excluded_incomplete_game_indices": completion["incomplete_indices"],
        "excluded_unstarted_game_indices": completion["unstarted_indices"],
        "derivative_path": output_path.as_posix(),
        "derivative_sha256": derivative_hash,
        "derivative_rows": len(selected),
        "provenance_path": provenance_path.as_posix(),
        "provenance_sha256": sha256(provenance_path.read_bytes()),
    }


def validate_source_manifest(manifest: dict[str, Any]) -> None:
    """Validate amended source population counts and roles."""
    sources = manifest["sources"]
    training = [s for s in sources if s["role"] == "training"]
    development = [s for s in sources if s["role"] == "development"]
    training_games = sum(int(source["games"]) for source in training)
    development_games = sum(int(source["games"]) for source in development)
    if (len(training), training_games, len(development), development_games) != (
        49,
        3072,
        16,
        1024,
    ):
        raise ValueError(
            "amended_source_totals_mismatch:"
            f"{len(training)}:{training_games}:{len(development)}:{development_games}"
        )
    if sum(s["games"] == 64 for s in training) != 47:
        raise ValueError("amended_full_training_shard_count_mismatch")
    if sum(s["games"] == 51 for s in training) != 1:
        raise ValueError("amended_recovered_training_source_count_mismatch")
    if sum(s["games"] == 13 for s in training) != 1:
        raise ValueError("amended_supplement_training_source_count_mismatch")
    if any(s["games"] != 64 for s in development):
        raise ValueError("amended_development_source_game_count_mismatch")
    expected_training = {
        *((seed, 64) for seed in range(542000, 542048) if seed != 542001),
        (542001, 51),
        (542064, 13),
    }
    actual_training = {
        (int(source["source_seed"]), int(source["games"])) for source in training
    }
    expected_development = {(seed, 64) for seed in range(542048, 542064)}
    actual_development = {
        (int(source["source_seed"]), int(source["games"])) for source in development
    }
    if actual_training != expected_training:
        raise ValueError("amended_training_source_assignment_mismatch")
    if actual_development != expected_development:
        raise ValueError("amended_development_source_assignment_mismatch")
    paths = [s["path"] for s in sources]
    if len(paths) != len(set(paths)):
        raise ValueError("amended_source_path_duplicate")
    if any(len(str(s["sha256"])) != 64 for s in sources):
        raise ValueError("amended_source_hash_invalid")
    recovered = next(source for source in training if source["source_seed"] == 542001)
    if not all(
        recovered.get(key)
        for key in (
            "original_source_path",
            "original_source_sha256",
            "provenance_path",
            "provenance_sha256",
        )
    ):
        raise ValueError("amended_recovered_source_provenance_missing")


def source_paths(root: Path, manifest: dict[str, Any], role: str) -> list[Path]:
    """Return authenticated sources for one partition, retaining file multiplicity."""
    validate_source_manifest(manifest)
    result = []
    for source in manifest["sources"]:
        if source["role"] != role:
            continue
        path = root / source["path"]
        if not path.is_file():
            raise FileNotFoundError(f"amended_source_missing:{source['path']}")
        actual = sha256(path.read_bytes())
        if actual != source["sha256"]:
            raise ValueError(f"amended_source_sha256_mismatch:{source['path']}")
        result.append(path)
    return result


def build_source_manifest(root: Path, corpus: Path) -> dict[str, Any]:
    """Build the explicit nonuniform source manifest after generation."""
    training_paths = [
        corpus / f"shard-{seed}.jsonl"
        for seed in range(542000, 542048)
        if seed != 542001
    ]
    training_paths.extend(
        [
            corpus / "recovered-seed542001-complete.jsonl",
            corpus / "shard-542064.jsonl",
        ]
    )
    development_paths = [
        corpus / f"shard-{seed}.jsonl" for seed in range(542048, 542064)
    ]
    sources = []
    for role, paths in (
        ("training", training_paths),
        ("development", development_paths),
    ):
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"amended_source_missing:{path}")
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            game_ids = {int(row["game_index"]) for row in rows}
            expected = (
                51
                if path.name == "recovered-seed542001-complete.jsonl"
                else 13
                if path.name == "shard-542064.jsonl"
                else 64
            )
            if len(game_ids) != expected:
                raise ValueError(
                    f"amended_source_game_count_mismatch:{path}:{len(game_ids)}:{expected}"
                )
            if game_ids != set(range(expected)):
                raise ValueError(f"amended_source_game_indices_mismatch:{path}")
            seed = (
                542001
                if path.name == "recovered-seed542001-complete.jsonl"
                else 542064
                if path.name == "shard-542064.jsonl"
                else int(path.stem.removeprefix("shard-"))
            )
            if {int(row["game_seed"]) for row in rows} != {seed}:
                raise ValueError(f"amended_source_game_seed_mismatch:{path}")
            if any(
                row.get("game_completed") is not True or row.get("game_capped") is True
                for row in rows
            ):
                raise ValueError(f"amended_source_contains_noncomplete_game:{path}")
            source_record = {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path.read_bytes()),
                "role": role,
                "games": expected,
                "source_seed": seed,
            }
            if path.name == "recovered-seed542001-complete.jsonl":
                provenance_path = corpus / "recovered-seed542001-provenance.jsonl"
                provenance_rows = [
                    json.loads(line)
                    for line in provenance_path.read_text(encoding="utf-8").splitlines()
                ]
                source_record.update(
                    {
                        "original_source_path": provenance_rows[0]["original_path"],
                        "original_source_sha256": provenance_rows[0]["source_sha256"],
                        "provenance_path": provenance_path.relative_to(root).as_posix(),
                        "provenance_sha256": sha256(provenance_path.read_bytes()),
                    }
                )
            sources.append(source_record)
    manifest: dict[str, Any] = {
        "status": "authenticated source manifest",
        "sources": sources,
    }
    validate_source_manifest(manifest)
    return manifest


def verify_extraction(
    staging_path: Path,
    derivative_path: Path,
    provenance_path: Path,
    *,
    source_path_label: str | None = None,
) -> dict[str, Any]:
    """Independently reconcile derivative bytes, rows, and original line bindings."""
    source_bytes = staging_path.read_bytes()
    source_hash = sha256(source_bytes)
    source_lines = source_bytes.splitlines()
    source_rows = [json.loads(line) for line in source_lines]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in source_rows:
        grouped.setdefault(int(row["game_index"]), []).append(row)
    complete_indices = []
    for index, game_rows in sorted(grouped.items()):
        ordered = sorted(game_rows, key=lambda row: int(row["move_index"]))
        expected = len(ordered[0].get("game_moves", []))
        if (
            len(ordered) == expected
            and [int(row["move_index"]) for row in ordered] == list(range(expected))
            and all(row.get("game_completed") is True for row in ordered)
            and not any(row.get("game_capped") is True for row in ordered)
        ):
            complete_indices.append(index)
    derivative_lines = derivative_path.read_bytes().splitlines()
    provenance = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(derivative_lines) != len(provenance):
        raise ValueError("recovered_provenance_row_count_mismatch")
    bound_source_lines: set[int] = set()
    for derivative_line, (line, binding) in enumerate(
        zip(derivative_lines, provenance, strict=True), 1
    ):
        source_line = int(binding["original_line"])
        if binding["derivative_line"] != derivative_line:
            raise ValueError("recovered_provenance_derivative_line_mismatch")
        if binding["source_sha256"] != source_hash:
            raise ValueError("recovered_provenance_source_hash_mismatch")
        if binding["original_path"] != (source_path_label or staging_path.as_posix()):
            raise ValueError("recovered_provenance_source_path_mismatch")
        if binding["source_line_sha256"] != sha256(source_lines[source_line - 1]):
            raise ValueError("recovered_provenance_source_line_hash_mismatch")
        if binding["derivative_line_sha256"] != sha256(line):
            raise ValueError("recovered_provenance_derivative_line_hash_mismatch")
        if source_line in bound_source_lines:
            raise ValueError("recovered_provenance_reused_original_line")
        bound_source_lines.add(source_line)
        if line != source_lines[source_line - 1]:
            raise ValueError("recovered_derivative_row_bytes_changed")
        row = json.loads(line)
        if int(row["game_index"]) not in complete_indices:
            raise ValueError("recovered_derivative_contains_incomplete_game")
    expected_source_lines = {
        index
        for index, row in enumerate(source_rows, 1)
        if int(row["game_index"]) in complete_indices
    }
    if bound_source_lines != expected_source_lines:
        raise ValueError("recovered_derivative_source_row_membership_mismatch")
    return {
        "complete_games": len(complete_indices),
        "rows": len(derivative_lines),
        "source_sha256": source_hash,
        "derivative_sha256": sha256(derivative_path.read_bytes()),
        "provenance_sha256": sha256(provenance_path.read_bytes()),
    }
