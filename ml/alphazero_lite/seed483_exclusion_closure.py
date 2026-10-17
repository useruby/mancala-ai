"""Read-only reconstruction and authorization for seed483 exclusions."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed482_fresh_confirmation import (
    canonical_identity,
    float32_identity,
    sha_bytes,
    state_from_encoded,
)
from ml.alphazero_lite.kalah_rules import KalahGame

BASE_MANIFEST = "docs/data/seed482-exclusion-manifest.json.gz"
BASE_MANIFEST_SHA256 = (
    "129391fc88118a99552029faafb1880f1daed62476b2d1eafd7e931f621dc60d"
)
BASE_UNION_SHA256 = "38e97a88041afd3a7f2e580f30155d17fae93f77ddf9267a6e663ca644b758b8"
ARENA_SOURCES = (
    "docs/data/seed429-canonical-policy-normalization/arena-evidence/A/A-games.jsonl",
    "docs/data/seed429-canonical-policy-normalization/arena-evidence/B/B-games.jsonl",
)
OPENING_SOURCES = (
    "docs/data/seed461-batch-order-sensitivity-exploratory-openings.jsonl",
    "docs/data/seed461-batch-order-sensitivity-confirmation-openings.jsonl",
    "docs/data/seed461-cosine-lr-ablation-openings.jsonl",
    "docs/data/seed461-cross-order-e4-average-openings.jsonl",
    "docs/data/seed461-e2-e4-average-openings.jsonl",
    "docs/data/seed461-e3-e4-openings.jsonl",
    "docs/data/seed461-lr-sensitivity-openings-v2.jsonl",
    "docs/data/seed461-e1-e4-corrected-diagnostic/seed397-openings-v2.jsonl",
)
SEED461_ARCHIVE = "docs/data/seed483-recovered-seed461-arena-games.tar.gz"


def _identity_record(
    state: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    return {
        "state": state,
        "canonical_identity": canonical_identity(state),
        "float32_identity": float32_identity(state),
        "provenance": provenance,
    }


def reconstruct_arena_bytes(
    raw: bytes, source: str, arm: str, experiment: int
) -> list[dict[str, Any]]:
    """Replay authenticated seed429 complete game trajectories; run no search."""
    records: list[dict[str, Any]] = []
    with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8") as stream:
        for line in stream:
            game_row = json.loads(line)
            game = KalahGame([4] * 12, [0, 0], 0)
            for move in game_row["opening_prefix_moves"]:
                absolute_move = (
                    game.pit_index(int(move)) if experiment == 429 else int(move)
                )
                if not game.move(absolute_move):
                    raise ValueError("seed429_opening_replay_invalid")
            for ply, move in enumerate(map(int, game_row["trajectory"].split(","))):
                if game.over():
                    raise ValueError("seed429_trajectory_continues_after_terminal")
                state = game.to_state()
                records.append(
                    _identity_record(
                        state,
                        {
                            "experiment": experiment,
                            "source": source,
                            "arm": arm,
                            "game_index": int(game_row["game_index"]),
                            "ply": ply,
                        },
                    )
                )
                if not game.move(move):
                    raise ValueError("seed429_trajectory_replay_invalid")
            if not game.over():
                raise ValueError("seed429_game_not_complete")
    return records


def reconstruct_arena_source(
    path: Path, arm: str, source_ref: str
) -> list[dict[str, Any]]:
    return reconstruct_arena_bytes(path.read_bytes(), source_ref, arm, 429)


def reconstruct_seed429(root: Path) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    occurrences = 0
    for relative, arm in zip(ARENA_SOURCES, ("A", "B"), strict=True):
        path = root / relative
        source_hashes[relative] = sha_bytes(path.read_bytes())
        reconstructed = reconstruct_arena_source(path, arm, relative)
        occurrences += len(reconstructed)
        for record in reconstructed:
            key = (record["canonical_identity"], record["float32_identity"])
            identities.setdefault(key, record)
    return list(identities.values()), source_hashes, occurrences


def _binding_game_sources(root: Path) -> dict[str, str]:
    paths = []
    for candidate in (root / "docs/data").glob("seed461*"):
        if candidate.is_dir():
            paths.extend(
                path.relative_to(root).as_posix() for path in candidate.rglob("*.json")
            )
        elif candidate.suffix == ".json":
            paths.append(candidate.relative_to(root).as_posix())
    sources: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            game_path, game_hash = value.get("games"), value.get("games_sha256")
            if (
                isinstance(game_path, str)
                and isinstance(game_hash, str)
                and game_path.endswith("-games.jsonl")
                and ".tmp/" in game_path
            ):
                member = game_path.split(".tmp/", 1)[1]
                if member in sources and sources[member] != game_hash:
                    raise ValueError(f"seed461_source_hash_conflict:{member}")
                sources[member] = game_hash
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for relative in paths:
        if "seed461" not in relative or not relative.endswith(".json"):
            continue
        try:
            visit(json.loads((root / relative).read_text(encoding="utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return sources


def reconstruct_seed461_arena(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    """Authenticate recovered logs; never guess identities after replay failure."""
    expected_sources = _binding_game_sources(root)
    hashes: dict[str, str] = {}
    move_occurrences = 0
    with tarfile.open(root / SEED461_ARCHIVE, mode="r:gz") as archive:
        for member_name, expected_hash in sorted(expected_sources.items()):
            member = archive.extractfile(member_name)
            if member is None:
                raise ValueError(
                    f"seed461_arena_source_missing_from_archive:{member_name}"
                )
            raw = member.read()
            actual_hash = sha_bytes(raw)
            if actual_hash != expected_hash:
                raise ValueError(f"seed461_arena_source_hash_mismatch:{member_name}")
            hashes[f"{SEED461_ARCHIVE}#{member_name}"] = actual_hash
            for line in raw.splitlines():
                if line:
                    row = json.loads(line)
                    move_occurrences += len(row.get("trajectory", "").split(","))
    return [], hashes, move_occurrences


def reconstruct_opening_source(path: Path, source_ref: str) -> list[dict[str, Any]]:
    """Recover explicit historical arena root inputs from published openings."""
    records = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            state = row.get("state")
            if not isinstance(state, dict):
                raise ValueError("seed461_opening_state_missing")
            records.append(
                _identity_record(
                    state,
                    {"experiment": 461, "source": source_ref, "row": index},
                )
            )
    return records


def reconstruct_union(root: Path) -> dict[str, Any]:
    """Independently rebuild the seed466/427 union plus seed429 arena probes."""
    root = root.resolve()
    training_path = root / "docs/data/seed466-replay-input-bundle/training.npz"
    evaluation_path = root / "docs/data/seed466-replay-input-bundle/evaluation.npz"
    membership_path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    records: dict[tuple[str, str], dict[str, Any]] = {}
    sources = {
        "seed466_training": sha_bytes(training_path.read_bytes()),
        "seed466_evaluation": sha_bytes(evaluation_path.read_bytes()),
        "seed427_validation_membership": sha_bytes(membership_path.read_bytes()),
    }

    def add(
        state: dict[str, Any], prov: dict[str, Any], vector: Any | None = None
    ) -> None:
        record = _identity_record(state, prov)
        if vector is not None:
            exact = sha_bytes(np.asarray(vector, dtype="<f4").tobytes())
            if exact != record["float32_identity"]:
                raise ValueError("reconstructed_float32_identity_mismatch")
        records[(record["canonical_identity"], record["float32_identity"])] = record

    with np.load(training_path, allow_pickle=False) as training:
        vectors = training["inputs"]
        for index, vector in enumerate(vectors):
            add(
                state_from_encoded(vector),
                {"experiment": 466, "partition": "compact_training", "row": index},
                vector,
            )
    with np.load(evaluation_path, allow_pickle=False) as evaluation:
        for index, compact_row in enumerate(evaluation["compact_rows"]):
            vector = vectors[int(compact_row)]
            expected = str(evaluation["input_identity"][index])
            if bytes.fromhex(expected) != np.asarray(vector, dtype="<f4").tobytes():
                raise ValueError("seed466_evaluation_identity_mismatch")
            add(
                state_from_encoded(vector),
                {"experiment": 466, "partition": "evaluation", "row": index},
                vector,
            )
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            vector = np.frombuffer(bytes.fromhex(row["input_identity"]), dtype="<f4")
            state = state_from_encoded(vector)
            canonical_key = json.loads(row["canonical_identity"])
            state_key = [
                state["player_pits"],
                state["opponent_pits"],
                state["player_store"],
                state["opponent_store"],
                state["current_player"],
            ]
            if canonical_key != state_key:
                raise ValueError("seed427_canonical_membership_mismatch")
            add(
                state,
                {"experiment": 427, "partition": row["subset"], "row": index},
                vector,
            )

    arena_records, arena_hashes, arena_occurrences = reconstruct_seed429(root)
    sources.update(arena_hashes)
    seed461_records, seed461_hashes, seed461_occurrences = reconstruct_seed461_arena(
        root
    )
    sources.update(seed461_hashes)
    sources[SEED461_ARCHIVE] = sha_bytes((root / SEED461_ARCHIVE).read_bytes())
    opening_records = []
    for relative in OPENING_SOURCES:
        source_path = root / relative
        sources[relative] = sha_bytes(source_path.read_bytes())
        opening_records.extend(reconstruct_opening_source(source_path, relative))
    baseline_pairs = set(records)
    for record in [*arena_records, *seed461_records, *opening_records]:
        key = (record["canonical_identity"], record["float32_identity"])
        records.setdefault(key, record)
    ordered = [records[key] for key in sorted(records)]
    additions = [
        record
        for record in ordered
        if (record["canonical_identity"], record["float32_identity"])
        not in baseline_pairs
    ]
    base_pair_digest = sha_bytes(
        json.dumps(
            sorted([list(pair) for pair in baseline_pairs]), separators=(",", ":")
        ).encode()
    )
    final_pair_digest = sha_bytes(
        json.dumps(
            sorted([list(pair) for pair in records]), separators=(",", ":")
        ).encode()
    )
    body: dict[str, Any] = {
        "schema": "seed483-exclusion-union-v1",
        "status": "incomplete_historical_source_coverage",
        "sources_sha256": sources,
        "base_manifest_sha256": BASE_UNION_SHA256,
        "base_manifest_file_sha256": BASE_MANIFEST_SHA256,
        "base_identity_pair_count": len(baseline_pairs),
        "base_identity_pairs_sha256": base_pair_digest,
        "seed429_probe_occurrence_count": arena_occurrences,
        "seed429_distinct_pairs": len(arena_records),
        "seed429_additional_pairs": len(
            {
                key
                for key in set(records) - baseline_pairs
                if key
                in {
                    (item["canonical_identity"], item["float32_identity"])
                    for item in arena_records
                }
            }
        ),
        "seed461_opening_occurrences": len(opening_records),
        "seed461_opening_distinct_pairs": len(
            {
                (item["canonical_identity"], item["float32_identity"])
                for item in opening_records
            }
        ),
        "all_reconstructed_additional_pairs": len(set(records) - baseline_pairs),
        "seed461_arena_probe_occurrences": seed461_occurrences,
        "seed461_arena_distinct_pairs": len(seed461_records),
        "additional_identity_count": len(additions),
        "final_identity_pair_count": len(records),
        "final_identity_pairs_sha256": final_pair_digest,
        "identities": additions,
    }
    body["union_sha256"] = final_pair_digest
    return body


def verify_base_manifest(root: Path) -> dict[str, Any]:
    """Check published baseline bytes, then independently rebuild and bind union."""
    base_path = root / BASE_MANIFEST
    if sha_bytes(base_path.read_bytes()) != BASE_MANIFEST_SHA256:
        raise ValueError("seed482_base_manifest_file_hash_mismatch")
    raw = gzip.decompress(base_path.read_bytes())
    base = json.loads(raw)
    claim = base.pop("manifest_sha256", None)
    actual = sha_bytes(json.dumps(base, sort_keys=True, separators=(",", ":")).encode())
    if claim != actual or claim != BASE_UNION_SHA256:
        raise ValueError("seed482_base_manifest_receipt_mismatch")
    result = reconstruct_union(root)
    claimed_pairs = {
        (record["canonical_identity"], record["float32_identity"])
        for record in base["identities"]
    }
    if len(claimed_pairs) != result["base_identity_pair_count"]:
        raise ValueError("seed482_base_manifest_duplicate_identity_pair")
    claimed_digest = sha_bytes(
        json.dumps(
            sorted([list(pair) for pair in claimed_pairs]), separators=(",", ":")
        ).encode()
    )
    if claimed_digest != result["base_identity_pairs_sha256"]:
        raise ValueError("seed482_base_manifest_identity_set_mismatch")
    return result


def verify_union_file(path: Path, root: Path) -> dict[str, Any]:
    """Read-only verify every published identity and receipt against reconstruction."""
    raw = path.read_bytes()
    decoded = gzip.decompress(raw) if path.suffix == ".gz" else raw
    published = json.loads(decoded)
    rebuilt = verify_base_manifest(root)
    if published != rebuilt:
        raise ValueError("seed483_union_identity_reconstruction_mismatch")
    pairs = {
        (row["canonical_identity"], row["float32_identity"])
        for row in published.get("identities", [])
    }
    if len(pairs) != published.get("additional_identity_count"):
        raise ValueError("seed483_union_duplicate_additional_identity")
    expected_digest = published.get("union_sha256")
    if expected_digest != published.get("final_identity_pairs_sha256"):
        raise ValueError("seed483_union_receipt_mismatch")
    return {
        "status": published["status"],
        "additional_identity_count": len(pairs),
        "final_identity_pair_count": published["final_identity_pair_count"],
        "union_sha256": expected_digest,
    }


def authorize_generation(
    coverage: dict[str, Any],
    union: dict[str, Any],
    *,
    root: Path,
    union_pairs: set[tuple[str, str]],
) -> None:
    """Reconstruct evidence before authorization; labels and experiment IDs do not suffice."""
    verify_coverage_document(coverage, root, union_pairs)
    if (
        coverage.get("status") != "complete"
        or coverage.get("independent_verification") != "passed"
        or any(
            row.get("contribution") == "uncovered" for row in coverage["experiments"]
        )
    ):
        raise ValueError("historical_probe_coverage_not_verified")
    coverage_union = coverage.get("reconstructed_union", {}).get("union_sha256")
    if coverage_union != union.get("union_sha256"):
        raise ValueError("historical_probe_union_binding_mismatch")
    if union.get("status") != "complete":
        raise ValueError("historical_probe_coverage_not_verified")
    pair_digest = sha_bytes(
        json.dumps(
            sorted([list(pair) for pair in union_pairs]), separators=(",", ":")
        ).encode()
    )
    if pair_digest != union.get("final_identity_pairs_sha256"):
        raise ValueError("historical_probe_union_binding_mismatch")


def verify_coverage_document(
    document: dict[str, Any], root: Path, union_pairs: set[tuple[str, str]]
) -> None:
    """Verify source receipts and identity-level coverage, never trust labels alone."""
    body = dict(document)
    receipt = body.pop("receipt_sha256", None)
    expected = sha_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    )
    if receipt != expected:
        raise ValueError("coverage_receipt_mismatch")
    rows = document.get("experiments")
    if not isinstance(rows, list) or [row.get("experiment") for row in rows] != list(
        range(427, 481)
    ):
        raise ValueError("coverage_experiment_inventory_mismatch")
    for row in rows:
        status = row.get("contribution")
        if status not in {
            "reconstructed_identities",
            "verified_subset",
            "evidence_backed_empty",
            "uncovered",
        }:
            raise ValueError("coverage_contribution_invalid")
        for source in row.get("source_artifacts", []):
            source_path = root / source["path"]
            if sha_bytes(source_path.read_bytes()) != source["sha256"]:
                raise ValueError(f"coverage_source_hash_mismatch:{source['path']}")
        if status in {"reconstructed_identities", "verified_subset"}:
            pairs = set()
            for item in row.get("identities", []):
                if item.get("state") is not None:
                    pair = (
                        canonical_identity(item["state"]),
                        float32_identity(item["state"]),
                    )
                else:
                    pair = (item["canonical_identity"], item["float32_identity"])
                pairs.add(pair)
            if status == "reconstructed_identities" and not pairs:
                population_ref = row.get("population")
                if not isinstance(population_ref, str):
                    raise ValueError("coverage_reconstruction_population_missing")
                pairs = _population_identity_pairs(root, population_ref)
                if not pairs:
                    raise ValueError("coverage_reconstruction_source_unavailable")
            if status == "verified_subset":
                proof = row.get("subset_proof")
                if not isinstance(proof, dict):
                    raise ValueError("coverage_subset_proof_missing")
                population_ref = proof.get("population_ref")
                if not isinstance(population_ref, str):
                    raise ValueError("coverage_subset_population_unavailable")
                population_source = proof.get("population_source")
                if population_source not in {
                    source.get("path") for source in row.get("source_artifacts", [])
                }:
                    raise ValueError("coverage_subset_source_unbound")
                population_pairs = _population_identity_pairs(root, population_ref)
                if not population_pairs:
                    raise ValueError("coverage_subset_population_unavailable")
                if proof.get("population_pair_count") != len(
                    population_pairs
                ) or proof.get("population_pairs_sha256") != _pair_set_sha256(
                    population_pairs
                ):
                    raise ValueError("coverage_subset_proof_mismatch")
                if pairs and not pairs <= population_pairs:
                    raise ValueError("coverage_identity_outside_subset_population")
                pairs = pairs or population_pairs
            if not pairs <= union_pairs:
                raise ValueError("coverage_identity_outside_final_union")
        elif status == "evidence_backed_empty":
            evidence = row.get("empty_contribution_evidence")
            source_paths = {
                source["path"] for source in row.get("source_artifacts", [])
            }
            if not evidence or any(
                item.get("path") not in source_paths for item in evidence
            ):
                raise ValueError("coverage_empty_evidence_missing")


def _population_identity_pairs(root: Path, population_ref: str) -> set[tuple[str, str]]:
    if population_ref == "seed427_validation_membership":
        membership = (
            root
            / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
        )
        pairs = set()
        with gzip.open(membership, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                vector = np.frombuffer(
                    bytes.fromhex(row["input_identity"]), dtype="<f4"
                )
                state = state_from_encoded(vector)
                pairs.add((canonical_identity(state), float32_identity(state)))
        return pairs
    if population_ref in {
        "seed427_or_seed466_authenticated_population",
        "seed466_base",
    }:
        base = json.loads(gzip.decompress((root / BASE_MANIFEST).read_bytes()))
        return {
            (row["canonical_identity"], row["float32_identity"])
            for row in base["identities"]
        }
    if population_ref == "seed429_arena_trajectories":
        rows, _, _ = reconstruct_seed429(root)
        return {(row["canonical_identity"], row["float32_identity"]) for row in rows}
    if population_ref == "seed461_opening_roots_and_missing_arena_continuations":
        rows = [
            reconstruct_opening_source(root / path, path) for path in OPENING_SOURCES
        ]
        return {
            (row["canonical_identity"], row["float32_identity"])
            for group in rows
            for row in group
        }
    return set()


def _pair_set_sha256(pairs: set[tuple[str, str]]) -> str:
    return sha_bytes(
        json.dumps(
            sorted([list(pair) for pair in pairs]), separators=(",", ":")
        ).encode()
    )


def tracked_source_inventory(root: Path) -> dict[int, list[dict[str, str]]]:
    """Return seed-specific source artifacts and their current byte hashes."""
    paths = [
        path.relative_to(root).as_posix()
        for path in (root / "docs/data").rglob("*")
        if path.is_file()
    ]
    paths.extend(
        path.relative_to(root).as_posix()
        for path in (root / "ml/alphazero_lite").glob("*seed*.py")
        if path.is_file()
    )
    result: dict[int, list[dict[str, str]]] = {seed: [] for seed in range(427, 481)}
    for relative in paths:
        match = re.search(r"seed(\d{3})", relative)
        if match is None or int(match.group(1)) not in result:
            continue
        path = root / relative
        if path.is_file():
            result[int(match.group(1))].append(
                {"path": relative, "sha256": sha_bytes(path.read_bytes())}
            )
    return result


def build_coverage_ledger(root: Path, union: dict[str, Any]) -> dict[str, Any]:
    """Publish an experiment-by-experiment inventory with explicit unresolved gaps."""
    root = root.resolve()
    inventory = tracked_source_inventory(root)
    base_hashes = {
        "554f3bd016a0f920ef253538d741a91cb27c9882d2dba94cf7509030af8ca385",
        "72950845b76cb0cb842d9ba8065591978469e74c9ac51cd1c0f76c120158b364",
        "2e3b503b4e5d64c40d2bca0b44560bfd9f380c2f57606ad8c2db949a1a9c93ae",
    }
    rows = []
    missing_arena_sources: dict[str, str] = {}
    subset_population_cache: dict[str, set[tuple[str, str]]] = {}
    for experiment in range(427, 481):
        sources = inventory[experiment]
        if experiment in {427, 428}:
            references = [
                "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
                "docs/data/seed426-canonical-overlap/seed427-prediction-evidence.jsonl.gz",
            ]
            if experiment == 428:
                references.extend(
                    [
                        "docs/data/seed426-canonical-overlap/seed428-analysis-plan.md",
                        "docs/data/seed426-canonical-overlap/seed428-source-contributions.json",
                    ]
                )
            known = {source["path"] for source in sources}
            for relative in references:
                if relative not in known:
                    sources.append(
                        {
                            "path": relative,
                            "sha256": sha_bytes((root / relative).read_bytes()),
                        }
                    )
        if experiment == 461:
            sources.append(
                {
                    "path": SEED461_ARCHIVE,
                    "sha256": sha_bytes((root / SEED461_ARCHIVE).read_bytes()),
                }
            )
        if experiment in {430, 431}:
            relative = f"docs/data/seed429-canonical-policy-normalization/seed{experiment}-verification-correction.json"
            sources.append(
                {"path": relative, "sha256": sha_bytes((root / relative).read_bytes())}
            )
        joined_text = " ".join(
            (root / source["path"]).read_text(errors="ignore")
            for source in sources
            if (root / source["path"]).suffix in {".json", ".md"}
            and (root / source["path"]).stat().st_size < 20_000_000
        )
        evidence_hashes = {digest for digest in base_hashes if digest in joined_text}
        if experiment == 427:
            contribution, method, population = (
                "verified_subset",
                "reconstructed from authenticated ordered membership; identities in base union",
                "seed427_validation_membership",
            )
        elif experiment == 428:
            contribution, method, population = (
                "verified_subset",
                "published analysis plan explicitly reuses seed427 predictions and membership",
                "seed427_validation_membership",
            )
        elif experiment == 429:
            contribution, method, population = (
                "reconstructed_identities",
                "strict replay of both committed A/B arena trajectories and opening prefixes",
                "seed429_arena_trajectories",
            )
        elif experiment in {430, 431, 432, 433, 434, 439, 443, 448, 456, 477}:
            contribution, method, population = (
                "evidence_backed_empty",
                "read-only census, correction, or verifier/accounting publication; no model inference, search, or game execution",
                "",
            )
        elif experiment == 461:
            contribution, method, population = (
                "uncovered",
                "reconstructed all committed opening roots; arena game continuations unavailable",
                "seed461_opening_roots_and_missing_arena_continuations",
            )
            for source in sources:
                path = root / source["path"]
                if path.name.endswith("-evaluation-binding.json"):
                    try:
                        binding = json.loads(path.read_text())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    def visit(value: Any) -> None:
                        if isinstance(value, dict):
                            game_path = value.get("games")
                            game_hash = value.get("games_sha256")
                            if isinstance(game_path, str) and isinstance(
                                game_hash, str
                            ):
                                relative = game_path
                                prefix = str(root) + "/"
                                if relative.startswith(prefix):
                                    relative = relative[len(prefix) :]
                                if not (root / relative).is_file():
                                    missing_arena_sources[relative] = game_hash
                            for child in value.values():
                                visit(child)
                        elif isinstance(value, list):
                            for child in value:
                                visit(child)

                    visit(binding)
        elif experiment == 467:
            contribution, method, population = (
                "uncovered",
                "seed467 diagnostic report identifies fresh self-play artifact, but bytes are absent",
                "seed467_fresh_self_play",
            )
        elif evidence_hashes:
            contribution, method, population = (
                "verified_subset",
                "registration/receipt binds probe inputs to authenticated seed427 or seed466 population",
                "seed427_or_seed466_authenticated_population",
            )
        elif sources:
            contribution, method, population = (
                "uncovered",
                "committed artifacts inventoried; source-to-probe identity mapping not independently established",
                "",
            )
        else:
            contribution, method, population = (
                "uncovered",
                "no committed seed-specific source artifact found; absence is not evidence of no probes",
                "",
            )
        empty_evidence = (
            [
                source
                for source in sources
                if source["path"].startswith("docs/data/")
                and not source["path"].endswith(".py")
            ][:1]
            if contribution == "evidence_backed_empty"
            else []
        )
        subset_proof = None
        if contribution == "verified_subset":
            population_source = (
                "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
                if population == "seed427_validation_membership"
                else BASE_MANIFEST
            )
            if population_source not in {item["path"] for item in sources}:
                sources.append(
                    {
                        "path": population_source,
                        "sha256": sha_bytes((root / population_source).read_bytes()),
                    }
                )
            if population not in subset_population_cache:
                subset_population_cache[population] = _population_identity_pairs(
                    root, population
                )
            population_pairs = subset_population_cache[population]
            subset_proof = {
                "population_ref": population,
                "population_source": population_source,
                "population_pair_count": len(population_pairs),
                "population_pairs_sha256": _pair_set_sha256(population_pairs),
            }
        rows.append(
            {
                "experiment": experiment,
                "source_artifacts": sources,
                "contribution": contribution,
                "reconstruction_method": method,
                "population": population,
                "source_population_bindings": sorted(evidence_hashes),
                "subset_proof": subset_proof,
                "identities_in_union": (
                    union["seed429_distinct_pairs"]
                    if experiment == 429
                    else union["seed461_opening_distinct_pairs"]
                    if experiment == 461
                    else None
                ),
                "empty_contribution_evidence": empty_evidence,
            }
        )
    missing_arena_gaps = [
        {
            "path": path,
            "expected_sha256": digest,
            "population_gap": "all per-ply model-probe states in the missing arena game continuations",
        }
        for path, digest in sorted(missing_arena_sources.items())
    ]
    replay_gaps = [
        {
            "path": f".tmp/{member}",
            "expected_sha256": digest,
            "recovered_copy": SEED461_ARCHIVE,
            "population_gap": "per-ply arena model-probe states; committed trajectory encoding does not replay under canonical Kalah transitions (first move legality mismatch)",
        }
        for key, digest in sorted(union["sources_sha256"].items())
        if key.startswith(f"{SEED461_ARCHIVE}#")
        for member in [key.split("#", 1)[1]]
    ]
    unrecovered = [
        {
            "experiment": experiment,
            "path": None,
            "path_search": f"docs/data/seed{experiment}-* and committed seed{experiment} registrations/receipts",
            "expected_sha256": None,
            "population_gap": "no committed seed-specific artifact or expected source identity was found; probe population cannot be bounded from committed evidence",
        }
        for experiment in (458, 459, 460, 463, 465)
    ]
    unrecovered.append(
        {
            "experiment": 467,
            "path": ".tmp/seed455-nextgen-s467-parent-anchor-w010/runs/seed455-nextgen-s467-parent-anchor-w010-iter1/self_play.jsonl",
            "expected_sha256": "6f7038dccf282fb9365a031a06f8c5a8af4a2b8a6eda32725fe125b06d654a76",
            "population_gap": "all seed467 self-play states used by fresh validation and training diagnostics; 749,547,469-byte source named in committed generation receipt",
        }
    )
    body = {
        "schema": "seed483-historical-probe-coverage-v1",
        "status": "incomplete",
        "experiment_range": [427, 480],
        "base_population": {
            "path": BASE_MANIFEST,
            "sha256": BASE_MANIFEST_SHA256,
            "union_sha256": BASE_UNION_SHA256,
            "pair_count": 62330,
        },
        "reconstructed_union": {
            "additional_pairs": union["all_reconstructed_additional_pairs"],
            "final_pair_count": union["final_identity_pair_count"],
            "union_sha256": union["union_sha256"],
        },
        "experiments": rows,
        "unavailable_sources": unrecovered,
        "reconstruction_gaps": [*missing_arena_gaps, *replay_gaps],
        "generation_authorized": False,
    }
    body["receipt_sha256"] = sha_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    )
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ledger-output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        union_path = args.root / "docs/data/seed483-historical-probe-identities.json.gz"
        coverage_path = args.root / "docs/data/seed483-source-coverage.json"
        summary = verify_union_file(union_path, args.root)
        base = json.loads(gzip.decompress((args.root / BASE_MANIFEST).read_bytes()))
        union = json.loads(gzip.decompress(union_path.read_bytes()))
        pairs = {
            (row["canonical_identity"], row["float32_identity"])
            for row in base["identities"]
        }
        pairs.update(
            (row["canonical_identity"], row["float32_identity"])
            for row in union["identities"]
        )
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        verify_coverage_document(coverage, args.root, pairs)
        print(
            json.dumps(
                {
                    **summary,
                    "coverage_status": coverage["status"],
                    "uncovered_experiments": [
                        row["experiment"]
                        for row in coverage["experiments"]
                        if row["contribution"] == "uncovered"
                    ],
                    "generation_authorized": False,
                },
                sort_keys=True,
            )
        )
        return
    result = verify_base_manifest(args.root)
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if args.output:
        if args.output.suffix == ".gz":
            args.output.write_bytes(gzip.compress(payload, mtime=0))
        else:
            args.output.write_bytes(payload)
    ledger = build_coverage_ledger(args.root, result)
    if args.ledger_output:
        ledger_payload = (json.dumps(ledger, sort_keys=True, indent=2) + "\n").encode()
        if args.ledger_output.suffix == ".gz":
            args.ledger_output.write_bytes(gzip.compress(ledger_payload, mtime=0))
        else:
            args.ledger_output.write_bytes(ledger_payload)
    print(
        json.dumps(
            {
                **{key: value for key, value in result.items() if key != "identities"},
                "coverage_status": ledger["status"],
                "unavailable_source_count": len(ledger["unavailable_sources"]),
                "generation_authorized": ledger["generation_authorized"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
