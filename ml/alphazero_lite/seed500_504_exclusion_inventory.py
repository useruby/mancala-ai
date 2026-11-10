"""Authenticate and index the seed500 corpus under the registered seed504 split."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
CORPUS = DATA / "seed500-selfplay-512"
SEED504 = DATA / "seed504"
OUT = DATA / "seed500-504-exclusion-inventory"
SHARDS = [CORPUS / f"shard-503{i:03d}.jsonl" for i in range(8)]
SPLIT_SHA256 = "080b362d334dcce81bbd04a490900ec78483d6c05ad38c88ebb33fd8b8695d89"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require_digest(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path) if path.is_file() else "missing"
    if observed != expected:
        raise ValueError(
            f"digest_mismatch:{label}:expected={expected}:observed={observed}"
        )


def _record(
    source: str, line: int, state: dict[str, Any], phase: str
) -> dict[str, Any]:
    canonical = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    encoded = encode_state(state, input_encoding="kalah_v3")
    float32_bytes = b"".join(
        __import__("struct").pack("<f", float(value)) for value in encoded
    )
    return {
        "source": source,
        "line": line,
        "phase": phase,
        "canonical_state_sha256": canonical,
        "kalah_v3_float32_sha256": hashlib.sha256(float32_bytes).hexdigest(),
    }


def _replayed_occurrences() -> dict[
    tuple[str, int], tuple[dict[str, Any], dict[str, Any]]
]:
    """Return each source row and the rule state replayed immediately before it."""
    result: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for path in SHARDS:
        by_game: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            row = json.loads(line)
            by_game[int(row["game_index"])].append((line_no, row))
        for game_index, indexed_rows in by_game.items():
            indexed_rows.sort(key=lambda pair: int(pair[1]["move_index"]))
            first = indexed_rows[0][1]
            game = KalahGame.from_state(first["game_start_state"])
            moves = first["game_moves"]
            if len(indexed_rows) != len(moves):
                raise ValueError(f"row_action_count_mismatch:{path.name}:{game_index}")
            for move_index, (line_no, row) in enumerate(indexed_rows):
                state = game.to_state()
                expected = b"".join(
                    __import__("struct").pack("<f", float(value))
                    for value in row["state"]
                )
                actual_values = encode_state(state, input_encoding="kalah_v3")
                actual = b"".join(
                    __import__("struct").pack("<f", float(value))
                    for value in actual_values
                )
                if actual != expected:
                    raise ValueError(
                        f"encoded_state_replay_mismatch:{path.name}:{line_no}"
                    )
                if int(row["move_index"]) != move_index:
                    raise ValueError(f"row_move_order_mismatch:{path.name}:{line_no}")
                result[(path.name, line_no)] = (row, state)
                move = int(moves[move_index])
                if move not in game.possible_moves() or not game.move(
                    game.pit_index(move)
                ):
                    raise ValueError(
                        f"illegal_recorded_move:{path.name}:{game_index}:{move_index}"
                    )
            if not game.over():
                raise ValueError(f"game_not_terminal:{path.name}:{game_index}")
    return result


def census() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # #504's manifest binds all eight #503 source files to their exact bytes.
    manifest = json.loads((SEED504 / "manifest.json").read_text(encoding="utf-8"))
    source_specs = manifest["sources"]["training"] + manifest["sources"]["development"]
    if len(source_specs) != 8:
        raise ValueError("expected_eight_registered_seed500_shards")
    for source in source_specs:
        path = ROOT / source["path"]
        require_digest(path, source["sha256"], source["path"])

    split_path = SEED504 / "split.json"
    observed_split = sha256_file(split_path)
    require_digest(split_path, SPLIT_SHA256, str(split_path.relative_to(ROOT)))
    corpus_accounting = json.loads((CORPUS / "accounting.json").read_text())
    corpus_rows = sum(
        len(path.read_text(encoding="utf-8").splitlines()) for path in SHARDS
    )
    if (
        corpus_rows != 18281
        or corpus_accounting.get("loader_accepted_rows") != 18281
        or corpus_accounting.get("production_loader") != "accepted"
    ):
        raise ValueError("loader_accepted_row_count_mismatch")
    if corpus_accounting["actions"] != corpus_rows:
        raise ValueError("seed500_accounting_row_count_mismatch")

    # Rebuild membership through the frozen #504 splitter, and compare every
    # occurrence against the publication rather than trusting its row totals.
    row_cache: dict[int, list[dict[str, Any]]] = {}
    for index, path in enumerate(SHARDS):
        row_cache[index] = []
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            row = json.loads(line)
            row["_source"], row["_line"] = path.name, line_no
            row_cache[index].append(row)
    train_rows = [row for i in range(6) for row in row_cache[i]]
    raw_dev = [row for i in range(6, 8) for row in row_cache[i]]

    def input_key(row: dict[str, Any]) -> tuple[float, ...]:
        return tuple(
            __import__("struct").unpack(
                "<f", __import__("struct").pack("<f", float(x))
            )[0]
            for x in row["state"]
        )

    def canonical_key(row: dict[str, Any]) -> tuple[float, ...]:
        return input_key(row)[:15]

    training_canonical = {canonical_key(row) for row in train_rows}
    training_input = {input_key(row) for row in train_rows}
    retained_dev = [
        row
        for row in raw_dev
        if canonical_key(row) not in training_canonical
        and input_key(row) not in training_input
    ]
    membership = {
        "training": [
            {"shard": row["_source"], "line": row["_line"]} for row in train_rows
        ],
        "development_original": [
            {"shard": row["_source"], "line": row["_line"]} for row in raw_dev
        ],
        "development_filtered": [
            {"shard": row["_source"], "line": row["_line"]} for row in retained_dev
        ],
    }
    rebuilt = {
        "counts": {},
        "membership": membership,
        "membership_sha256": sha256_bytes(
            json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()
        ),
    }
    for name, group in (
        ("training", train_rows),
        ("development_original", raw_dev),
        ("development_filtered", retained_dev),
    ):
        cids = {canonical_key(row) for row in group}
        fids = {input_key(row) for row in group}
        active_counts = defaultdict(int)
        for row in group:
            active = int(round(sum(input_key(row)[:12]) * 48.0))
            active_counts[
                ">32" if active > 32 else "17-32" if active >= 17 else "<=16"
            ] += 1
        rebuilt["counts"][name] = {
            "rows": len(group),
            "canonical_unique": len(cids),
            "float32_unique": len(fids),
            "active_stones": dict(active_counts),
        }
    rebuilt["counts"]["excluded_development_rows"] = len(raw_dev) - len(retained_dev)
    rebuilt["counts"]["remaining_overlap_canonical"] = len(
        training_canonical & {canonical_key(row) for row in retained_dev}
    )
    rebuilt["counts"]["remaining_overlap_float32"] = len(
        training_input & {input_key(row) for row in retained_dev}
    )
    published = json.loads(split_path.read_text(encoding="utf-8"))
    if rebuilt != published:
        raise ValueError("seed504_split_membership_reconstruction_mismatch")
    if (len(train_rows), len(raw_dev), len(retained_dev)) != (13724, 4557, 2671):
        raise ValueError("seed504_split_count_mismatch")

    results_path = SEED504 / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    expected_scored = {(row["_source"], int(row["_line"])) for row in retained_dev}
    for endpoint, evidence in results.items():
        predictions = evidence.get("predictions", [])
        scored_membership = {(row["source"], int(row["line"])) for row in predictions}
        if len(predictions) != 2671 or scored_membership != expected_scored:
            raise ValueError(
                f"seed504_scored_development_membership_mismatch:{endpoint}"
            )
        if evidence.get("metrics", {}).get("pooled", {}).get("rows") != 2671:
            raise ValueError(f"seed504_scored_development_count_mismatch:{endpoint}")

    replayed = _replayed_occurrences()
    occurrence_rows: list[dict[str, Any]] = []
    membership_names = {
        "training": train_rows,
        "raw_development": raw_dev,
        "retained_development": retained_dev,
    }
    memberships = {}
    for group, rows in membership_names.items():
        keys = {(row["_source"], int(row["_line"])) for row in rows}
        memberships[group] = keys
        for source, line in sorted(keys):
            try:
                _, state = replayed[(source, line)]
            except KeyError as exc:
                raise ValueError(
                    f"membership_row_missing_from_replay:{group}:{source}:{line}"
                ) from exc
            occurrence_rows.append(
                {"population": group, **_record(source, line, state, "pre_move")}
            )

    counts = {}
    for group, keys in memberships.items():
        group_rows = [row for row in occurrence_rows if row["population"] == group]
        counts[group] = {
            "occurrences": len(keys),
            "canonical_distinct": len(
                {row["canonical_state_sha256"] for row in group_rows}
            ),
            "kalah_v3_float32_distinct": len(
                {row["kalah_v3_float32_sha256"] for row in group_rows}
            ),
        }
    counts["loader_accepted_corpus"] = {
        "occurrences": 18281,
        "canonical_distinct": corpus_accounting["canonical_unique_states"],
        "kalah_v3_float32_distinct": corpus_accounting["encoded_input_unique_states"],
    }
    if (
        counts["training"]["occurrences"] + counts["raw_development"]["occurrences"]
        != 18281
    ):
        raise ValueError("training_plus_raw_development_does_not_reconcile")
    return {
        "source": "#503 archived seed500 shards; #504 frozen split",
        "source_status": "verified",
        "counts": counts,
        "retained_development_scored": 2671,
        "excluded_raw_development": 1886,
        "split_sha256": observed_split,
        "scored_results_sha256": sha256_file(results_path),
        "scored_endpoints": sorted(results),
    }, occurrence_rows


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report, rows = census()
    index = (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in sorted(
                rows, key=lambda row: (row["population"], row["source"], row["line"])
            )
        )
        + "\n"
    ).encode()
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "identity-index.jsonl").write_bytes(index)
        (OUT / "count-report.json").write_bytes(report_bytes)
    else:
        for name, content in (
            ("identity-index.jsonl", index),
            ("count-report.json", report_bytes),
        ):
            if not (OUT / name).is_file() or (OUT / name).read_bytes() != content:
                raise ValueError(f"published_artifact_mismatch:{name}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
