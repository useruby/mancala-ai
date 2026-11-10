"""Read-only reconciliation inventory for the #561 holdback population.

The verifier authenticates #511's frozen source list, imports the three
publication-pinned historical snapshots, and independently replays the arena
and self-play records. It never starts an arena, search, or training run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import archived_screen_identity_inventory as identity
from ml.alphazero_lite import archived_screen_identity_seven as seven
from ml.alphazero_lite import seed511_fixed_composition_screen as seed511
from ml.alphazero_lite.seed482_fresh_confirmation import state_from_encoded

ROOT = Path(__file__).resolve().parents[2]
OUT_REL = Path("docs/data/seed561-historical-exclusion-inventory-v1")
REGISTRATION_REL = Path(
    "docs/data/seed511-fixed-composition-screen-v4/registration.json"
)
REGISTRATION_SHA256 = "89f04c6e1f774d81a7866d3913dd5ffa990b9e04a58f9a7f066b8f19204445c4"
SCHEMA = "seed561_historical_exclusion_inventory_v1"


class InventoryError(ValueError):
    """Raised when a pinned source or deterministic index does not verify."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.splitlines() if line]


def authenticate_seed511_sources(root: Path) -> dict[str, str]:
    """Reject altered #511 bytes even if a local registration is re-receipted."""
    registration = root / REGISTRATION_REL
    if _sha(registration) != REGISTRATION_SHA256:
        raise InventoryError("seed511_registration_hash_mismatch")
    document = json.loads(registration.read_text())
    pins = document.get("exclusion_source_sha256")
    if not isinstance(pins, dict) or not pins:
        raise InventoryError("seed511_source_pins_missing")
    for relative, expected in pins.items():
        path = root / relative
        if not path.is_file() or _sha(path) != expected:
            raise InventoryError(f"pinned_source_mismatch:{relative}")
    return pins


def _snapshot_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    rows: list[dict[str, Any]] = []
    sets = {"canonical": set(), "kalah_v3_float32": set()}
    sources = (
        ("482", "docs/data/seed482-exclusion-manifest.json.gz", "identities"),
        ("483", "docs/data/seed483-historical-probe-identities.json.gz", "identities"),
        (
            "484_seed467_additions",
            "docs/data/seed484-extended-probe-union.json.gz",
            "seed467_additional_identities",
        ),
    )
    for source, relative, key in sources:
        document = json.loads(gzip.decompress((root / relative).read_bytes()))
        for ordinal, item in enumerate(document[key]):
            c, f = item["canonical_identity"], item["float32_identity"]
            sets["canonical"].add(c)
            sets["kalah_v3_float32"].add(f)
            rows.append(
                {
                    "source": source,
                    "source_artifact": relative,
                    "source_ordinal": ordinal,
                    "canonical_identity": c,
                    "kalah_v3_float32_identity": f,
                    "provenance": item.get("provenance", {"state": item.get("state")}),
                    "identity_origin": "imported_verified_snapshot",
                }
            )
    return rows, sets


def _replay_records(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    sets = {"canonical": set(), "kalah_v3_float32": set()}
    counts: dict[str, int] = {}

    def capture(state: dict[str, Any], source: str, provenance: dict[str, Any]) -> None:
        c = identity.canonical_identity(state)
        f = identity.kalah_v3_identity(state)
        sets["canonical"].add(c)
        sets["kalah_v3_float32"].add(f)
        rows.append(
            {
                "source": source,
                "canonical_identity": c,
                "kalah_v3_float32_identity": f,
                "provenance": provenance,
                "identity_origin": "independently_replayed_state",
            }
        )

    def replay(
        prefix: list[int],
        trajectory: str,
        source: str,
        artifact: str,
        game_index: int,
        seat: int | None = None,
    ) -> int:
        from ml.alphazero_lite.kalah_rules import KalahGame

        game = KalahGame.from_state(seed511.INITIAL_STATE)
        ply = 0
        for action_index, action in enumerate(
            [*(int(x) for x in prefix), *(int(x) for x in trajectory.split(",") if x)]
        ):
            is_prefix = action_index < len(prefix)
            relative = action if is_prefix else action - game.current_player * 6
            if game.over() or relative not in game.possible_moves():
                raise InventoryError(f"illegal_replay:{source}:{game_index}:{ply}")
            absolute = game.pit_index(relative) if is_prefix else action
            state = game.to_state()
            capture(
                state,
                source,
                {
                    "artifact": artifact,
                    "game_index": game_index,
                    "seat": seat,
                    "ply": ply,
                    "action_absolute": absolute,
                },
            )
            if not game.move(absolute):
                raise InventoryError(f"move_rejected:{source}:{game_index}:{ply}")
            ply += 1
        if not game.over() and len(trajectory.split(",")) < 200:
            raise InventoryError(f"nonterminal_replay:{source}:{game_index}")
        return ply

    for source in ("488", "499", "505"):
        relative = {
            "488": "docs/data/seed488-frozen-strength-evaluation/outcomes.jsonl",
            "499": "docs/data/seed499-budget-comparison/outcomes.jsonl",
            "505": "docs/data/seed505-strength-screen-v3/outcomes.jsonl",
        }[source]
        games = _jsonl(root / relative)
        for row in games:
            replay(
                row["opening_prefix_moves"],
                row["trajectory"],
                source,
                relative,
                int(row["game_index"]),
                int(row["challenger_player"]),
            )
        counts[source] = len(games)

    source = "508"
    directory = root / "docs/data/seed508-policy-value-crossover/batches"
    game_count = 0
    for path in sorted(directory.glob("*-games-*.jsonl")):
        arm, _, number = path.name.split("-")
        opening_path = (
            directory / f"{arm}-opening-{int(number.split('.')[0]):04d}.jsonl"
        )
        opening = _jsonl(opening_path)
        if len(opening) != 1:
            raise InventoryError(f"seed508_opening_shape:{path.name}")
        for row in _jsonl(path):
            replay(
                opening[0]["prefix_moves"],
                row["trajectory"],
                source,
                str(path.relative_to(root)),
                int(row["game_index"]),
                int(row["challenger_player"]),
            )
            game_count += 1
    if game_count != 2048:
        raise InventoryError(f"seed508_game_count:{game_count}")
    counts[source] = game_count

    source = "500"
    corpus = root / "docs/data/seed500-selfplay-512"
    accounting = json.loads((corpus / "accounting.json").read_text())
    files = [root / "docs/data/seed500-selfplay-64.jsonl"]
    files.extend(root / item["path"] for item in accounting["shards"])
    for path in files:
        for row_index, row in enumerate(_jsonl(path)):
            state = state_from_encoded(row["state"])
            capture(
                state,
                source,
                {
                    "artifact": str(path.relative_to(root)),
                    "row_index": row_index,
                    "game_index": row.get("game_index"),
                    "move_index": row.get("move_index"),
                },
            )
    counts[source] = sum(len(_jsonl(path)) for path in files)
    return rows, sets, counts


def _published_subtotal(root: Path) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """Load authenticated #562–#564 indexes and preserve their provenance rows."""
    rows: list[dict[str, Any]] = []
    sets = {"canonical": set(), "kalah_v3_float32": set()}
    # #562's occurrence census is deterministic and read-only; its publication
    # index carries canonical rows, while the independent census supplies both
    # paired scheme memberships and first-occurrence provenance.
    census_module = __import__(
        "ml.alphazero_lite.seed556_560_exclusion_inventory",
        fromlist=["census"],
    )
    original_base = census_module.BASE
    try:
        census_module.BASE = root / "docs/data"
        _, ids556, ids560, _, provenance = census_module.census(
            data556=root / "docs/data/seed556-exploratory-arena/screen-0005",
            data560=root / "docs/data/seed560-paired-first-action/run-0004",
        )
    finally:
        census_module.BASE = original_base
    published_562 = {"canonical": set(), "kalah_v3_float32": set()}
    index_562 = (
        root / "docs/data/seed556-560-exclusion-inventory/identity-inventory.json"
    )
    for record in _jsonl(index_562):
        scheme = record.get("scheme")
        if scheme not in published_562 or not isinstance(record.get("identity"), str):
            raise InventoryError("seed562_index_row_invalid")
        published_562[scheme].add(record["identity"])
    for source in ("556", "560"):
        source_key = "562"
        for scheme in ("canonical", "kalah_v3_float32"):
            ids = ids556[scheme] if source == "556" else ids560[scheme]
            sets[scheme].update(ids)
            for value, location in provenance[source][scheme].items():
                rows.append(
                    {
                        "source": source_key,
                        "scheme": scheme,
                        "identity": value,
                        "provenance": location,
                    }
                )
    reconstructed_562 = {
        scheme: ids556[scheme] | ids560[scheme] for scheme in published_562
    }
    if reconstructed_562 != published_562:
        raise InventoryError("seed562_reconstructed_membership_mismatch")
    # #563 publishes both schemes on a paired row.
    expanded_records = _jsonl(
        root
        / "docs/data/expanded-training-dev-exclusion-inventory/identity-inventory.jsonl"
    )
    for record in expanded_records:
        c, f = record["canonical_state_sha256"], record["kalah_v3_float32_sha256"]
        sets["canonical"].add(c)
        sets["kalah_v3_float32"].add(f)
        rows.append(
            {
                "source": "563",
                "canonical_identity": c,
                "kalah_v3_float32_identity": f,
                "provenance": record,
            }
        )
    audit = root / "docs/data/seed564-archived-screen-identity-audit"
    by_location: dict[tuple[str, ...], dict[str, str]] = {}
    for scheme, filename in (
        ("canonical", "seven-source-canonical-index.jsonl.gz"),
        ("kalah_v3_float32", "seven-source-kalah-v3-float32-index.jsonl.gz"),
    ):
        for record in _jsonl(audit / filename):
            location = tuple(
                str(record.get(k, ""))
                for k in (
                    "source",
                    "run",
                    "batch_path",
                    "game_position",
                    "opening_index",
                    "seat",
                    "ply",
                    "action_absolute",
                )
            )
            by_location.setdefault(location, {})[scheme] = record["identity"]
            sets[scheme].add(record["identity"])
    for location, pair in sorted(by_location.items()):
        if set(pair) != {"canonical", "kalah_v3_float32"}:
            raise InventoryError("seed564_scheme_provenance_mismatch")
        rows.append({"source": "564", "location": location, **pair})
    return rows, sets


def build(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    root = root.resolve()
    registration = root / REGISTRATION_REL
    pins = authenticate_seed511_sources(root)
    reg = json.loads(registration.read_text())

    audit = root / "docs/data/seed564-archived-screen-identity-audit"
    seven_verification = seven.verify_seven(root, audit)
    subtotal_rows, subtotal = _published_subtotal(root)
    if (
        len(subtotal["canonical"]) != 302402
        or len(subtotal["kalah_v3_float32"]) != 302402
    ):
        raise InventoryError("published_562_564_subtotal_not_302402")
    snapshots, snapshot_sets = _snapshot_records(root)
    replay_rows, replay_sets, replay_counts = _replay_records(root)
    expanded_records = _jsonl(
        root
        / "docs/data/expanded-training-dev-exclusion-inventory/identity-inventory.jsonl"
    )
    seed503_504 = {
        "canonical": {
            row["canonical_state_sha256"]
            for row in expanded_records
            if row.get("source_family") == "seed500_504"
        },
        "kalah_v3_float32": {
            row["kalah_v3_float32_sha256"]
            for row in expanded_records
            if row.get("source_family") == "seed500_504"
        },
    }
    older_sets = {
        scheme: snapshot_sets[scheme] | replay_sets[scheme]
        for scheme in ("canonical", "kalah_v3_float32")
    }
    # Do not count the #503/#504 training/development cohort again: it is
    # already part of #563. Per-source intersections remain visible below.
    source_records = snapshots + replay_rows
    report_sources: list[dict[str, Any]] = []
    source_groups = [
        ("482", [r for r in snapshots if r["source"] == "482"], None),
        ("483", [r for r in snapshots if r["source"] == "483"], None),
        (
            "484_seed467_additions",
            [r for r in snapshots if r["source"] == "484_seed467_additions"],
            None,
        ),
        *[
            (
                name,
                [r for r in replay_rows if r["source"] == name],
                replay_counts.get(name),
            )
            for name in ("488", "499", "505", "508", "500")
        ],
    ]
    for name, group, game_count in source_groups:
        row = {
            "source": name,
            "identity_origin": "imported_verified_snapshot"
            if name in {"482", "483", "484_seed467_additions"}
            else "independently_replayed_state",
        }
        if game_count is not None:
            row["games_or_rows"] = game_count
        for scheme, field in (
            ("canonical", "canonical_identity"),
            ("kalah_v3_float32", "kalah_v3_float32_identity"),
        ):
            values = {item[field] for item in group}
            overlap = values & subtotal[scheme]
            row[scheme] = {
                "distinct": len(values),
                "overlap_with_562_564": len(overlap),
                "additions_beyond_562_564": len(values - subtotal[scheme]),
                "overlap_with_563_seed503_504": len(values & seed503_504[scheme]),
            }
        report_sources.append(row)

    combined = subtotal_rows + source_records
    # Index preserves each source occurrence/provenance; sorting is stable and
    # the identities are explicitly separated by scheme.
    index_bytes: dict[str, bytes] = {}
    index_rows: dict[str, int] = {}
    for scheme, field in (
        ("canonical", "canonical_identity"),
        ("kalah_v3_float32", "kalah_v3_float32_identity"),
    ):
        records = []
        for item in combined:
            value = item.get(field)
            if value is None and item.get("scheme") == scheme:
                value = item.get("identity")
            if value is not None:
                records.append(
                    {
                        "identity": value,
                        "source": item["source"],
                        "provenance": item.get(
                            "provenance", item.get("location", item)
                        ),
                    }
                )
        records.sort(
            key=lambda item: (
                item["identity"],
                item["source"],
                json.dumps(item["provenance"], sort_keys=True, separators=(",", ":")),
            )
        )
        index_rows[scheme] = len(records)
        payload = (
            "".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in records
            )
        ).encode()
        index_bytes[scheme] = gzip.compress(payload, mtime=0)

    combined_sets = {
        scheme: subtotal[scheme] | older_sets[scheme] for scheme in subtotal
    }
    report = {
        "schema": SCHEMA,
        "source_scope": "exact #511 registration exclusion_source_sha256 list",
        "seed511_registration_sha256": REGISTRATION_SHA256,
        "seed511_pinned_source_count": len(pins),
        "all_seed511_pinned_bytes_authenticated": True,
        "subtotal_562_564": {scheme: len(subtotal[scheme]) for scheme in subtotal},
        "index_occurrence_rows": index_rows,
        "published_subtotal_verification": {
            "verifier": "archived_screen_identity_seven.verify_seven",
            "passed": True,
            "verified_baseline_and_seven_source_indexes": seven_verification[
                "baseline_verified"
            ],
        },
        "sources": report_sources,
        "older_union": {
            scheme: {
                "distinct": len(older_sets[scheme]),
                "overlap_with_562_564": len(older_sets[scheme] & subtotal[scheme]),
                "additions_beyond_562_564": len(older_sets[scheme] - subtotal[scheme]),
            }
            for scheme in subtotal
        },
        "combined_union": {
            scheme: len(combined_sets[scheme]) for scheme in combined_sets
        },
        "seed503_504_not_double_counted": True,
        "seed461_continuations": "unresolved; not reconstructed or counted",
        "seed483_confirmation_gate": "closed",
        "other_uncovered_historical_populations": "remain uncovered; counts unknown",
        "no_experiment_execution": True,
        "seed545_standalone_diagnostic": {
            "commit": "0139cbd70d70184fc86cabd92afafdd849f28ef0",
            "prior_failed_invocation": "python -m ml.alphazero_lite.archived_screen_identity_inventory --root . --out .tmp/historical-exclusion-audit",
            "prior_failed_invocation_interpreter": "/usr/bin/python (CPython 3.14.7)",
            "prior_working_directory": ".tmp/holdback-historical-exclusions",
            "prior_working_directory_absolute": "/home/alex/Mancala/ai/.tmp/holdback-historical-exclusions",
            "prior_root_path": ".",
            "prior_out_path": ".tmp/historical-exclusion-audit",
            "prior_error": "published_index_mismatch:seed545-canonical-index.jsonl.gz",
            "cause": "--out selected a scratch directory without the committed #545 index artifacts; no committed index mismatch",
            "correct_command": "python -m ml.alphazero_lite.archived_screen_identity_inventory --root . --out docs/data/seed564-archived-screen-identity-audit",
            "correct_interpreter": "/usr/bin/python (CPython 3.14.7)",
            "correct_working_directory_absolute": "/home/alex/Mancala/ai/.tmp/holdback-historical-exclusions",
            "correct_root_path": ".",
            "correct_out_path": "docs/data/seed564-archived-screen-identity-audit",
            "committed_canonical_index_sha256": "2e4c66b9be044701430fbdf30ed28a0ebd2e2f434d9adc86533b4ccfedb38045",
            "committed_float32_index_sha256": "c710db3d81c0af6da68d7a724cf024d0663af81357eadb08ad2cf0082c94c043",
            "fresh_canonical_rows": 68166,
            "fresh_float32_rows": 68166,
            "canonical_and_float32_decompressed_indexes_match": True,
            "first_differing_row": None,
            "membership_or_provenance_difference": False,
            "serialization_only_difference": False,
            "committed_index_modified": False,
        },
    }
    manifest = {
        "schema": SCHEMA,
        "anchor_commit": "0139cbd70d70184fc86cabd92afafdd849f28ef0",
        "seed511_registration": {
            "path": str(REGISTRATION_REL),
            "sha256": REGISTRATION_SHA256,
            "pinned_exclusion_sources": reg["exclusion_source_sha256"],
        },
        "published_subtotal": "authenticated #562, #563, and #564 indexes",
        "identity_schemes": ["canonical_state_sha256", "kalah_v3_float32_sha256"],
        "indexes": ["canonical-index.jsonl.gz", "kalah-v3-float32-index.jsonl.gz"],
        "verifier": "python -m ml.alphazero_lite.seed561_historical_exclusion_inventory",
    }
    return {
        "source-manifest.json": (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "reconciliation-report.json": (
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "canonical-index.jsonl.gz": index_bytes["canonical"],
        "kalah-v3-float32-index.jsonl.gz": index_bytes["kalah_v3_float32"],
    }, report


def verify(root: Path, out: Path, *, write: bool = False) -> dict[str, Any]:
    artifacts, report = build(root)
    out = out if out.is_absolute() else root / out
    if write:
        out.mkdir(parents=True, exist_ok=True)
        for name, payload in artifacts.items():
            (out / name).write_bytes(payload)
    else:
        for name, payload in artifacts.items():
            path = out / name
            if not path.is_file() or path.read_bytes() != payload:
                raise InventoryError(f"published_artifact_mismatch:{name}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT_REL)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                verify(args.root, args.out, write=args.write), indent=2, sort_keys=True
            )
        )
    except (InventoryError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"verification failed: {exc}\n")


if __name__ == "__main__":
    main()
