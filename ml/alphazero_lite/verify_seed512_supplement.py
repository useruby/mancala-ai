"""Append-only semantic and provenance checks for the seed512 publication."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from ml.alphazero_lite import seed512_evaluation as protocol  # noqa: E402
from ml.alphazero_lite.verify_seed512_evaluation import (  # noqa: E402
    independent_analysis,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_ledger(ledger: list[dict[str, Any]]) -> None:
    """Require exactly one record for each opening/seat and each raw position."""
    if len(ledger) != 2048:
        raise ValueError("seed512_supplement_ledger_count")
    seen: set[tuple[int, int]] = set()
    positions: dict[int, set[int]] = {i: set() for i in range(1024)}
    for row in ledger:
        opening = int(row["opening_index"])
        seat = int(row["challenger_player"])
        position = int(row["batch_game_position"])
        if opening not in range(1024):
            raise ValueError("seed512_supplement_opening_range")
        if seat not in {0, 1}:
            raise ValueError(f"seed512_supplement_seat_range:{opening}")
        if position not in {0, 1}:
            raise ValueError(f"seed512_supplement_position_range:{opening}")
        key = opening, seat
        if key in seen:
            raise ValueError(f"seed512_supplement_duplicate_seat:{opening}:{seat}")
        seen.add(key)
        if position in positions[opening]:
            raise ValueError(
                f"seed512_supplement_duplicate_position:{opening}:{position}"
            )
        positions[opening].add(position)
    expected = {(i, seat) for i in range(1024) for seat in (0, 1)}
    if seen != expected or any(v != {0, 1} for v in positions.values()):
        raise ValueError("seed512_supplement_pair_coverage")


def audit_assets(root: Path) -> dict[str, Any]:
    """Check checkpoint/artifact/runtime identities against seed511 bindings."""
    reg = json.loads((root / protocol.OUT / "registration.json").read_text())
    prior = json.loads((root / protocol.SEED511_OUT / "registration.json").read_text())
    checked: dict[str, str] = {}
    gaps: list[str] = []
    for name, component in reg["components"].items():
        artifact = root / component["artifact"]
        checkpoint = root / component["checkpoint_source"]
        if checkpoint.exists():
            if sha(checkpoint) != component["checkpoint_sha256"]:
                raise ValueError(f"seed512_supplement_checkpoint_hash:{name}")
            checked[f"checkpoint:{name}"] = sha(checkpoint)
        else:
            gaps.append(f"checkpoint:{component['checkpoint_source']}")
        for rel, digest in component["files"].items():
            path = artifact / rel
            if not path.exists():
                gaps.append(f"artifact:{path.relative_to(root)}")
            elif sha(path) != digest:
                raise ValueError(f"seed512_supplement_artifact_hash:{name}:{rel}")
            else:
                checked[f"artifact:{name}:{rel}"] = digest
    for key in ("exact_root_native_probe", "exact_root_tablebase"):
        asset = prior["runtime_assets"][key]
        path = root / asset["artifact"]
        if path.exists():
            if sha(path) != asset["sha256"]:
                raise ValueError(f"seed512_supplement_runtime_asset_hash:{key}")
            checked[key] = asset["sha256"]
        else:
            gaps.append(f"runtime_asset:{asset['artifact']}")
    return {"checked": checked, "unresolved_identity_paths": gaps}


def require_result_match(computed: dict[str, Any], published: dict[str, Any]) -> None:
    if computed != published:
        raise ValueError("seed512_supplement_result_mismatch")


def verify(root: Path = protocol.ROOT) -> dict[str, Any]:
    root = root.resolve()
    protocol.preflight(root, allow_started=True)
    out = root / protocol.OUT
    ledger = protocol.read_rows(out / "outcomes.jsonl")
    openings = protocol.read_rows(out / "openings.jsonl")
    audit_ledger(ledger)
    for i, opening in enumerate(openings):
        rows = [r for r in ledger if int(r["opening_index"]) == i]
        protocol.verify_pair(root, i, opening, rows)
        games = protocol.read_rows(out / "batches" / f"games-{i:04d}.jsonl")
        if {int(r["batch_game_position"]) for r in rows} != {0, 1}:
            raise ValueError(f"seed512_supplement_raw_positions:{i}")
        for row in rows:
            pos = int(row["batch_game_position"])
            if int(games[pos]["challenger_player"]) != int(row["challenger_player"]):
                raise ValueError(f"seed512_supplement_raw_assignment:{i}:{pos}")
    computed = independent_analysis(root, ledger)
    require_result_match(computed, json.loads((out / "results.json").read_text()))
    return {
        "status": "verified_with_provenance_gaps"
        if audit_assets(root)["unresolved_identity_paths"]
        else "verified_complete",
        "games_replayed": len(ledger),
        "results": computed,
        "assets": audit_assets(root),
        "imported_execution_bindings": {
            "source": "seed511 recorded execution bindings",
            "limitations": [
                "seed512 registration binds seed512 protocol files and seed511 runtime identities, but does not freeze every imported Python dependency source file individually"
            ],
            "retrospective_freeze_created": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(verify(REPO_ROOT), indent=2, sort_keys=True))
