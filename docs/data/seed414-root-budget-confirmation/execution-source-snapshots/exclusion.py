"""Extend the verified seed398 exclusion proof through the #411 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.arena import apply_opening_moves
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed414-root-budget-confirmation"
BASE = ROOT / "docs/data/seed398-policy-value-composition/opening-exclusion-proof.json"
FALLBACK = (
    ROOT
    / "docs/data/seed398-paired-first-action/fallback-archive/outcomes.python-fallback.jsonl"
)
NEW_411 = ROOT / "docs/data/seed398-ff1536-confirmation/new-outcomes.jsonl"
PROBES_406 = ROOT / "docs/data/seed398-search-budget-persistence/raw-probes.json"
CONTINUATION_LEDGERS = (
    ROOT / "docs/data/seed398-paired-first-action/native-outcomes.jsonl",
    ROOT / "docs/data/seed398-e4-reference-diagnostic/outcomes.jsonl",
)
SOURCE_SUITES = (
    ROOT / "docs/data/seed398-policy-value-composition/seed398-openings-v2.jsonl",
    ROOT / "docs/data/seed398-search-budget-persistence/registration.json",
    ROOT / "docs/data/seed398-paired-first-action/registration.json",
    ROOT / "docs/data/seed398-e4-reference-diagnostic/registration.json",
    ROOT / "docs/data/seed398-ff1536-confirmation/registration.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_id(state: dict[str, Any]) -> str:
    return suites.canonical_key(state)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replay_rows(
    rows: list[dict[str, Any]],
    *,
    start_from_state: bool,
    starts: dict[int, dict[str, Any]] | None = None,
) -> set[str]:
    """Collect start and consumed state identities, failing on illegal replay."""
    identities: set[str] = set()
    for row_index, row in enumerate(rows):
        wrapped = row.get("outcome", row)
        if start_from_state:
            raw_state = wrapped.get("state") or row.get("state")
            if raw_state is None:
                raw_state = (starts or {}).get(int(row["opening_index"]))
            if raw_state is None:
                raise ValueError(f"missing_registered_start:{row_index}")
            game = KalahGame.from_state(raw_state)
            if wrapped.get("state_hash") != state_id(game.to_state()):
                raise ValueError(f"trajectory_start_identity_mismatch:{row_index}")
        else:
            game = KalahGame.from_state(suites.INITIAL_STATE)
            prefix = row.get("prefix_moves")
            if prefix is None:
                continue
            applied = apply_opening_moves(game, [int(move) for move in prefix])
            if applied != len(prefix):
                raise ValueError(f"truncated_prefix:{row_index}")
        identities.add(state_id(game.to_state()))
        trajectory = wrapped.get("trajectory", [])
        for move_index, item in enumerate(trajectory):
            if game.over() or int(item["actor"]) != game.current_player:
                raise ValueError(f"trajectory_actor_mismatch:{row_index}:{move_index}")
            action = int(item["action_relative"])
            if action not in game.possible_moves() or not game.move(
                game.pit_index(action)
            ):
                raise ValueError(f"illegal_trajectory:{row_index}:{move_index}")
            if "state" in item and item["state"] != game.to_state():
                raise ValueError(f"trajectory_state_mismatch:{row_index}:{move_index}")
            identities.add(state_id(game.to_state()))
        if start_from_state and not game.over():
            raise ValueError(f"consumed_trajectory_not_terminal:{row_index}")
        if start_from_state and "score" in wrapped:
            root_player = int(wrapped["root_player"])
            margin = (
                game.captured_seeds[root_player] - game.captured_seeds[1 - root_player]
            )
            score = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            if wrapped["stores"] != game.captured_seeds or wrapped["score"] != score:
                raise ValueError(f"trajectory_terminal_accounting_mismatch:{row_index}")
    return identities


def build_extension() -> dict[str, Any]:
    base = json.loads(BASE.read_text())
    if base.get("schema") != "seed398-complete-opening-exclusion-proof-v1":
        raise ValueError("historical_proof_schema_mismatch")
    declared = set(base["declared_state_identities"])
    consumed = set(base["actual_state_identities"])
    sources: list[dict[str, Any]] = [
        {
            "path": str(BASE.relative_to(ROOT)),
            "sha256": sha(BASE),
            "role": "verified_historical_proof",
        }
    ]
    added_declared: set[str] = set()
    added_consumed: set[str] = set()
    for path in SOURCE_SUITES:
        if not path.is_file():
            raise FileNotFoundError(f"missing_declared_source:{path}")
        if path.suffix == ".jsonl":
            rows = load_jsonl(path)
            ids = {state_id(row["state"]) for row in rows}
            declared.update(ids)
            added_declared.update(ids)
            added_consumed.update(replay_rows(rows, start_from_state=False))
        else:
            reg = json.loads(path.read_text())
            rows = reg.get("states", [])
            ids = {state_id(row["state"]) for row in rows if "state" in row}
            for action_row in reg.get("action_mapping", []):
                if action_row.get("state_hash"):
                    ids.add(str(action_row["state_hash"]))
            declared.update(ids)
            added_declared.update(ids)
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
                "kind": "declared_suite"
                if path.suffix == ".jsonl"
                else "registered_states",
                "identity_count": len(ids),
            }
        )
    probe_registration_path = (
        ROOT / "docs/data/seed398-search-budget-persistence/registration.json"
    )
    probe_registration = json.loads(probe_registration_path.read_text())
    probes = json.loads(PROBES_406.read_text())
    if probes.get("registration_sha256") != sha(probe_registration_path):
        raise ValueError("registered_probe_registration_binding_mismatch")
    probe_state_ids = {str(row["state_hash"]) for row in probes["probes"]}
    registered_probe_ids = {
        str(row["state_hash"]) for row in probe_registration["states"]
    }
    if not probe_state_ids <= registered_probe_ids:
        raise ValueError("probe_states_not_covered_by_registration")
    declared.update(probe_state_ids)
    added_declared.update(probe_state_ids)
    sources.append(
        {
            "path": str(PROBES_406.relative_to(ROOT)),
            "sha256": sha(PROBES_406),
            "kind": "registered_probe_states",
            "probe_count": len(probes["probes"]),
            "unique_state_count": len(probe_state_ids),
        }
    )
    state_registration = json.loads(
        (ROOT / "docs/data/seed398-paired-first-action/registration.json").read_text()
    )
    starts = {
        int(row["opening_index"]): row["state"] for row in state_registration["states"]
    }
    trajectory_paths = (FALLBACK, *CONTINUATION_LEDGERS, NEW_411)
    for path in trajectory_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing_consumed_trajectory_source:{path}")
        rows = load_jsonl(path)
        ids = replay_rows(rows, start_from_state=True, starts=starts)
        consumed.update(ids)
        added_consumed.update(ids)
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
                "kind": "replayed_consumed_trajectories",
                "row_count": len(rows),
                "unique_state_count": len(ids),
            }
        )
    union = declared | consumed
    result = {
        "schema": "seed414-extended-opening-exclusion-proof-v1",
        "historical_proof_sha256": sha(BASE),
        "sources": sources,
        "historical_declared_count": len(base["declared_state_identities"]),
        "historical_consumed_count": len(base["actual_state_identities"]),
        "additional_declared_unique_count": len(
            added_declared - set(base["declared_state_identities"])
        ),
        "additional_consumed_unique_count": len(
            added_consumed - set(base["actual_state_identities"])
        ),
        "declared_identity_count": len(declared),
        "consumed_identity_count": len(consumed),
        "union_identity_count": len(union),
        "declared_state_identities": sorted(declared),
        "consumed_state_identities": sorted(consumed),
        "excluded_state_identities": sorted(union),
        "excluded_identity_sha256": hashlib.sha256(
            "\n".join(sorted(union)).encode()
        ).hexdigest(),
        "deduplication": {
            "declared_consumed_intersection": len(declared & consumed),
            "declared_plus_consumed_minus_intersection": len(declared)
            + len(consumed)
            - len(declared & consumed),
            "union_reconciles": len(union)
            == len(declared) + len(consumed) - len(declared & consumed),
        },
    }
    if not result["deduplication"]["union_reconciles"]:
        raise ValueError("exclusion_union_reconciliation_failed")
    return result


def publish() -> dict[str, Any]:
    record = build_extension()
    target = OUT / "opening-exclusion-proof.json"
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if (
        target.exists()
        and target.read_text() != payload
        and json.loads(target.read_text()).get("schema")
        != "seed414-extended-opening-exclusion-proof-v1"
    ):
        raise ValueError("exclusion_proof_is_immutable")
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(payload)
    return record


if __name__ == "__main__":
    proof = publish()
    print(
        json.dumps(
            {key: value for key, value in proof.items() if "identities" not in key},
            indent=2,
        )
    )
