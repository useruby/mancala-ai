"""Read-only replay and identity census for authenticated #556 and #560."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game
from ml.alphazero_lite.verify_seed560_paired_first_action import validate_continuation

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/data"
S556 = BASE / "seed556-exploratory-arena/screen-0005"
S560 = BASE / "seed560-paired-first-action/run-0004"
OUT = BASE / "seed556-560-exclusion-inventory"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def validate_alias_records(arms, roots):
    """Validate every arm then reject aliases whose complete outcomes differ."""
    aliases = {}
    for arm in arms:
        key = (int(arm["opening_index"]), int(arm["action"]))
        if key[0] not in roots:
            raise ValueError("560_root_missing")
        validate_continuation(roots[key[0]], arm, arm["outcome"])
        prior = aliases.get(key)
        if prior is not None and prior != arm["outcome"]:
            raise ValueError("560_alias_conflicting_outcomes")
        aliases[key] = arm["outcome"]
    return aliases


def census(data556=S556, data560=S560):
    # Verify committed #560 freeze-controlled input bytes before interpreting them.
    freeze_path = data560 / "execution-freeze.json"
    if (
        digest(freeze_path)
        != "fb47912171f8d3ba07353654b6beb2eae5227f4aee28315799921413a9975edc"
    ):
        raise ValueError("560_freeze_bytes_mismatch")
    freeze = json.loads(freeze_path.read_text())
    cohort_path, continuation_path = (
        data560 / "cohort-manifest.json",
        data560 / "continuations.jsonl",
    )
    if (
        digest(cohort_path) != freeze["cohort_sha256"]
        or freeze.get("cohort_sha256")
        != "fe24d68aea9a158cb441aa886a60adc26631805c5b1213f14fceb4aa4ce4a254"
    ):
        raise ValueError("560_cohort_bytes_mismatch")
    if (
        digest(continuation_path)
        != "5c47e4d161077f3b438d737887da35b73ca3dddab7d60db22fed19bd6dd3d8a3"
    ):
        raise ValueError("560_continuation_bytes_mismatch")
    receipt = json.loads((data556 / "verification-receipt.json").read_text())
    if (
        digest(data556 / "verification-receipt.json")
        != "a318933da72b8ed23cb5487fbbd8e57edd0d1d407ba9bc1f8485b0736fcdfbe7"
    ):
        raise ValueError("556_verification_receipt_identity_mismatch")
    relevant = [
        "registration.json",
        "execution-freeze.json",
        "openings.jsonl",
        "outcomes.jsonl",
    ]
    relevant += [f"batches/games-{i:04d}.jsonl" for i in range(1024)]
    for relative in relevant:
        expected = receipt["evidence_sha256"].get(relative)
        if expected is None:
            raise ValueError(f"556_receipt_binding_missing:{relative}")
        path = data556 / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"556_receipt_bytes_mismatch:{relative}")
    cohort = json.loads(cohort_path.read_text())["states"]
    arms = jsonl(continuation_path)
    if len(arms) != 512:
        raise ValueError("560_arm_count_mismatch")
    roots = {i: r for i, r in enumerate(cohort)}
    # Validate every arm before identifying physical continuations.
    aliases = {}
    ids560 = {"canonical": set(), "kalah_v3_float32": set()}
    provenance560 = {"canonical": {}, "kalah_v3_float32": {}}

    def add560(state, index, action, phase):
        for scheme, identity in (
            ("canonical", canonical_identity(state)),
            ("kalah_v3_float32", production_float32_identity(state)),
        ):
            ids560[scheme].add(identity)
            provenance560[scheme].setdefault(
                identity,
                {
                    "opening_index": index,
                    "forced_action": action,
                    "phase": phase,
                    "state_sha256": canonical_identity(state),
                },
            )

    moves = 0
    aliases = validate_alias_records(arms, roots)
    for (index, action), outcome in aliases.items():
        game = KalahGame.from_state(roots[index]["state"])
        if action not in game.possible_moves() or not game.move(game.pit_index(action)):
            raise ValueError("560_forced_move_illegal")
        # Include forced child and every reached state through terminal.
        state = game.to_state()
        add560(state, index, action, "forced_child")
        row = next(
            r
            for r in arms
            if (int(r["opening_index"]), int(r["action"])) == (index, action)
        )
        for step in row["outcome"]["trajectory"]:
            move = int(step["action"])
            if move not in game.possible_moves() or not game.move(game.pit_index(move)):
                raise ValueError("560_action_illegal")
            state = game.to_state()
            if state != step["state_after"]:
                raise ValueError("560_state_mismatch")
            add560(state, index, action, f"post_move_{step['ply']}")
            moves += 1
    # Replay registered 1,024 opening prefixes and all 2,048 game streams.
    opening_records = jsonl(S556 / "openings.jsonl")
    if len(opening_records) != 1024:
        raise ValueError("556_opening_count_mismatch")
    ids556 = {"canonical": set(), "kalah_v3_float32": set()}
    provenance556 = {"canonical": {}, "kalah_v3_float32": {}}

    def add556(state, index, seat, ply, phase):
        for scheme, identity in (
            ("canonical", canonical_identity(state)),
            ("kalah_v3_float32", production_float32_identity(state)),
        ):
            ids556[scheme].add(identity)
            provenance556[scheme].setdefault(
                identity,
                {
                    "opening_index": index,
                    "game_position": seat,
                    "ply": ply,
                    "phase": phase,
                    "state_sha256": canonical_identity(state),
                },
            )

    for i, opening in enumerate(opening_records):
        initial = {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        game = KalahGame.from_state(initial)
        for action in opening["prefix_moves"]:
            if (
                game.over()
                or int(action) not in game.possible_moves()
                or not game.move(game.pit_index(int(action)))
            ):
                raise ValueError(f"556_prefix_illegal:{i}")
        if game.to_state() != opening["state"]:
            raise ValueError(f"556_prefix_state_mismatch:{i}")
        add556(game.to_state(), i, None, None, "opening_root")
        for seat in (0, 1):
            path = S556 / "batches" / f"games-{i:04d}.jsonl"
            rows = jsonl(path)
            if len(rows) != 2:
                raise ValueError(f"556_pair_size:{i}")
            row = rows[seat]
            _replay_game(opening, row, seat)
            # Reconstruct full trajectory independently, indexing source files globally.
            state = KalahGame.from_state(initial)
            for action in opening["prefix_moves"]:
                state.move(state.pit_index(int(action)))
            moves_abs = [int(x) for x in row["trajectory"].split(",") if x]
            for move_index, absolute in enumerate(moves_abs):
                s = state.to_state()
                add556(s, i, seat, move_index, "pre_move")
                move = absolute - state.current_player * 6
                if (
                    state.over()
                    or move not in state.possible_moves()
                    or not state.move(state.pit_index(move))
                ):
                    raise ValueError(f"556_game_illegal:{i}:{seat}")
                s = state.to_state()
                add556(s, i, seat, move_index, "post_move")
    report = {}
    for scheme in ids556:
        inter = ids556[scheme] & ids560[scheme]
        report[scheme] = {
            "source_556": len(ids556[scheme]),
            "source_560": len(ids560[scheme]),
            "intersection": len(inter),
            "unique_556": len(ids556[scheme] - ids560[scheme]),
            "unique_560": len(ids560[scheme] - ids556[scheme]),
            "union": len(ids556[scheme] | ids560[scheme]),
        }
    # Check published diagnostic-root cohorts against the replayed #556 population.
    subset_counts = {}
    for name, path in (
        ("557", BASE / "seed557-retrospective-diagnostic/cohort.json"),
        (
            "559",
            BASE / "seed559-fixed-position-feasibility/run-0003/cohort-manifest.json",
        ),
        ("560", data560 / "cohort-manifest.json"),
    ):
        doc = json.loads(path.read_text())
        rows = doc.get("states", doc if isinstance(doc, list) else [])
        for row in rows:
            state = row.get("state")
            if state is None:
                continue
            if (
                canonical_identity(state) not in ids556["canonical"]
                or production_float32_identity(state) not in ids556["kalah_v3_float32"]
            ):
                raise ValueError(f"{name}_root_outside_556_population")
        subset_counts[name] = len(rows)
    cohort557 = BASE / "seed557-retrospective-diagnostic/cohort.json"
    cohort558 = BASE / "seed558-value-only-swap/cohort.json"
    if cohort557.read_bytes() != cohort558.read_bytes():
        raise ValueError("558_cohort_not_exact_557_reuse")
    return (
        report,
        ids556,
        ids560,
        {
            "arms": len(arms),
            "physical_continuations": len(aliases),
            "actions_physical": moves,
            "actions_arm_records": sum(len(r["outcome"]["trajectory"]) for r in arms),
            "identities_556": {k: len(v) for k, v in ids556.items()},
            "identities_560": {k: len(v) for k, v in ids560.items()},
            "declared_root_subset_counts": subset_counts,
            "558_exact_reuse": True,
        },
        {"556": provenance556, "560": provenance560},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report, a, b, observed, provenance = census()
    inventory_rows = [
        {
            "source": source,
            "scheme": scheme,
            "identity": identity,
            "provenance": location,
        }
        for source in ("556", "560")
        for scheme in ("canonical", "kalah_v3_float32")
        for identity, location in sorted(provenance[source][scheme].items())
    ]
    inventory_bytes = (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in inventory_rows
        )
        + "\n"
    ).encode()
    report_bytes = (
        json.dumps(
            {
                "scope": "verified #556–#560 only",
                "observed": observed,
                "overlap": report,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    manifest = {
        "scope": "#556–#560 only",
        "anchor_commit": "2e92bbfd3f90c9517c4c4449d3bebd750f22ee71",
        "sources": {
            "556": {
                "merged_commit": "f38838e6ffe20a44c1e8c161a9daed0f1243f0ae",
                "receipt_sha256": digest(S556 / "verification-receipt.json"),
                "registration_sha256": digest(S556 / "registration.json"),
                "opening_sha256": digest(S556 / "openings.jsonl"),
                "receipt_bound_evidence_files": json.loads(
                    (S556 / "verification-receipt.json").read_text()
                )["evidence_sha256"],
            },
            "560": {
                "cohort_sha256": digest(S560 / "cohort-manifest.json"),
                "freeze_sha256": digest(S560 / "execution-freeze.json"),
                "continuations_sha256": digest(S560 / "continuations.jsonl"),
                "merged_commit": "8bf3d551c678066f0863002e1cba295c8834689b",
            },
        },
        "identity_schemes": [
            "canonical_state_sha256",
            "kalah_v3_little_endian_float32_sha256",
        ],
        "verifier": "python -m ml.alphazero_lite.seed556_560_exclusion_inventory",
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "identity-inventory.json").write_bytes(inventory_bytes)
        (OUT / "count-report.json").write_bytes(report_bytes)
        (OUT / "source-manifest.json").write_bytes(manifest_bytes)
    else:
        expected = {
            "identity-inventory.json": inventory_bytes,
            "count-report.json": report_bytes,
            "source-manifest.json": manifest_bytes,
        }
        for filename, contents in expected.items():
            if (
                not (OUT / filename).is_file()
                or (OUT / filename).read_bytes() != contents
            ):
                raise ValueError(f"published_inventory_verification_failed:{filename}")
    print(
        json.dumps({"observed": observed, "overlap": report}, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
