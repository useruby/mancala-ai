"""Retrospective search-budget persistence diagnostic for seed398 SS/FF.

Register selection and execution identities before probes; then run probes and
analyze the hash-bound raw evidence with this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena, seed398_composition_diagnostic as composition
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-policy-value-composition"
OUT = ROOT / "docs/data/seed398-search-budget-persistence"
LEDGER = DATA / "validated-outcome-ledger.jsonl"
SUITE = DATA / "seed398-openings-v2.jsonl"
TREATMENTS = ("SS", "FS", "SF", "FF")
CHECKPOINTS = (384, 768, 1536)
OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def replay_state(row: dict[str, Any], ply: int) -> dict[str, Any]:
    game = KalahGame.from_state(composition.INITIAL_STATE)
    prefix = [int(move) for move in row["opening_prefix_moves"]]
    if composition.apply_opening_moves(game, prefix) != len(prefix):
        raise ValueError("opening_prefix_replay_incomplete")
    trajectory = [int(move) for move in row["trajectory"].split(",") if move]
    for move in trajectory[:ply]:
        if not game.move(move):  # recorded trajectory moves are absolute pit indices
            raise ValueError("trajectory_contains_illegal_move")
    return game.to_state()


def selection() -> list[dict[str, Any]]:
    ledger = jsonl(LEDGER)
    paired: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in ledger:
        if row["treatment"] in ("SS", "FF"):
            paired.setdefault(
                (int(row["opening_index"]), int(row["challenger_player"])), {}
            )[row["treatment"]] = row
    candidates: dict[int, dict[str, Any]] = {}
    for (opening, seat), pair in sorted(paired.items()):
        if set(pair) != {"SS", "FF"}:
            raise ValueError("missing_paired_trajectory")
        ss = [int(x) for x in pair["SS"]["trajectory"].split(",") if x]
        ff = [int(x) for x in pair["FF"]["trajectory"].split(",") if x]
        for ply, (left, right) in enumerate(zip(ss, ff)):
            if left == right:
                continue
            state = replay_state(pair["SS"], ply)
            game = KalahGame.from_state(state)
            if game.over() or sum(game.pits) <= 32 or game.current_player != seat:
                break
            legal = game.possible_moves()
            if left not in {game.pit_index(move) for move in legal} or right not in {
                game.pit_index(move) for move in legal
            }:
                raise ValueError("recorded_disagreement_action_illegal")
            state_hash = arena.canonical_game_state_hash(game)
            candidate = {
                "opening_index": opening,
                "challenger_player": seat,
                "game_within_opening": int(pair["SS"]["game_within_opening"]),
                "game_index": int(pair["SS"]["game_index"]),
                "ply": ply,
                "state_hash": state_hash,
                "state": state,
                "legal_actions": legal,
                "ss_action_384": left % 6,
                "ff_action_384": right % 6,
                "ledger_provenance": {
                    name: {
                        key: pair[name][key]
                        for key in (
                            "treatment",
                            "game_index",
                            "opening_index",
                            "challenger_player",
                            "game_within_opening",
                            "opening_state_hash",
                            "opening_prefix_moves",
                            "opening_applied_prefix_length",
                            "trajectory",
                            "first_move_challenger",
                            "first_move_current",
                            "winner",
                            "margin",
                        )
                    }
                    for name in ("SS", "FF")
                },
            }
            previous = candidates.get(opening)
            if previous is None or state_hash < previous["state_hash"]:
                candidates[opening] = candidate
            break
    ordered = sorted(
        candidates.values(),
        key=lambda row: (
            hashlib.sha256(
                f"seed398-budget-persistence-v1:{row['opening_index']}:{row['state_hash']}".encode()
            ).hexdigest(),
            row["opening_index"],
        ),
    )
    if len(ordered) < 64:
        raise ValueError(f"insufficient_distinct_openings:{len(ordered)}")
    return ordered[:64]


def register() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = composition.DATA
    selection_rows = selection()
    manifest = {
        "schema": "seed398-search-budget-persistence-registration-v1",
        "scope": "retrospective diagnostic states; not a fresh holdout",
        "publication_verifier": "ml.alphazero_lite.verify_seed398_publication.verify_publication",
        "publication_registration_sha256": sha256(evidence / "registration.json"),
        "publication_binding_sha256": sha256(evidence / "evaluation-binding.json"),
        "ledger_sha256": sha256(LEDGER),
        "suite_sha256": sha256(SUITE),
        "source_sha256": {
            "arena.py": sha256(ROOT / "ml/alphazero_lite/arena.py"),
            "self_play.py": sha256(ROOT / "ml/alphazero_lite/self_play.py"),
            "diagnostic.py": sha256(Path(__file__)),
        },
        "components": composition.component_bindings(),
        "treatments": composition.TREATMENTS,
        "search": {
            "simulations": 1536,
            "snapshots": list(CHECKPOINTS),
            "c_puct": 1.25,
            "options": OPTIONS,
            "exact_root_threshold": 16,
            "exact_leaf_solving": "disabled",
            "native_runtime_contract": "seed398 frozen root-16 contract",
        },
        "prefix_protocol": "For each treatment/state, one 1536-simulation PUCT run with the same seed derived from the original 384-search azlite_eval_seed_v2 context; deterministic prefix draws are captured at 384/768/1536. No reseeding or restart between snapshots.",
        "selection_rule": "first SS/FF differing action on paired opening+challenger-seat replay; require nonterminal, >32 stones, challenger to move, both moves legal; at most one per opening, lexicographically lowest canonical state hash; choose 64 by SHA256('seed398-budget-persistence-v1:<opening_index>:<state_hash>') ascending, opening index tie-break.",
        "classification_rule": {
            "budget_sensitive_disagreement": "SS and FF same selected move at 1536 in >=32/64 states",
            "persistent_model_disagreement": "otherwise, >=48/64 retain own 384 move at both 768 and 1536 and each has normalized top-two visit gap >=0.10 at 1536",
            "otherwise": "mixed_or_unresolved",
        },
        "decision_scope": "decision stability only; no move-quality, playing-strength, or promotion inference",
        "states": selection_rows,
    }
    path = OUT / "registration.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("immutable_registration_conflict")
    path.write_text(payload)
    return manifest


def action_stats(result: dict[str, Any], legal: list[int]) -> dict[str, Any]:
    stats = {int(row["move"]): row for row in result["child_stats"]}
    visits = {move: int(stats[move]["visits"]) for move in legal}
    ordered = sorted(visits.values(), reverse=True)
    total = sum(ordered)
    gap = (
        (ordered[0] - (ordered[1] if len(ordered) > 1 else 0)) / total if total else 0.0
    )
    return {
        "selected_action": result["selected_move"],
        "visits": {str(m): visits[m] for m in legal},
        "q_values": {str(m): float(stats[m]["q_value"]) for m in legal},
        "priors": {str(m): float(result["policy"][m]) for m in legal},
        "normalized_top_two_visit_gap": gap,
    }


def run() -> dict[str, Any]:
    manifest = json.loads((OUT / "registration.json").read_text())
    from ml.alphazero_lite.verify_seed398_publication import verify_publication

    verify_publication()
    expected_sources = manifest["source_sha256"]
    current_sources = {
        "arena.py": sha256(ROOT / "ml/alphazero_lite/arena.py"),
        "self_play.py": sha256(ROOT / "ml/alphazero_lite/self_play.py"),
        "diagnostic.py": sha256(Path(__file__)),
    }
    if current_sources != expected_sources:
        raise ValueError("execution_source_changed_after_registration")
    evaluators = {
        source: arena.ArtifactEvaluator(path)
        for source, path in composition.SOURCE_ARTIFACTS.items()
    }
    rows = []
    for state_row in manifest["states"]:
        state = state_row["state"]
        context = dict(
            contract_version="azlite_eval_seed_v2",
            base_seed=398,
            suite_sha256=sha256(SUITE),
            opening_index=state_row["opening_index"],
            opening_state_hash=state_row["ledger_provenance"]["SS"][
                "opening_state_hash"
            ],
            challenger_player=state_row["challenger_player"],
            game_within_opening=state_row["game_within_opening"],
            ply=state_row["ply"],
            canonical_current_state_hash=state_row["state_hash"],
            acting_role="challenger",
        )
        seed, context_hash = derive_search_seed(**context)
        for treatment, mix in composition.TREATMENTS.items():
            policy_ev = evaluators[mix["policy_source"]]
            value_ev = evaluators[mix["value_source"]]

            class Composed:
                def evaluate(self, game):
                    policy, _ = policy_ev.evaluate(game)
                    _, value = value_ev.evaluate(game)
                    return policy, value

            snapshots: list[dict[str, Any]] = []
            started = time.perf_counter()
            result = arena.evaluate_artifact_position(
                evaluator=Composed(),
                state=state,
                simulations=1536,
                seed=seed,
                c_puct=1.25,
                search_options=OPTIONS,
                exact_root_solve_threshold=16,
                root_snapshot_checkpoints=set(CHECKPOINTS),
            )
            elapsed = time.perf_counter() - started
            for snapshot in result.get("root_snapshots", []):
                count = int(snapshot["simulation"])
                if count in CHECKPOINTS:
                    visits = {
                        move: int(snapshot["visits"][move])
                        for move in state_row["legal_actions"]
                    }
                    ordered_visits = sorted(visits.values(), reverse=True)
                    total_visits = sum(ordered_visits)
                    gap = (
                        ordered_visits[0]
                        - (ordered_visits[1] if len(ordered_visits) > 1 else 0)
                    ) / total_visits
                    move_rows = {int(item["move"]): item for item in snapshot["moves"]}
                    snapshots.append(
                        {
                            "simulations": count,
                            "selected_action": snapshot["selected_move"],
                            "visits": {
                                str(m): visits[m] for m in state_row["legal_actions"]
                            },
                            "q_values": {
                                str(m): float(move_rows[m]["q_value"])
                                for m in state_row["legal_actions"]
                            },
                            "priors": {
                                str(m): float(move_rows[m]["prior"])
                                for m in state_row["legal_actions"]
                            },
                            "normalized_top_two_visit_gap": gap,
                            "elapsed_wall_seconds": elapsed,
                        }
                    )
            if [row["simulations"] for row in snapshots] != list(CHECKPOINTS):
                raise ValueError("root_snapshot_checkpoints_missing")
            standalone = arena.evaluate_artifact_position(
                evaluator=Composed(),
                state=state,
                simulations=384,
                seed=seed,
                c_puct=1.25,
                search_options=OPTIONS,
                exact_root_solve_threshold=16,
            )
            if (
                snapshots[0]["visits"]
                != {
                    str(move): int(standalone["visits"][move])
                    for move in state_row["legal_actions"]
                }
                or snapshots[0]["selected_action"] != standalone["selected_move"]
            ):
                raise ValueError(
                    f"snapshot_prefix_mismatch:{state_row['opening_index']}:{treatment}"
                )
            if treatment in ("SS", "FF"):
                original_action = state_row["ledger_provenance"][treatment][
                    "trajectory"
                ].split(",")[state_row["ply"]]
                if int(original_action) % 6 != standalone["selected_move"]:
                    raise ValueError(
                        f"recorded_384_action_mismatch:{state_row['opening_index']}:{treatment}"
                    )
            rows.append(
                {
                    "opening_index": state_row["opening_index"],
                    "state_hash": state_row["state_hash"],
                    "treatment": treatment,
                    "seed": seed,
                    "seed_context_hash": context_hash,
                    "final_action": result["selected_move"],
                    "snapshots": snapshots,
                    "elapsed_wall_seconds": elapsed,
                }
            )
    raw = {
        "schema": "seed398-search-budget-persistence-probes-v1",
        "registration_sha256": sha256(OUT / "registration.json"),
        "probes": rows,
    }
    path = OUT / "raw-probes.json"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return raw


def analyze(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = raw or json.loads((OUT / "raw-probes.json").read_text())
    registration = json.loads((OUT / "registration.json").read_text())

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(
        raw["registration_sha256"] == sha256(OUT / "registration.json"),
        "raw_registration_hash_mismatch",
    )
    grouped = {(row["opening_index"], row["treatment"]): row for row in raw["probes"]}
    require(len(grouped) == 256, "probe_count_mismatch")
    states = []
    for selected in registration["states"]:
        key = selected["opening_index"]
        treatment_rows = {name: grouped[(key, name)] for name in TREATMENTS}
        for record in treatment_rows.values():
            require(
                [snap["simulations"] for snap in record["snapshots"]]
                == list(CHECKPOINTS),
                "snapshot_checkpoints_mismatch",
            )
        states.append(
            {
                "opening_index": key,
                "ss_ff_same_1536": treatment_rows["SS"]["final_action"]
                == treatment_rows["FF"]["final_action"],
                "ss_retains_both": all(
                    next(
                        s
                        for s in treatment_rows["SS"]["snapshots"]
                        if s["simulations"] == b
                    )["selected_action"]
                    == selected["ss_action_384"]
                    for b in (768, 1536)
                ),
                "ff_retains_both": all(
                    next(
                        s
                        for s in treatment_rows["FF"]["snapshots"]
                        if s["simulations"] == b
                    )["selected_action"]
                    == selected["ff_action_384"]
                    for b in (768, 1536)
                ),
                "ss_gap_1536": treatment_rows["SS"]["snapshots"][-1][
                    "normalized_top_two_visit_gap"
                ],
                "ff_gap_1536": treatment_rows["FF"]["snapshots"][-1][
                    "normalized_top_two_visit_gap"
                ],
                "treatments": treatment_rows,
            }
        )
    same = sum(row["ss_ff_same_1536"] for row in states)
    persistent = sum(
        row["ss_retains_both"]
        and row["ff_retains_both"]
        and row["ss_gap_1536"] >= 0.1
        and row["ff_gap_1536"] >= 0.1
        for row in states
    )
    classification = (
        "budget_sensitive_disagreement"
        if same >= 32
        else "persistent_model_disagreement"
        if persistent >= 48
        else "mixed_or_unresolved"
    )
    summary = {
        "schema": "seed398-search-budget-persistence-summary-v1",
        "classification": classification,
        "sampled_states": len(states),
        "ss_ff_same_at_1536": same,
        "persistent_with_margin": persistent,
        "secondary_FS_SF": {
            name: {
                "same_as_SS_at_1536": sum(
                    grouped[(r["opening_index"], name)]["final_action"]
                    == grouped[(r["opening_index"], "SS")]["final_action"]
                    for r in states
                )
            }
            for name in ("FS", "SF")
        },
        "states": states,
        "interpretation": "Decision stability only; neither classification establishes move quality or model strength, or authorizes promotion.",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "run", "analyze", "verify"))
    args = parser.parse_args()
    if args.stage == "register":
        register()
    elif args.stage == "run":
        run()
    else:
        from ml.alphazero_lite.verify_seed398_publication import verify_publication

        verify_publication()
        if args.stage == "analyze":
            print(json.dumps(analyze(), indent=2))
        else:
            print(
                json.dumps(
                    {
                        "publication": "verified",
                        "registration_sha256": sha256(OUT / "registration.json"),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
