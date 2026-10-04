"""Read-only, model-artifact-independent verifier for seed414 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import seed414_exclusion_proof as exclusion
from ml.alphazero_lite import seed414_root_budget_analysis as analysis
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed414_root_budget_confirmation import (
    REFERENCES,
    ROOT,
    sha,
)

DATA = ROOT / "docs/data/seed414-root-budget-confirmation"


def verify() -> dict[str, Any]:
    paths = {
        "suite": DATA / "suite.jsonl",
        "registration": DATA / "registration.json",
        "proof": DATA / "opening-exclusion-proof.json",
        "probes": DATA / "root-probes.jsonl",
        "trajectories": DATA / "continuations.jsonl",
        "aliases": DATA / "aliases.jsonl",
        "analysis": DATA / "analysis.json",
        "matrix": DATA / "paired-matrix.json",
        "results": DATA / "results.md",
        "binding": DATA / "publication-binding.json",
    }
    before = {key: path.read_bytes() for key, path in paths.items()}
    reg = _json(paths["registration"])
    states = _jsonl(paths["suite"])
    proof = _json(paths["proof"])
    require(
        reg["schema"] == "seed414-root-budget-confirmation-registration-v1",
        "registration_schema",
    )
    require(sha(paths["suite"]) == reg["suite_sha256"], "suite_binding")
    require(sha(paths["proof"]) == reg["exclusion_proof_sha256"], "proof_binding")
    registration_hash = sha(paths["registration"])
    require(reg["source_sha256"] == _execution_sources(), "execution_source_binding")
    require(exclusion.build_extension() == proof, "exclusion_proof_reconciliation")
    require(len(states) == 128 == reg["suite_count"], "suite_count")
    suites.validate_arena_entries(states)
    identity_list = [_state_id(row["state"]) for row in states]
    require(len(set(identity_list)) == 128, "suite_unique_states")
    require(
        not (set(identity_list) & set(proof["excluded_state_identities"])),
        "suite_exclusion_overlap",
    )
    require(
        all(sum(KalahGame.from_state(row["state"]).pits) > 32 for row in states),
        "suite_stone_floor",
    )
    probes = _jsonl(paths["probes"])
    trajectories = _jsonl(paths["trajectories"])
    aliases = _jsonl(paths["aliases"])
    state_by_index = {row["opening_index"]: row for row in states}
    require(len(probes) == 128, "probe_count")
    for probe in probes:
        state = state_by_index[probe["opening_index"]]
        require(probe["state_hash"] == state["state_hash"], "probe_state_identity")
        require(
            probe["registration_sha256"] == registration_hash,
            "probe_registration_binding",
        )
        snaps = probe["snapshots"]
        require(set(snaps) == {"384", "1536"}, "prefix_snapshot_coverage")
        require(snaps["384"]["action"] in state["legal_actions"], "illegal_384_action")
        require(
            snaps["1536"]["action"] in state["legal_actions"], "illegal_1536_action"
        )
        require(
            sum(map(int, snaps["384"]["visits"].values())) == 384,
            "prefix_384_visit_total",
        )
        require(
            sum(map(int, snaps["1536"]["visits"].values())) == 1536,
            "prefix_1536_visit_total",
        )
        require(
            set(snaps["384"]["visits"])
            == {str(move) for move in state["legal_actions"]}
            and set(snaps["1536"]["visits"])
            == {str(move) for move in state["legal_actions"]},
            "snapshot_legal_move_coverage",
        )
        require(
            all(
                int(snaps["384"]["visits"][move]) <= int(snaps["1536"]["visits"][move])
                for move in snaps["384"]["visits"]
            ),
            "prefix_visit_monotonicity",
        )
        for budget in ("384", "1536"):
            entries = {int(item["move"]): item for item in snaps[budget]["moves"]}
            winner = max(
                state["legal_actions"],
                key=lambda move: (
                    int(snaps[budget]["visits"][str(move)]),
                    float(entries[move]["q_value"]),
                    float(entries[move]["prior"]),
                    -move,
                ),
            )
            require(winner == snaps[budget]["action"], f"snapshot_selection:{budget}")
        _verify_root_seed(reg, state, probe)
    trajectory_by_identity = {}
    for row in trajectories:
        key = (row["opening_index"], row["reference"], row["forced_action"])
        require(key not in trajectory_by_identity, "duplicate_trajectory")
        require(
            row["registration_sha256"] == registration_hash,
            "trajectory_registration_binding",
        )
        _replay(
            row["outcome"],
            state_by_index[row["opening_index"]],
            reg,
            int(row["forced_action"]),
        )
        trajectory_by_identity[key] = row
    require(len(aliases) == 512, "logical_case_count")
    alias_keys = set()
    for alias in aliases:
        key = (alias["opening_index"], alias["reference"], alias["root_budget"])
        require(key not in alias_keys, "duplicate_alias")
        alias_keys.add(key)
        require(alias["reference"] in REFERENCES, "unknown_reference")
        require(
            alias["registration_sha256"] == registration_hash,
            "alias_registration_binding",
        )
        target = trajectory_by_identity.get(
            (alias["opening_index"], alias["reference"], alias["action"])
        )
        require(target is not None, "alias_target_missing")
        identity = hashlib.sha256(
            json.dumps(target["outcome"], sort_keys=True).encode()
        ).hexdigest()
        require(identity == alias["trajectory_identity"], "alias_trajectory_identity")
        if alias["alias_of"] is not None:
            source_key = (
                alias["opening_index"],
                alias["reference"],
                alias["alias_of"]["root_budget"],
            )
            source_alias = next(
                (
                    item
                    for item in aliases
                    if (item["opening_index"], item["reference"], item["root_budget"])
                    == source_key
                ),
                None,
            )
            require(source_alias is not None, "alias_source_missing")
            require(source_alias["action"] == alias["action"], "alias_source_action")
            require(
                source_alias["trajectory_identity"] == alias["trajectory_identity"],
                "alias_source_identity",
            )
    require(len(trajectory_by_identity) <= 512, "physical_trajectory_count")
    report, matrix = analysis.calculate(reg, probes, trajectories, aliases)
    require(_json(paths["analysis"]) == report, "analysis_reconciliation")
    require(_json(paths["matrix"]) == matrix, "matrix_reconciliation")
    binding = _json(paths["binding"])
    require(
        binding["hashes"]
        == {
            k: hashlib.sha256(v).hexdigest()
            for k, v in before.items()
            if k != "binding"
        },
        "publication_hashes",
    )
    require(
        binding["source_snapshot_hashes"] == reg["source_sha256"],
        "source_snapshot_hashes",
    )
    snapshot_dir = DATA / "execution-source-snapshots"
    for name, digest in reg["source_sha256"].items():
        require(sha(snapshot_dir / f"{name}.py") == digest, f"source_snapshot:{name}")
    after = {key: path.read_bytes() for key, path in paths.items()}
    require(before == after, "verifier_mutated_evidence")
    return {
        "status": "verified",
        "states": len(states),
        "logical_cases": len(aliases),
        "physical_continuations": len(trajectories),
        "decision": report["decision"],
        "mean_gain": report["primary"]["mean_gain"],
        "interval": report["primary"]["bootstrap_95_interval"],
    }


def _execution_sources() -> dict[str, str]:
    runner = ROOT / "ml/alphazero_lite/seed414_root_budget_confirmation.py"
    paths = {
        "runner": runner,
        "analysis": ROOT / "ml/alphazero_lite/seed414_root_budget_analysis.py",
        "verifier": Path(__file__),
        "exclusion": ROOT / "ml/alphazero_lite/seed414_exclusion_proof.py",
        "publisher": ROOT
        / "ml/alphazero_lite/publish_seed414_root_budget_confirmation.py",
        "arena": ROOT / "ml/alphazero_lite/arena.py",
        "rules": ROOT / "ml/alphazero_lite/kalah_rules.py",
        "seed_contract": ROOT / "ml/alphazero_lite/evaluation_seed_contract.py",
        "native_adapter": ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py",
        "runtime_search_policy": ROOT / "ml/alphazero_lite/runtime_search_policy.py",
        "opening_suite": ROOT / "ml/alphazero_lite/build_opening_suite.py",
        "population": ROOT / "ml/alphazero_lite/seed461_order_population.py",
    }
    return {name: sha(path) for name, path in paths.items()}


def _verify_root_seed(
    reg: dict[str, Any], state: dict[str, Any], probe: dict[str, Any]
) -> None:
    game = KalahGame.from_state(state["state"])
    seed, context_hash = derive_search_seed(
        contract_version="azlite_eval_seed_v2",
        base_seed=414,
        suite_sha256=reg["suite_sha256"],
        opening_index=state["opening_index"],
        opening_state_hash=state["state_hash"],
        challenger_player=game.current_player,
        game_within_opening=0,
        ply=0,
        canonical_current_state_hash=_canonical_game_hash(game),
        acting_role="challenger",
    )
    require(
        (seed, context_hash) == (probe["seed"], probe["seed_context_hash"]),
        "root_seed_contract",
    )


def _replay(
    outcome: dict[str, Any],
    state: dict[str, Any],
    reg: dict[str, Any],
    forced_action: int,
) -> None:
    game = KalahGame.from_state(state["state"])
    root = game.current_player
    require(outcome["state_hash"] == state["state_hash"], "trajectory_state_hash")
    require(bool(outcome["trajectory"]), "trajectory_empty")
    require(outcome["trajectory"][0]["forced"], "forced_action_missing")
    require(
        int(outcome["trajectory"][0]["action_relative"]) == forced_action,
        "forced_action_identity",
    )
    for index, move in enumerate(outcome["trajectory"]):
        require(
            not game.over() and game.current_player == move["actor"], "trajectory_actor"
        )
        action = int(move["action_relative"])
        require(action in game.possible_moves(), "trajectory_legal_action")
        require(
            int(move["action_absolute"]) == game.pit_index(action),
            "trajectory_absolute_action",
        )
        before = sum(game.pits)
        require(game.move(game.pit_index(action)), "trajectory_move")
        require(game.to_state() == move["state"], "trajectory_exact_state")
        if index:
            require(move["simulations"] == 1536, "continuation_simulations")
            require(move["c_puct"] == 1.25, "continuation_cpuct")
            require(
                move["search_options"]
                == {
                    "fpu_mode": "zero",
                    "reuse_subtree": False,
                    "normalize_values": False,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.0,
                    "root_temperature": 0.0,
                },
                "continuation_search_options",
            )
            before_game = KalahGame.from_state(
                outcome["trajectory"][index - 1]["state"]
            )
            seed, context_hash = derive_search_seed(
                contract_version="azlite_eval_seed_v2",
                base_seed=414,
                suite_sha256=reg["suite_sha256"],
                opening_index=state["opening_index"],
                opening_state_hash=state["state_hash"],
                challenger_player=root,
                game_within_opening=0,
                ply=index,
                canonical_current_state_hash=_canonical_game_hash(before_game),
                acting_role="challenger" if move["actor"] == root else "current",
            )
            require(
                (seed, context_hash) == (move["seed"], move["seed_context_hash"]),
                "continuation_seed_contract",
            )
            require(
                move["active_pit_stones_before_move"] == before,
                "trajectory_stone_count",
            )
            exact = before <= 16
            require(
                move["exact_root_backend"]
                == ("native_kvtb_root_action_probe_v1" if exact else None),
                "root16_backend",
            )
            require(
                move["exact_root_solver_calls"] == (1 if exact else 0),
                "root16_solver_calls",
            )
    require(game.over(), "trajectory_not_terminal")
    margin = game.captured_seeds[root] - game.captured_seeds[1 - root]
    score = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    require(outcome["stores"] == game.captured_seeds, "terminal_stores")
    require(outcome["store_margin_root_perspective"] == margin, "terminal_margin")
    require(outcome["score"] == score, "terminal_score")


def _canonical_game_hash(game: KalahGame) -> str:
    from ml.alphazero_lite.arena import canonical_game_state_hash

    return canonical_game_state_hash(game)


def _state_id(state: dict[str, Any]) -> str:
    from ml.alphazero_lite.build_opening_suite import canonical_key

    return canonical_key(state)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
