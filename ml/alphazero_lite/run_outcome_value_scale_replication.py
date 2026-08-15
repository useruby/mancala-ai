#!/usr/bin/env python3
# ruff: noqa: E402
"""Independent, diagnostic-only replication of PR #189's outcome-value scale.

This runner never trains, promotes, or changes a runtime artifact.  Its causal
corpus is freshly generated and explicitly excludes every PR #188/#189 state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.evaluation_seed_contract import stable_hash, stable_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_frozen_trunk_value_sufficiency_audit import (
    EXPECTED_CURRENT_SHA256,
    entropy,
    phase,
)
from ml.alphazero_lite.run_value_scale_causal_preflight import (
    AffineValueEvaluator,
    forced_continuation,
    js,
    q_rank,
    q_u_ratio,
    visit_margin,
)
from ml.alphazero_lite.self_play import PUCT, build_eval_search_options

DEFAULT_WORKDIR = Path("/tmp/azlite_outcome_value_scale_replication")
OUTCOME_A = 1.425456519226987
OUTCOME_B = 0.061720579402024627
BUDGETS = (384, 768, 1200)
CONTINUATION_BUDGETS = (768, 1200)
SEARCH_OPTIONS = build_eval_search_options()
_CURRENT: ArtifactEvaluator | None = None
_AFFINE: AffineValueEvaluator | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def prior_hashes() -> set[str]:
    """Load every state persisted by the two earlier preflights, never suites."""
    hashes: set[str] = set()
    for path in (
        Path("/tmp/azlite_frozen_trunk_value_sufficiency/value_probe_states.jsonl"),
        Path(
            "/tmp/azlite_frozen_trunk_value_sufficiency/evaluation_domain_states.jsonl"
        ),
        Path("/tmp/azlite_value_scale_causal_preflight/corpus.json"),
        Path("/tmp/azlite_value_scale_causal_preflight/evaluation_labels.json"),
    ):
        if not path.is_file():
            raise RuntimeError(f"missing prior-preflight state ledger: {path}")
        rows = (
            [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if path.suffix == ".jsonl"
            else json.loads(path.read_text(encoding="utf-8")).get("rows", [])
        )
        hashes.update(str(row["state_hash"]) for row in rows)
    return hashes


def state_row(
    game: KalahGame, evaluator: ArtifactEvaluator, *, source: str, game_id: int
) -> dict[str, Any]:
    policy, value = evaluator.evaluate(game)
    state = game.to_state()
    return {
        "state": state,
        "state_hash": stable_hash(state),
        "source": source,
        "source_game": f"{source}:{game_id}",
        "player": int(game.current_player),
        "phase": phase(game),
        "legal_move_count": len(game.possible_moves()),
        "current_value": float(value),
        "policy_entropy": float(entropy(policy)),
    }


def fresh_selfplay(
    evaluator: ArtifactEvaluator, *, seed: int, forbidden: set[str], count: int
) -> list[dict[str, Any]]:
    rows, seen, game_id = [], set(forbidden), 0
    while len(rows) < count:
        game = KalahGame([4] * 12, [0, 0], 0)
        rng = random.Random(stable_seed(seed, "fresh-current-selfplay", game_id))
        while not game.over():
            if game.possible_moves():
                row = state_row(
                    game, evaluator, source="fresh_current_selfplay", game_id=game_id
                )
                if row["state_hash"] not in seen:
                    rows.append(row)
                    seen.add(row["state_hash"])
                    if len(rows) == count:
                        return rows
            search = PUCT(evaluator, 128, 1.25, rng, **SEARCH_OPTIONS)
            visits, _ = search.run(game)
            legal = game.possible_moves()
            weights = [float(visits[move]) for move in legal]
            move = rng.choices(legal, weights=weights if sum(weights) else None, k=1)[0]
            if not game.move(game.pit_index(move)):
                raise RuntimeError("self-play selected an illegal move")
        game_id += 1
        if game_id > count * 4:
            raise RuntimeError("could not harvest enough fresh self-play states")
    return rows


def random_prefix_rows(
    evaluator: ArtifactEvaluator,
    *,
    seed: int,
    forbidden: set[str],
    count: int,
    source: str,
    phases: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows, seen = [], set(forbidden)
    for game_id in range(count * 32):
        game = KalahGame([4] * 12, [0, 0], 0)
        rng = random.Random(stable_seed(seed, source, game_id))
        target_plies = (
            1 + rng.randrange(10) if phases is None else 12 + rng.randrange(36)
        )
        for _ in range(target_plies):
            legal = game.possible_moves()
            if game.over() or not legal:
                break
            game.move(game.pit_index(legal[rng.randrange(len(legal))]))
        if (
            game.over()
            or not game.possible_moves()
            or (phases and phase(game) not in phases)
        ):
            continue
        row = state_row(game, evaluator, source=source, game_id=game_id)
        if row["state_hash"] not in seen:
            rows.append(row)
            seen.add(row["state_hash"])
            if len(rows) == count:
                return rows
    raise RuntimeError(f"could not harvest {count} {source} states")


def build_corpus(
    evaluator: ArtifactEvaluator, *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forbidden = prior_hashes()
    selfplay = fresh_selfplay(evaluator, seed=seed, forbidden=forbidden, count=512)
    forbidden.update(row["state_hash"] for row in selfplay)
    opening = random_prefix_rows(
        evaluator,
        seed=seed,
        forbidden=forbidden,
        count=256,
        source="independent_opening_family",
    )
    forbidden.update(row["state_hash"] for row in opening)
    enriched = random_prefix_rows(
        evaluator,
        seed=seed,
        forbidden=forbidden,
        count=256,
        source="mid_late_enriched",
        phases={"midgame", "late"},
    )
    rows = selfplay + opening + enriched
    if len(rows) != 1024 or len({row["state_hash"] for row in rows}) != 1024:
        raise RuntimeError("replication corpus must contain 1,024 unique states")
    manifest = {
        "schema": "azlite_outcome_value_scale_replication_v1",
        "state_corpus_hash": stable_hash([row["state_hash"] for row in rows]),
        "prior_state_exclusion_count": len(prior_hashes()),
        "sources": {
            source: {"count": sum(row["source"] == source for row in rows)}
            for source in (
                "fresh_current_selfplay",
                "independent_opening_family",
                "mid_late_enriched",
            )
        },
        "frozen_fields": [
            "state_hash",
            "source",
            "source_game",
            "player",
            "phase",
            "legal_move_count",
            "current_value",
            "policy_entropy",
        ],
    }
    return rows, manifest


def _init_worker(artifact: str) -> None:
    global _CURRENT, _AFFINE
    _CURRENT = ArtifactEvaluator(Path(artifact))
    _AFFINE = AffineValueEvaluator(_CURRENT, OUTCOME_A, OUTCOME_B)


def _search_worker(task: tuple[dict[str, Any], int, int]) -> tuple[str, dict[str, Any]]:
    row, budget, seed = task
    if _CURRENT is None or _AFFINE is None:
        raise RuntimeError("worker was not initialized")
    common = dict(
        state=row["state"],
        simulations=budget,
        seed=stable_seed(seed, row["state_hash"], "puct", budget),
        c_puct=1.25,
        search_options=SEARCH_OPTIONS,
    )
    current = evaluate_artifact_position(evaluator=_CURRENT, **common)
    affine = evaluate_artifact_position(evaluator=_AFFINE, **common)
    if current["policy"] != affine["policy"]:
        raise RuntimeError("policy changed under value-only treatment")
    return row["state_hash"], {"current": current, "outcome_affine": affine}


def _causal_worker(
    task: tuple[dict[str, Any], int, int, int, int],
) -> tuple[int, dict[str, Any]]:
    row, current_move, affine_move, budget, seed = task
    if _CURRENT is None:
        raise RuntimeError("worker was not initialized")
    current = forced_continuation(
        row["state"], _CURRENT, current_move, budget=budget, seed=seed
    )
    affine = forced_continuation(
        row["state"], _CURRENT, affine_move, budget=budget, seed=seed
    )
    return budget, {
        "state_hash": row["state_hash"],
        "normalized_final_margin_delta": affine["normalized_final_margin"]
        - current["normalized_final_margin"],
        "binary_outcome_delta": affine["outcome"] - current["outcome"],
    }


def bootstrap(
    rows: list[dict[str, Any]], key: str, *, seed: int
) -> dict[str, float | int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["state_hash"]].append(float(row[key]))
    values = np.asarray([np.mean(group) for group in grouped.values()])
    if not len(values):
        return {"n": 0, "unique_states": 0, "mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(
        axis=1
    )
    return {
        "n": len(rows),
        "unique_states": len(values),
        "mean": float(values.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def run_comparison(
    rows: list[dict[str, Any]], *, artifact: Path, seed: int, workers: int
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(artifact),)
    ) as executor:
        for budget in BUDGETS:
            searches = dict(
                executor.map(
                    _search_worker, ((row, budget, seed) for row in rows), chunksize=1
                )
            )
            diagnostics, tasks = [], []
            for row in rows:
                current, affine = searches[row["state_hash"]].values()
                changed = int(current["selected_move"]) != int(affine["selected_move"])
                diagnostics.append(
                    {
                        "changed": changed,
                        "visit_js": js(current["visits"], affine["visits"]),
                        "q_ranking_changed": q_rank(current) != q_rank(affine),
                        "root_value_delta": float(
                            affine.get("search_root_value", 0)
                            - current.get("search_root_value", 0)
                        ),
                        "q_u_ratio_delta": q_u_ratio(affine) - q_u_ratio(current),
                        "visit_margin_delta": visit_margin(affine)
                        - visit_margin(current),
                    }
                )
                if changed:
                    for continuation_budget in CONTINUATION_BUDGETS:
                        tasks.append(
                            (
                                row,
                                int(current["selected_move"]),
                                int(affine["selected_move"]),
                                continuation_budget,
                                stable_seed(
                                    seed,
                                    row["state_hash"],
                                    "forced-continuation",
                                    budget,
                                    continuation_budget,
                                ),
                            )
                        )
            causal: dict[str, list[dict[str, Any]]] = {
                str(value): [] for value in CONTINUATION_BUDGETS
            }
            for continuation_budget, item in executor.map(
                _causal_worker, tasks, chunksize=1
            ):
                causal[str(continuation_budget)].append(item)
            report[f"D{budget}"] = {
                "states": len(rows),
                "selected_move_disagreement": float(
                    np.mean([row["changed"] for row in diagnostics])
                ),
                "visit_js": float(np.mean([row["visit_js"] for row in diagnostics])),
                "q_ranking_changed": float(
                    np.mean([row["q_ranking_changed"] for row in diagnostics])
                ),
                "root_value_delta": float(
                    np.mean([row["root_value_delta"] for row in diagnostics])
                ),
                "q_u_ratio_delta": float(
                    np.mean([row["q_u_ratio_delta"] for row in diagnostics])
                ),
                "visit_margin_delta": float(
                    np.mean([row["visit_margin_delta"] for row in diagnostics])
                ),
                "causal": {
                    budget_key: {
                        "normalized_final_margin_delta": bootstrap(
                            items,
                            "normalized_final_margin_delta",
                            seed=stable_seed(seed, budget, budget_key, "margin"),
                        ),
                        "binary_outcome_delta": bootstrap(
                            items,
                            "binary_outcome_delta",
                            seed=stable_seed(seed, budget, budget_key, "outcome"),
                        ),
                    }
                    for budget_key, items in causal.items()
                },
            }
    return report


def d1200_passes(comparison: dict[str, Any]) -> bool:
    causal = comparison["D1200"]["causal"]
    return all(
        causal[str(budget)]["normalized_final_margin_delta"]["unique_states"] >= 64
        and causal[str(budget)]["normalized_final_margin_delta"]["mean"] > 0
        and causal[str(budget)]["normalized_final_margin_delta"]["lower"] >= 0
        and causal[str(budget)]["binary_outcome_delta"]["mean"] >= 0
        for budget in CONTINUATION_BUDGETS
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--seed", type=int, default=190)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    artifact, workdir = Path(args.current), Path(args.workdir)
    if sha256(artifact / "weights.json") != EXPECTED_CURRENT_SHA256:
        raise RuntimeError("current weights hash mismatch")
    current = ArtifactEvaluator(artifact)
    rows, manifest = build_corpus(current, seed=args.seed)
    write_jsonl(workdir / "corpus.jsonl", rows)
    manifest["source_hashes"] = {}
    for source in manifest["sources"]:
        source_path = workdir / f"{source}.jsonl"
        write_jsonl(source_path, [row for row in rows if row["source"] == source])
        manifest["source_hashes"][source] = sha256(source_path)
    manifest["current_weights_sha256"] = sha256(artifact / "weights.json")
    write_json(workdir / "corpus_manifest.json", manifest)
    comparison = run_comparison(
        rows, artifact=artifact, seed=args.seed, workers=args.workers
    )
    passed = d1200_passes(comparison)
    report = {
        "schema": manifest["schema"],
        "current_weights_sha256": manifest["current_weights_sha256"],
        "frozen_outcome_affine": {"a": OUTCOME_A, "b": OUTCOME_B, "clip": [-1, 1]},
        "search_configuration": {
            "budgets": list(BUDGETS),
            "continuation_budgets": list(CONTINUATION_BUDGETS),
            "c_puct": 1.25,
            "tactical_root_bias": 0,
            "zero_dirichlet": True,
            "deterministic_root": True,
            "normalize_values": False,
            "treatment_invariant_seed_identity": True,
        },
        "corpus_manifest": manifest,
        "comparison": comparison,
        "d1200_forced_move_replication_passed": passed,
        "classification": "outcome_value_scale_improves_moves_not_games"
        if passed
        else "outcome_value_scale_effect_not_replicated",
        "canonical_arena_run": False,
    }
    write_json(workdir / "summary.json", report)
    write_json(
        REPO_ROOT
        / "docs/data/alphazero-lite-outcome-value-scale-replication-summary.json",
        report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
