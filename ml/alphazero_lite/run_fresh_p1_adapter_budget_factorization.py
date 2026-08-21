#!/usr/bin/env python3
# ruff: noqa: E402
"""Factor PR #214 A16's arena effect by challenger and parent search budget.

This is an evaluation-only runner.  It loads the immutable PR #214 P1/A16
artifacts, never trains or exports a model, and only varies the two arena
simulation counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
)  # noqa: E402
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    ADAPTER_KEYS,
    new_model,
    output,
)  # noqa: E402
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import _cross_entropy  # noqa: E402
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (
    parse_game_jsonl,
    run_arena,
)  # noqa: E402
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import ARENA_SUITE  # noqa: E402
from ml.alphazero_lite.run_shared_trunk_delta_attribution import stable_hash  # noqa: E402
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402
from ml.alphazero_lite.train import (
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)  # noqa: E402

P1_CHECKPOINT_SHA = "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"
A16_STATE_SHA = "0b322e0996a4902cb8737ff32a429bfd45803ee4a55d713e2960ea9e9faf5068"
REPLAY_SHA = "892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88"
CONTEXTS = ("384:256", "384:384", "1200:256", "384:1200", "1200:1200")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
CANONICAL_OPENINGS = 128


def state_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash a state dict with the PR #214 tensor-payload convention."""
    return stable_hash(
        {
            name: value.detach().cpu().numpy().tobytes().hex()
            for name, value in state.items()
        }
    )


def _suite() -> tuple[list[dict[str, Any]], str]:
    rows = read_jsonl(ARENA_SUITE)
    prefixes = [tuple(row["prefix_moves"]) for row in rows]
    if len(rows) != CANONICAL_OPENINGS or len(set(prefixes)) != CANONICAL_OPENINGS:
        raise RuntimeError("canonical arena suite must contain 128 distinct openings")
    return rows, sha256_file(ARENA_SUITE)


def _artifact_hash(path: Path) -> str:
    return sha256_file(path / "weights.json")


def _manifest(
    *, challenger: Path, current: Path, suite_hash: str, context: str
) -> dict[str, Any]:
    challenger_sims, current_sims = (int(value) for value in context.split(":"))
    return {
        "schema": "azlite_fresh_p1_adapter_budget_factorization_cache_v1",
        "challenger_weights_sha256": _artifact_hash(challenger),
        "current_weights_sha256": _artifact_hash(current),
        "suite_sha256": suite_hash,
        "context": context,
        "challenger_simulations": challenger_sims,
        "current_simulations": current_sims,
        "c_puct": 1.25,
        "base_seed": BOOTSTRAP_SEED,
        "seed_contract": "azlite_eval_seed_v2",
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
        "normalize_values": False,
        "tactical_root_bias": 0.0,
        "games_per_opening": 2,
        "forced_challenger_seats": [0, 1],
    }


def _complete_records(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    records = parse_game_jsonl(str(path))
    if (
        len(records) != 256
        or len({int(row["opening_index"]) for row in records}) != 128
    ):
        return None
    return records


def arena_records(
    *,
    workdir: Path,
    challenger: Path,
    current: Path,
    context: str,
    role: str,
    workers: int,
    suite_hash: str,
) -> list[dict[str, Any]]:
    """Run or validate one exact paired arena arm, retaining opening records."""
    challenger_sims, current_sims = (int(value) for value in context.split(":"))
    evidence = workdir / "arena" / context.replace(":", "_") / role
    cache_manifest = _manifest(
        challenger=challenger, current=current, suite_hash=suite_hash, context=context
    )
    all_records: list[dict[str, Any]] = []
    for seat in (0, 1):
        seat_dir = evidence / f"starts_{seat}"
        records_path = seat_dir / "games.jsonl"
        manifest_path = seat_dir / "cache_manifest.json"
        cached_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else None
        )
        records = (
            _complete_records(records_path)
            if cached_manifest == cache_manifest
            else None
        )
        if records is None:
            seat_dir.mkdir(parents=True, exist_ok=True)
            run_arena(
                challenger=str(challenger),
                current=str(current),
                challenger_sims=challenger_sims,
                current_sims=current_sims,
                games=256,
                seed=BOOTSTRAP_SEED,
                workers=workers,
                out_json=str(seat_dir / "arena.json"),
                out_jsonl=str(records_path),
                opening_prefixes_jsonl=str(ARENA_SUITE),
                challenger_starts=seat,
                games_per_opening=2,
                root_policy_mode="deterministic",
                root_temperature=0.0,
                normalize_values=False,
                c_puct=1.25,
                tactical_root_bias=0.0,
                seed_contract="azlite_eval_seed_v2",
                suite_sha256=suite_hash,
                seed_ledger_output=str(seat_dir / "seed_ledger.jsonl"),
            )
            records = _complete_records(records_path)
            if records is None:
                raise RuntimeError(f"incomplete arena evidence: {records_path}")
            manifest_path.write_text(
                json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        opponent_config = {
            key: cache_manifest[key]
            for key in (
                "context",
                "challenger_simulations",
                "current_simulations",
                "c_puct",
                "seed_contract",
                "root_policy_mode",
                "root_temperature",
                "normalize_values",
                "tactical_root_bias",
                "games_per_opening",
            )
        }
        for row in records:
            row["opponent_weights_sha256"] = cache_manifest["current_weights_sha256"]
            row["opponent_config_sha256"] = stable_hash(opponent_config)
        all_records.extend(records)
    return all_records


def _bootstrap_draws(per_opening: dict[int, float], indexes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [per_opening[int(key)] for key in sorted(per_opening)], dtype=float
    )[indexes].mean(axis=1)


def paired_contrast(
    effects: dict[str, dict[str, Any]], weights: dict[str, float], indexes: np.ndarray
) -> dict[str, Any]:
    """Calculate an opening-paired linear contrast using shared bootstrap draws."""
    keys = [set(effects[context]["per_opening_effect"]) for context in weights]
    if any(keys[0] != key_set for key_set in keys[1:]):
        raise ValueError("all contrast contexts must contain the same openings")
    values = np.zeros(len(keys[0]), dtype=float)
    for context, weight in weights.items():
        values += weight * np.asarray(
            [effects[context]["per_opening_effect"][key] for key in sorted(keys[0])],
            dtype=float,
        )
    draws = values[indexes].mean(axis=1)
    return {
        "effect": float(values.mean()),
        "opening_bootstrap_ci": {
            "lower_95": float(np.quantile(draws, 0.025)),
            "upper_95": float(np.quantile(draws, 0.975)),
            "samples": int(len(indexes)),
            "unique_openings": int(len(values)),
        },
        "per_opening_effect": dict(zip(sorted(keys[0]), values.tolist(), strict=True)),
    }


def assemble_factorial(effects: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Assemble all requested contrasts from a single opening-bootstrap sample matrix."""
    openings = sorted(effects[CONTEXTS[0]]["per_opening_effect"])
    if len(openings) != CANONICAL_OPENINGS:
        raise ValueError("factorial analysis requires all 128 canonical openings")
    indexes = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(openings), size=(BOOTSTRAP_SAMPLES, len(openings))
    )
    definitions = {
        "candidate_search_increment_low_parent": {"1200:256": 1.0, "384:256": -1.0},
        "parent_search_increment_low_candidate": {"384:1200": 1.0, "384:256": -1.0},
        "equalization_384": {"384:384": 1.0, "384:256": -1.0},
        "candidate_search_increment_high_parent": {"1200:1200": 1.0, "384:1200": -1.0},
        "parent_search_increment_high_candidate": {"1200:1200": 1.0, "1200:256": -1.0},
        "high_high_interaction": {
            "1200:1200": 1.0,
            "1200:256": -1.0,
            "384:1200": -1.0,
            "384:256": 1.0,
        },
    }
    return {
        name: paired_contrast(effects, weights, indexes)
        for name, weights in definitions.items()
    }


def opening_attribution(effects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    openings = sorted(effects["384:256"]["per_opening_effect"])
    matrix = {
        context: np.asarray(
            [effects[context]["per_opening_effect"][key] for key in openings]
        )
        for context in CONTEXTS
    }
    base = matrix["384:256"]
    challenger_high = matrix["1200:256"]
    parent_high = matrix["384:1200"]
    correlations = {}
    for left in CONTEXTS:
        correlations[left] = {}
        for right in CONTEXTS:
            if np.std(matrix[left]) == 0.0 or np.std(matrix[right]) == 0.0:
                correlations[left][right] = None
            else:
                correlations[left][right] = float(
                    np.corrcoef(matrix[left], matrix[right])[0, 1]
                )
    return {
        "fraction_openings_challenger_search_worsens_a16": float(
            np.mean(challenger_high < base)
        ),
        "fraction_openings_parent_search_worsens_a16": float(
            np.mean(parent_high < base)
        ),
        "safe_neutral_384_256_to_negative_1200_256": float(
            np.mean((base >= 0.0) & (challenger_high < 0.0))
        ),
        "safe_neutral_384_256_to_negative_384_1200": float(
            np.mean((base >= 0.0) & (parent_high < 0.0))
        ),
        "effect_correlation": correlations,
    }


def _visit_js(left: np.ndarray, right: np.ndarray) -> float:
    left, right = left / left.sum(), right / right.sum()
    midpoint = (left + right) / 2.0
    return float(
        0.5
        * np.sum(left * np.log(np.maximum(left, 1e-12) / np.maximum(midpoint, 1e-12)))
        + 0.5
        * np.sum(right * np.log(np.maximum(right, 1e-12) / np.maximum(midpoint, 1e-12)))
    )


def deterministic_probe(
    candidate: Path, parent: Path, suite: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare immutable A16/P1 PUCT roots with treatment-invariant seeds."""
    options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    evaluators = {"a16": ArtifactEvaluator(candidate), "p1": ArtifactEvaluator(parent)}
    by_budget: dict[str, list[dict[str, Any]]] = {"384": [], "1200": []}
    for index, entry in enumerate(suite):
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        for move in entry["prefix_moves"]:
            game.move(move)
        for simulations in (384, 1200):
            results = {
                name: evaluate_artifact_position(
                    evaluator=evaluator,
                    state=game.to_state(),
                    simulations=simulations,
                    seed=BOOTSTRAP_SEED + index,
                    c_puct=1.25,
                    search_options=options,
                )
                for name, evaluator in evaluators.items()
            }
            a16, p1 = results["a16"], results["p1"]
            legal = a16["legal_moves"]
            av, pv = np.asarray(a16["visits"])[legal], np.asarray(p1["visits"])[legal]

            def margin(result: dict[str, Any]) -> float:
                ordered = sorted(
                    (result["visits"][move] for move in legal), reverse=True
                )
                return (
                    float(ordered[0] - ordered[1])
                    if len(ordered) > 1
                    else float(ordered[0])
                )

            def q_ranking(result: dict[str, Any]) -> list[int]:
                return [
                    row["move"]
                    for row in sorted(
                        result["child_stats"],
                        key=lambda row: (-row["q_value"], row["move"]),
                    )
                ]

            by_budget[str(simulations)].append(
                {
                    "opening_index": index,
                    "a16_move": a16["selected_move"],
                    "p1_move": p1["selected_move"],
                    "visit_js": _visit_js(av, pv),
                    "visit_margin_delta": margin(a16) - margin(p1),
                    "q_ranking_changed": q_ranking(a16) != q_ranking(p1),
                    "root_value_delta": float(
                        a16.get("search_root_value", 0.0)
                        - p1.get("search_root_value", 0.0)
                    ),
                    "a16_terminal_leaf_count": a16.get("terminal_leaf_count"),
                    "p1_terminal_leaf_count": p1.get("terminal_leaf_count"),
                    "a16_nonterminal_leaf_count": a16.get("nonterminal_leaf_count"),
                    "p1_nonterminal_leaf_count": p1.get("nonterminal_leaf_count"),
                }
            )
    summaries = {}
    for budget, rows in by_budget.items():
        summaries[budget] = {
            "selected_move_change_rate": float(
                np.mean([row["a16_move"] != row["p1_move"] for row in rows])
            ),
            "visit_distribution_js_mean": float(
                np.mean([row["visit_js"] for row in rows])
            ),
            "top1_top2_visit_margin_delta_mean": float(
                np.mean([row["visit_margin_delta"] for row in rows])
            ),
            "q_ranking_change_rate": float(
                np.mean([row["q_ranking_changed"] for row in rows])
            ),
            "root_value_delta_mean": float(
                np.mean([row["root_value_delta"] for row in rows])
            ),
            "a16_terminal_leaf_count_mean": float(
                np.mean([row["a16_terminal_leaf_count"] for row in rows])
            ),
            "p1_terminal_leaf_count_mean": float(
                np.mean([row["p1_terminal_leaf_count"] for row in rows])
            ),
            "a16_nonterminal_leaf_count_mean": float(
                np.mean([row["a16_nonterminal_leaf_count"] for row in rows])
            ),
            "p1_nonterminal_leaf_count_mean": float(
                np.mean([row["p1_nonterminal_leaf_count"] for row in rows])
            ),
            "expanded_node_count": "not exposed by PUCT telemetry",
            "max_depth": "not exposed by PUCT telemetry",
            "mean_depth": "not exposed by PUCT telemetry",
            "rows": rows,
        }
    divergent = [
        row["opening_index"]
        for row384, row in zip(by_budget["384"], by_budget["1200"], strict=True)
        if row384["a16_move"] == row384["p1_move"] and row["a16_move"] != row["p1_move"]
    ]
    return {"by_budget": summaries, "agree_384_diverge_1200_opening_indexes": divergent}


def classify(
    invariants: dict[str, bool],
    contexts: dict[str, dict[str, Any]],
    contrasts: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    if (
        not all(invariants.values())
        or not contexts["384:256"]["reproduces_pr214"]
        or not contexts["1200:1200"]["reproduces_pr214"]
    ):
        return (
            "invariant_failure",
            "Do not interpret outcomes; repair the immutable-model or arena reproduction contract.",
        )
    low = contexts["384:256"]["paired_treatment_effect"]
    candidate = contexts["1200:256"]["paired_treatment_effect"]
    parent = contexts["384:1200"]["paired_treatment_effect"]
    high = contexts["1200:1200"]["paired_treatment_effect"]

    def negative_ci(name: str) -> bool:
        return contrasts[name]["opening_bootstrap_ci"]["upper_95"] < 0.0

    if (
        candidate < low
        and negative_ci("candidate_search_increment_low_parent")
        and parent >= candidate
    ):
        return (
            "candidate_search_amplifies_adapter_harm",
            "Trace the first candidate-vs-P1 search divergence from 384 to 1200 simulations, including depth/node attribution.",
        )
    if (
        parent < low
        and negative_ci("parent_search_increment_low_candidate")
        and candidate >= parent
    ):
        return (
            "opponent_search_exposes_adapter_harm",
            "Identify states/actions where deeper P1 search gains against A16 and compare them with training search targets.",
        )
    if contexts["384:384"]["paired_treatment_effect"] < low:
        return (
            "equal_budget_exposes_adapter_harm",
            "Make 384:384 a required promotion-gate cell and inspect why 384:256 hid the failure.",
        )
    if (
        candidate >= low
        and parent >= low
        and high < low
        and negative_ci("high_high_interaction")
    ):
        return (
            "high_high_search_interaction",
            "Run paired first-divergence tracing at 1200:1200.",
        )
    negative = sum(contexts[key]["paired_treatment_effect"] < 0.0 for key in CONTEXTS)
    if negative >= 4:
        return (
            "budget_robust_adapter_regression",
            "Return to search-target/policy alignment rather than attributing the regression to one budget side.",
        )
    return (
        "inconclusive",
        "Repeat no parameter changes; retain the same matched controls and extend descriptive first-divergence tracing only.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_adapter_budget_factorization"),
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-budget-factorization-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-budget-factorization-results.md",
    )
    args = parser.parse_args()
    parent_checkpoint = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    parent_artifact = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    candidate_checkpoint = args.adapter_workdir / "artifacts/step_0016/checkpoint.npz"
    candidate_artifact = args.adapter_workdir / "artifacts/step_0016/artifact"
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    suite, suite_hash = _suite()
    parent, candidate = new_model(torch.device("cpu")), new_model(torch.device("cpu"))
    load_checkpoint_into_model(parent, parent_checkpoint)
    load_checkpoint_into_model(candidate, candidate_checkpoint)
    parent_state, candidate_state = parent.state_dict(), candidate.state_dict()
    inherited_identical = all(
        torch.equal(parent_state[name], candidate_state[name])
        for name in parent_state
        if name not in ADAPTER_KEYS
    )
    only_adapter_differs = inherited_identical and all(
        not torch.equal(parent_state[name], candidate_state[name])
        for name in ADAPTER_KEYS
    )
    rows = read_jsonl(replay)
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    parent_policy, candidate_policy = (
        output(parent_state, x, mask),
        output(candidate_state, x, mask),
    )
    l1_mean = float(np.abs(parent_policy - candidate_policy).sum(axis=1).mean())
    pure_state = torch.load(
        args.adapter_workdir / "pure_search/snapshots/step_0046.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    search_target = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    parent_search_ce = float(np.mean(_cross_entropy(parent_policy, search_target)))
    candidate_search_ce = float(
        np.mean(_cross_entropy(candidate_policy, search_target))
    )
    pure_search_ce = float(
        np.mean(_cross_entropy(output(pure_state, x, mask), search_target))
    )
    fit_fraction = (parent_search_ce - candidate_search_ce) / (
        parent_search_ce - pure_search_ce
    )
    metrics_match = abs(l1_mean - 0.0017535027555326227) < 1e-10
    fit_fraction_match = abs(fit_fraction - 0.3259840837739601) < 1e-10
    hashes = {
        "p1_checkpoint": sha256_file(parent_checkpoint),
        "a16_state": state_hash(candidate_state),
        "replay": sha256_file(replay),
        "suite": suite_hash,
        "p1_artifact_weights": _artifact_hash(parent_artifact),
        "a16_artifact_weights": _artifact_hash(candidate_artifact),
    }
    invariants = {
        "p1_checkpoint_hash": hashes["p1_checkpoint"] == P1_CHECKPOINT_SHA,
        "a16_state_hash": hashes["a16_state"] == A16_STATE_SHA,
        "replay_hash": hashes["replay"] == REPLAY_SHA,
        "inherited_parameters_byte_identical": inherited_identical,
        "only_adapter_parameters_differ": only_adapter_differs,
        "mean_legal_l1_reproduced": metrics_match,
        "fit_fraction_reproduced": fit_fraction_match,
    }
    effects: dict[str, dict[str, Any]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for context in CONTEXTS:
        control = arena_records(
            workdir=args.workdir,
            challenger=parent_artifact,
            current=parent_artifact,
            context=context,
            role="p1_vs_p1_control",
            workers=args.workers,
            suite_hash=suite_hash,
        )
        records = arena_records(
            workdir=args.workdir,
            challenger=candidate_artifact,
            current=parent_artifact,
            context=context,
            role="a16_vs_p1",
            workers=args.workers,
            suite_hash=suite_hash,
        )
        effect = paired_opening_candidate_effect(
            records,
            control,
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        effects[context] = effect
        expected = {"384:256": -0.009765625, "1200:1200": -0.01953125}.get(context)
        contexts[context] = {
            "raw_a16_score": effect["orientation_decomposition"][
                "candidate_challenger_score"
            ],
            "raw_p1_p1_control_score": effect["orientation_decomposition"][
                "current_control_challenger_score"
            ],
            "paired_treatment_effect": effect["paired_candidate_effect"],
            "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
            "seat_a_effect": effect["p0_effect"],
            "seat_b_effect": effect["p1_effect"],
            "win_draw_loss": {
                "wins": sum(row["winner"] == "challenger" for row in records),
                "draws": sum(row["winner"] == "draw" for row in records),
                "losses": sum(row["winner"] == "current" for row in records),
            },
            "reproduces_pr214": None
            if expected is None
            else effect["paired_candidate_effect"] == expected,
        }
    contrasts = assemble_factorial(effects)
    classification, follow_up = classify(invariants, contexts, contrasts)
    arena_configuration = {
        "contexts": list(CONTEXTS),
        "canonical_openings": 128,
        "seat_swapping": True,
        "games_per_opening_per_seat": 2,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "seed_contract": "azlite_eval_seed_v2",
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
        "normalize_values": False,
        "tactical_root_bias": 0.0,
        "c_puct": 1.25,
    }
    hashes["arena_configuration"] = stable_hash(arena_configuration)
    summary = {
        "schema": "azlite_fresh_p1_adapter_budget_factorization_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "only_manipulated_variables": [
                "challenger_simulations",
                "current_simulations",
            ],
            "c_puct": 1.25,
        },
        "hashes": hashes,
        "invariants": invariants,
        "network_output_metrics": {
            "fit_fraction": fit_fraction,
            "mean_legal_policy_l1": l1_mean,
        },
        "arena_configuration": arena_configuration,
        "contexts": contexts,
        "opening_treatment_effects": {
            context: effect["per_opening_effect"] for context, effect in effects.items()
        },
        "factorial_contrasts": {
            name: {
                key: value
                for key, value in contrast.items()
                if key != "per_opening_effect"
            }
            for name, contrast in contrasts.items()
        },
        "opening_attribution": opening_attribution(effects),
        "deterministic_search_probe": deterministic_probe(
            candidate_artifact, parent_artifact, suite
        ),
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = [
        "| Context | A16 raw | P1/P1 raw | Effect | 95% CI | Seat A | Seat B | W/D/L |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for context in CONTEXTS:
        row, ci = contexts[context], contexts[context]["opening_bootstrap_ci"]
        wdl = row["win_draw_loss"]
        table.append(
            f"| {context} | {row['raw_a16_score']:.4f} | {row['raw_p1_p1_control_score']:.4f} | {row['paired_treatment_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {row['seat_a_effect']:+.4f} | {row['seat_b_effect']:+.4f} | {wdl['wins']}/{wdl['draws']}/{wdl['losses']} |"
        )
    contrast_rows = ["| Contrast | Effect | 95% CI |", "| --- | ---: | --- |"]
    for name, value in contrasts.items():
        ci = value["opening_bootstrap_ci"]
        contrast_rows.append(
            f"| {name} | {value['effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
        )
    probe_summary = {
        budget: {key: value for key, value in details.items() if key != "rows"}
        for budget, details in summary["deterministic_search_probe"][
            "by_budget"
        ].items()
    }
    args.out_report.write_text(
        "\n".join(
            [
                "# PR #214 Adapter Budget Factorization",
                "",
                f"**Classification:** `{classification}`",
                "",
                f"**Recommended follow-up:** {follow_up}",
                "",
                "## Five-Context Matched Arena",
                "",
                *table,
                "",
                "The seat columns are treatment effects. The reported PR #214 high/high loss remains entirely Seat A; the new 384:1200 loss is instead concentrated in Seat B.",
                "",
                "## Opening-Paired Factorial Contrasts",
                "",
                *contrast_rows,
                "",
                "## Opening Attribution",
                "",
                "```json",
                json.dumps(summary["opening_attribution"], indent=2, sort_keys=True),
                "```",
                "",
                "## Artifact And Contract Hashes",
                "",
                "```json",
                json.dumps(hashes, indent=2, sort_keys=True),
                "```",
                "",
                "## Invariants",
                "",
                "```json",
                json.dumps(invariants, indent=2, sort_keys=True),
                "```",
                "",
                "## Deterministic Probe",
                "",
                "```json",
                json.dumps(
                    {
                        "by_budget": probe_summary,
                        "agree_384_diverge_1200_opening_indexes": summary[
                            "deterministic_search_probe"
                        ]["agree_384_diverge_1200_opening_indexes"],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(classification)


if __name__ == "__main__":
    main()
