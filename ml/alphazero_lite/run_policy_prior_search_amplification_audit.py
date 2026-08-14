#!/usr/bin/env python3
# ruff: noqa: E402
"""Diagnostic-only causal audit of PR #182 policy-prior search amplification.

The runner never writes an artifact, replay, teacher target, or production
configuration. Detailed state and trace records stay in ``workdir``; only a
compact aggregate summary is committed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.evaluation_seed_contract import (
    SEED_CONTRACT_VERSION,
    derive_search_seed,
    stable_hash,
    stable_seed,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.run_denoised_puct_convergence_audit import phase_for_state
from ml.alphazero_lite.run_stronger_policy_teacher_ablation import (
    EXPECTED_CURRENT_SHA256,
)
from ml.alphazero_lite.self_play import PUCT, Node
from ml.alphazero_lite.train import legal_mask_matrix_for_encoded_states

EXPECTED = {
    "current": EXPECTED_CURRENT_SHA256,
    "D384": "6fe1444d6c82cb4a443d62111c3adb9ccd028d73071de52e1be2d186b6dec779",
    "D1200": "03dc5145a1d2fd471b6eb5a8dfacb4c57b5612594d1a5fd6d07acfc3dd8f02eb",
}
POLICY_KEYS = frozenset({"w_policy", "b_policy", "w_policy_hidden", "b_policy_hidden"})
BUDGETS = (128, 384, 768, 1200)
ALPHAS = (0.25, 0.50, 0.75, 1.00)
SCHEMA = "azlite_policy_prior_search_amplification_audit_v1"
_MEDIUM_CURRENT: ArtifactEvaluator | None = None
_MEDIUM_INTERPOLATIONS: dict[str, Any] = {}
_CONTINUATION_CURRENT: ArtifactEvaluator | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, sort_keys=True, separators=(",", ":"), default=_json_default
                )
                + "\n"
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def legal_distribution(values: np.ndarray, legal: list[int]) -> np.ndarray:
    result = np.zeros(6, dtype=float)
    legal_values = np.asarray(values, dtype=float)[legal]
    total = float(legal_values.sum())
    result[legal] = legal_values / total if total > 0 else 1.0 / len(legal)
    return result


def entropy(policy: np.ndarray, legal: list[int]) -> float:
    values = policy[legal]
    return float(-np.sum(values * np.log(np.maximum(values, 1e-12))))


def kl(left: np.ndarray, right: np.ndarray, legal: list[int]) -> float:
    a, b = left[legal], right[legal]
    return float(np.sum(a * np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12))))


def js(left: np.ndarray, right: np.ndarray, legal: list[int]) -> float:
    midpoint = (left + right) / 2
    return (kl(left, midpoint, legal) + kl(right, midpoint, legal)) / 2


def ranked(policy: np.ndarray, legal: list[int]) -> list[int]:
    return sorted(legal, key=lambda move: (-float(policy[move]), move))


def selected(root: Node, legal: list[int]) -> int:
    return max(
        legal,
        key=lambda move: (
            root.children[move].visit_count,
            root.children[move].q_value,
            root.children[move].prior,
            -move,
        ),
    )


class InterpolatedPolicyEvaluator:
    """Diagnostic evaluator: legal policy logits interpolate; current value is exact."""

    def __init__(
        self, current: ArtifactEvaluator, candidate: ArtifactEvaluator, alpha: float
    ):
        if alpha not in ALPHAS:
            raise ValueError("diagnostic alpha must be one of the configured values")
        self.current, self.candidate, self.alpha = current, candidate, float(alpha)

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        current_policy, current_value = self.current.evaluate(game)
        if self.alpha == 0.0:
            return current_policy, float(current_value)
        candidate_policy, _ = self.candidate.evaluate(game)
        if self.alpha == 1.0:
            # Endpoint equivalence is bitwise, not merely algebraic: a log/exp
            # reconstruction perturbs float32 artifact probabilities.
            return candidate_policy, float(current_value)
        legal = game.possible_moves()
        policy = np.zeros(6, dtype=np.float32)
        if legal:
            # Legal log-probabilities differ from legal logits only by a constant.
            logits = (1.0 - self.alpha) * np.log(
                np.maximum(current_policy[legal], 1e-30)
            )
            logits += self.alpha * np.log(np.maximum(candidate_policy[legal], 1e-30))
            logits -= np.max(logits)
            policy[legal] = np.exp(logits) / np.exp(logits).sum()
        return policy, float(current_value)


class TracePUCT(PUCT):
    """PUCT observer; it overrides no selection or backup semantics."""

    def run(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, Node]:
        self.selections: list[dict[str, Any]] = []
        return super().run(*args, **kwargs)

    def _select_child(self, node: Node) -> Node:
        entries, _move, _reference, _trust = self._selection_entries(
            node, sort_moves=True
        )
        child = super()._select_child(node)
        self.selections.append(
            {
                "state_hash": stable_hash(node.game.to_state()),
                "move": next(
                    move for move, item in node.children.items() if item is child
                ),
                "entries": entries,
            }
        )
        return child


def run_search(
    evaluator: Any, state: dict[str, Any], budget: int, seed: int, trace: bool = False
) -> tuple[dict[str, Any], Any]:
    game = KalahGame.from_state(state)
    engine_type = TracePUCT if trace else PUCT
    engine = engine_type(
        evaluator=evaluator,
        simulations=budget,
        c_puct=1.25,
        rng=random.Random(seed),
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        normalize_values=False,
    )
    visits, root = engine.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    legal = [int(move) for move in game.possible_moves()]
    visit_policy = legal_distribution(visits, legal)
    move = selected(root, legal)
    children = {move: root.children[move] for move in legal}
    q_values = sorted((float(children[item].q_value) for item in legal), reverse=True)
    return {
        "selected_move": move,
        "legal_moves": legal,
        "visits": visits.tolist(),
        "visit_policy": visit_policy.tolist(),
        "selected_visit_share": float(visit_policy[move]),
        "root_value": float(root.q_value),
        "child_q": float(children[move].q_value),
        "root_q_margin": float(q_values[0] - q_values[1]) if len(q_values) > 1 else 0.0,
        "visit_margin": float(
            sorted([children[m].visit_count for m in legal], reverse=True)[0]
            - sorted([children[m].visit_count for m in legal], reverse=True)[1]
        )
        if len(legal) > 1
        else 0.0,
    }, engine


def parameter_provenance(
    current: Path, d384: Path, d1200: Path, workdir: Path
) -> dict[str, Any]:
    checkpoints = {
        "current": materialize_weights_json_checkpoint(
            weights_path=current / "weights.json", out_path=workdir / "current.npz"
        ),
        "D384": d384.parent / "checkpoint.npz",
        "D1200": d1200.parent / "checkpoint.npz",
    }
    hashes = {
        "current": sha256_file(current / "weights.json"),
        "D384": sha256_file(checkpoints["D384"]),
        "D1200": sha256_file(checkpoints["D1200"]),
    }
    if hashes != EXPECTED:
        raise RuntimeError(f"artifact hash mismatch: {hashes}")
    arrays = {name: dict(np.load(path)) for name, path in checkpoints.items()}
    for candidate in ("D384", "D1200"):
        changed = {
            key
            for key in arrays["current"]
            if not np.array_equal(arrays["current"][key], arrays[candidate][key])
        }
        if changed - POLICY_KEYS:
            raise RuntimeError(
                f"{candidate} changed non-policy tensors: {sorted(changed - POLICY_KEYS)}"
            )
        if not changed:
            raise RuntimeError(f"{candidate} has no policy change")
    return {
        "hashes": hashes,
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "trunk_identical": True,
        "value_head_identical": True,
    }


def verify_pr182_records(workdir: Path, source: Path) -> dict[str, Any]:
    summary = json.loads((source / "summary_metrics.json").read_text(encoding="utf-8"))
    trajectory, low, high = (
        source / "trajectory_replay.jsonl",
        source / "replay_D384.jsonl",
        source / "replay_D1200.jsonl",
    )
    observed = {
        "trajectory_replay_sha256": sha256_file(trajectory),
        "D384_replay_sha256": sha256_file(low),
        "D1200_replay_sha256": sha256_file(high),
    }
    expected_trajectory = summary["trajectory_corpus_manifest"][
        "trajectory_replay_sha256"
    ]
    invariants = summary["paired_data_invariants"]["hashes"]
    if (
        observed["trajectory_replay_sha256"] != expected_trajectory
        or sha256_file(low) != invariants["D384_policies"]
        and False
    ):
        raise RuntimeError("PR #182 trajectory/replay provenance mismatch")
    # The historic summary hashes policy sequences, not complete JSONL files.
    low_rows, high_rows = read_jsonl(low), read_jsonl(high)
    if (
        stable_hash([row["policy"] for row in low_rows]) != invariants["D384_policies"]
        or stable_hash([row["policy"] for row in high_rows])
        != invariants["D1200_policies"]
    ):
        raise RuntimeError("PR #182 teacher-policy replay hash mismatch")
    return {
        **observed,
        "teacher_policy_hashes": {
            "D384": invariants["D384_policies"],
            "D1200": invariants["D1200_policies"],
        },
        "rows": len(low_rows),
        "validation_indexes_sha256": sha256_file(
            source / "validation_source_indexes.npy"
        ),
    }


def validation_metrics(source: Path, artifacts: dict[str, Path]) -> dict[str, Any]:
    rows = read_jsonl(source / "replay_D384.jsonl")
    high = read_jsonl(source / "replay_D1200.jsonl")
    indexes = np.load(source / "validation_source_indexes.npy", allow_pickle=False)
    states = np.asarray([rows[int(i)]["state"] for i in indexes], dtype=np.float32)
    masks = legal_mask_matrix_for_encoded_states(states)
    # Artifact evaluators avoid depending on raw_state omitted from replay rows; reconstruct raw rows from trajectory.
    corpus = read_jsonl(source / "trajectory_replay.jsonl")
    result: dict[str, Any] = {}
    evaluators = {name: ArtifactEvaluator(path) for name, path in artifacts.items()}
    for name, evaluator in evaluators.items():
        probs = np.asarray(
            [
                evaluator.evaluate(KalahGame.from_state(corpus[int(i)]["raw_state"]))[0]
                for i in indexes
            ],
            dtype=float,
        )
        current = np.asarray(
            [
                evaluators["current"].evaluate(
                    KalahGame.from_state(corpus[int(i)]["raw_state"])
                )[0]
                for i in indexes
            ],
            dtype=float,
        )
        result[name] = {}
        for teacher, lane in (("D384", rows), ("D1200", high)):
            target = np.asarray([lane[int(i)]["policy"] for i in indexes], dtype=float)
            legal = masks > 0
            ce = -np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1).mean()
            midpoint = (target + probs) / 2
            result[name][teacher] = {
                "ce": float(ce),
                "js": float(
                    np.mean(
                        np.sum(
                            target
                            * np.log(
                                np.maximum(target, 1e-12) / np.maximum(midpoint, 1e-12)
                            ),
                            axis=1,
                        )
                        + np.sum(
                            probs
                            * np.log(
                                np.maximum(probs, 1e-12) / np.maximum(midpoint, 1e-12)
                            ),
                            axis=1,
                        )
                    )
                    / 2
                ),
                "legal_top1_agreement": float(
                    np.mean(
                        np.argmax(np.where(legal, probs, -1), 1)
                        == np.argmax(np.where(legal, target, -1), 1)
                    )
                ),
                "entropy": float(
                    np.mean(-np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1))
                ),
                "policy_kl_from_current": float(
                    np.mean(
                        np.sum(
                            probs
                            * np.log(
                                np.maximum(probs, 1e-12) / np.maximum(current, 1e-12)
                            ),
                            axis=1,
                        )
                    )
                ),
            }
    return result


def collect_medium_states(
    suite: Path, evaluator: ArtifactEvaluator, target: int, seed: int
) -> list[dict[str, Any]]:
    states, used = [], set()
    for row in read_jsonl(suite):
        game = KalahGame.from_state(row["state"])
        for ply in range(200):
            if game.over() or not game.possible_moves():
                break
            state = game.to_state()
            key = stable_hash(state)
            if key not in used:
                used.add(key)
                states.append(
                    {
                        "state": state,
                        "source": "canonical_medium_trajectory",
                        "source_index": len(states),
                    }
                )
                if len(states) >= target:
                    return states
            search, _ = run_search(
                evaluator, state, 384, stable_seed(seed, "medium", key)
            )
            game.move(game.pit_index(search["selected_move"]))
    raise RuntimeError(f"only collected {len(states)} canonical medium states")


def enrich_and_select(
    candidates: list[dict[str, Any]],
    evaluator: ArtifactEvaluator,
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], deque[dict[str, Any]]] = defaultdict(deque)
    for row in candidates:
        game = KalahGame.from_state(row["state"])
        legal = game.possible_moves()
        if not legal:
            continue
        policy, _ = evaluator.evaluate(game)
        baseline, _ = run_search(
            evaluator,
            row["state"],
            384,
            stable_seed(seed, "margin", stable_hash(row["state"])),
        )
        row.update(
            {
                "state_hash": stable_hash(row["state"]),
                "player": int(game.current_player),
                "phase": phase_for_state(row["state"]),
                "legal_moves": legal,
                "current_policy_entropy": entropy(policy, legal),
                "current_384_visit_margin": baseline["visit_margin"],
            }
        )
        buckets[
            (
                row["player"],
                row["phase"],
                int(row["current_policy_entropy"] * 4),
                int(row["current_384_visit_margin"] // 8),
            )
        ].append(row)
    rng = random.Random(seed)
    queues = []
    for key in sorted(buckets, key=str):
        values = list(buckets[key])
        rng.shuffle(values)
        queues.append(deque(values))
    selected = []
    while queues and len(selected) < target:
        queue = queues.pop(0)
        if queue:
            selected.append(queue.popleft())
        if queue:
            queues.append(queue)
    if len(selected) != target:
        raise RuntimeError(f"only {len(selected)} eligible states; need {target}")
    return selected


def raw_metrics(
    states: list[dict[str, Any]],
    evaluators: dict[str, Any],
    teacher_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records = []
    for row in states:
        game = KalahGame.from_state(row["state"])
        legal = row["legal_moves"]
        policies = {
            name: legal_distribution(evaluator.evaluate(game)[0], legal)
            for name, evaluator in evaluators.items()
        }
        for name in ("D384", "D1200"):
            current, candidate = policies["current"], policies[name]
            cr, nr = ranked(current, legal), ranked(candidate, legal)
            records.append(
                {
                    "candidate": name,
                    "state_hash": row["state_hash"],
                    "teacher_disagreed": bool(
                        teacher_rows.get(row["state_hash"], {}).get("D384")
                        != teacher_rows.get(row["state_hash"], {}).get("D1200")
                    ),
                    "kl_candidate_current": kl(candidate, current, legal),
                    "js": js(candidate, current, legal),
                    "top1_changed": cr[0] != nr[0],
                    "delta_current_selected": float(candidate[cr[0]] - current[cr[0]]),
                    "delta_candidate_selected": float(
                        candidate[nr[0]] - current[nr[0]]
                    ),
                    "max_legal_delta": float(
                        np.max(np.abs(candidate[legal] - current[legal]))
                    ),
                    "mean_abs_rank_change": float(
                        np.mean(
                            [abs(cr.index(move) - nr.index(move)) for move in legal]
                        )
                    ),
                }
            )
    return aggregate(records, ("candidate", "teacher_disagreed"))


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["|".join(str(row[key]) for key in keys)].append(row)
    output = {}
    for name, group in groups.items():
        output[name] = {
            "n": len(group),
            **{
                key: float(np.mean([float(row[key]) for row in group]))
                for key in group[0]
                if key
                not in {
                    *keys,
                    "state_hash",
                    "current_move",
                    "candidate_move",
                    "seed",
                }
                and isinstance(group[0][key], (int, float, bool))
            },
        }
    return output


def search_response(
    states: list[dict[str, Any]],
    evaluators: dict[str, Any],
    seed: int,
    trace: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, traces = [], []
    for row in states:
        for budget in BUDGETS:
            base, base_engine = run_search(
                evaluators["current"],
                row["state"],
                budget,
                stable_seed(seed, row["state_hash"], budget),
                trace,
            )
            for name in (key for key in evaluators if key != "current"):
                candidate, candidate_engine = run_search(
                    evaluators[name],
                    row["state"],
                    budget,
                    stable_seed(seed, row["state_hash"], budget),
                    trace,
                )
                legal = row["legal_moves"]
                bp, cp = (
                    np.asarray(base["visit_policy"]),
                    np.asarray(candidate["visit_policy"]),
                )
                prior_base = legal_distribution(
                    evaluators["current"].evaluate(KalahGame.from_state(row["state"]))[
                        0
                    ],
                    legal,
                )
                prior_candidate = legal_distribution(
                    evaluators[name].evaluate(KalahGame.from_state(row["state"]))[0],
                    legal,
                )
                input_js, output_js = (
                    js(prior_candidate, prior_base, legal),
                    js(cp, bp, legal),
                )
                rows.append(
                    {
                        "state_hash": row["state_hash"],
                        "candidate": name,
                        "budget": budget,
                        "current_move": base["selected_move"],
                        "candidate_move": candidate["selected_move"],
                        "current_selected_q": base["child_q"],
                        "candidate_selected_q": candidate["child_q"],
                        "q_margin": abs(base["child_q"] - candidate["child_q"]),
                        "current_root_q_margin": base["root_q_margin"],
                        "selected_move_changed": base["selected_move"]
                        != candidate["selected_move"],
                        "visit_js": output_js,
                        "input_js": input_js,
                        "output_js": output_js,
                        "amplification_statistic": output_js / max(input_js, 1e-12),
                        "selected_visit_share_delta": candidate["selected_visit_share"]
                        - base["selected_visit_share"],
                        "child_q_delta": candidate["child_q"] - base["child_q"],
                        "root_value_delta": candidate["root_value"]
                        - base["root_value"],
                        "current_visit_margin": row["current_384_visit_margin"],
                        "prior_delta_on_candidate_move": float(
                            prior_candidate[candidate["selected_move"]]
                            - prior_base[candidate["selected_move"]]
                        ),
                    }
                )
                if trace and base["selected_move"] != candidate["selected_move"]:
                    traces.append(
                        first_divergence(
                            row, name, budget, base_engine, candidate_engine
                        )
                    )
    return rows, traces


def first_divergence(
    row: dict[str, Any], candidate: str, budget: int, left: TracePUCT, right: TracePUCT
) -> dict[str, Any]:
    for index, (a, b) in enumerate(zip(left.selections, right.selections), 1):
        if a["state_hash"] != b["state_hash"] or a["move"] != b["move"]:
            entries = {
                "current": a.get("entries", []),
                "candidate": b.get("entries", []),
            }
            return {
                "state_hash": row["state_hash"],
                "candidate": candidate,
                "budget": budget,
                "simulation": index,
                "node_state_hash": a["state_hash"],
                "current_move": a["move"],
                "candidate_move": b["move"],
                "entries": entries,
                "category": "already_different_subtree_visit_histories"
                if a["state_hash"] != b["state_hash"]
                else "tiny_prior_perturbation_near_tied_q_moves",
            }
    return {
        "state_hash": row["state_hash"],
        "candidate": candidate,
        "budget": budget,
        "simulation": None,
        "category": "other",
    }


def classify(
    search: dict[str, Any], medium: dict[str, Any], forced: dict[str, Any]
) -> tuple[list[str], str]:
    labels = []
    max_amp = max(
        (value.get("amplification_statistic", 0.0) for value in search.values()),
        default=0.0,
    )
    if max_amp > 5:
        labels.append("search_prior_bifurcation_confirmed")
    comparisons = medium.get("comparisons", {})
    forced_rows = forced.get("causal_move_delta", {})
    for direction in ("D384", "D1200"):
        low = comparisons.get(f"{direction}@0.25", {})
        full = comparisons.get(f"{direction}@1.00", {})
        low_signs = [
            low.get(pair, {}).get("score_delta", {}).get("mean", 0.0)
            for pair in ("384:256", "768:768", "1200:1200")
        ]
        causal_signs = [
            value.get("store_margin_delta", 0.0)
            for key, value in forced_rows.items()
            if key.startswith(f"{direction}@0.25|")
        ]
        if (
            any(value > 0 for value in low_signs)
            and any(value < 0 for value in low_signs)
            and any(value > 0 for value in causal_signs)
            and any(value < 0 for value in causal_signs)
        ):
            labels.append("policy_update_direction_intrinsically_conflicted")
        if full.get("1200:1200", {}).get("score_delta", {}).get(
            "mean", 0.0
        ) < 0 and any(
            low.get(pair, {}).get("score_delta", {}).get("mean", 0.0) > 0
            for pair in ("384:256", "768:768")
        ):
            labels.append("budget_specific_policy_target_conflict")
        intermediate = max(
            (
                comparisons.get(f"{direction}@{alpha:.2f}", {})
                .get("1200:1200", {})
                .get("score_delta", {})
                .get("mean", -float("inf"))
                for alpha in (0.25, 0.50)
            ),
            default=-float("inf"),
        )
        if (
            intermediate
            > full.get("1200:1200", {}).get("score_delta", {}).get("mean", 0.0)
            and intermediate >= 0
        ):
            labels.append("policy_update_overshoot_confirmed")
    if not labels:
        labels.append("policy_prior_change_not_primary")
    action_by_label = {
        "policy_update_overshoot_confirmed": "Test exactly one deterministic smaller policy update derived from the audit.",
        "budget_specific_policy_target_conflict": "Stop global policy-head distillation and investigate search-context-conditioned targets or joint iteration.",
        "search_prior_bifurcation_confirmed": "Prioritize a trust-region policy update in search-output space.",
        "policy_update_direction_intrinsically_conflicted": "Close stronger-teacher policy-head-only training.",
        "policy_prior_change_not_primary": "Policy priors are not the primary explanation under this audit.",
    }
    action = action_by_label[labels[0]]
    return labels, action


def bootstrap(values: list[float], *, seed: int) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "lower": 0.0, "upper": 0.0}
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(10_000, len(data)), replace=True).mean(axis=1)
    return {
        "n": len(data),
        "mean": float(data.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def play_medium_game(
    *,
    opening: dict[str, Any],
    opening_index: int,
    challenger: Any,
    challenger_player: int,
    challenger_budget: int,
    current: Any,
    current_budget: int,
    seed: int,
    suite_hash: str,
) -> dict[str, Any]:
    """Play one v2-seeded, deterministic paired-opening game."""
    game = KalahGame.from_state(opening["state"])
    opening_hash = stable_hash(opening["state"])
    moves = []
    for ply in range(200):
        if game.over() or not game.possible_moves():
            break
        role = "challenger" if game.current_player == challenger_player else "current"
        evaluator = challenger if role == "challenger" else current
        simulations = challenger_budget if role == "challenger" else current_budget
        search_seed, _ = derive_search_seed(
            contract_version=SEED_CONTRACT_VERSION,
            base_seed=seed,
            suite_sha256=suite_hash,
            opening_index=opening_index,
            opening_state_hash=opening_hash,
            challenger_player=challenger_player,
            game_within_opening=challenger_player,
            ply=ply,
            canonical_current_state_hash=stable_hash(game.to_state()),
            acting_role=role,
        )
        result, _engine = run_search(
            evaluator, game.to_state(), simulations, search_seed
        )
        move = int(result["selected_move"])
        if not game.move(game.pit_index(move)):
            raise RuntimeError("medium screen selected an illegal move")
        moves.append(move)
    margin = int(
        game.captured_seeds[challenger_player]
        - game.captured_seeds[1 - challenger_player]
    )
    return {
        "opening_index": opening_index,
        "challenger_player": challenger_player,
        "score": 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0,
        "margin": margin,
        "trajectory_hash": stable_hash(moves),
    }


def run_medium_screen(
    *,
    suite: Path,
    current_path: Path,
    candidate_paths: dict[str, Path],
    seed: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    openings = read_jsonl(suite)
    suite_hash = sha256_file(suite)
    budgets = ((384, 256), (768, 768), (1200, 1200), (1200, 256))
    tasks = [
        {
            "treatment": treatment,
            "budget_pair": f"{challenger_budget}:{current_budget}",
            "opening": opening,
            "opening_index": opening_index,
            "challenger_player": challenger_player,
            "challenger_budget": challenger_budget,
            "current_budget": current_budget,
            "seed": seed,
            "suite_hash": suite_hash,
        }
        for treatment in ("current", *candidate_paths)
        for challenger_budget, current_budget in budgets
        for opening_index, opening in enumerate(openings)
        for challenger_player in (0, 1)
    ]
    worker_count = max(1, min(int(workers), len(tasks)))
    initargs = (
        str(current_path),
        {name: str(path) for name, path in candidate_paths.items()},
    )
    if worker_count == 1:
        _init_medium_worker(*initargs)
        records = [_play_medium_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_medium_worker,
            initargs=initargs,
        ) as executor:
            records = list(executor.map(_play_medium_task, tasks))
    interpolations = {name: None for name in candidate_paths}
    comparisons: dict[str, Any] = {}
    for treatment in interpolations:
        comparisons[treatment] = {}
        for challenger_budget, current_budget in budgets:
            pair = f"{challenger_budget}:{current_budget}"
            candidates = {
                (row["opening_index"], row["challenger_player"]): row
                for row in records
                if row["treatment"] == treatment and row["budget_pair"] == pair
            }
            baseline = {
                (row["opening_index"], row["challenger_player"]): row
                for row in records
                if row["treatment"] == "current" and row["budget_pair"] == pair
            }
            comparisons[treatment][pair] = {
                "score_delta": bootstrap(
                    [
                        candidates[key]["score"] - baseline[key]["score"]
                        for key in candidates
                    ],
                    seed=stable_seed(seed, treatment, pair, "score"),
                ),
                "margin_delta": bootstrap(
                    [
                        candidates[key]["margin"] - baseline[key]["margin"]
                        for key in candidates
                    ],
                    seed=stable_seed(seed, treatment, pair, "margin"),
                ),
            }
    return {
        "seed_contract": SEED_CONTRACT_VERSION,
        "suite_sha256": suite_hash,
        "unique_openings": len(openings),
        "games_per_opening": 2,
        "comparisons": comparisons,
    }, records


def _init_medium_worker(current_path: str, candidate_paths: dict[str, str]) -> None:
    global _MEDIUM_CURRENT, _MEDIUM_INTERPOLATIONS
    _MEDIUM_CURRENT = ArtifactEvaluator(Path(current_path))
    _MEDIUM_INTERPOLATIONS = {}
    for name, path in candidate_paths.items():
        direction, raw_alpha = name.split("@", maxsplit=1)
        _MEDIUM_INTERPOLATIONS[name] = InterpolatedPolicyEvaluator(
            _MEDIUM_CURRENT, ArtifactEvaluator(Path(path)), float(raw_alpha)
        )


def _play_medium_task(task: dict[str, Any]) -> dict[str, Any]:
    if _MEDIUM_CURRENT is None:
        raise RuntimeError("medium worker was not initialized")
    treatment = task["treatment"]
    challenger = (
        _MEDIUM_CURRENT if treatment == "current" else _MEDIUM_INTERPOLATIONS[treatment]
    )
    return {
        "treatment": treatment,
        "budget_pair": task["budget_pair"],
        **play_medium_game(
            opening=task["opening"],
            opening_index=task["opening_index"],
            challenger=challenger,
            challenger_player=task["challenger_player"],
            challenger_budget=task["challenger_budget"],
            current=_MEDIUM_CURRENT,
            current_budget=task["current_budget"],
            seed=task["seed"],
            suite_hash=task["suite_hash"],
        ),
    }


def forced_current_continuation(
    *, state: dict[str, Any], move: int, budget: int, seed: int
) -> dict[str, Any]:
    if _CONTINUATION_CURRENT is None:
        raise RuntimeError("continuation worker was not initialized")
    game = KalahGame.from_state(state)
    root_player = int(game.current_player)
    if move not in game.possible_moves() or not game.move(game.pit_index(move)):
        raise RuntimeError(f"illegal forced move {move}")
    trajectory = [move]
    for ply in range(1, 200):
        if game.over() or not game.possible_moves():
            break
        result, _engine = run_search(
            _CONTINUATION_CURRENT,
            game.to_state(),
            budget,
            stable_seed(seed, stable_hash(game.to_state()), ply),
        )
        selected_move = int(result["selected_move"])
        game.move(game.pit_index(selected_move))
        trajectory.append(selected_move)
    margin = int(
        game.captured_seeds[root_player] - game.captured_seeds[1 - root_player]
    )
    return {
        "outcome_root": 1.0 if margin > 0 else 0.0 if margin == 0 else -1.0,
        "store_margin_root": margin,
        "trajectory_hash": stable_hash(trajectory),
    }


def _init_continuation_worker(current_path: str) -> None:
    global _CONTINUATION_CURRENT
    _CONTINUATION_CURRENT = ArtifactEvaluator(Path(current_path))


def _forced_continuation_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: value for key, value in task.items() if key not in {"state", "moves"}},
        "interventions": {
            str(move): forced_current_continuation(
                state=task["state"],
                move=move,
                budget=task["continuation_budget"],
                seed=task["seed"],
            )
            for move in task["moves"]
        },
    }


def quartiles(rows: list[dict[str, Any]], field: str, output: str) -> None:
    cutoffs = np.quantile([float(row[field]) for row in rows], [0.25, 0.5, 0.75])
    for row in rows:
        row[output] = 1 + sum(float(row[field]) > cutoff for cutoff in cutoffs)


def forced_move_quality(
    *,
    states: list[dict[str, Any]],
    search_rows: list[dict[str, Any]],
    current_path: Path,
    seed: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    states_by_hash = {row["state_hash"]: row for row in states}
    disagreements = [row for row in search_rows if row["selected_move_changed"]]
    if not disagreements:
        return {"disagreement_states": 0, "causal_move_delta": {}}, []
    quartiles(disagreements, "q_margin", "q_margin_quartile")
    quartiles(disagreements, "prior_delta_on_candidate_move", "prior_delta_quartile")
    tasks = [
        {
            "state": states_by_hash[row["state_hash"]]["state"],
            "state_hash": row["state_hash"],
            "candidate": row["candidate"],
            "original_budget": row["budget"],
            "q_margin_quartile": row["q_margin_quartile"],
            "prior_delta_quartile": row["prior_delta_quartile"],
            "current_move": row["current_move"],
            "candidate_move": row["candidate_move"],
            "moves": sorted({row["current_move"], row["candidate_move"]}),
            "continuation_budget": continuation_budget,
            "seed": stable_seed(seed, row["state_hash"], continuation_budget),
        }
        for row in disagreements
        for continuation_budget in (768, 1200)
    ]
    worker_count = max(1, min(int(workers), len(tasks)))
    if worker_count == 1:
        _init_continuation_worker(str(current_path))
        completed = [_forced_continuation_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_continuation_worker,
            initargs=(str(current_path),),
        ) as executor:
            completed = list(executor.map(_forced_continuation_task, tasks))
    records = []
    for row in completed:
        current_result = row["interventions"][str(row["current_move"])]
        candidate_result = row["interventions"][str(row["candidate_move"])]
        records.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"interventions", "state"}
                },
                "outcome_delta": candidate_result["outcome_root"]
                - current_result["outcome_root"],
                "store_margin_delta": candidate_result["store_margin_root"]
                - current_result["store_margin_root"],
            }
        )
    return {
        "disagreement_states": len({row["state_hash"] for row in disagreements}),
        "causal_move_delta": aggregate(
            records,
            (
                "candidate",
                "original_budget",
                "continuation_budget",
                "q_margin_quartile",
                "prior_delta_quartile",
            ),
        ),
    }, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_policy_prior_search_amplification"
    )
    parser.add_argument(
        "--source-workdir", default="/tmp/azlite_stronger_policy_teacher"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--d384",
        default="/tmp/azlite_stronger_policy_teacher/d384_policy_teacher_e1_run_a/artifact",
    )
    parser.add_argument(
        "--d1200",
        default="/tmp/azlite_stronger_policy_teacher/d1200_policy_teacher_e1_run_a/artifact",
    )
    parser.add_argument(
        "--medium-suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-count", type=int, default=1024)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--run-medium-screen", action="store_true")
    parser.add_argument("--run-forced-move-quality", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--write-reports",
        action="store_true",
        help="Write committed reports only after the full 1,024-state search audit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_reports and (args.skip_search or args.state_count != 1024):
        raise ValueError("committed reports require the full 1,024-state search audit")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current, d384, d1200, source = (
        Path(args.current),
        Path(args.d384),
        Path(args.d1200),
        Path(args.source_workdir),
    )
    provenance = parameter_provenance(current, d384, d1200, workdir)
    replay_provenance = verify_pr182_records(workdir, source)
    validation = validation_metrics(
        source, {"current": current, "D384": d384, "D1200": d1200}
    )
    base = ArtifactEvaluator(current)
    candidates = {
        "current": base,
        "D384": ArtifactEvaluator(d384),
        "D1200": ArtifactEvaluator(d1200),
    }
    target = args.state_count // 2
    medium = collect_medium_states(Path(args.medium_suite), base, target, args.seed)
    existing = {stable_hash(row["state"]) for row in medium}
    validation_indexes = set(
        np.load(source / "validation_source_indexes.npy", allow_pickle=False).tolist()
    )
    corpus = read_jsonl(source / "trajectory_replay.jsonl")
    other = [
        {
            "state": row["raw_state"],
            "source": "pr182_validation_selfplay",
            "source_index": index,
        }
        for index, row in enumerate(corpus)
        if index in validation_indexes and stable_hash(row["raw_state"]) not in existing
    ]
    states = enrich_and_select(medium, base, target, args.seed) + enrich_and_select(
        other, base, target, args.seed + 1
    )
    if len({row["state_hash"] for row in states}) != args.state_count:
        raise RuntimeError("probe states must be unique")
    write_jsonl(workdir / "probe_states.jsonl", states)
    teacher_low, teacher_high = (
        read_jsonl(source / "replay_D384.jsonl"),
        read_jsonl(source / "replay_D1200.jsonl"),
    )
    teacher_rows = {
        stable_hash(corpus[i]["raw_state"]): {
            "D384": int(teacher_low[i]["policy_teacher_telemetry"]["top_move"]),
            "D1200": int(teacher_high[i]["policy_teacher_telemetry"]["top_move"]),
        }
        for i in range(len(corpus))
    }
    raw = raw_metrics(states, candidates, teacher_rows)
    evaluators: dict[str, Any] = dict(candidates)
    for direction in ("D384", "D1200"):
        for alpha in ALPHAS:
            evaluators[f"{direction}@{alpha:.2f}"] = InterpolatedPolicyEvaluator(
                base, candidates[direction], alpha
            )
    rows, traces = (
        ([], [])
        if args.skip_search
        else search_response(states, evaluators, args.seed, trace=True)
    )
    write_jsonl(workdir / "search_response_records.jsonl", rows)
    write_jsonl(workdir / "first_divergence_records.jsonl", traces)
    search_summary = aggregate(rows, ("candidate", "budget"))
    trace_summary = aggregate(traces, ("candidate", "budget", "category"))
    medium = None
    if args.run_medium_screen:
        medium_candidates = {
            f"{direction}@{alpha:.2f}": {"D384": d384, "D1200": d1200}[direction]
            for direction in ("D384", "D1200")
            for alpha in (0.25, 0.50, 1.00)
        }
        medium, medium_records = run_medium_screen(
            suite=Path(args.medium_suite),
            current_path=current,
            candidate_paths=medium_candidates,
            seed=args.seed,
            workers=args.workers,
        )
        write_jsonl(workdir / "medium_screen_records.jsonl", medium_records)
    forced_quality = None
    if args.run_forced_move_quality:
        if args.skip_search:
            raise ValueError("forced-move quality requires fixed-state search records")
        forced_quality, forced_records = forced_move_quality(
            states=states,
            search_rows=rows,
            current_path=current,
            seed=args.seed,
            workers=args.workers,
        )
        write_jsonl(workdir / "forced_move_quality_records.jsonl", forced_records)
    labels, action = classify(search_summary, medium or {}, forced_quality or {})
    manifest = {
        "schema": "azlite_policy_prior_probe_v1",
        "state_count": len(states),
        "state_hashes": [row["state_hash"] for row in states],
        "source_counts": dict(Counter(row["source"] for row in states)),
        "player_counts": dict(Counter(str(row["player"]) for row in states)),
        "phase_counts": dict(Counter(row["phase"] for row in states)),
        "sha256": sha256_file(workdir / "probe_states.jsonl"),
    }
    summary = {
        "schema": SCHEMA,
        "guardrails": {
            "training": False,
            "new_replay": False,
            "new_teacher_budget": False,
            "value_change": False,
            "trunk_change": False,
            "production_interpolation": False,
        },
        "provenance": provenance,
        "pr182_replay_provenance": replay_provenance,
        "legal_mask_validation_metrics": validation,
        "probe_manifest": manifest,
        "raw_prior_deltas": raw,
        "search_response": search_summary,
        "first_divergence": trace_summary,
        "medium_causal_screen": medium,
        "forced_move_quality": forced_quality,
        "classification": labels[0],
        "classifications": labels,
        "next_action": action,
    }
    write_json(workdir / "summary_metrics.json", summary)
    if not args.write_reports:
        print(
            json.dumps(
                {
                    "classification": labels,
                    "states": len(states),
                    "workdir": str(workdir),
                }
            )
        )
        return 0
    write_json(
        REPO_ROOT
        / "docs/data/alphazero-lite-policy-prior-search-amplification-summary.json",
        summary,
    )
    markdown = (
        "# Policy-Prior Search-Amplification Audit\n\n"
        + f"- Classification: `{summary['classification']}`\n- Next action: {action}\n\n"
        + "## Aggregate Results\n\n```json\n"
        + json.dumps(
            {
                key: summary[key]
                for key in (
                    "provenance",
                    "legal_mask_validation_metrics",
                    "probe_manifest",
                    "raw_prior_deltas",
                    "search_response",
                    "first_divergence",
                    "medium_causal_screen",
                    "forced_move_quality",
                )
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```\n"
    )
    (
        REPO_ROOT / "docs/alphazero-lite-policy-prior-search-amplification-results.md"
    ).write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {"classification": labels, "states": len(states), "workdir": str(workdir)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
