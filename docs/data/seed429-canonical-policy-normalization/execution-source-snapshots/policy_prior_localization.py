#!/usr/bin/env python3
"""Per-depth incumbent-prior substitution for the PR #200 policy-head localization.

This module provides the single override primitive reused by both the canonical
arena (``arena.py``) and the PUCT probe without introducing a second MCTS. The
``PUCT`` search (``self_play.py``) is extended with a ``prior_override`` callable
applied at *every* expanded node with the node's tree-search depth; this module
defines that callable.

Modes (``depth`` is tree-search depth from the current move's root):

- ``candidate_all``     : candidate policy at every node (no override; baseline).
- ``incumbent_root``    : incumbent policy at root only (depth 0).
- ``incumbent_depth1``  : incumbent policy at depths 0 and 1.
- ``incumbent_depth2``  : incumbent policy at depths 0, 1, and 2.
- ``incumbent_all``     : incumbent policy at every node (candidate value path
  unchanged).

The override never touches value outputs, ``c_puct``, simulations, or weights.
It only replaces the legal-normalized policy prior at expanded nodes whose depth
satisfies the mode, using the incumbent evaluator's legal-masked policy. The
candidate evaluator continues to provide the value at every node, so the
candidate/value path is otherwise unchanged (and for the PR #200 frozen-trunk
``policy_head`` lane the candidate value stack is byte-identical to the
incumbent, hence ``incumbent_all`` is search-equivalent to the incumbent).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

PRIOR_OVERRIDE_MODES: tuple[str, ...] = (
    "candidate_all",
    "incumbent_root",
    "incumbent_depth1",
    "incumbent_depth2",
    "incumbent_all",
)

# Maximum depth (inclusive) at which the incumbent prior is substituted.
# ``candidate_all`` uses the incumbent at no depth (threshold = -1); the root
# is depth 0, so ``incumbent_root`` substitutes at depth <= 0, and so on.
MODE_DEPTH_THRESHOLD: dict[str, int] = {
    "candidate_all": -1,
    "incumbent_root": 0,
    "incumbent_depth1": 1,
    "incumbent_depth2": 2,
    "incumbent_all": 2**31 - 1,
}


def mode_uses_incumbent_at_depth(mode: str, depth: int) -> bool:
    """Return whether ``mode`` substitutes the incumbent prior at ``depth``."""
    if mode not in MODE_DEPTH_THRESHOLD:
        raise ValueError(f"unknown prior override mode: {mode!r}")
    return depth <= MODE_DEPTH_THRESHOLD[mode]


def _legal_normalize(prior: np.ndarray, legal_moves: list[int]) -> np.ndarray:
    out = np.zeros_like(prior, dtype=np.float32)
    if not legal_moves:
        return out
    out[legal_moves] = np.asarray(prior, dtype=np.float32)[legal_moves]
    total = float(np.sum(out[legal_moves]))
    if not np.isfinite(total) or total <= 0.0:
        out[legal_moves] = 1.0 / len(legal_moves)
        return out.astype(np.float32)
    out[legal_moves] /= total
    return out.astype(np.float32)


class PriorSubstitutionOverride:
    """Per-depth incumbent/candidate prior selector for ``PUCT.prior_override``.

    ``incumbent_evaluator`` is any object exposing
    ``evaluate(game) -> (policy_vector, value)`` (e.g. ``ArtifactEvaluator`` or
    ``CheckpointEvaluator``); only its policy output is consumed. The callable
    signature matches the PUCT per-depth override contract
    ``(*, game, legal_moves, priors, depth) -> np.ndarray``.
    """

    def __init__(
        self,
        mode: str,
        incumbent_evaluator: Any,
        *,
        record_telemetry: bool = True,
        state_hasher: Callable[..., str] | None = None,
    ) -> None:
        if mode not in MODE_DEPTH_THRESHOLD:
            raise ValueError(f"unknown prior override mode: {mode!r}")
        if incumbent_evaluator is None and mode != "candidate_all":
            raise ValueError(
                "incumbent_evaluator is required for non-candidate_all modes"
            )
        self.mode = mode
        self.incumbent_evaluator: Any = incumbent_evaluator
        self.threshold = MODE_DEPTH_THRESHOLD[mode]
        self.record_telemetry = bool(record_telemetry)
        self.state_hasher = state_hasher
        self.last_telemetry: dict[str, Any] | None = None
        # Per-node telemetry log consumed by the PUCT probe reporting.
        self.telemetry_log: list[dict[str, Any]] = []

    # PUCT calls this with keyword arguments; ``depth`` defaults to 0 so the
    # callable is also safe to use as a root-only override if needed.
    def __call__(
        self,
        *,
        game,
        legal_moves: list[int],
        priors: np.ndarray,
        depth: int = 0,
    ) -> np.ndarray:
        substituted = depth <= self.threshold and bool(legal_moves)
        entry: dict[str, Any] = {
            "depth": int(depth),
            "legal_move_count": int(len(legal_moves)),
            "substituted": bool(substituted),
        }
        if self.state_hasher is not None:
            try:
                entry["state_hash"] = self.state_hasher(game)
            except Exception:
                entry["state_hash"] = None

        if not substituted:
            if self.record_telemetry:
                self.telemetry_log.append(entry)
            self.last_telemetry = entry
            return np.asarray(priors, dtype=np.float32)

        incumbent_policy, _ = self.incumbent_evaluator.evaluate(game)
        incumbent_policy = np.asarray(incumbent_policy, dtype=np.float32)
        incumbent_masked = _legal_normalize(incumbent_policy, legal_moves)

        if self.record_telemetry and legal_moves:
            candidate_masked = _legal_normalize(
                np.asarray(priors, dtype=np.float32), legal_moves
            )
            diff = incumbent_masked[legal_moves] - candidate_masked[legal_moves]
            entry["candidate_legal_l1"] = float(
                np.sum(np.abs(candidate_masked[legal_moves]))
            )
            entry["incumbent_legal_l1"] = float(
                np.sum(np.abs(incumbent_masked[legal_moves]))
            )
            entry["pairwise_legal_l1"] = float(np.sum(np.abs(diff)))
            entry["pairwise_legal_js"] = float(
                _js(incumbent_masked, candidate_masked, legal_moves)
            )
            entry["top1_incumbent_move"] = int(
                legal_moves[int(np.argmax(incumbent_masked[legal_moves]))]
            )
            entry["top1_candidate_move"] = int(
                legal_moves[int(np.argmax(candidate_masked[legal_moves]))]
            )
            entry["top1_changed"] = (
                entry["top1_incumbent_move"] != entry["top1_candidate_move"]
            )

        if self.record_telemetry:
            self.telemetry_log.append(entry)
        self.last_telemetry = entry
        return incumbent_masked


class TailPriorSubstitutionOverride:
    """Replace a candidate prior with P1 only when legal-policy L1 is in a tail.

    The PUCT evaluator remains P2, so this callable changes only the expanded
    node's prior.  ``threshold=None`` is the incumbent-all control.
    """

    def __init__(
        self,
        incumbent_evaluator: Any,
        *,
        threshold: float | None,
        record_telemetry: bool = True,
    ) -> None:
        if threshold is not None and (not np.isfinite(threshold) or threshold < 0.0):
            raise ValueError("threshold must be a finite non-negative L1 value")
        self.incumbent_evaluator = incumbent_evaluator
        self.threshold = threshold
        self.record_telemetry = bool(record_telemetry)
        self.last_telemetry: dict[str, Any] | None = None
        self.telemetry_log: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        game,
        legal_moves: list[int],
        priors: np.ndarray,
        depth: int = 0,
    ) -> np.ndarray:
        candidate = _legal_normalize(np.asarray(priors, dtype=np.float32), legal_moves)
        incumbent_policy, _ = self.incumbent_evaluator.evaluate(game)
        incumbent = _legal_normalize(
            np.asarray(incumbent_policy, dtype=np.float32), legal_moves
        )
        l1 = float(np.sum(np.abs(candidate[legal_moves] - incumbent[legal_moves])))
        substituted = bool(legal_moves) and (
            self.threshold is None or l1 >= self.threshold
        )
        entry = {
            "depth": int(depth),
            "player_to_move": int(game.current_player),
            "legal_move_count": int(len(legal_moves)),
            "candidate_vs_parent_legal_l1": l1,
            "substituted": substituted,
        }
        if self.record_telemetry:
            self.telemetry_log.append(entry)
        self.last_telemetry = entry
        return incumbent if substituted else candidate


def build_tail_prior_substitution_override(
    incumbent_evaluator: Any, *, threshold: float | None, **kwargs: Any
) -> TailPriorSubstitutionOverride:
    """Build an all-depth calibrated-tail override (``None`` means all nodes)."""
    return TailPriorSubstitutionOverride(
        incumbent_evaluator, threshold=threshold, **kwargs
    )


def _js(left: np.ndarray, right: np.ndarray, legal_moves: list[int]) -> float:
    """Jensen-Shannon divergence (nats) restricted to legal moves."""
    p = np.clip(left[legal_moves].astype(np.float64), 1e-12, None)
    q = np.clip(right[legal_moves].astype(np.float64), 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def build_prior_substitution_override(
    mode: str,
    incumbent_evaluator: Any | None = None,
    **kwargs: Any,
) -> PriorSubstitutionOverride | None:
    """Return the override callable for ``mode`` (``None`` for ``candidate_all``)."""
    if mode == "candidate_all":
        return None
    if incumbent_evaluator is None:
        raise ValueError(
            "incumbent_evaluator is required for non-candidate_all prior override modes"
        )
    return PriorSubstitutionOverride(mode, incumbent_evaluator, **kwargs)


def summarize_override_telemetry(
    log: Iterable[dict[str, Any]],
    *,
    max_depth: int = 4,
) -> dict[str, Any]:
    """Aggregate the per-node override telemetry into a by-depth summary."""
    records = list(log)
    by_depth: dict[int, dict[str, Any]] = {}

    def bucket(depth: int) -> dict[str, Any]:
        d = min(depth, max_depth)
        if d not in by_depth:
            by_depth[d] = {
                "depth": d,
                "expanded_nodes": 0,
                "affected_nodes": 0,
                "pairwise_legal_l1": [],
                "pairwise_legal_js": [],
                "top1_changed": 0,
            }
        return by_depth[d]

    for record in records:
        depth = min(int(record["depth"]), max_depth)
        b = bucket(depth)
        b["expanded_nodes"] += 1
        if record.get("substituted"):
            b["affected_nodes"] += 1
            if "pairwise_legal_l1" in record:
                b["pairwise_legal_l1"].append(float(record["pairwise_legal_l1"]))
            if "pairwise_legal_js" in record:
                b["pairwise_legal_js"].append(float(record["pairwise_legal_js"]))
            if record.get("top1_changed"):
                b["top1_changed"] += 1

    summary: dict[str, Any] = {}
    total_expanded = sum(b["expanded_nodes"] for b in by_depth.values())
    total_affected = sum(b["affected_nodes"] for b in by_depth.values())
    for depth in sorted(by_depth):
        b = by_depth[depth]
        affected = b["affected_nodes"]
        summary[str(depth)] = {
            "expanded_nodes": b["expanded_nodes"],
            "affected_nodes": affected,
            "affected_fraction": (
                affected / b["expanded_nodes"] if b["expanded_nodes"] else 0.0
            ),
            "mean_pairwise_legal_l1": (
                float(np.mean(b["pairwise_legal_l1"]))
                if b["pairwise_legal_l1"]
                else 0.0
            ),
            "mean_pairwise_legal_js": (
                float(np.mean(b["pairwise_legal_js"]))
                if b["pairwise_legal_js"]
                else 0.0
            ),
            "top1_change_rate": (
                affected / b["expanded_nodes"] if b["expanded_nodes"] else 0.0
            ),
        }
    summary["total_expanded_nodes"] = total_expanded
    summary["total_affected_nodes"] = total_affected
    summary["overall_affected_fraction"] = (
        total_affected / total_expanded if total_expanded else 0.0
    )
    return summary
