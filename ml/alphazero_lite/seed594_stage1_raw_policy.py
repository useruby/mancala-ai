"""Preparation and analysis helpers for Seed 594 raw-policy Stage 1.

Qualification access is deliberately separated from the synthetic-only model
parity check. This module does not run PUCT, games, or qualification inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import arena, self_play, train
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
P_DIR = ROOT / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4"
SEED594_CHECKPOINT = (
    ROOT
    / "docs/data/seed594-exact-oracle-distillation/run-0001/seed594-exact-oracle-update-512.npz"
)
P_SHA256 = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
SEED594_SHA256 = "7fddf5d6cba7eea4e7b8e06ba89c5b936d28727470934a51c603a739768e9bdd"
INPUT_ENCODING = "kalah_v3"
PARITY_TOLERANCE = 1e-6
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 594
MEAN_REGRET_THRESHOLD = -0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model(checkpoint: Path) -> train.PolicyValueNet:
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    skipped = train.load_checkpoint_into_model(model, checkpoint)
    if skipped:
        raise ValueError(f"checkpoint_tensor_mismatch:{checkpoint.name}")
    model.eval()
    return model


def _synthetic_states() -> list[dict[str, Any]]:
    """Small explicit non-terminal fixtures; not loaded from any cohort."""
    return [
        {
            "player_pits": [1, 0, 4, 2, 0, 1],
            "opponent_pits": [0, 3, 1, 0, 2, 1],
            "player_store": 10,
            "opponent_store": 8,
            "current_player": 0,
        },
        {
            "player_pits": [0, 2, 0, 5, 1, 0],
            "opponent_pits": [4, 0, 1, 2, 0, 3],
            "player_store": 7,
            "opponent_store": 11,
            "current_player": 1,
        },
        {
            "player_pits": [2, 1, 0, 0, 6, 0],
            "opponent_pits": [1, 0, 3, 0, 2, 0],
            "player_store": 12,
            "opponent_store": 9,
            "current_player": 0,
        },
    ]


def _export_evaluator(
    model: train.PolicyValueNet, directory: Path
) -> arena.ArtifactEvaluator:
    arrays = train.checkpoint_from_model(model)
    weights = {key: value.tolist() for key, value in arrays.items()}
    (directory / "weights.json").write_text(
        json.dumps(weights, sort_keys=True, separators=(",", ":"))
    )
    metadata = {
        "input_encoding": INPUT_ENCODING,
        "architecture": {"model_type": "residual_v3"},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))
    return arena.ArtifactEvaluator(directory)


def _torch_policy(model: train.PolicyValueNet, state: dict[str, Any]) -> np.ndarray:
    game = KalahGame.from_state(state)
    encoded = torch.tensor(
        [self_play.encode_state(state, input_encoding=INPUT_ENCODING)],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        logits, _ = model(encoded)
    legal = game.possible_moves()
    if not legal:
        raise ValueError("synthetic_fixture_has_no_legal_action")
    masked = logits[0].clone()
    masked[[action for action in range(6) if action not in legal]] = torch.finfo(
        masked.dtype
    ).min
    return torch.softmax(masked, dim=0).cpu().numpy().astype(np.float32)


def deterministic_top_action(policy: list[float] | np.ndarray, legal: list[int]) -> int:
    """Return the lowest player-relative action index among exact maxima."""
    values = np.asarray(policy, dtype=np.float64)
    legal_actions = sorted(set(int(action) for action in legal))
    if (
        values.shape != (6,)
        or not legal_actions
        or any(action < 0 or action >= 6 for action in legal_actions)
    ):
        raise ValueError("invalid_policy_or_legal_actions")
    return min(legal_actions, key=lambda action: (-values[action], action))


def compare_root_policies(
    *,
    opening_index: int,
    current_player: int,
    p_policy: list[float] | np.ndarray,
    candidate_policy: list[float] | np.ndarray,
    legal_actions: list[int],
    action_margins: dict[str, int] | dict[int, int],
) -> dict[str, Any]:
    """Measure exact regret and WDL downgrade on one fully covered root."""
    legal = sorted(int(action) for action in legal_actions)
    margin_actions = [int(action) for action in action_margins]
    if (
        not legal
        or len(legal) != len(set(legal))
        or len(margin_actions) != len(set(margin_actions))
        or set(margin_actions) != set(legal)
    ):
        raise ValueError("incomplete_exact_action_coverage")
    p_top = deterministic_top_action(p_policy, legal)
    candidate_top = deterministic_top_action(candidate_policy, legal)
    if current_player not in (0, 1):
        raise ValueError("invalid_current_player")
    # The authenticated #559 native oracle records margins from the supplied
    # perspective_player, which was the side to move for every root.
    margins = {int(action): int(value) for action, value in action_margins.items()}
    best = max(margins.values())
    optimal_actions = sorted(
        action for action, margin in margins.items() if margin == best
    )
    p_values = np.asarray(p_policy, dtype=np.float64)
    candidate_values = np.asarray(candidate_policy, dtype=np.float64)
    if (
        not np.isfinite(p_values).all()
        or not np.isfinite(candidate_values).all()
        or np.any(p_values < 0)
        or np.any(candidate_values < 0)
    ):
        raise ValueError("invalid_policy_probabilities")

    def rank(value: int) -> int:
        return 1 if value > 0 else -1 if value < 0 else 0

    best_rank = rank(best)
    p_rank = rank(margins[p_top])
    candidate_rank = rank(margins[candidate_top])
    return {
        "opening_index": int(opening_index),
        "p_action": p_top,
        "candidate_action": candidate_top,
        "exact_optimal_actions": optimal_actions,
        "p_exact_optimal_top_action": p_top in optimal_actions,
        "candidate_exact_optimal_top_action": candidate_top in optimal_actions,
        "p_exact_optimal_prior_mass": float(
            sum(p_values[action] for action in optimal_actions)
        ),
        "candidate_exact_optimal_prior_mass": float(
            sum(candidate_values[action] for action in optimal_actions)
        ),
        "p_regret": best - margins[p_top],
        "candidate_regret": best - margins[candidate_top],
        "p_wdl_rank": p_rank,
        "candidate_wdl_rank": candidate_rank,
        "p_wdl_downgrade": p_rank < best_rank,
        "candidate_wdl_downgrade": candidate_rank < best_rank,
    }


def verify_synthetic_export_parity() -> dict[str, Any]:
    """Compare each checkpoint's PyTorch forward to existing NumPy evaluator.

    The only states consumed here are the literal fixtures in
    :func:`_synthetic_states`; qualification membership is not opened.
    """
    if sha256_file(P_DIR / "model.npz") != P_SHA256:
        raise ValueError("P_E4_checkpoint_identity_mismatch")
    if sha256_file(SEED594_CHECKPOINT) != SEED594_SHA256:
        raise ValueError("seed594_checkpoint_identity_mismatch")
    report: dict[str, Any] = {"states": len(_synthetic_states()), "models": {}}
    for label, checkpoint in (
        ("P-E4", P_DIR / "model.npz"),
        ("Seed 594", SEED594_CHECKPOINT),
    ):
        model = _load_model(checkpoint)
        with tempfile.TemporaryDirectory(
            prefix="seed594-stage1-parity-", dir=ROOT / ".tmp"
        ) as scratch:
            exported_evaluator = _export_evaluator(model, Path(scratch))
            serving_evaluator = (
                arena.ArtifactEvaluator(P_DIR)
                if label == "P-E4"
                else exported_evaluator
            )
            errors: list[float] = []
            for state in _synthetic_states():
                game = KalahGame.from_state(state)
                torch_prior = _torch_policy(model, state)
                serving_prior, _ = serving_evaluator.evaluate(game)
                exported_prior, _ = exported_evaluator.evaluate(game)
                legal = game.possible_moves()
                if any(
                    torch_prior[action] != 0
                    for action in range(6)
                    if action not in legal
                ):
                    raise ValueError("torch_legal_mask_failure")
                if any(
                    serving_prior[action] != 0
                    for action in range(6)
                    if action not in legal
                ):
                    raise ValueError("serving_legal_mask_failure")
                if any(
                    exported_prior[action] != 0
                    for action in range(6)
                    if action not in legal
                ):
                    raise ValueError("exported_legal_mask_failure")
                errors.append(
                    max(
                        float(np.max(np.abs(torch_prior - serving_prior))),
                        float(np.max(np.abs(torch_prior - exported_prior))),
                    )
                )
            maximum = max(errors)
            if maximum > PARITY_TOLERANCE:
                raise ValueError(f"export_inference_parity_failure:{label}:{maximum}")
            report["models"][label] = {
                "checkpoint_sha256": sha256_file(checkpoint),
                "synthetic_forward_count": len(errors),
                "max_abs_probability_difference": maximum,
                "legal_masks_exact": True,
            }
    report["status"] = "passed"
    report["qualification_roots_used_for_inference"] = False
    return report


def evaluate_membership_roots(*, authorization: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute frozen raw-policy scoring; do not call before review/authorization.

    This function is deliberately not called by preparation verification. It
    loads only the authenticated #570 membership and #559 oracle after the
    synthetic parity gate passes; it performs one forward per model/root and no
    search or games.
    """
    freeze_path = ROOT / "docs/data/seed594-stage1-raw-policy/execution-freeze.json"
    if (
        authorization.get("schema")
        != "seed594-stage1-reviewed-execution-authorization-v1"
        or authorization.get("status") != "reviewed_and_authorized"
        or authorization.get("qualification_inference_authorized") is not True
        or not authorization.get("review_url")
        or not authorization.get("reviewed_by")
        or authorization.get("execution_freeze_sha256") != sha256_file(freeze_path)
    ):
        raise PermissionError("reviewed_execution_authorization_required")
    verify_preparation_freeze()
    membership_path = (
        ROOT
        / "docs/data/seed570-policy-prior-distillation/qualification-membership.json"
    )
    oracle_path = (
        ROOT
        / "docs/data/seed559-fixed-position-feasibility/run-0003/oracle-records.jsonl"
    )
    roots = json.loads(membership_path.read_bytes())["roots"]
    oracle_rows = [
        json.loads(line) for line in oracle_path.read_text().splitlines() if line
    ]
    oracle_by_state_hash = {str(row["state_hash"]): row for row in oracle_rows}
    if len(roots) != 423 or len({int(row["opening_index"]) for row in roots}) != 423:
        raise ValueError("qualification_membership_count_or_opening_mismatch")
    if sha256_file(P_DIR / "model.npz") != P_SHA256:
        raise ValueError("P_E4_checkpoint_identity_mismatch")
    if sha256_file(SEED594_CHECKPOINT) != SEED594_SHA256:
        raise ValueError("seed594_checkpoint_identity_mismatch")
    candidate = _load_model(SEED594_CHECKPOINT)
    p_evaluator = arena.ArtifactEvaluator(P_DIR)
    output: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="seed594-stage1-candidate-", dir=ROOT / ".tmp"
    ) as scratch:
        candidate_evaluator = _export_evaluator(candidate, Path(scratch))
        for root in roots:
            state_hash = str(root["state_hash"])
            oracle = oracle_by_state_hash.get(state_hash)
            if oracle is None or oracle.get("coverage_complete") is not True:
                raise ValueError(f"missing_or_incomplete_oracle:{state_hash}")
            state = root["state"]
            game = KalahGame.from_state(state)
            legal = sorted(int(action) for action in game.possible_moves())
            if legal != sorted(int(action) for action in root["legal_moves"]):
                raise ValueError(f"membership_legal_action_mismatch:{state_hash}")
            if sorted(int(action) for action in oracle.get("legal_moves", [])) != legal:
                raise ValueError(f"oracle_legal_action_mismatch:{state_hash}")
            p_policy, _ = p_evaluator.evaluate(game)
            candidate_policy, _ = candidate_evaluator.evaluate(game)
            for policy in (p_policy, candidate_policy):
                if np.asarray(policy).shape != (6,) or not np.isfinite(policy).all():
                    raise ValueError(f"invalid_policy_output:{state_hash}")
                if any(
                    float(policy[action]) != 0.0
                    for action in range(6)
                    if action not in legal
                ):
                    raise ValueError(f"illegal_action_probability:{state_hash}")
            metrics = compare_root_policies(
                opening_index=int(root["opening_index"]),
                current_player=int(state["current_player"]),
                p_policy=p_policy,
                candidate_policy=candidate_policy,
                legal_actions=legal,
                action_margins=oracle["action_margins"],
            )
            output.append({"state_hash": state_hash, **metrics})
    return output


def verify_paired_result_membership(
    result_rows: list[dict[str, Any]], membership_roots: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require exact one-to-one state-hash/opening mapping to frozen membership."""
    if len(membership_roots) != 423:
        raise ValueError("membership_population_mismatch")
    expected: dict[int, str] = {}
    expected_hashes: set[str] = set()
    for root in membership_roots:
        opening = root["opening_index"]
        state_hash = root["state_hash"]
        if type(opening) is not int or not isinstance(state_hash, str):
            raise ValueError("membership_identity_field_type_mismatch")
        if opening in expected or state_hash in expected_hashes:
            raise ValueError("membership_contains_duplicate_identity")
        expected[opening] = state_hash
        expected_hashes.add(state_hash)
    if len(result_rows) != 423:
        raise ValueError("result_population_mismatch")
    seen_openings: set[int] = set()
    seen_hashes: set[str] = set()
    for row in result_rows:
        opening = row["opening_index"]
        state_hash = row["state_hash"]
        if type(opening) is not int or not isinstance(state_hash, str):
            raise ValueError("result_identity_field_type_mismatch")
        if opening in seen_openings or state_hash in seen_hashes:
            raise ValueError("result_contains_duplicate_identity")
        seen_openings.add(opening)
        seen_hashes.add(state_hash)
        if expected.get(opening) != state_hash:
            raise ValueError("result_membership_mapping_mismatch")
    if seen_openings != set(expected) or seen_hashes != expected_hashes:
        raise ValueError("result_membership_incomplete")
    return {
        "status": "verified",
        "root_count": 423,
        "state_hashes": sorted(seen_hashes),
        "opening_to_state_hash": {
            str(opening): expected[opening] for opening in sorted(expected)
        },
    }


def verify_results_against_authenticated_membership(
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Authenticate #570 membership, then verify every published result identity."""
    verify_preparation_freeze()
    membership_path = (
        ROOT
        / "docs/data/seed570-policy-prior-distillation/qualification-membership.json"
    )
    roots = json.loads(membership_path.read_bytes())["roots"]
    checked = verify_paired_result_membership(result_rows, roots)
    checked["membership_sha256"] = (
        "94e126af6ea88fb99a4230f84e5cf2a20248daa39ea66d25b68730ea0426d668"
    )
    return checked


def verify_preparation_freeze() -> dict[str, Any]:
    """Verify frozen bytes, then parity, then membership/oracle coverage.

    The order is intentional: no qualification package is opened until both
    synthetic model parity checks have passed.
    """
    package = ROOT / "docs/data/seed594-stage1-raw-policy"
    freeze = json.loads((package / "execution-freeze.json").read_bytes())
    if freeze.get("status") != "prepared-pending-independent-review-no-inference":
        raise ValueError("preparation_status_mismatch")
    if freeze.get("base_revision") != "7c2d36ea1ee09987d0b6f7a8f972bf03e9c5525e":
        raise ValueError("preparation_base_revision_mismatch")
    if freeze.get("checkpoint_identity") != {
        "Seed 594": SEED594_SHA256,
        "P-E4": P_SHA256,
    }:
        raise ValueError("preparation_checkpoint_identity_mismatch")
    deviation = freeze.get("access_order_deviation", {})
    if (
        deviation.get("public_membership_file_opened_before_synthetic_parity_checks")
        is not True
        or deviation.get("historical_deviation_preserved") is not True
        or deviation.get("qualification_inference_performed") is not False
        or deviation.get("qualification_outcome_analysis_performed") is not False
        or deviation.get("cohort_characterization")
        != "existing-preexisting-membership-not-fresh-or-sealed"
    ):
        raise ValueError("access_order_deviation_disclosure_mismatch")
    analysis = freeze.get("analysis", {})
    if (
        analysis.get("mean_gate")
        != "mean candidate-minus-P exact regret <= -0.10 seeds"
        or analysis.get("paired_ci_gate")
        != "95% opening-cluster bootstrap interval upper endpoint < 0"
        or analysis.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES
        or analysis.get("bootstrap_rng") != "numpy.default_rng(594)"
        or analysis.get("bootstrap_quantile_method") != "linear"
        or analysis.get("bootstrap_quantiles") != [0.025, 0.975]
        or analysis.get("bootstrap_clusters")
        != "423 opening indices, one retained root per opening"
        or analysis.get("result_membership_verification")
        != "exact 423 state hashes and opening-index-to-state-hash mapping; reject omission, substitution, duplicate hash/opening, or mismapped opening"
        or set(analysis.get("per_model_metrics", {}))
        != {
            "exact_optimal_top_action_rate",
            "exact_optimal_prior_probability_mass",
            "exact_regret",
            "wdl_downgrade",
        }
    ):
        raise ValueError("preparation_analysis_binding_mismatch")
    limits = freeze.get("execution_limits", {})
    if any(
        limits.get(key) is not False
        for key in (
            "qualification_inference",
            "puct",
            "games",
            "sealed_suite_access",
            "execution_authorized",
        )
    ):
        raise ValueError("preparation_execution_limit_mismatch")
    torch.set_num_threads(1)
    for relative, expected in freeze["pre_parity_file_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"preparation_binding_mismatch:{relative}")
    expected_runtime = freeze["runtime"]
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "execution_threads": torch.get_num_threads(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    if observed_runtime != expected_runtime:
        raise ValueError(f"runtime_mismatch:{observed_runtime}")
    parity = verify_synthetic_export_parity()
    for relative, expected in freeze["post_parity_identity_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"qualification_identity_binding_mismatch:{relative}")
    from ml.alphazero_lite.seed570_qualification_cohort import verify_package

    membership = verify_package(
        ROOT,
        ROOT
        / "docs/data/seed570-policy-prior-distillation/qualification-membership.json",
    )
    if membership["population"] != 423:
        raise ValueError("qualification_membership_population_mismatch")
    return {
        "status": "verified-pending-review-no-inference",
        "synthetic_parity": parity,
        "membership_population": membership["population"],
        "membership_sha256": membership["membership_sha256"],
        "oracle_identity_only": True,
        "qualification_inference_performed": False,
        "qualification_outcomes_analyzed": False,
    }


def paired_opening_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct paired exact-regret CI, resampling opening clusters.

    Each row has ``opening_index``, ``candidate_regret``, and ``p_regret``.
    """
    by_opening: dict[int, list[float]] = {}
    for row in rows:
        opening = int(row["opening_index"])
        difference = float(row["candidate_regret"]) - float(row["p_regret"])
        if not np.isfinite(difference):
            raise ValueError("nonfinite_regret_difference")
        by_opening.setdefault(opening, []).append(difference)
    if len(by_opening) != 423 or any(
        len(values) != 1 for values in by_opening.values()
    ):
        raise ValueError("incomplete_or_duplicate_opening_results")
    cluster_means = np.asarray(
        [by_opening[index][0] for index in sorted(by_opening)], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(
        0, len(cluster_means), size=(BOOTSTRAP_RESAMPLES, len(cluster_means))
    )
    means = cluster_means[samples].mean(axis=1)
    return {
        "mean_candidate_minus_p_regret": float(cluster_means.mean()),
        "paired_opening_bootstrap_ci95": [
            float(value)
            for value in np.quantile(means, [0.025, 0.975], method="linear")
        ],
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "rng": "numpy.default_rng(594)",
        "quantile_method": "linear",
        "opening_clusters": len(cluster_means),
    }


def analyze_root_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build registered paired aggregate from one result per frozen opening."""
    opening_ids = [int(row["opening_index"]) for row in rows]
    if len(rows) != 423 or len(set(opening_ids)) != 423:
        raise ValueError("incomplete_or_duplicate_opening_results")
    bootstrap = paired_opening_bootstrap(rows)
    return {
        **bootstrap,
        "p_mean_exact_regret": float(np.mean([row["p_regret"] for row in rows])),
        "candidate_mean_exact_regret": float(
            np.mean([row["candidate_regret"] for row in rows])
        ),
        "p_exact_optimal_top_action_rate": float(
            np.mean([bool(row["p_exact_optimal_top_action"]) for row in rows])
        ),
        "candidate_exact_optimal_top_action_rate": float(
            np.mean([bool(row["candidate_exact_optimal_top_action"]) for row in rows])
        ),
        "p_exact_optimal_prior_probability_mass": float(
            np.mean([row["p_exact_optimal_prior_mass"] for row in rows])
        ),
        "candidate_exact_optimal_prior_probability_mass": float(
            np.mean([row["candidate_exact_optimal_prior_mass"] for row in rows])
        ),
        "candidate_wdl_downgrade_rate": float(
            np.mean([bool(row["candidate_wdl_downgrade"]) for row in rows])
        ),
        "p_wdl_downgrade_rate": float(
            np.mean([bool(row["p_wdl_downgrade"]) for row in rows])
        ),
        "candidate_wdl_downgrade_count": sum(
            row["candidate_wdl_downgrade"] for row in rows
        ),
        "p_wdl_downgrade_count": sum(row["p_wdl_downgrade"] for row in rows),
    }


def qualify_gate(analysis: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen Stage 1 advancement gate without imputing missing data."""
    if (
        analysis.get("opening_clusters") != 423
        or analysis.get("resamples") != BOOTSTRAP_RESAMPLES
        or analysis.get("seed") != BOOTSTRAP_SEED
        or analysis.get("quantile_method") != "linear"
    ):
        raise ValueError("incomplete_analysis")
    ci = analysis.get("paired_opening_bootstrap_ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("invalid_bootstrap_interval")
    mean_difference = float(analysis.get("mean_candidate_minus_p_regret", float("nan")))
    ci_upper = float(ci[1])
    p_wdl = float(analysis.get("p_wdl_downgrade_rate", float("nan")))
    candidate_wdl = float(analysis.get("candidate_wdl_downgrade_rate", float("nan")))
    if not all(
        np.isfinite(value)
        for value in (mean_difference, ci_upper, p_wdl, candidate_wdl)
    ):
        raise ValueError("incomplete_analysis_values")
    checks = {
        "mean_regret_improvement": mean_difference <= MEAN_REGRET_THRESHOLD,
        "ci_upper_below_zero": ci_upper < 0.0,
        "no_wdl_downgrade_increase": candidate_wdl <= p_wdl,
    }
    return {"checks": checks, "decision": "keep" if all(checks.values()) else "reject"}
