"""Independent, read-only verifier and analysis of a #588 prediction package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed588_raw_prior_diagnostic import (
    EXPECTED_ARTIFACTS,
    EXPECTED_INPUTS,
    VerificationError,
    _jsonl,
    _sha,
    verify as verify_sources,
)

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 58_820_261_114


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_membership(
    records: list[dict[str, Any]], expected_hashes: set[str], label: str
) -> None:
    """Require exactly one complete record for every frozen root."""
    observed = [row.get("state_hash") for row in records]
    if (
        len(records) != 512
        or len(set(observed)) != 512
        or set(observed) != expected_hashes
    ):
        raise VerificationError(f"{label}_membership_or_count")


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = int(location)
    high = min(low + 1, len(ordered) - 1)
    fraction = location - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _wdl(margin: int) -> int:
    return (margin > 0) - (margin < 0)


def _root_metrics(
    p: list[float], t: list[float], legal: list[int], margins: dict[int, int]
) -> dict[str, float]:
    best_margin = max(margins[action] for action in legal)
    optimal = {action for action in legal if margins[action] == best_margin}
    p_top = min(legal, key=lambda action: (-p[action], action))
    t_top = min(legal, key=lambda action: (-t[action], action))
    best_wdl = max(_wdl(margins[action]) for action in legal)
    p_regret = float(best_margin - margins[p_top])
    t_regret = float(best_margin - margins[t_top])
    p_selected = float(p_top in optimal)
    t_selected = float(t_top in optimal)
    p_downgrade = float(_wdl(margins[p_top]) < best_wdl)
    t_downgrade = float(_wdl(margins[t_top]) < best_wdl)
    return {
        "p_optimal_mass": sum(p[action] for action in optimal),
        "t512_optimal_mass": sum(t[action] for action in optimal),
        "optimal_mass_delta": sum(t[action] for action in optimal)
        - sum(p[action] for action in optimal),
        "p_top_prior_regret": p_regret,
        "t512_top_prior_regret": t_regret,
        "top_prior_regret_delta": t_regret - p_regret,
        "p_optimal_selection": p_selected,
        "t512_optimal_selection": t_selected,
        "optimal_selection_delta": t_selected - p_selected,
        "p_wdl_downgrade": p_downgrade,
        "t512_wdl_downgrade": t_downgrade,
        "wdl_downgrade_delta": t_downgrade - p_downgrade,
    }


def independently_analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute paired estimates and linear percentile bootstrap from roots."""
    if len(rows) != 512 or len({row["opening_index"] for row in rows}) != 512:
        raise VerificationError("analysis_root_count_or_cluster_uniqueness")
    names = (
        "optimal_mass_delta",
        "top_prior_regret_delta",
        "optimal_selection_delta",
        "wdl_downgrade_delta",
    )
    values = [row["metrics"] for row in rows]
    mean = {name: sum(row[name] for row in values) / 512 for name in names}
    arm_fields = (
        "p_optimal_mass",
        "t512_optimal_mass",
        "p_top_prior_regret",
        "t512_top_prior_regret",
        "p_optimal_selection",
        "t512_optimal_selection",
        "p_wdl_downgrade",
        "t512_wdl_downgrade",
    )
    arm_means = {name: sum(row[name] for row in values) / 512 for name in arm_fields}
    rng = random.Random(BOOTSTRAP_SEED)
    draws = {name: [] for name in names}
    for _ in range(BOOTSTRAP_RESAMPLES):
        selected = [values[rng.randrange(512)] for _ in range(512)]
        for name in names:
            draws[name].append(sum(row[name] for row in selected) / 512)
    intervals = {
        name: [_quantile(draws[name], 0.025), _quantile(draws[name], 0.975)]
        for name in names
    }
    p_regret = sum(row["p_top_prior_regret"] for row in values) / 512
    t_regret = sum(row["t512_top_prior_regret"] for row in values) / 512
    should_advance = (
        mean["optimal_mass_delta"] > 0
        and intervals["optimal_mass_delta"][0] > 0
        and t_regret <= p_regret
    )
    return {
        "roots": 512,
        "paired_point_estimates": mean,
        "arm_point_estimates": arm_means,
        "paired_bootstrap_95_linear": intervals,
        "mean_p_top_prior_exact_regret": p_regret,
        "mean_t512_top_prior_exact_regret": t_regret,
        "bootstrap": {
            "cluster": "opening_index",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "rng": "random.Random; randrange with replacement",
            "quantiles": [0.025, 0.975],
            "interpolation": "linear",
        },
        "decision": "may_propose_separately_frozen_search_diagnostic"
        if should_advance
        else "do_not_propose_search_followup",
        "model_promotion": False,
        "generalization_claim": False,
    }


def verify_result(
    *,
    root: Path,
    package: Path,
    trusted_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify external manifest trust, source parity, membership, and metrics."""
    if not trusted_manifest_sha256:
        raise VerificationError("trusted_run_manifest_digest_required")
    package = package if package.is_absolute() else root / package
    manifest_path = package / "run-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if _sha(manifest_bytes) != trusted_manifest_sha256:
        raise VerificationError("run_manifest_trust_digest_mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "seed588-raw-prior-run-v1":
        raise VerificationError("run_manifest_schema")
    if manifest.get("source_evidence_sha256") != EXPECTED_INPUTS:
        raise VerificationError("run_source_binding")
    expected_freeze = (
        root / "docs/data/seed588-retrospective-raw-prior/execution-freeze-v2.json"
    )
    if manifest.get("execution_freeze_sha256") != _sha_file(expected_freeze):
        raise VerificationError("run_execution_freeze_binding")
    if manifest.get("artifacts") != EXPECTED_ARTIFACTS:
        raise VerificationError("run_artifact_binding")
    if manifest.get("p_parity_passed") is not True:
        raise VerificationError("p_parity_not_passed")
    if (
        manifest.get("p_parity_forward_evaluations") != 512
        or manifest.get("t512_forward_evaluations") != 512
    ):
        raise VerificationError("run_forward_count")
    if manifest.get("mcts") or manifest.get("games") or manifest.get("training"):
        raise VerificationError("run_scope_violation")
    authorization_digest = manifest.get("authorization_sha256")
    if (
        not isinstance(authorization_digest, str)
        or len(authorization_digest) != 64
        or any(
            character not in "0123456789abcdef" for character in authorization_digest
        )
    ):
        raise VerificationError("run_authorization_binding")
    file_bindings = manifest.get("result_files_sha256")
    required_files = {
        "p-parity.jsonl",
        "t512-predictions.jsonl",
        "root-results.jsonl",
        "analysis.json",
    }
    if not isinstance(file_bindings, dict) or set(file_bindings) != required_files:
        raise VerificationError("result_file_binding_shape")
    for name, expected in file_bindings.items():
        if _sha_file(package / name) != expected:
            raise VerificationError(f"result_file_hash:{name}")

    source_status = verify_sources(root)
    if source_status["roots"] != 512:
        raise VerificationError("source_cohort_count")
    source_dir = root / "docs/data/seed559-fixed-position-feasibility/run-0003"
    roots = json.loads((source_dir / "cohort-manifest.json").read_bytes())["states"]
    archived = {
        row["state_hash"]: row
        for row in _jsonl((source_dir / "search-records.jsonl").read_bytes())
    }
    oracles = {
        row["state_hash"]: row
        for row in _jsonl((source_dir / "oracle-records.jsonl").read_bytes())
    }
    parity = _jsonl((package / "p-parity.jsonl").read_bytes())
    predictions = _jsonl((package / "t512-predictions.jsonl").read_bytes())
    published_root_results = _jsonl((package / "root-results.jsonl").read_bytes())
    expected_hashes = {row["state_hash"] for row in roots}
    validate_membership(parity, expected_hashes, "parity")
    validate_membership(predictions, expected_hashes, "prediction")
    validate_membership(published_root_results, expected_hashes, "root_metrics")
    parity_by_hash = {row["state_hash"]: row for row in parity}
    predictions_by_hash = {row["state_hash"]: row for row in predictions}
    result_rows = []
    observed_parity_max = 0.0
    for root_row in roots:
        key = root_row["state_hash"]
        p_row = parity_by_hash[key]
        t_row = predictions_by_hash[key]
        if int(p_row["opening_index"]) != int(root_row["opening_index"]) or int(
            t_row["opening_index"]
        ) != int(root_row["opening_index"]):
            raise VerificationError("opening_identity_mismatch")
        legal = sorted(int(move) for move in archived[key]["legal_moves"])
        if sorted(int(move) for move in p_row["legal_relative_actions"]) != legal:
            raise VerificationError("relative_action_encoding_mismatch")
        archived_priors = [float(value) for value in archived[key]["priors"]]
        p_forward = [float(value) for value in p_row["p_forward_priors"]]
        p_recorded_archive = [float(value) for value in p_row["archived_p_priors"]]
        t_prior = [float(value) for value in t_row["t512_priors"]]
        if p_recorded_archive != archived_priors:
            raise VerificationError("archived_p_prior_evidence_mismatch")
        for label, vector in (("p", p_forward), ("t512", t_prior)):
            if len(vector) != 6 or any(not math.isfinite(value) for value in vector):
                raise VerificationError(f"{label}_prior_shape_or_finiteness")
            if any(vector[action] != 0 for action in range(6) if action not in legal):
                raise VerificationError(f"{label}_illegal_action_prior")
            if abs(sum(vector[action] for action in legal) - 1.0) > 2e-6:
                raise VerificationError(f"{label}_prior_normalization")
        maximum_error = max(
            abs(p_forward[action] - archived_priors[action]) for action in range(6)
        )
        observed_parity_max = max(observed_parity_max, maximum_error)
        if (
            maximum_error > 1e-6
            or abs(float(p_row["max_abs_difference"]) - maximum_error) > 1e-12
        ):
            raise VerificationError("p_prior_parity_failure")
        margins = {
            int(move): int(value)
            for move, value in oracles[key]["action_margins"].items()
        }
        if set(margins) != set(legal):
            raise VerificationError("oracle_margin_coverage")
        recomputed_metrics = _root_metrics(archived_priors, t_prior, legal, margins)
        result_rows.append(
            {
                "opening_index": int(root_row["opening_index"]),
                "state_hash": key,
                "metrics": recomputed_metrics,
            }
        )
    if (
        abs(float(manifest.get("parity_max_abs_difference", -1)) - observed_parity_max)
        > 1e-12
    ):
        raise VerificationError("run_manifest_parity_summary_mismatch")
    expected_by_hash = {row["state_hash"]: row for row in result_rows}
    if published_root_results != [expected_by_hash[row["state_hash"]] for row in roots]:
        raise VerificationError("per_root_metrics_tampered")
    recomputed_analysis = independently_analyze(result_rows)
    if json.loads((package / "analysis.json").read_bytes()) != recomputed_analysis:
        raise VerificationError("analysis_tampered_or_mismatch")
    return {
        "status": "verified",
        "source_roots": 512,
        "p_parity_roots": 512,
        "t512_roots": 512,
        "inference_performed_by_verifier": False,
        "analysis": recomputed_analysis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--trusted-run-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        result = verify_result(
            root=args.root,
            package=args.package,
            trusted_manifest_sha256=args.trusted_run_manifest_sha256,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"result verification failed: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
