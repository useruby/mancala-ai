"""Authorization-gated P-parity then T512 raw-policy evaluation for #588."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed588_raw_prior_analysis import analyze, root_metrics
from ml.alphazero_lite.seed588_raw_prior_diagnostic import (
    ARTIFACTS,
    EXPECTED_ARTIFACTS,
    EXPECTED_INPUTS,
    VerificationError,
    _jsonl,
    _sha,
    verify as verify_sources,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "docs/data/seed588-retrospective-raw-prior"
FREEZE_PATH = PACKAGE / "execution-freeze-v2.json"
AUTH_SCHEMA = "seed588-reviewed-execution-authorization-v1"
MAX_PARITY_ERROR = 1e-6


class ExecutionBlocked(PermissionError):
    """Execution did not satisfy its external reviewed-authorization gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_dependencies(root: Path = ROOT) -> dict[str, str]:
    relative = (
        "ml/alphazero_lite/run_seed588_raw_prior.py",
        "ml/alphazero_lite/seed588_raw_prior_diagnostic.py",
        "ml/alphazero_lite/seed588_raw_prior_analysis.py",
        "ml/alphazero_lite/verify_seed588_raw_prior_result.py",
        "ml/alphazero_lite/test_seed588_raw_prior_execution.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/arena.py",
        "ml/alphazero_lite/self_play.py",
        "ml/alphazero_lite/input_encodings.py",
        "ml/alphazero_lite/seed557_retrospective.py",
        "docs/data/seed588-retrospective-raw-prior/README.md",
        "docs/data/seed588-retrospective-raw-prior/execution-freeze.json",
    )
    return {name: sha256_file(root / name) for name in relative}


def verify_execution_freeze(root: Path = ROOT) -> dict[str, Any]:
    """Authenticate the complete executable freeze without loading a model."""
    freeze_path = root / FREEZE_PATH.relative_to(ROOT)
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    if freeze.get("schema") != "seed588-execution-readiness-freeze-v1":
        raise ExecutionBlocked("execution_freeze_schema")
    if freeze.get("status") != "frozen-pending-review-no-inference":
        raise ExecutionBlocked("execution_freeze_status")
    if freeze.get("implementation_sha256") != frozen_dependencies(root):
        raise ExecutionBlocked("freeze_implementation_binding_mismatch")
    preparation = (
        root / "docs/data/seed588-retrospective-raw-prior/execution-freeze.json"
    )
    protocol = root / "docs/data/seed588-retrospective-raw-prior/README.md"
    if freeze.get("preparation_freeze", {}).get("sha256") != sha256_file(preparation):
        raise ExecutionBlocked("preparation_freeze_binding_mismatch")
    if freeze.get("protocol", {}).get("sha256") != sha256_file(protocol):
        raise ExecutionBlocked("protocol_binding_mismatch")
    if freeze.get("source_evidence_sha256") != EXPECTED_INPUTS:
        raise ExecutionBlocked("freeze_source_binding_mismatch")
    if freeze.get("forward_backend") != {
        "implementation": "NumPy arena.ArtifactEvaluator using authenticated weights.json",
        "pytorch_required": False,
        "historical_preparation_pytorch_pin_enforced": False,
    }:
        raise ExecutionBlocked("freeze_forward_backend_mismatch")
    artifact_bindings = freeze.get("artifacts", {})
    if set(artifact_bindings) != set(EXPECTED_ARTIFACTS):
        raise ExecutionBlocked("freeze_artifact_set_mismatch")
    for model, artifact_record in artifact_bindings.items():
        if {
            key: value for key, value in artifact_record.items() if key != "source"
        } != EXPECTED_ARTIFACTS[model]:
            raise ExecutionBlocked(f"freeze_{model}_binding_mismatch")
    for model, files in EXPECTED_ARTIFACTS.items():
        observed = {
            name: sha256_file(root / ARTIFACTS[model].relative_to(ROOT) / name)
            for name in files
        }
        if observed != files:
            raise ExecutionBlocked(f"{model}_artifact_identity_mismatch")
    forward = freeze.get("forward_contract", {})
    if (
        forward.get("P_prior_parity_evaluations") != 512
        or forward.get("T512_policy_evaluations") != 512
        or forward.get("parity_max_abs_tolerance") != MAX_PARITY_ERROR
        or forward.get("legal_mask") != "exact player-relative indices 0..5"
        or forward.get("all_P_parity_before_any_T512") is not True
        or any(forward.get(field) for field in ("PUCT", "MCTS", "games", "training"))
    ):
        raise ExecutionBlocked("freeze_forward_contract_mismatch")
    analysis = freeze.get("analysis", {})
    if (
        analysis.get("bootstrap_resamples") != 20_000
        or analysis.get("seed") != 58_820_261_114
        or analysis.get("rng") != "random.Random; randrange with replacement"
        or analysis.get("quantiles") != [0.025, 0.975]
        or analysis.get("interpolation") != "linear"
    ):
        raise ExecutionBlocked("freeze_analysis_contract_mismatch")
    source_status = verify_sources(root)
    if source_status["roots"] != 512 or source_status["inference_performed"]:
        raise ExecutionBlocked("source_preflight_failed")
    return {
        "status": "verified_pending_review",
        "freeze_sha256": _sha(freeze_raw),
        "implementation_sha256": freeze["implementation_sha256"],
        "roots": 512,
        "inference_performed": False,
    }


def verify_authorization(
    *,
    root: Path,
    authorization_path: Path | None,
    trusted_sha256: str | None,
) -> dict[str, Any]:
    """Require an externally trusted, review-bound authorization record."""
    if authorization_path is None or trusted_sha256 is None:
        raise ExecutionBlocked("reviewed_authorization_and_trust_digest_required")
    raw = authorization_path.read_bytes()
    if _sha(raw) != trusted_sha256:
        raise ExecutionBlocked("authorization_digest_mismatch")
    authorization = json.loads(raw)
    if (
        authorization.get("schema") != AUTH_SCHEMA
        or authorization.get("status") != "reviewed_and_authorized"
        or not authorization.get("review_url")
        or not authorization.get("reviewed_by")
    ):
        raise ExecutionBlocked("authorization_not_reviewed")
    freeze_path = root / FREEZE_PATH.relative_to(ROOT)
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    if authorization.get("execution_freeze_sha256") != _sha(freeze_raw):
        raise ExecutionBlocked("execution_freeze_hash_mismatch")
    verify_execution_freeze(root)
    if authorization.get("source_and_implementation_sha256") != frozen_dependencies(
        root
    ):
        raise ExecutionBlocked("implementation_dependency_hash_mismatch")
    if authorization.get("artifacts") != EXPECTED_ARTIFACTS:
        raise ExecutionBlocked("authorization_artifact_binding_mismatch")
    if authorization.get("source_evidence_sha256") != EXPECTED_INPUTS:
        raise ExecutionBlocked("authorization_source_binding_mismatch")
    expected_runtime = authorization.get("runtime")
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device": "cpu",
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "execution_threads": 1,
    }
    if (
        expected_runtime != observed_runtime
        or freeze.get("runtime") != observed_runtime
    ):
        raise ExecutionBlocked("runtime_identity_mismatch")
    return authorization


def _read_inputs(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_result = verify_sources(root)
    data = root / "docs/data/seed559-fixed-position-feasibility/run-0003"
    cohort = json.loads((data / "cohort-manifest.json").read_bytes())["states"]
    search_rows = _jsonl((data / "search-records.jsonl").read_bytes())
    oracle_rows = _jsonl((data / "oracle-records.jsonl").read_bytes())
    if source_result["roots"] != 512:
        raise VerificationError("source_root_count")
    return (
        cohort,
        {row["state_hash"]: row for row in search_rows},
        {row["state_hash"]: row for row in oracle_rows},
    )


def _evaluate(evaluator: arena.ArtifactEvaluator, state: dict[str, Any]) -> list[float]:
    game = KalahGame.from_state(state)
    priors, _unused_value = evaluator.evaluate(game)
    vector = np.asarray(priors, dtype=np.float32)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("policy_output_shape_or_finiteness")
    return [float(value) for value in vector]


def compare_parity(
    observed: list[float], archived: list[float], legal_relative: list[int]
) -> float:
    """Return maximum absolute difference for player-relative policy indices."""
    if len(observed) != 6 or len(archived) != 6:
        return float("inf")
    if any(not np.isfinite(value) for value in observed + archived):
        return float("inf")
    return max(
        abs(float(observed[action]) - float(archived[action])) for action in range(6)
    )


def require_parity_pass(rows: list[dict[str, Any]]) -> None:
    """Fail closed unless all 512 P forwards pass mask and value parity."""
    if len(rows) != 512:
        raise VerificationError("P_parity_forward_count")
    if any(
        not row.get("legal_mask_exact")
        or float(row.get("max_abs_difference", float("inf"))) > MAX_PARITY_ERROR
        for row in rows
    ):
        raise VerificationError("P_archived_prior_parity_failed; T512_not_started")


def _create_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def execute(
    *,
    root: Path,
    output: Path,
    authorization_path: Path | None,
    trusted_authorization_sha256: str | None,
) -> dict[str, Any]:
    """Run P parity first, then exactly 512 T512 passes, create-only output."""
    output = output if output.is_absolute() else root / output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output_must_not_exist:{output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output_parent_missing:{output.parent}")
    authorization = verify_authorization(
        root=root,
        authorization_path=authorization_path,
        trusted_sha256=trusted_authorization_sha256,
    )
    cohort, archived_p, oracles = _read_inputs(root)
    p_dir = root / ARTIFACTS["P"].relative_to(ROOT)
    t_dir = root / ARTIFACTS["T512"].relative_to(ROOT)
    p_evaluator = arena.ArtifactEvaluator(p_dir)
    t_evaluator = arena.ArtifactEvaluator(t_dir)
    # Close the verification-to-load gap before any forward evaluation.
    for model, directory in (("P", p_dir), ("T512", t_dir)):
        observed = {
            name: sha256_file(directory / name) for name in EXPECTED_ARTIFACTS[model]
        }
        if observed != EXPECTED_ARTIFACTS[model]:
            raise ExecutionBlocked(f"{model}_artifact_changed_after_load")

    parity: list[dict[str, Any]] = []
    for root_row in cohort:
        state_hash = root_row["state_hash"]
        observed = _evaluate(p_evaluator, root_row["state"])
        expected = [float(value) for value in archived_p[state_hash]["priors"]]
        legal_relative = sorted(
            int(move) for move in archived_p[state_hash]["legal_moves"]
        )
        mask_exact = len(observed) == 6 and all(
            observed[action] == 0.0
            for action in range(6)
            if action not in legal_relative
        )
        difference = compare_parity(observed, expected, legal_relative)
        parity.append(
            {
                "state_hash": state_hash,
                "opening_index": root_row["opening_index"],
                "p_forward_priors": observed,
                "archived_p_priors": expected,
                "max_abs_difference": difference,
                "legal_mask_exact": mask_exact,
                "legal_relative_actions": legal_relative,
            }
        )
    require_parity_pass(parity)

    results: list[dict[str, Any]] = []
    for root_row in cohort:
        state_hash = root_row["state_hash"]
        if state_hash not in oracles:
            raise VerificationError(f"oracle_missing:{state_hash}")
        t_priors = _evaluate(t_evaluator, root_row["state"])
        legal_relative = sorted(
            int(move) for move in archived_p[state_hash]["legal_moves"]
        )
        if any(
            t_priors[action] != 0.0
            for action in range(6)
            if action not in legal_relative
        ):
            raise VerificationError(f"T512_legal_mask:{state_hash}")
        results.append(
            {
                "state_hash": state_hash,
                "opening_index": root_row["opening_index"],
                "t512_priors": t_priors,
            }
        )
    if len(results) != 512 or len({row["state_hash"] for row in results}) != 512:
        raise VerificationError("T512_evaluation_count_or_membership")

    root_results = []
    for root_row, prediction in zip(cohort, results, strict=True):
        key = root_row["state_hash"]
        legal = sorted(int(move) for move in archived_p[key]["legal_moves"])
        margins = {
            int(move): int(value)
            for move, value in oracles[key]["action_margins"].items()
        }
        metrics = root_metrics(
            p_priors=[float(value) for value in archived_p[key]["priors"]],
            t_priors=prediction["t512_priors"],
            legal=legal,
            margins=margins,
        )
        root_results.append(
            {
                "state_hash": key,
                "opening_index": int(root_row["opening_index"]),
                "metrics": metrics,
            }
        )
    analysis = analyze(root_results)

    output.mkdir(mode=0o700)
    _create_jsonl(output / "p-parity.jsonl", parity)
    _create_jsonl(output / "t512-predictions.jsonl", results)
    _create_jsonl(output / "root-results.jsonl", root_results)
    with (output / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(analysis, stream, indent=2, sort_keys=True)
        stream.write("\n")
    manifest = {
        "schema": "seed588-raw-prior-run-v1",
        "execution_freeze_sha256": sha256_file(root / FREEZE_PATH.relative_to(ROOT)),
        "implementation_sha256": authorization["source_and_implementation_sha256"],
        "source_evidence_sha256": EXPECTED_INPUTS,
        "artifacts": EXPECTED_ARTIFACTS,
        "authorization_sha256": trusted_authorization_sha256,
        "runtime": authorization["runtime"],
        "p_parity_forward_evaluations": len(parity),
        "t512_forward_evaluations": len(results),
        "parity_max_abs_difference": max(row["max_abs_difference"] for row in parity),
        "p_parity_passed": True,
        "mcts": False,
        "games": False,
        "training": False,
        "result_files_sha256": {
            name: sha256_file(output / name)
            for name in (
                "p-parity.jsonl",
                "t512-predictions.jsonl",
                "root-results.jsonl",
                "analysis.json",
            )
        },
    }
    with (output / "run-manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        try:
            print(json.dumps(verify_execution_freeze(args.root), sort_keys=True))
        except (
            ExecutionBlocked,
            VerificationError,
            OSError,
            ValueError,
            KeyError,
        ) as exc:
            parser.exit(2, f"freeze verification failed: {exc}\n")
        return
    if args.output is None:
        parser.error("--output is required unless --verify-only is specified")
    if args.authorization is not None and not args.authorization.is_absolute():
        args.authorization = args.root / args.authorization
    try:
        result = execute(
            root=args.root,
            output=args.output,
            authorization_path=args.authorization,
            trusted_authorization_sha256=args.authorization_sha256,
        )
    except (ExecutionBlocked, VerificationError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"execution blocked/failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
