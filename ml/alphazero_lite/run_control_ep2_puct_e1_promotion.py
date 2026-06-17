#!/usr/bin/env python3
"""Promote the selected control_ep2 PUCT policy-head e1 artifact with guards.

Does not train, generate self-play, or change promotion thresholds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402

VENV_PYTHON = REPO_ROOT / ".venv/bin/python"
DEFAULT_GATE_BUDGETS = "384:256,1200:1200,1200:256,256:768"
DEFAULT_LARGE_BUDGETS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
EXPECTED_CHECKPOINT_SHA256 = (
    "a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357"
)
EXPECTED_WEIGHTS_SHA256 = (
    "6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece"
)
CONTROL_EP2_CHECKPOINT_SHA256 = (
    "619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9"
)
PROMOTION_DOC = (
    REPO_ROOT / "docs/alphazero-lite-control-ep2-puct-e1-promotion-results.md"
)
EVIDENCE_DOC = "docs/alphazero-lite-control-ep2-puct-head-preflight-results.md"
EXPECTED_FIXED_LARGE_384_256_DS = -0.0208
MAX_1200_1200_REGRESSION = 0.03


def _python() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_command(*, cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    return result


def copy_deployable_artifact(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "metadata.json", dest / "metadata.json")
    shutil.copy2(src / "weights.json", dest / "weights.json")


def build_promoted_metadata(
    *,
    candidate_metadata: dict[str, Any],
    checkpoint_sha256: str,
    weights_sha256: str,
) -> dict[str, Any]:
    metadata = json.loads(json.dumps(candidate_metadata))
    metadata["version"] = "azlite-control-ep2-puct-policy-head-e1"
    artifacts = metadata.setdefault("artifacts", {})
    if isinstance(artifacts, dict):
        artifacts.pop("weights_fallback_file", None)
        artifacts["weights_file"] = "weights.json"
        artifacts["weights_sha256"] = checkpoint_sha256
        artifacts["weights_json_sha256"] = weights_sha256
    metadata["promotion"] = {
        "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_name": "puct_policy_head_e1",
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_artifact_weights_sha256": weights_sha256,
        "parent_control_ep2_checkpoint_sha256": CONTROL_EP2_CHECKPOINT_SHA256,
        "promotion_evidence_doc": EVIDENCE_DOC,
        "training_scope": "policy_head",
        "replay_source": "control_ep2_puct_selfplay",
        "architecture_change": "none",
    }
    return metadata


def runtime_probe_states() -> list[dict[str, Any]]:
    return [
        {
            "name": "opening_start",
            "state": {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
        },
        {
            "name": "sparse_legal_mask",
            "state": {
                "player_pits": [0, 1, 0, 0, 0, 2],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
        },
    ]


def validate_runtime_loader(artifact_dir: Path) -> dict[str, Any]:
    evaluator = arena.ArtifactEvaluator(artifact_dir)
    search_options = arena.build_eval_search_options(root_policy_mode="deterministic")
    probes: list[dict[str, Any]] = []
    for probe in runtime_probe_states():
        game = KalahGame.from_state(probe["state"])
        legal_moves = game.possible_moves()
        priors, value = evaluator.evaluate(game)
        if priors.shape != (6,):
            raise RuntimeError(
                f"policy shape mismatch for {probe['name']}: {priors.shape}"
            )
        for move in range(6):
            if move not in legal_moves and float(priors[move]) != 0.0:
                raise RuntimeError(
                    f"illegal move {move} received policy mass {float(priors[move])}"
                )
        if legal_moves and abs(float(priors[legal_moves].sum()) - 1.0) > 1e-6:
            raise RuntimeError(
                f"legal move policy does not sum to 1.0 for {probe['name']}"
            )
        summary = arena.evaluate_artifact_position(
            artifact_path=artifact_dir,
            evaluator=evaluator,
            state=probe["state"],
            simulations=0,
            seed=42,
            c_puct=1.25,
            search_options=search_options,
        )
        selected_move = summary.get("selected_move")
        if legal_moves and selected_move not in legal_moves:
            raise RuntimeError(
                f"selected move {selected_move} is not legal for {probe['name']}"
            )
        probes.append(
            {
                "name": probe["name"],
                "legal_moves": legal_moves,
                "selected_move": selected_move,
                "value": float(value),
                "policy": [round(float(p), 6) for p in priors.tolist()],
            }
        )
    return {
        "model_type": evaluator.model_type,
        "input_encoding": evaluator.input_encoding,
        "probes": probes,
    }


def benchmark_candidate_report(
    report: dict[str, Any], candidate_name: str
) -> dict[str, Any]:
    for temperature_report in report.get("temperature_reports", []):
        for seed_report in temperature_report.get("seed_reports", []):
            for candidate_report in seed_report.get("candidate_reports", []):
                if candidate_report.get("candidate") == candidate_name:
                    return candidate_report
    raise KeyError(f"candidate report not found: {candidate_name}")


def verify_opening_suite_results(
    *,
    previous_report: dict[str, Any],
    promoted_report: dict[str, Any],
) -> dict[str, Any]:
    previous_standard = previous_report["budget_results"]["standard"]
    promoted_standard = promoted_report["budget_results"]["standard"]
    previous_equal_high = previous_report["budget_results"]["equal_high"]
    promoted_equal_high = promoted_report["budget_results"]["equal_high"]

    promoted_standard_ds = float(promoted_standard["ds"])
    previous_standard_ds = float(previous_standard["ds"])
    promoted_equal_high_ds = float(promoted_equal_high["ds"])
    previous_equal_high_ds = float(previous_equal_high["ds"])

    exact_384_256_match = round(promoted_standard_ds, 4) == round(
        EXPECTED_FIXED_LARGE_384_256_DS, 4
    )
    stronger_than_previous = promoted_standard_ds > previous_standard_ds
    equal_high_regression = previous_equal_high_ds - promoted_equal_high_ds
    equal_high_within_tolerance = equal_high_regression <= MAX_1200_1200_REGRESSION

    return {
        "expected_standard_ds": EXPECTED_FIXED_LARGE_384_256_DS,
        "promoted_standard_ds": promoted_standard_ds,
        "previous_standard_ds": previous_standard_ds,
        "exact_384_256_match": exact_384_256_match,
        "stronger_than_previous": stronger_than_previous,
        "promoted_equal_high_ds": promoted_equal_high_ds,
        "previous_equal_high_ds": previous_equal_high_ds,
        "equal_high_regression": equal_high_regression,
        "equal_high_within_tolerance": equal_high_within_tolerance,
    }


def build_results_markdown(summary: dict[str, Any]) -> str:
    opening = summary["opening_suite"]
    gate = summary["gate"]
    promoted = opening["promoted_current"]
    previous = opening["previous_current_ref"]
    lines = [
        "# AlphaZero-Lite Control EP2 PUCT e1 Promotion Results",
        "",
        f"**Date**: {dt.date.today().isoformat()}",
        "",
        f"**Classification**: `{summary['final_classification']}`",
        "",
        "## Summary",
        "",
        f"- Previous current weights SHA256: `{summary['previous_current_weights_sha256']}`",
        f"- Promoted current weights SHA256: `{summary['promoted_current_weights_sha256']}`",
        f"- Expected e1 weights SHA256: `{summary['expected_candidate_weights_sha256']}`",
        f"- Metadata version: `{summary['metadata_version']}`",
        "- Changed files under `model-artifact/current`: `metadata.json`, `weights.json`",
        "- Training run: not run",
        "- Self-play generation: not run",
        "",
        "## Artifact Integrity",
        "",
        f"- Candidate artifact verified at `{summary['candidate_artifact_path']}`",
        f"- Candidate checkpoint SHA256: `{summary['candidate_checkpoint_sha256']}`",
        f"- Candidate weights SHA256: `{summary['candidate_weights_sha256']}`",
        f"- Checked-in `model-artifact/current/weights.json` SHA256 after replacement: `{summary['promoted_current_weights_sha256']}`",
        f"- Metadata JSON parse: `{summary['artifact_integrity']['metadata_json_parse']}`",
        f"- Runtime loader: `{summary['artifact_integrity']['runtime_loader']}`",
        "",
        "## Gate Results",
        "",
        f"- Classification: `{gate['classification']}`",
        f"- Standard disadvantaged-seat score (`384:256`): `{gate['budget_results']['standard']['disadvantaged_seat_score']:+.4f}`",
        f"- Equal-high disadvantaged-seat score (`1200:1200`): `{gate['budget_results']['equal_high']['disadvantaged_seat_score']:+.4f}`",
        f"- Challenger-high disadvantaged-seat score (`1200:256`): `{gate['budget_results']['challenger_high']['disadvantaged_seat_score']:+.4f}`",
        f"- Current-high disadvantaged-seat score (`256:768`): `{gate['budget_results']['current_high_asymmetry']['disadvantaged_seat_score']:+.4f}`",
        "",
        "## Fixed Large-Suite Before/After",
        "",
        "| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| previous_current_ref | "
        f"{previous['budget_results']['standard']['ds']:+.4f} | "
        f"{previous['budget_results']['challenger_768_vs_256']['ds']:+.4f} | "
        f"{previous['budget_results']['equal_768']['ds']:+.4f} | "
        f"{previous['budget_results']['equal_high']['ds']:+.4f} | "
        f"{previous['budget_results']['1200_vs_256']['ds']:+.4f} | "
        f"{previous['budget_results']['current_high_asymmetry']['ds']:+.4f} |",
        "| promoted_current | "
        f"{promoted['budget_results']['standard']['ds']:+.4f} | "
        f"{promoted['budget_results']['challenger_768_vs_256']['ds']:+.4f} | "
        f"{promoted['budget_results']['equal_768']['ds']:+.4f} | "
        f"{promoted['budget_results']['equal_high']['ds']:+.4f} | "
        f"{promoted['budget_results']['1200_vs_256']['ds']:+.4f} | "
        f"{promoted['budget_results']['current_high_asymmetry']['ds']:+.4f} |",
        "",
        "## Acceptance Checks",
        "",
        f"- `384:256` exact deterministic reproduction of PR #119 e1 value `-0.0208`: `{opening['verification']['exact_384_256_match']}`",
        f"- `384:256` promoted current stronger than previous current: `{opening['verification']['stronger_than_previous']}`",
        f"- `1200:1200` regression within accepted tolerance (`<= {MAX_1200_1200_REGRESSION:.2f}`): `{opening['verification']['equal_high_within_tolerance']}`",
        "- No training or self-play was run during promotion: `true`",
        "",
        "## Runtime Probes",
        "",
    ]
    for probe in summary["artifact_integrity"]["runtime_probe_summary"]["probes"]:
        lines.append(
            f"- `{probe['name']}` legal moves `{probe['legal_moves']}` selected `{probe['selected_move']}`"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Promotion workdir: `{summary['workdir']}`")
    lines.append(f"- Gate report: `{summary['gate_report_path']}`")
    lines.append(f"- Opening-suite report: `{summary['opening_suite_report_path']}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote control_ep2 PUCT policy-head e1 into model-artifact/current."
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path(
            "/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/checkpoint_epoch1.npz"
        ),
    )
    parser.add_argument(
        "--expected-candidate-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-candidate-weights-sha256",
        default=EXPECTED_WEIGHTS_SHA256,
    )
    parser.add_argument("--current", type=Path, default=Path("model-artifact/current"))
    parser.add_argument("--previous-current-ref", type=Path, default=None)
    parser.add_argument(
        "--large-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/large_eval.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = args.workdir.resolve()
    candidate_artifact = args.candidate_artifact.resolve()
    candidate_checkpoint = args.candidate_checkpoint.resolve()
    current_dir = (REPO_ROOT / args.current).resolve()
    large_suite = args.large_suite.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if not (candidate_artifact / "weights.json").is_file():
        raise FileNotFoundError(f"missing candidate weights.json: {candidate_artifact}")
    if not (candidate_artifact / "metadata.json").is_file():
        raise FileNotFoundError(
            f"missing candidate metadata.json: {candidate_artifact}"
        )
    if not candidate_checkpoint.is_file():
        raise FileNotFoundError(f"missing candidate checkpoint: {candidate_checkpoint}")

    candidate_checkpoint_sha = sha256_file(candidate_checkpoint)
    if candidate_checkpoint_sha != args.expected_candidate_checkpoint_sha256:
        raise RuntimeError(
            "candidate checkpoint SHA256 mismatch: "
            f"expected {args.expected_candidate_checkpoint_sha256}, got {candidate_checkpoint_sha}"
        )

    candidate_weights_sha = sha256_file(candidate_artifact / "weights.json")
    if candidate_weights_sha != args.expected_candidate_weights_sha256:
        raise RuntimeError(
            "candidate weights SHA256 mismatch: "
            f"expected {args.expected_candidate_weights_sha256}, got {candidate_weights_sha}"
        )

    if args.previous_current_ref is not None:
        previous_current_dir = args.previous_current_ref.resolve()
    else:
        previous_current_dir = workdir / "previous_current_ref"
        copy_deployable_artifact(current_dir, previous_current_dir)
    previous_current_sha = sha256_file(previous_current_dir / "weights.json")

    candidate_metadata = load_json(candidate_artifact / "metadata.json")
    promoted_metadata = build_promoted_metadata(
        candidate_metadata=candidate_metadata,
        checkpoint_sha256=candidate_checkpoint_sha,
        weights_sha256=candidate_weights_sha,
    )
    write_json(current_dir / "metadata.json", promoted_metadata)
    shutil.copy2(candidate_artifact / "weights.json", current_dir / "weights.json")

    promoted_current_sha = sha256_file(current_dir / "weights.json")
    metadata_json = load_json(current_dir / "metadata.json")
    runtime_probe_summary = validate_runtime_loader(current_dir)

    gate_out = workdir / "gate_report.json"
    gate_workdir = workdir / "gate_eval"
    run_command(
        cmd=[
            _python(),
            str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
            "--candidate-path",
            str(current_dir),
            "--current-path",
            str(previous_current_dir),
            "--out",
            str(gate_out),
            "--budget-pairs",
            DEFAULT_GATE_BUDGETS,
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
            "--workdir",
            str(gate_workdir),
        ],
        cwd=REPO_ROOT,
    )
    gate_report = load_json(gate_out)

    opening_workdir = workdir / "opening_suite_eval"
    run_command(
        cmd=[
            _python(),
            str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
            "--workdir",
            str(opening_workdir),
            "--suite",
            str(large_suite),
            "--current",
            str(previous_current_dir),
            "--candidates",
            f"{previous_current_dir},{current_dir}",
            "--budget-pairs",
            DEFAULT_LARGE_BUDGETS,
            "--games-per-opening",
            "2",
            "--root-policy-mode",
            "deterministic",
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
        ],
        cwd=REPO_ROOT,
    )
    opening_report_path = opening_workdir / "temperature_benchmark_report.json"
    opening_report = load_json(opening_report_path)
    previous_report = benchmark_candidate_report(opening_report, "previous_current_ref")
    promoted_report = benchmark_candidate_report(opening_report, "current")
    opening_verification = verify_opening_suite_results(
        previous_report=previous_report,
        promoted_report=promoted_report,
    )

    gate_ok = gate_report.get("classification") == "high_search_breakthrough"
    gate_standard_ok = (
        float(gate_report["budget_results"]["standard"]["disadvantaged_seat_score"])
        >= 0.0
    )
    gate_equal_high_ok = (
        float(gate_report["budget_results"]["equal_high"]["disadvantaged_seat_score"])
        > 0.1
    )
    runtime_ok = promoted_current_sha == args.expected_candidate_weights_sha256
    opening_ok = all(
        [
            opening_verification["exact_384_256_match"],
            opening_verification["stronger_than_previous"],
            opening_verification["equal_high_within_tolerance"],
        ]
    )

    final_classification = "promoted_e1_current"
    if promoted_current_sha != args.expected_candidate_weights_sha256:
        final_classification = "promotion_blocked_artifact_mismatch"
    elif not runtime_ok:
        final_classification = "promotion_blocked_runtime_failure"
    elif not gate_ok or not gate_standard_ok or not gate_equal_high_ok:
        final_classification = "promotion_blocked_gate_regression"
    elif not opening_ok:
        final_classification = "promotion_blocked_gate_regression"

    summary = {
        "schema": "azlite_control_ep2_puct_e1_promotion_v1",
        "workdir": str(workdir),
        "candidate_artifact_path": str(candidate_artifact),
        "candidate_checkpoint_path": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": candidate_checkpoint_sha,
        "candidate_weights_sha256": candidate_weights_sha,
        "expected_candidate_weights_sha256": args.expected_candidate_weights_sha256,
        "previous_current_weights_sha256": previous_current_sha,
        "promoted_current_weights_sha256": promoted_current_sha,
        "metadata_version": str(metadata_json.get("version")),
        "artifact_integrity": {
            "metadata_json_parse": True,
            "runtime_loader": True,
            "runtime_probe_summary": runtime_probe_summary,
        },
        "gate": gate_report,
        "opening_suite": {
            "report_path": str(opening_report_path),
            "previous_current_ref": previous_report,
            "promoted_current": promoted_report,
            "verification": opening_verification,
        },
        "gate_report_path": str(gate_out),
        "opening_suite_report_path": str(opening_report_path),
        "final_classification": final_classification,
        "guardrails": {
            "training_run": False,
            "self_play_generation": False,
            "architecture_change": False,
            "replay_change": False,
            "seed_sweep": False,
            "threshold_change": False,
        },
    }

    summary_path = workdir / "promotion_summary.json"
    write_json(summary_path, summary)
    PROMOTION_DOC.write_text(build_results_markdown(summary), encoding="utf-8")

    if final_classification != "promoted_e1_current":
        raise RuntimeError(
            f"promotion failed with classification: {final_classification}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
