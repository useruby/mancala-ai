#!/usr/bin/env python3
"""Promote the balanced_w8s4_policy_head_e1 artifact with guardrails."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_promoted_current_puct_iter2_smoke import (  # noqa: E402
    heldout_summary,
    run_default_gate,
    run_opening_suite_benchmark,
)

VENV_PYTHON = REPO_ROOT / ".venv/bin/python"
PROMOTION_DOC = REPO_ROOT / "docs/alphazero-lite-balanced-w8s4-e1-promotion-results.md"
SUMMARY_FILENAME = "summary_metrics.json"
DEFAULT_LARGE_BUDGETS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
EXPECTED_PREVIOUS_CURRENT_WEIGHTS_SHA256 = (
    "6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece"
)
EXPECTED_CANDIDATE_NAME = "balanced_w8s4_policy_head_e1"
EXPECTED_PARENT_VERSION = "azlite-control-ep2-puct-policy-head-e1"
PROMOTED_VERSION = "azlite-balanced-w8s4-policy-head-e1"
SOURCE_EXPERIMENT = "docs/alphazero-lite-balanced-opening-puct-replay-results.md"
SOURCE_RUNNER = "ml/alphazero_lite/run_balanced_opening_puct_replay.py"
REPLAY_SOURCES = [
    "generic_bootstrap",
    "random_teacher",
    "opening_puct_disagreement_replay",
    "equal_budget_stability_replay",
]
REPLAY_WEIGHTS = [4, 1, 8, 4]
ARCHITECTURE_NOTE = "residual_v3 / kalah_v3 / hidden_sizes 96,3"
MIN_384_256_IMPROVEMENT = 0.15
MAX_768_768_REGRESSION = 0.10
MAX_HIGH_BUDGET_REGRESSION = 0.03
PROMOTION_SCHEMA = "azlite_balanced_w8s4_e1_promotion_v1"
CHANGED_CURRENT_FILES = ["metadata.json", "weights.json"]
ALLOWED_GATE_CLASSIFICATIONS = {
    "high_search_breakthrough",
    "standard_budget_breakthrough",
}


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


def parse_csv_paths(text: str) -> list[Path]:
    return [Path(item.strip()) for item in text.split(",") if item.strip()]


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def copy_deployable_artifact(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "metadata.json", dest / "metadata.json")
    shutil.copy2(src / "weights.json", dest / "weights.json")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def find_candidate(candidates: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for candidate in candidates:
        if str(candidate.get("name")) == name:
            return candidate
    raise RuntimeError(f"candidate missing from manifest: {name}")


def find_manifest_parent_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for candidate in candidates:
        if str(candidate.get("name")) == "promoted_current_ref":
            return candidate
    raise RuntimeError("manifest missing promoted_current_ref entry")


def verify_candidate_manifest(
    *, manifest: dict[str, Any], candidate: dict[str, Any], expected_parent_sha256: str
) -> dict[str, Any]:
    if str(candidate.get("name")) != EXPECTED_CANDIDATE_NAME:
        raise RuntimeError(
            "candidate name mismatch: expected "
            f"{EXPECTED_CANDIDATE_NAME}, got {candidate.get('name')}"
        )

    artifact_dir_text = candidate.get("artifact_dir")
    artifact_weights_sha256 = candidate.get("artifact_weights_sha256")
    if not isinstance(artifact_dir_text, str) or not artifact_dir_text:
        raise RuntimeError("candidate manifest missing artifact directory")
    if not isinstance(artifact_weights_sha256, str) or not artifact_weights_sha256:
        raise RuntimeError("candidate manifest missing artifact weights SHA256")

    checkpoint_path = candidate.get("checkpoint_path")
    checkpoint_sha256 = candidate.get("checkpoint_sha256")
    if checkpoint_path is not None and not isinstance(checkpoint_path, str):
        raise RuntimeError("candidate manifest checkpoint path must be a string")
    if checkpoint_sha256 is not None and not isinstance(checkpoint_sha256, str):
        raise RuntimeError("candidate manifest checkpoint SHA256 must be a string")

    replay_weights = candidate.get("replay_weights")
    if not isinstance(replay_weights, dict):
        raise RuntimeError("candidate manifest missing replay_weights")
    if replay_weights.get("mined_disagreement") != 8:
        raise RuntimeError("candidate manifest disagreement replay weight must be 8")
    if replay_weights.get("stability_anchor") != 4:
        raise RuntimeError("candidate manifest stability replay weight must be 4")

    train_metrics = candidate.get("train_metrics")
    if not isinstance(train_metrics, dict):
        raise RuntimeError("candidate manifest missing train_metrics")
    if train_metrics.get("trainable_scope") != "policy_head":
        raise RuntimeError("candidate manifest trainable_scope must be policy_head")

    manifest_parent = find_manifest_parent_candidate(
        list(manifest.get("candidates", []))
    )
    parent_sha256 = manifest_parent.get("artifact_weights_sha256")
    if parent_sha256 != expected_parent_sha256:
        raise RuntimeError(
            "manifest parent current weights SHA256 mismatch: "
            f"expected {expected_parent_sha256}, got {parent_sha256}"
        )

    return {
        "artifact_dir": Path(artifact_dir_text).resolve(),
        "artifact_weights_sha256": artifact_weights_sha256,
        "checkpoint_path": Path(checkpoint_path).resolve()
        if isinstance(checkpoint_path, str) and checkpoint_path
        else None,
        "checkpoint_sha256": checkpoint_sha256,
        "trainable_scope": "policy_head",
        "parent_weights_sha256": parent_sha256,
    }


def validate_candidate_artifact(candidate_info: dict[str, Any]) -> None:
    artifact_dir = candidate_info["artifact_dir"]
    weights_path = artifact_dir / "weights.json"
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"candidate artifact directory missing: {artifact_dir}")
    require_file(weights_path, "candidate weights.json")
    require_file(artifact_dir / "metadata.json", "candidate metadata.json")

    actual_weights_sha256 = sha256_file(weights_path)
    if actual_weights_sha256 != candidate_info["artifact_weights_sha256"]:
        raise RuntimeError(
            "computed candidate weights SHA256 differs from manifest: "
            f"expected {candidate_info['artifact_weights_sha256']}, got {actual_weights_sha256}"
        )

    checkpoint_path = candidate_info.get("checkpoint_path")
    checkpoint_sha256 = candidate_info.get("checkpoint_sha256")
    if checkpoint_path is not None:
        require_file(checkpoint_path, "candidate checkpoint")
        if checkpoint_sha256 is not None:
            actual_checkpoint_sha256 = sha256_file(checkpoint_path)
            if actual_checkpoint_sha256 != checkpoint_sha256:
                raise RuntimeError(
                    "computed candidate checkpoint SHA256 differs from manifest: "
                    f"expected {checkpoint_sha256}, got {actual_checkpoint_sha256}"
                )


def build_promoted_metadata(
    *,
    candidate_metadata: dict[str, Any],
    candidate: dict[str, Any],
    previous_current_weights_sha256: str,
    promoted_current_weights_sha256: str,
) -> dict[str, Any]:
    metadata = json.loads(json.dumps(candidate_metadata))
    metadata["version"] = PROMOTED_VERSION
    metadata["parent_version"] = EXPECTED_PARENT_VERSION
    metadata["parent_weights_sha256"] = previous_current_weights_sha256
    metadata["source_experiment"] = SOURCE_EXPERIMENT
    metadata["source_runner"] = SOURCE_RUNNER
    metadata["selected_lane"] = EXPECTED_CANDIDATE_NAME
    metadata["trainable_scope"] = "policy_head"
    metadata["replay_sources"] = list(REPLAY_SOURCES)
    metadata["replay_weights"] = list(REPLAY_WEIGHTS)
    metadata["architecture_note"] = ARCHITECTURE_NOTE
    metadata["architecture_change"] = "none"

    artifacts = metadata.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise RuntimeError("candidate metadata artifacts field must be an object")
    artifacts.pop("weights_fallback_file", None)
    artifacts["weights_file"] = "weights.json"
    if isinstance(candidate.get("checkpoint_sha256"), str):
        artifacts["weights_sha256"] = candidate["checkpoint_sha256"]
    artifacts["weights_json_sha256"] = promoted_current_weights_sha256

    metadata["promotion"] = {
        "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_name": EXPECTED_CANDIDATE_NAME,
        "parent_version": EXPECTED_PARENT_VERSION,
        "parent_weights_sha256": previous_current_weights_sha256,
        "source_checkpoint_path": candidate.get("checkpoint_path"),
        "source_checkpoint_sha256": candidate.get("checkpoint_sha256"),
        "source_artifact_weights_sha256": promoted_current_weights_sha256,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_runner": SOURCE_RUNNER,
        "selected_lane": EXPECTED_CANDIDATE_NAME,
        "trainable_scope": "policy_head",
        "replay_sources": list(REPLAY_SOURCES),
        "replay_weights": list(REPLAY_WEIGHTS),
        "architecture": ARCHITECTURE_NOTE,
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


def budget_ds(candidate_report: dict[str, Any], label: str) -> float:
    return float(candidate_report["budget_results"][label]["ds"])


def gate_keeps_high_search_signature(gate_report: dict[str, Any]) -> bool:
    classification = gate_report.get("classification")
    if classification not in ALLOWED_GATE_CLASSIFICATIONS:
        return False
    budget_results = gate_report.get("budget_results", {})
    equal_high = float(
        budget_results.get("equal_high", {}).get("disadvantaged_seat_score", 0.0)
    )
    challenger_high = float(
        budget_results.get("challenger_high", {}).get("disadvantaged_seat_score", 0.0)
    )
    return equal_high > 0.1 or challenger_high > 0.1


def verify_fixed_large_suite(
    *, previous_report: dict[str, Any], promoted_report: dict[str, Any]
) -> dict[str, Any]:
    delta_384_256 = budget_ds(promoted_report, "standard") - budget_ds(
        previous_report, "standard"
    )
    delta_768_768 = budget_ds(promoted_report, "equal_768") - budget_ds(
        previous_report, "equal_768"
    )
    delta_1200_1200 = budget_ds(promoted_report, "equal_high") - budget_ds(
        previous_report, "equal_high"
    )
    delta_1200_256 = budget_ds(promoted_report, "1200_vs_256") - budget_ds(
        previous_report, "1200_vs_256"
    )
    return {
        "delta_384_256": delta_384_256,
        "delta_768_768": delta_768_768,
        "delta_1200_1200": delta_1200_1200,
        "delta_1200_256": delta_1200_256,
        "improved_384_256": delta_384_256 >= MIN_384_256_IMPROVEMENT,
        "within_768_768_limit": delta_768_768 >= -MAX_768_768_REGRESSION,
        "within_1200_1200_limit": delta_1200_1200 >= -MAX_HIGH_BUDGET_REGRESSION,
        "within_1200_256_limit": delta_1200_256 >= -MAX_HIGH_BUDGET_REGRESSION,
    }


def heldout_table_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    previous_rows = {
        row["suite"]: row for row in summary["previous_current_ref"]["rows"]
    }
    promoted_rows = {row["suite"]: row for row in summary["promoted_current"]["rows"]}
    for suite in sorted(set(previous_rows) | set(promoted_rows)):
        previous_ds = previous_rows.get(suite, {}).get("ds")
        promoted_ds = promoted_rows.get(suite, {}).get("ds")
        delta = None
        if previous_ds is not None and promoted_ds is not None:
            delta = float(promoted_ds) - float(previous_ds)
        rows.append(
            [
                suite,
                fmt(previous_ds),
                fmt(promoted_ds),
                fmt(delta),
            ]
        )
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_results_markdown(summary: dict[str, Any]) -> str:
    fixed_large = summary["fixed_large_suite"]
    gate = summary["gate"]
    metadata_provenance = summary["metadata_provenance"]
    lines = [
        "# AlphaZero-Lite Balanced w8s4 e1 Promotion Results",
        "",
        f"**Date**: {dt.date.today().isoformat()}",
        "",
        f"**Classification**: `{summary['final_classification']}`",
        "",
        "## Summary",
        "",
        f"- Previous current weights SHA256: `{summary['previous_current_weights_sha256']}`",
        f"- Promoted current weights SHA256: `{summary['promoted_current_weights_sha256']}`",
        f"- Manifest candidate hash: `{summary['manifest_candidate_weights_sha256']}`",
        "- Exact files changed under `model-artifact/current`: `metadata.json`, `weights.json`",
        "- Training run: not run",
        "- Replay generation: not run",
        "- Self-play generation: not run",
        "",
        "## Artifact Integrity",
        "",
        f"- Previous current weights SHA256 matched expected parent: `{summary['artifact_integrity']['previous_current_matches_expected']}`",
        f"- Candidate artifact verified at `{summary['candidate_artifact_path']}`",
        f"- Candidate checkpoint SHA256: `{summary['candidate_checkpoint_sha256'] or 'n/a'}`",
        f"- Promoted `model-artifact/current/weights.json` SHA256: `{summary['promoted_current_weights_sha256']}`",
        f"- Metadata JSON parse: `{summary['artifact_integrity']['metadata_json_parse']}`",
        f"- Runtime loader: `{summary['artifact_integrity']['runtime_loader']}`",
        f"- Runtime test: `{summary['artifact_integrity']['runtime_test_passed']}`",
        "",
        "## Metadata Provenance",
        "",
    ]
    for key, value in metadata_provenance.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Fixed Large-Suite Before/After",
            "",
            markdown_table(
                [
                    "Candidate",
                    "384:256 DS",
                    "768:256 DS",
                    "768:768 DS",
                    "1200:1200 DS",
                    "1200:256 DS",
                    "256:768 DS",
                ],
                [
                    [
                        "previous_current_ref",
                        fmt(
                            fixed_large["previous_current_ref"]["budget_results"][
                                "standard"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["previous_current_ref"]["budget_results"][
                                "challenger_768_vs_256"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["previous_current_ref"]["budget_results"][
                                "equal_768"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["previous_current_ref"]["budget_results"][
                                "equal_high"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["previous_current_ref"]["budget_results"][
                                "1200_vs_256"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["previous_current_ref"]["budget_results"][
                                "current_high_asymmetry"
                            ]["ds"]
                        ),
                    ],
                    [
                        "promoted_current",
                        fmt(
                            fixed_large["promoted_current"]["budget_results"][
                                "standard"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["promoted_current"]["budget_results"][
                                "challenger_768_vs_256"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["promoted_current"]["budget_results"][
                                "equal_768"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["promoted_current"]["budget_results"][
                                "equal_high"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["promoted_current"]["budget_results"][
                                "1200_vs_256"
                            ]["ds"]
                        ),
                        fmt(
                            fixed_large["promoted_current"]["budget_results"][
                                "current_high_asymmetry"
                            ]["ds"]
                        ),
                    ],
                ],
            ),
            "",
            "## Held-Out Spot Check",
            "",
            markdown_table(
                ["Suite", "Previous 384:256 DS", "Promoted 384:256 DS", "Delta"],
                heldout_table_rows(summary["heldout_spot_check"]),
            ),
            "",
            "## Gate Classification Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Classification",
                    "384:256 DS",
                    "1200:1200 DS",
                    "1200:256 DS",
                    "256:768 DS",
                ],
                [
                    [
                        "previous_current_ref",
                        gate["previous_current_ref"]["classification"],
                        fmt(
                            gate["previous_current_ref"]["budget_results"]["standard"][
                                "disadvantaged_seat_score"
                            ]
                        ),
                        fmt(
                            gate["previous_current_ref"]["budget_results"][
                                "equal_high"
                            ]["disadvantaged_seat_score"]
                        ),
                        fmt(
                            gate["previous_current_ref"]["budget_results"][
                                "challenger_high"
                            ]["disadvantaged_seat_score"]
                        ),
                        fmt(
                            gate["previous_current_ref"]["budget_results"][
                                "current_high_asymmetry"
                            ]["disadvantaged_seat_score"]
                        ),
                    ],
                    [
                        "promoted_current",
                        gate["promoted_current"]["classification"],
                        fmt(
                            gate["promoted_current"]["budget_results"]["standard"][
                                "disadvantaged_seat_score"
                            ]
                        ),
                        fmt(
                            gate["promoted_current"]["budget_results"]["equal_high"][
                                "disadvantaged_seat_score"
                            ]
                        ),
                        fmt(
                            gate["promoted_current"]["budget_results"][
                                "challenger_high"
                            ]["disadvantaged_seat_score"]
                        ),
                        fmt(
                            gate["promoted_current"]["budget_results"][
                                "current_high_asymmetry"
                            ]["disadvantaged_seat_score"]
                        ),
                    ],
                ],
            ),
            "",
            "## Acceptance Checks",
            "",
            f"- `384:256` improvement >= `{MIN_384_256_IMPROVEMENT:.2f}` DS: `{fixed_large['verification']['improved_384_256']}`",
            f"- `768:768` regression >= `-{MAX_768_768_REGRESSION:.2f}` DS: `{fixed_large['verification']['within_768_768_limit']}`",
            f"- `1200:1200` regression >= `-{MAX_HIGH_BUDGET_REGRESSION:.2f}` DS: `{fixed_large['verification']['within_1200_1200_limit']}`",
            f"- `1200:256` regression >= `-{MAX_HIGH_BUDGET_REGRESSION:.2f}` DS: `{fixed_large['verification']['within_1200_256_limit']}`",
            f"- Gate keeps high-search signature: `{summary['gate']['verification']['promoted_keeps_high_search_signature']}`",
            "",
            "## Artifacts",
            "",
            f"- Promotion workdir: `{summary['workdir']}`",
            f"- Previous current backup: `{summary['previous_current_artifact_path']}`",
            f"- Fixed large-suite report: `{summary['fixed_large_suite']['report_path']}`",
            f"- Held-out report dir: `{summary['heldout_spot_check']['reports_dir']}`",
            f"- Gate report dir: `{summary['gate']['reports_dir']}`",
            f"- Summary metrics: `{summary['summary_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote balanced_w8s4_policy_head_e1 into model-artifact/current."
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--source-workdir", type=Path, required=True)
    parser.add_argument(
        "--candidate-name", default=EXPECTED_CANDIDATE_NAME, required=False
    )
    parser.add_argument(
        "--current", type=Path, default=Path("model-artifact/current"), required=False
    )
    parser.add_argument(
        "--expected-previous-current-weights-sha256",
        default=EXPECTED_PREVIOUS_CURRENT_WEIGHTS_SHA256,
    )
    parser.add_argument(
        "--large-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/large_eval.jsonl"),
    )
    parser.add_argument("--heldout-suites", default="")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_name != EXPECTED_CANDIDATE_NAME:
        raise RuntimeError(
            "candidate name must exactly match balanced_w8s4_policy_head_e1"
        )

    workdir = args.workdir.resolve()
    source_workdir = args.source_workdir.resolve()
    current_dir = resolve_repo_path(args.current)
    large_suite = args.large_suite.resolve()
    heldout_suites = [path.resolve() for path in parse_csv_paths(args.heldout_suites)]
    workdir.mkdir(parents=True, exist_ok=True)

    manifest_path = source_workdir / "candidate_manifest.json"
    source_summary_path = source_workdir / "summary_metrics.json"
    require_file(manifest_path, "candidate manifest")
    require_file(source_summary_path, "summary metrics")
    require_file(current_dir / "metadata.json", "current metadata.json")
    require_file(current_dir / "weights.json", "current weights.json")
    require_file(large_suite, "large suite")
    for heldout_suite in heldout_suites:
        require_file(heldout_suite, "held-out suite")

    manifest = load_json(manifest_path)
    manifest_candidates = list(manifest.get("candidates", []))
    if not isinstance(manifest_candidates, list):
        raise RuntimeError("candidate manifest candidates field must be a list")
    candidate = find_candidate(manifest_candidates, args.candidate_name)
    candidate_info = verify_candidate_manifest(
        manifest=manifest,
        candidate=candidate,
        expected_parent_sha256=args.expected_previous_current_weights_sha256,
    )
    validate_candidate_artifact(candidate_info)

    previous_current_sha = sha256_file(current_dir / "weights.json")
    previous_current_matches_expected = (
        previous_current_sha == args.expected_previous_current_weights_sha256
    )
    if not previous_current_matches_expected:
        raise RuntimeError(
            "previous current weights SHA256 mismatch: "
            f"expected {args.expected_previous_current_weights_sha256}, got {previous_current_sha}"
        )

    previous_current_dir = workdir / "previous_current_artifact"
    copy_deployable_artifact(current_dir, previous_current_dir)

    candidate_artifact_dir = candidate_info["artifact_dir"]
    candidate_metadata = load_json(candidate_artifact_dir / "metadata.json")
    shutil.copy2(candidate_artifact_dir / "weights.json", current_dir / "weights.json")
    promoted_current_sha = sha256_file(current_dir / "weights.json")
    promoted_metadata = build_promoted_metadata(
        candidate_metadata=candidate_metadata,
        candidate={
            **candidate,
            "checkpoint_path": str(candidate_info["checkpoint_path"])
            if candidate_info["checkpoint_path"] is not None
            else None,
        },
        previous_current_weights_sha256=previous_current_sha,
        promoted_current_weights_sha256=promoted_current_sha,
    )
    write_json(current_dir / "metadata.json", promoted_metadata)

    runtime_probe_summary = validate_runtime_loader(current_dir)

    # Keep the test invocation explicit here so the promotion summary records the
    # repository-level runtime validation, not just direct ArtifactEvaluator probes.
    runtime_test = subprocess.run(
        [
            str(VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable)),
            "-m",
            "unittest",
            "ml.alphazero_lite.test_current_artifact_runtime",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    runtime_test_passed = runtime_test.returncode == 0
    if not runtime_test_passed:
        raise RuntimeError(
            "current artifact runtime test failed:\n"
            f"stdout:\n{runtime_test.stdout[-4000:]}\n"
            f"stderr:\n{runtime_test.stderr[-4000:]}"
        )

    fixed_large_workdir = workdir / "fixed_large_suite"
    fixed_large_report = run_opening_suite_benchmark(
        workdir=str(fixed_large_workdir),
        suite=str(large_suite),
        current=str(previous_current_dir),
        candidates=f"{previous_current_dir},{current_dir}",
        budget_pairs=DEFAULT_LARGE_BUDGETS,
        games_per_opening=2,
        seed=args.seed,
        workers=args.workers,
        timeout=14400,
    )
    previous_candidate_name = previous_current_dir.name
    previous_fixed_large = benchmark_candidate_report(
        fixed_large_report, previous_candidate_name
    )
    promoted_fixed_large = benchmark_candidate_report(
        fixed_large_report, current_dir.name
    )
    fixed_large_verification = verify_fixed_large_suite(
        previous_report=previous_fixed_large,
        promoted_report=promoted_fixed_large,
    )

    heldout_reports: dict[str, dict[str, Any]] = {}
    heldout_reports_dir = workdir / "heldout_spot_check"
    for heldout_suite in heldout_suites:
        suite_name = heldout_suite.stem
        report = run_opening_suite_benchmark(
            workdir=str(heldout_reports_dir / suite_name),
            suite=str(heldout_suite),
            current=str(previous_current_dir),
            candidates=f"{previous_current_dir},{current_dir}",
            budget_pairs=DEFAULT_LARGE_BUDGETS,
            games_per_opening=2,
            seed=args.seed,
            workers=args.workers,
            timeout=14400,
        )
        heldout_reports[suite_name] = report

    heldout_spot_check = {
        "reports_dir": str(heldout_reports_dir),
        "previous_current_ref": heldout_summary(
            heldout_reports, previous_candidate_name
        ),
        "promoted_current": heldout_summary(heldout_reports, current_dir.name),
    }

    gate_reports_dir = workdir / "gate"
    gate_reports_dir.mkdir(parents=True, exist_ok=True)
    previous_gate_report = run_default_gate(
        candidate_path=str(previous_current_dir),
        current_path=str(previous_current_dir),
        out=str(gate_reports_dir / "previous_current_ref.json"),
        seed=args.seed,
        workers=args.workers,
    )
    promoted_gate_report = run_default_gate(
        candidate_path=str(current_dir),
        current_path=str(previous_current_dir),
        out=str(gate_reports_dir / "promoted_current.json"),
        seed=args.seed,
        workers=args.workers,
    )
    gate_verification = {
        "promoted_keeps_high_search_signature": gate_keeps_high_search_signature(
            promoted_gate_report
        )
    }

    metadata_json = load_json(current_dir / "metadata.json")
    metadata_provenance = {
        "version": metadata_json.get("version"),
        "parent_version": metadata_json.get("parent_version"),
        "parent_weights_sha256": metadata_json.get("parent_weights_sha256"),
        "source_experiment": metadata_json.get("source_experiment"),
        "source_runner": metadata_json.get("source_runner"),
        "selected_lane": metadata_json.get("selected_lane"),
        "trainable_scope": metadata_json.get("trainable_scope"),
        "replay_sources": ",".join(metadata_json.get("replay_sources", [])),
        "replay_weights": ",".join(
            str(x) for x in metadata_json.get("replay_weights", [])
        ),
        "architecture": metadata_json.get("architecture_note"),
        "architecture_change": metadata_json.get("architecture_change"),
    }
    metadata_complete = all(
        value not in (None, "") for value in metadata_provenance.values()
    )

    final_classification = "promoted_balanced_w8s4_e1_current"
    if promoted_current_sha != candidate_info["artifact_weights_sha256"]:
        final_classification = "promotion_blocked_artifact_mismatch"
    elif not metadata_complete:
        final_classification = "promotion_blocked_artifact_mismatch"
    elif not runtime_test_passed:
        final_classification = "promotion_blocked_runtime_failure"
    elif not all(
        [
            fixed_large_verification["improved_384_256"],
            fixed_large_verification["within_768_768_limit"],
            fixed_large_verification["within_1200_1200_limit"],
            fixed_large_verification["within_1200_256_limit"],
        ]
    ):
        final_classification = "promotion_blocked_eval_regression"
    elif not gate_verification["promoted_keeps_high_search_signature"]:
        final_classification = "promotion_blocked_gate_regression"

    promotion_summary_path = workdir / SUMMARY_FILENAME
    summary = {
        "schema": PROMOTION_SCHEMA,
        "workdir": str(workdir),
        "summary_path": str(promotion_summary_path),
        "source_workdir": str(source_workdir),
        "candidate_name": EXPECTED_CANDIDATE_NAME,
        "candidate_artifact_path": str(candidate_artifact_dir),
        "candidate_checkpoint_sha256": candidate_info.get("checkpoint_sha256"),
        "previous_current_artifact_path": str(previous_current_dir),
        "previous_current_weights_sha256": previous_current_sha,
        "promoted_current_weights_sha256": promoted_current_sha,
        "manifest_candidate_weights_sha256": candidate_info["artifact_weights_sha256"],
        "changed_current_files": list(CHANGED_CURRENT_FILES),
        "artifact_integrity": {
            "previous_current_matches_expected": previous_current_matches_expected,
            "metadata_json_parse": True,
            "runtime_loader": True,
            "runtime_probe_summary": runtime_probe_summary,
            "runtime_test_passed": runtime_test_passed,
        },
        "metadata_provenance": metadata_provenance,
        "fixed_large_suite": {
            "report_path": str(
                fixed_large_workdir / "temperature_benchmark_report.json"
            ),
            "previous_current_ref": previous_fixed_large,
            "promoted_current": promoted_fixed_large,
            "verification": fixed_large_verification,
        },
        "heldout_spot_check": heldout_spot_check,
        "gate": {
            "reports_dir": str(gate_reports_dir),
            "previous_current_ref": previous_gate_report,
            "promoted_current": promoted_gate_report,
            "verification": gate_verification,
        },
        "guardrails": {
            "training_run": False,
            "replay_generation": False,
            "self_play_generation": False,
            "architecture_change": False,
            "residual_v4": False,
            "lr_change": False,
            "threshold_change": False,
        },
        "final_classification": final_classification,
    }
    write_json(promotion_summary_path, summary)
    PROMOTION_DOC.write_text(build_results_markdown(summary), encoding="utf-8")

    if final_classification != "promoted_balanced_w8s4_e1_current":
        raise RuntimeError(
            f"promotion failed with classification: {final_classification}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
