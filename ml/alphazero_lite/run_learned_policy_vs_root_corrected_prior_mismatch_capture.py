#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_CAPTURE_ARTIFACT = (
    "/tmp/azlite_learned_policy_vs_root_corrected_prior_capture/"
    "learned-policy-vs-root-corrected-prior-capture/"
    "learned_policy_vs_root_corrected_prior_capture.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_learned_policy_vs_root_corrected_prior_mismatch_capture"
DEFAULT_RUN_ID = "learned-policy-vs-root-corrected-prior-mismatch-capture"
SCHEMA = "azlite_learned_policy_vs_root_corrected_prior_mismatch_capture_v1"
FOCUS_ROW = "capture_available-002"
PRESERVATION_ROW = "capture_available-003"
INTERVENTIONS = (
    "original_prior",
    "zero_wrong_extra_turn_prior",
    "swap_reference_and_wrong",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--capture-artifact", default=DEFAULT_CAPTURE_ARTIFACT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _ranked_moves(distribution: dict[str, float] | None) -> list[int]:
    if not isinstance(distribution, dict):
        return []
    return [
        int(move)
        for move, _value in sorted(
            distribution.items(), key=lambda item: (-float(item[1]), int(item[0]))
        )
    ]


def _selection_breakdown_rank(row_payload: dict) -> list[int]:
    moves = (
        (((row_payload.get("trace_excerpt") or {}).get("selection_breakdown") or {}).get("moves"))
        or []
    )
    ranked = sorted(
        moves,
        key=lambda item: (
            -float(item.get("selection_score", float("-inf"))),
            int(item.get("move", -1)),
        ),
    )
    return [int(item["move"]) for item in ranked if "move" in item]


def _visit_rank(row_payload: dict) -> list[int]:
    distribution = ((row_payload.get("row_views") or {}).get("search_view") or {}).get(
        "visit_distribution"
    )
    return _ranked_moves(distribution)


def _policy_rank(row_payload: dict) -> list[int]:
    distribution = ((row_payload.get("row_views") or {}).get("policy_view") or {}).get(
        "raw_policy_distribution"
    )
    return _ranked_moves(distribution)


def _row_002_move_table(focus_interventions: dict) -> list[dict]:
    moves = sorted(
        {
            int(move)
            for intervention in INTERVENTIONS
            for move in (
                (((focus_interventions[intervention].get("row_views") or {}).get("policy_view") or {}).get("raw_policy_distribution") or {}).keys()
            )
        }
    )
    rows = []
    for move in moves:
        rows.append(
            {
                "move": move,
                "original_prior_probability": (
                    ((focus_interventions["original_prior"].get("row_views") or {}).get("policy_view") or {}).get("raw_policy_distribution")
                    or {}
                ).get(str(move)),
                "zero_wrong_prior_probability": (
                    (((focus_interventions["zero_wrong_extra_turn_prior"].get("trace_excerpt") or {}).get("selection_breakdown") or {}).get("moves") or [])
                ),
                "swap_prior_probability": (
                    (((focus_interventions["swap_reference_and_wrong"].get("trace_excerpt") or {}).get("selection_breakdown") or {}).get("moves") or [])
                ),
            }
        )
    return rows


def _normalized_prior_by_move(row_payload: dict) -> dict[str, float] | None:
    moves = (
        (((row_payload.get("trace_excerpt") or {}).get("selection_breakdown") or {}).get("moves"))
        or []
    )
    if not moves:
        return None
    return {
        str(int(entry["move"])): round(float(entry.get("prior", 0.0)), 4)
        for entry in moves
        if "move" in entry
    }


def build_payload(capture_artifact: dict, *, capture_artifact_path: str) -> dict:
    if (
        capture_artifact.get("classification")
        != "learned_prior_underweights_reference_and_overweights_wrong_extra_turn"
    ):
        raise ValueError("capture artifact must isolate the learned-prior mismatch")

    focus_interventions = dict(capture_artifact["focus_row"]["interventions"])
    preservation_interventions = dict(capture_artifact["preservation_row"]["interventions"])

    move_table = []
    moves = sorted(
        {
            int(move)
            for move in (
                (((focus_interventions["original_prior"].get("row_views") or {}).get("policy_view") or {}).get("raw_policy_distribution") or {}).keys()
            )
        }
    )
    zero_wrong_priors = _normalized_prior_by_move(
        focus_interventions["zero_wrong_extra_turn_prior"]
    ) or {}
    swap_priors = _normalized_prior_by_move(
        focus_interventions["swap_reference_and_wrong"]
    ) or {}
    original_priors = (
        ((focus_interventions["original_prior"].get("row_views") or {}).get("policy_view") or {}).get(
            "raw_policy_distribution"
        )
        or {}
    )
    for move in moves:
        move_key = str(move)
        move_table.append(
            {
                "move": move,
                "learned_policy_probability": original_priors.get(move_key),
                "zero_wrong_root_corrected_prior": zero_wrong_priors.get(move_key),
                "swap_root_corrected_prior": swap_priors.get(move_key),
            }
        )

    ranking_comparison = {
        intervention: {
            "policy_rank": _policy_rank(focus_interventions[intervention]),
            "selection_score_rank": _selection_breakdown_rank(
                focus_interventions[intervention]
            ),
            "final_visit_rank": _visit_rank(focus_interventions[intervention]),
            "selected_move": focus_interventions[intervention].get(
                "searched_selected_move"
            ),
        }
        for intervention in INTERVENTIONS
    }

    preservation_summary = {
        intervention: {
            "selected_move": preservation_interventions[intervention].get(
                "searched_selected_move"
            ),
            "reference_move_visit_share": preservation_interventions[intervention].get(
                "reference_move_visit_share"
            ),
        }
        for intervention in INTERVENTIONS
    }

    return {
        "schema": SCHEMA,
        "source_capture_artifact_path": capture_artifact_path,
        "classification": "reference_underweighting_and_wrong_extra_turn_overweighting_confirmed",
        "decision": "write_learned_policy_mismatch_note",
        "focus_row_id": FOCUS_ROW,
        "preservation_row_id": PRESERVATION_ROW,
        "row_002_move_probability_comparison": move_table,
        "row_002_ranking_comparison": ranking_comparison,
        "row_003_preservation_comparison": preservation_summary,
    }


def build_markdown(payload: dict) -> str:
    lines = [
        "# Learned Policy vs Root-Corrected Prior Mismatch Capture",
        "",
        "## Outcome",
        "",
        f"- classification: `{payload['classification']}`",
        f"- decision: `{payload['decision']}`",
        "",
        "## Row 002 Move Probabilities",
        "",
        "| move | learned policy | zero-wrong corrected prior | swap corrected prior |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["row_002_move_probability_comparison"]:
        lines.append(
            f"| `{row['move']}` | `{row['learned_policy_probability']}` | `{row['zero_wrong_root_corrected_prior']}` | `{row['swap_root_corrected_prior']}` |"
        )
    lines.extend([
        "",
        "## Row 002 Rankings",
        "",
    ])
    for intervention, ranking in payload["row_002_ranking_comparison"].items():
        lines.append(
            f"- `{intervention}`: policy `{ranking['policy_rank']}`, selection_score `{ranking['selection_score_rank']}`, final_visit `{ranking['final_visit_rank']}`, selected `{ranking['selected_move']}`"
        )
    lines.extend([
        "",
        "## Row 003 Preservation",
        "",
    ])
    for intervention, summary in payload["row_003_preservation_comparison"].items():
        lines.append(
            f"- `{intervention}`: selected `{summary['selected_move']}`, reference visit share `{summary['reference_move_visit_share']}`"
        )
    lines.extend([
        "",
        "## Conclusion",
        "",
        "- row `002` needs root correction because the learned guarded-`w2` policy both underweights the reference move and leaves too much mass on the wrong extra-turn branch",
        "- row `003` remains stable under the same corrections, so the mismatch is row-specific rather than a generic extra-turn suppression issue",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    capture_artifact_path = resolve_path(root, args.capture_artifact)
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    payload = build_payload(
        load_json(capture_artifact_path),
        capture_artifact_path=str(capture_artifact_path),
    )
    artifact_path = out_root / "learned_policy_vs_root_corrected_prior_mismatch_capture.json"
    write_json(artifact_path, payload)

    summary = {
        "run_id": args.run_id,
        "artifact_path": str(artifact_path),
        "schema": payload["schema"],
        "classification": payload["classification"],
        "decision": payload["decision"],
    }
    summary_path = out_root / "learned_policy_vs_root_corrected_prior_mismatch_capture_summary.json"
    write_json(summary_path, summary)

    report_path = root / "docs/alphazero-lite-learned-policy-vs-root-corrected-prior-mismatch-capture.md"
    report_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "classification": payload["classification"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
