#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_MISMATCH_CAPTURE = (
    "/tmp/azlite_learned_policy_vs_root_corrected_prior_mismatch_capture/"
    "learned-policy-vs-root-corrected-prior-mismatch-capture/"
    "learned_policy_vs_root_corrected_prior_mismatch_capture.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_learned_policy_mismatch_note"
DEFAULT_RUN_ID = "learned-policy-mismatch-note"
SCHEMA = "azlite_learned_policy_mismatch_note_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mismatch-capture", default=DEFAULT_MISMATCH_CAPTURE)
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


def build_payload(mismatch_capture: dict, *, mismatch_capture_path: str) -> dict:
    classification = mismatch_capture.get("classification")
    if (
        classification
        != "reference_underweighting_and_wrong_extra_turn_overweighting_confirmed"
    ):
        raise ValueError("mismatch capture must confirm the learned-policy mismatch")
    return {
        "schema": SCHEMA,
        "classification": classification,
        "decision": "note_learned_policy_mismatch_confirmed",
        "input_artifacts": {
            "mismatch_capture_path": mismatch_capture_path,
        },
        "conclusion": {
            "focus_row_id": mismatch_capture.get("focus_row_id"),
            "preservation_row_id": mismatch_capture.get("preservation_row_id"),
            "summary": "The guarded-w2 learned policy on row 002 both underweights the reference move and overweights the wrong extra-turn move, while row 003 remains stable under the same root corrections.",
        },
    }


def build_markdown(payload: dict) -> str:
    return "\n".join(
        [
            "# AlphaZero-lite Learned Policy Mismatch Note",
            "",
            "## Outcome",
            "",
            f"- classification: `{payload['classification']}`",
            f"- decision: `{payload['decision']}`",
            "",
            "## Conclusion",
            "",
            f"- {payload['conclusion']['summary']}",
            f"- focus row: `{payload['conclusion']['focus_row_id']}`",
            f"- preservation row: `{payload['conclusion']['preservation_row_id']}`",
            "",
            "Supporting artifact:",
            "",
            f"- `{payload['input_artifacts']['mismatch_capture_path']}`",
        ]
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    mismatch_capture_path = resolve_path(root, args.mismatch_capture)
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    payload = build_payload(
        load_json(mismatch_capture_path), mismatch_capture_path=str(mismatch_capture_path)
    )
    summary_path = out_root / "learned_policy_mismatch_note.json"
    write_json(summary_path, payload)

    report_path = root / "docs/alphazero-lite-learned-policy-mismatch-note.md"
    report_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "decision": payload["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
