"""Freeze and publish seed476 read-only direction transport evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.alphazero_lite.seed476_direction_transport import (
    INPUTS,
    OUT,
    calculate,
    freeze,
    sha,
)

ROOT = Path(__file__).resolve().parents[2]


def publish(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    registration = json.loads((out / "registration.json").read_text())
    if registration.get("status") != "frozen_before_gradient_calculation":
        raise ValueError("registration_not_frozen")
    for name, expected in registration["inputs_sha256"].items():
        if sha(root / INPUTS[name]) != expected:
            raise ValueError(f"input_changed:{name}")
    for name, expected in registration["source_closure_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError(f"source_changed:{name}")
    report, arrays = calculate(root)
    np.savez_compressed(out / "directions.npz", **arrays)
    (out / "evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out / "results.md").write_text(_markdown(report))
    receipt = {
        path.name: sha(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "receipt.json"
    }
    receipt["status"] = "seed476_publication_bound"
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def _markdown(report: dict) -> str:
    lines = [
        "# Seed476: full-population versus minibatch direction transport",
        "",
        "Retrospective and exploratory read-only diagnostic; descriptive attribution only, with no optimizer or sampling causality and no strength claim.",
        "",
        "| Step | Weighting | Full negative | Minibatch negative | Recorded Adam | Category |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for step in report["steps"]:
        for weighting, result in step["unseen"].items():
            scores = result["directions"]
            lines.append(
                f"| {step['step']} | {weighting} | {scores['full_negative']['cosine']:.9g} | {scores['minibatch_negative']['cosine']:.9g} | {scores['recorded_adam']['cosine']:.9g} | {result['category']} |"
            )
    lines.extend(
        [
            "",
            "No training arm, optimizer sweep, searches, games, runtime export, promotion, or strength claim was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "publish"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root)
    else:
        publish(args.root)
    print(f"seed476 {args.action} complete")


if __name__ == "__main__":
    main()
