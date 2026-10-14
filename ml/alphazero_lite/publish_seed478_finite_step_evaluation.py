"""Freeze and publish seed478 finite-step evaluation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.alphazero_lite.seed478_finite_step_evaluation import (
    OUT,
    ROOT,
    calculate,
    freeze,
    sha,
)


def publish(root: Path) -> None:
    root = root.resolve()
    freeze(root)
    report, arrays = calculate(root)
    out = root / OUT
    (out / "evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(out / "occurrence-evidence.npz", **arrays)
    lines = [
        "# Seed478 finite-step evaluation",
        "",
        "Retrospective local counterfactual diagnostic; no sequential trajectory is implied and no training is authorized.",
        "",
        "Canonical seed476 verifier passed in the frozen environment; seed477 semantic correction passed but supplied no numerical replay.",
        "",
        "## Counts and summaries",
        "",
    ]
    for key, values in report["curves"].items():
        lines.append(
            f"- `{key}`: improvement {values['improvement']}, regression {values['regression']}, ambiguous {values['ambiguous']}, mean {values['mean']}, median {values['median']}."
        )
    lines += ["", f"Registration SHA-256: `{sha(out / 'registration.json')}`", ""]
    (out / "results.md").write_text("\n".join(lines))
    receipt = {
        "status": "seed478_finite_step_evaluation_bound",
        **{
            name: sha(out / name)
            for name in (
                "registration.json",
                "evidence.json",
                "occurrence-evidence.npz",
                "results.md",
            )
        },
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    publish(args.root)
    print("seed478 finite-step evaluation published")


if __name__ == "__main__":
    main()
