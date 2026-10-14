"""Freeze, execute, and publish seed479 population-ablation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.alphazero_lite.seed479_population_ablation import (
    OUT,
    ROOT,
    calculate,
    freeze,
    sha,
)


def publish(root: Path) -> None:
    root = root.resolve()
    registration = freeze(root)
    evidence, arrays = calculate(root)
    out = root / OUT
    (out / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(out / "trajectory-evidence.npz", **arrays)
    result = [
        "# Seed479 full-population versus minibatch normalized-gradient ablation",
        "",
        "Exploratory only: the existing evaluation cohort and seed478 findings informed the protocol. This is not fresh confirmation.",
        "",
        "M divides each minibatch fresh-policy numerator by that minibatch's fresh policy mass; F divides the full fresh training-policy numerator by the complete fresh policy mass.",
        "",
        f"Fresh training occurrences: {registration['population']['fresh_training_occurrences']}; denominator: {registration['population']['full_fresh_policy_mass']:.17g}.",
        "",
        "## Endpoint metrics",
        "",
        "```json",
        json.dumps(
            {
                "initializer": evidence["initializer_metrics"],
                "endpoints": evidence["endpoint_metrics"],
                "training_policy_ce": evidence["full_training_policy_ce"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Gates",
        "",
    ]
    result.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in evidence["gates"].items()
    )
    result.extend(
        [
            "",
            f"Decision: `{evidence['decision']}`.",
            "",
            "No search, game, runtime export, promotion, or strength claim is made. Passing authorizes only a separately designed fresh confirmation.",
            "",
        ]
    )
    (out / "results.md").write_text("\n".join(result))
    receipt = {
        "status": "seed479_population_ablation_bound",
        **{
            name: sha(out / name)
            for name in (
                "registration.json",
                "evidence.json",
                "trajectory-evidence.npz",
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
    publish(parser.parse_args().root)
    print("seed479 population ablation published")


if __name__ == "__main__":
    main()
