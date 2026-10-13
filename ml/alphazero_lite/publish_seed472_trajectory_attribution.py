"""Publish append-only seed472 trajectory-attribution evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.alphazero_lite.seed472_trajectory_attribution import (
    INPUTS,
    OUT,
    SOURCES,
    calculate,
    sha,
)


def publish(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    registration = json.loads((out / "registration.json").read_text())
    if registration.get("status") != "frozen_before_execution":
        raise ValueError("registration_not_frozen")
    if set(registration["source_closure_sha256"]) != set(SOURCES):
        raise ValueError("frozen_source_inventory_mismatch")
    for rel, expected in registration["source_closure_sha256"].items():
        if sha(root / rel) != expected:
            raise ValueError(f"frozen_source_changed:{rel}")
    for key, rel in INPUTS.items():
        if sha(root / rel) != registration["inputs_sha256"][key]:
            raise ValueError(f"frozen_input_changed:{key}")
    evidence, diagnostic = calculate(root)
    np.savez_compressed(out / "evidence.npz", **evidence)
    (out / "diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Seed472 read-only trajectory attribution",
        "",
        "Retrospective local attribution of archived seed471 T; descriptive only and not proof of optimizer-memory causality.",
        "",
        f"**Classification:** `{diagnostic['classification']}`",
        "",
        "| Weighting | A signed / absolute | R signed / absolute | Delta signed / absolute | Classification |",
        "|---|---:|---:|---:|---|",
    ]
    for weighting, row in diagnostic["totals"].items():
        lines.append(
            f"| {weighting} | {row['A_signed']:.12g} / {row['A_absolute']:.12g} | {row['R_signed']:.12g} / {row['R_absolute']:.12g} | {row['Delta_signed']:.12g} / {row['Delta_absolute']:.12g} | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            "Forward-schedule endpoint CE and difference from seed471's historical endpoint (historical metrics are preserved):",
            "",
        ]
    )
    for weighting in ("exposure_weighted", "equal_input"):
        lines.append(
            f"- {weighting}: seed472={diagnostic['forward_schedule_endpoint'][f'post_{weighting}']:.12g}; seed471={diagnostic['seed471_historical_endpoint_fresh_ce'][weighting]:.12g}; difference={diagnostic['forward_schedule_difference_from_seed471'][weighting]:.12g}"
        )
    lines += [
        "",
        "No training trajectory was run. Carried-Adam, zero-moment first-step Adam, and negative-J alignments are local comparisons and do not establish optimizer-memory causality.",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    receipt = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "receipt.json"
    }
    receipt["status"] = "post_execution_bound_receipt"
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    publish(parser.parse_args().root)


if __name__ == "__main__":
    main()
