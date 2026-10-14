"""Freeze and publish seed475 retrospective gradient attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.alphazero_lite.seed475_policy_population_attribution import (
    INPUTS,
    OUT,
    SOURCES,
    calculate,
    sha,
)
from ml.alphazero_lite.seed466_runtime_bundle import MANIFEST_SHA256

ROOT = Path(__file__).resolve().parents[2]


def freeze(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(out.iterdir()):
        raise ValueError("seed475_output_exists")
    # Require the archived seed474 endpoint to pass its independent verifier first.
    from ml.alphazero_lite.verify_seed474_fresh_value_removal import (
        verify as verify_474,
    )

    verify_474(root)
    from ml.alphazero_lite.seed474_verification_supplement import (
        verify as verify_supplement,
    )

    verify_supplement(root)
    receipt = json.loads((root / INPUTS["seed474_receipt"]).read_text())
    for name, expected in receipt.items():
        if (
            name != "status"
            and sha(root / OUT.parent / "seed474-fresh-value-removal" / name)
            != expected
        ):
            raise ValueError(f"seed474_receipt_mismatch:{name}")
    out.mkdir(parents=True, exist_ok=True)
    arrays = __import__(
        "ml.alphazero_lite.seed466_runtime_bundle",
        fromlist=["read_authenticated_bundle"],
    ).read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = arrays["training"], arrays["evaluation"]
    train_rows = tr["expanded_replay_indices"][tr["train_positions"]]
    fresh_rows = train_rows[tr["fresh_source_mask"][train_rows]]
    unseen = np.flatnonzero(ev["source_ids"] == 0)
    reg = {
        "schema": "seed475-policy-population-attribution-registration-v1",
        "status": "frozen_before_gradient_calculation",
        "interpretation": "retrospective and exploratory; descriptive gradient attribution, no causal inference or strength claim",
        "authorization": "frozen-checkpoint diagnostic only; no training, optimizer sweep, searches, games, runtime export, or promotion",
        "inputs_sha256": {key: sha(root / path) for key, path in INPUTS.items()},
        "source_closure_sha256": {path: sha(root / path) for path in SOURCES},
        "authenticated_manifest_sha256": MANIFEST_SHA256,
        "dependencies": {
            "python": __import__("sys").version,
            "torch": __import__("torch").__version__,
            "numpy": np.__version__,
            "platform": __import__("platform").platform(),
        },
        "populations": {
            "training_selection": "expanded_replay_indices[train_positions], then fresh_source_mask; stored occurrence order and multiplicity preserved",
            "training_occurrences": int(len(fresh_rows)),
            "strict_unseen_selection": "evaluation source_ids == 0; occurrence order and own target preserved",
            "unseen_occurrences": int(len(unseen)),
            "unseen_exact_inputs": len(set(ev["input_identity"][unseen].tolist())),
            "bucket_edges": [16, 32],
            "stone_count": "sum decoded integer pits from invertible kalah_v3 input prefix; verify equals authenticated canonical pit count",
        },
        "checkpoints": {
            "initializer": "authenticated original initializer",
            "seed474_P": "independently verified P-steps.npz endpoint parameter arrays at step 15 post; reconstruct from initializer",
        },
        "objective": "fresh training policy CE sum(c_i*CE_i)/sum(c_i), identical total fresh policy mass denominator for full and each active-stone bucket; no per-bucket normalization",
        "unseen_weighting": {
            "exposure": "1/859 each occurrence",
            "equal_input": "1/(851*k_identity) each occurrence",
        },
        "numerical_protocol": {
            "forward": "float32 eval-mode model",
            "autograd": "float32",
            "accumulation": "contiguous occurrence order, chunks of 512 training and 128 unseen; gradients accumulated in chunk order",
            "loss_numerators": "float64 reduction in occurrence order",
            "geometry": "float64 flattened gradients, dot products, norms, and cosines",
            "threads": 1,
            "rtol": 2e-6,
            "atol": 2e-8,
            "classification_threshold": 1e-6,
            "buffers": "must remain bitwise unchanged",
        },
        "classification": {
            "full_population_opposition": "cosine(G_unseen, -g_full) >= 1e-6",
            "full_population_descent": "cosine <= -1e-6",
            "otherwise": "ambiguous_or_zero",
            "bucket_interpretation": "descriptive signed first-order contributions only; do not infer causality or authorize reweighting",
        },
    }
    (out / "registration.json").write_text(
        json.dumps(reg, indent=2, sort_keys=True) + "\n"
    )


def publish(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    reg = json.loads((out / "registration.json").read_text())
    if reg.get("status") != "frozen_before_gradient_calculation":
        raise ValueError("registration_not_frozen")
    for name, expected in reg["inputs_sha256"].items():
        if sha(root / INPUTS[name]) != expected:
            raise ValueError(f"input_changed:{name}")
    for name, expected in reg["source_closure_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError(f"source_changed:{name}")
    report, arrays = calculate(root)
    np.savez_compressed(out / "gradients.npz", **arrays)
    (out / "evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out / "results.md").write_text(_markdown(report))
    receipt = {
        p.name: sha(p)
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "receipt.json"
    }
    receipt["status"] = "seed475_publication_bound"
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def _markdown(report: dict) -> str:
    lines = [
        "# Seed475: fresh-policy gradient attribution",
        "",
        "Retrospective and exploratory frozen-checkpoint diagnostic. This is not causal evidence, a reweighting recommendation, or a strength claim.",
        "",
        "| Checkpoint | Unseen weighting | Classification | cosine with -g_full | <=16 contribution | 17–32 contribution | >32 contribution |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for checkpoint, data in report["checkpoints"].items():
        for weighting, unseen in data["unseen"].items():
            c = unseen["bucket_negative_gradient_contributions"]
            lines.append(
                f"| {checkpoint} | {weighting} | {unseen['classification']} | {unseen['cosine_with_full_negative_training_gradient']:.9g} | {c['<=16']:.9g} | {c['17-32']:.9g} | {c['>32']:.9g} |"
            )
    lines += [
        "",
        "No training, optimizer sweep, searches, games, runtime export, promotion, or strength claim was performed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "publish"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    (freeze if args.action == "freeze" else publish)(args.root)
    print(f"seed475 {args.action} complete")


if __name__ == "__main__":
    main()
