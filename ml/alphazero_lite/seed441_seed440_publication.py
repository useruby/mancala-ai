"""Append-only completion of seed440's archive-defined attribution evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite import run_seed437_training_unseen_gradient_alignment as seed437
from ml.alphazero_lite.seed437_gradient_alignment import equal_input_weights
from ml.alphazero_lite.seed440_recorded_update_attribution import (
    _apply,
    _membership,
    _hash_parameters,
    run as seed440_run,
)

ROOT = Path(__file__).resolve().parents[2]
REL = Path("docs/data/seed441-seed440-publication-completion")
WEIGHTINGS = ("exposure_weighted", "equal_input")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_root(root: Path) -> None:
    """Rebind the imported legacy array loader's complete filesystem roots."""
    root = root.resolve()
    seed437.ROOT = root
    seed437.OUT = root / "docs/data/seed437-training-unseen-gradient-alignment"
    seed437.DATA = root / "docs/data/seed426-canonical-overlap"
    seed437.REGISTRATION = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    seed437.INIT = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    seed437.ADAM = root / "docs/data/seed435-adam-direction-screen/A-final.npz"


def generate(root: Path, publish: bool = False) -> dict[str, Any]:
    """Reconstruct the two frozen paths and optionally write supplemental archives."""
    root = root.resolve()
    configure_root(root)
    arrays = seed437.prepare_arrays()
    x, policy, value, _coeff, _mult, _source, _bucket, ids, _old_weights = arrays
    selected = [
        row
        for row in _membership(root)
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    identities = [row["input_identity"] for row in selected]
    if (
        len(ids) != len(identities)
        or [int(r["compact_row"]) for r in selected] != ids.tolist()
    ):
        raise ValueError("selected_membership_mismatch")
    weights = {
        "exposure_weighted": np.full(len(ids), 1.0 / len(ids), dtype=np.float64),
        "equal_input": equal_input_weights(identities),
    }
    attribution = seed440_run(root)
    bindings_path = (
        root / "docs/data/seed440-recorded-update-attribution/evidence-bindings.json"
    )
    protocol_path = root / "docs/data/seed440-recorded-update-attribution/protocol.json"
    ledger_path = (
        root / "docs/data/seed440-recorded-update-attribution/step-ledger.json"
    )
    layout = attribution["parameter_layout"]
    size = layout[-1]["stop"]
    predictions: dict[str, np.ndarray] = {}
    gradients: dict[str, np.ndarray] = {}
    state_ids: list[str] = []
    row_ids: list[str] = []
    row_indices: list[int] = []

    def record_predictions(model: torch.nn.Module, key: str) -> None:
        model.eval()
        logits_parts, value_parts = [], []
        with torch.inference_mode():
            for offset in range(0, len(ids), 512):
                batch = ids[offset : offset + 512]
                logits, pred_value = model(torch.from_numpy(x[batch]))
                logits_parts.append(logits.detach().cpu().numpy())
                value_parts.append(pred_value.detach().cpu().numpy())
        predictions[f"{key}_policy_logits"] = np.concatenate(logits_parts)
        predictions[f"{key}_value"] = np.concatenate(value_parts)
        state_ids.append(key)

    for arm in ("A", "B"):
        delta_path = (
            root / f"docs/data/seed435-adam-direction-screen/{arm}-audit-deltas.npz"
        )
        with np.load(delta_path, allow_pickle=False) as archive:
            deltas = np.asarray(archive["deltas"], dtype=np.float64)
        if deltas.shape != (16, size) or not np.isfinite(deltas).all():
            raise ValueError(f"invalid_delta_archive:{arm}")
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(
            model,
            root
            / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        )
        record_predictions(model, f"{arm}_state_00")
        for step, delta in enumerate(deltas, start=1):
            before = [parameter.detach().clone() for parameter in model.parameters()]
            for weighting in WEIGHTINGS:
                gradients[f"{arm}_step_{step:02d}_{weighting}"] = seed437.flat_grad(
                    model,
                    ids,
                    weights[weighting],
                    x,
                    policy,
                    value,
                    "policy",
                    512,
                )
            _apply(model, delta, layout, 0.5)
            record_predictions(model, f"{arm}_midpoint_{step:02d}")
            with torch.no_grad():
                for parameter, original in zip(model.parameters(), before, strict=True):
                    parameter.copy_(original)
            _apply(model, delta, layout)
            record_predictions(model, f"{arm}_state_{step:02d}")
        endpoint = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(
            endpoint,
            root / f"docs/data/seed435-adam-direction-screen/{arm}-final.npz",
        )
        if not all(
            torch.equal(left, right)
            for left, right in zip(
                model.parameters(), endpoint.parameters(), strict=True
            )
        ):
            raise ValueError(f"endpoint_parameters_differ:{arm}")
        if (
            _hash_parameters(model)
            != attribution["arms"][arm]["path_parameter_sha256"][-1]
        ):
            raise ValueError(f"endpoint_parameter_hash_differ:{arm}")

    row_indices = ids.tolist()
    row_ids = identities
    result = {
        "schema": "seed441-publication-completion-v1",
        "status": "post-execution publication completion",
        "interpretation": "Supplement to, and not a replacement for, seed440 historical evidence.",
        "seed440_protocol_sha256": sha(protocol_path),
        "seed440_bindings_sha256": sha(bindings_path),
        "seed440_ledger_sha256": sha(ledger_path),
        "membership": {
            "selected_exposures": len(ids),
            "unique_input_identities": len(set(identities)),
        },
        "classification": attribution["classification"],
        "archives": {
            "predictions": "predictions.npz",
            "weighting_gradients": "weighting-gradients.npz",
            "row-identities": "row-identities.json",
            "parameter-layout": "parameter-layout.json",
        },
        "_predictions": predictions,
        "_gradients": gradients,
        "_layout": layout,
        "_rows": row_indices,
        "_identities": row_ids,
        "_state_order": state_ids,
        "_ledger": attribution["arms"],
        "_classification": attribution["classification"],
    }
    if publish:
        output = root / REL
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions.npz", **predictions)
        np.savez_compressed(output / "weighting-gradients.npz", **gradients)
        (output / "row-identities.json").write_text(
            json.dumps(
                {
                    "compact_rows": row_indices,
                    "input_identities": row_ids,
                    "state_order": state_ids,
                },
                indent=2,
            )
            + "\n"
        )
        (output / "parameter-layout.json").write_text(
            json.dumps(layout, indent=2) + "\n"
        )
        public_result = {
            key: value for key, value in result.items() if not key.startswith("_")
        }
        (output / "completion.json").write_text(
            json.dumps(public_result, indent=2, sort_keys=True) + "\n"
        )
        return public_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args.root, args.publish), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
