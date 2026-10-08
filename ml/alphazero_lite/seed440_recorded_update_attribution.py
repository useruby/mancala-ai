"""Retrospective attribution of the archive-defined seed435 update paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.run_seed437_training_unseen_gradient_alignment import (
    flat_grad,
    prepare_arrays,
)
from ml.alphazero_lite.seed437_gradient_alignment import equal_input_weights

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed440-recorded-update-attribution"
DATA435 = ROOT / "docs/data/seed435-adam-direction-screen"
INIT = (
    ROOT
    / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
GROUPS = ("shared_trunk", "policy_head", "value_head")
TOL = 3e-5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(d: dict[str, dict[str, float]]) -> str:
    finite = all(d[w]["D"] > 0 and d[w]["R"] >= 0.75 * d[w]["D"] for w in d)
    first = all(d[w]["D"] > 0 and d[w]["S"] >= 0.75 * d[w]["D"] for w in d)
    if finite:
        return "finite_step_residual_dominant"
    if first:
        return "first_order_direction_dominant"
    return "mixed_or_no_common_dominance"


def _layout(model: torch.nn.Module) -> list[dict[str, Any]]:
    out, start = [], 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = (
            "shared_trunk"
            if name.startswith(("input_layer.", "residual_layers."))
            else "policy_head"
            if name.startswith(("policy_hidden_layer.", "policy_head."))
            else "value_head"
            if name.startswith(("value_hidden_layer.", "value_head."))
            else None
        )
        if group is None:
            raise ValueError(f"unclassified_parameter:{name}")
        stop = start + param.numel()
        out.append(
            {
                "name": name,
                "shape": list(param.shape),
                "start": start,
                "stop": stop,
                "group": group,
            }
        )
        start = stop
    return out


def _apply(
    model: torch.nn.Module,
    delta: np.ndarray,
    layout: list[dict[str, Any]],
    scale: float = 1.0,
) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for entry in layout:
            piece = (
                (delta[entry["start"] : entry["stop"]] * scale)
                .astype(np.float32)
                .reshape(entry["shape"])
            )
            params[entry["name"]].add_(torch.from_numpy(piece))


def _hash_parameters(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for param in model.parameters():
        digest.update(param.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _ce(
    model: torch.nn.Module,
    arrays: tuple[Any, ...],
    ids: np.ndarray,
    weights: np.ndarray,
    chunk: int = 512,
) -> float:
    x, p = arrays[0], arrays[1]
    total = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), chunk):
            chosen, w = ids[start : start + chunk], weights[start : start + chunk]
            xx = torch.from_numpy(x[chosen])
            logits, _ = model(xx)
            masks = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[chosen])
            )
            losses = train.compute_policy_cross_entropy(
                logits.masked_fill(masks <= 0, -1e9), torch.from_numpy(p[chosen])
            )
            total += float(torch.sum(losses.double() * torch.from_numpy(w).double()))
    return total


def run(root: Path = ROOT, publish: bool = False) -> dict[str, Any]:
    """Recompute the frozen attribution and optionally publish all derived outputs."""
    global ROOT, OUT, DATA435, INIT
    ROOT = root.resolve()
    OUT = ROOT / "docs/data/seed440-recorded-update-attribution"
    DATA435 = ROOT / "docs/data/seed435-adam-direction-screen"
    INIT = (
        ROOT
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    arrays = prepare_arrays()
    x, p, _v, coeff, mult, _src, _bucket, ids, _uw = arrays
    with np.load(
        ROOT
        / "docs/data/seed438-seed437-gradient-correction/corrected-gradient-vectors.npz",
        allow_pickle=False,
    ) as vectors:
        # Seed439's independently verified U-gradients bind the endpoint only;
        # gradients at each reconstructed state are recomputed below.
        if "adam_a16__U_exposure" not in vectors:
            raise ValueError("seed439_gradient_archive_missing")
    selected_membership = [
        row
        for row in _membership(ROOT)
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    identities = [r["input_identity"] for r in selected_membership]
    if len(identities) != len(ids):
        raise ValueError("unseen_membership_identity_mismatch")
    weights = {
        "exposure_weighted": np.full(len(ids), 1.0 / len(ids)),
        "equal_input": equal_input_weights(identities),
    }
    model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(model, INIT)
    layout = _layout(model)
    size = layout[-1]["stop"]
    result: dict[str, Any] = {
        "schema": "seed440-attribution-v1",
        "interpretation": "observational retrospective diagnosis only; no intervention authorized",
        "tolerances": {"absolute": TOL, "relative": TOL},
        "membership": {"exposures": len(ids), "identities": len(set(identities))},
        "parameter_layout": layout,
        "arms": {},
    }
    reconstructed: dict[str, np.ndarray] = {}
    for arm in ("A", "B"):
        with np.load(
            DATA435 / f"{arm}-audit-deltas.npz", allow_pickle=False
        ) as archive:
            deltas = np.asarray(archive["deltas"], dtype=np.float64)
        if deltas.shape != (16, size):
            raise ValueError(f"delta_shape_invalid:{arm}")
        model_arm = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(model_arm, INIT)
        state_hashes = [_hash_parameters(model_arm)]
        ledger = []
        for step, delta in enumerate(deltas):
            before_parameters = [
                parameter.detach().clone() for parameter in model_arm.parameters()
            ]
            before_ce, midpoint_ce, after_ce = {}, {}, {}
            dots, groups_dot = {}, {}
            for weighting in weights:
                grad = flat_grad(
                    model_arm, ids, weights[weighting], x, p, arrays[2], "policy", 512
                )
                before_ce[weighting] = _ce(model_arm, arrays, ids, weights[weighting])
                dots[weighting] = float(np.dot(grad, delta))
                groups_dot[weighting] = {
                    g: float(
                        np.dot(
                            grad[
                                [
                                    i
                                    for e in layout
                                    if e["group"] == g
                                    for i in range(e["start"], e["stop"])
                                ]
                            ],
                            delta[
                                [
                                    i
                                    for e in layout
                                    if e["group"] == g
                                    for i in range(e["start"], e["stop"])
                                ]
                            ],
                        )
                    )
                    for g in GROUPS
                }
            _apply(model_arm, delta, layout, 0.5)
            for weighting in weights:
                midpoint_ce[weighting] = _ce(model_arm, arrays, ids, weights[weighting])
            with torch.no_grad():
                for parameter, original in zip(
                    model_arm.parameters(), before_parameters, strict=True
                ):
                    parameter.copy_(original)
            _apply(model_arm, delta, layout)
            for weighting in weights:
                after_ce[weighting] = _ce(model_arm, arrays, ids, weights[weighting])
            ledger.append(
                {
                    "step": step + 1,
                    "delta_sha256": hashlib.sha256(
                        delta.astype("<f8").tobytes()
                    ).hexdigest(),
                    "pre_ce": before_ce,
                    "midpoint_ce": midpoint_ce,
                    "post_ce": after_ce,
                    "s": dots,
                    "group_s": groups_dot,
                    "D": {w: after_ce[w] - before_ce[w] for w in weights},
                    "r": {w: after_ce[w] - before_ce[w] - dots[w] for w in weights},
                    "midpoint_linear_residual": {
                        w: midpoint_ce[w] - before_ce[w] - 0.5 * dots[w]
                        for w in weights
                    },
                }
            )
            state_hashes.append(_hash_parameters(model_arm))
        endpoint = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(endpoint, DATA435 / f"{arm}-final.npz")
        if any(
            not torch.equal(a, b)
            for a, b in zip(model_arm.parameters(), endpoint.parameters(), strict=True)
        ):
            raise ValueError(f"path_endpoint_mismatch:{arm}")
        reconstructed[arm] = np.asarray(state_hashes)
        sums = {
            w: {
                key: float(sum(row[key][w] for row in ledger))
                for key in ("D", "s", "r")
            }
            for w in weights
        }
        for w in weights:
            sums[w]["telescoping_error"] = sums[w]["D"] - (
                ledger[-1]["post_ce"][w] - ledger[0]["pre_ce"][w]
            )
            sums[w]["identity_error"] = sums[w]["D"] - sums[w]["s"] - sums[w]["r"]
            sums[w]["group_reconciliation_error"] = sums[w]["s"] - sum(
                row["group_s"][w][g] for row in ledger for g in GROUPS
            )
        result["arms"][arm] = {
            "step_ledger": ledger,
            "path_parameter_sha256": state_hashes,
            "totals": sums,
            "endpoint_sha256": sha(DATA435 / f"{arm}-final.npz"),
        }
    primary = {
        w: {key: result["arms"]["A"]["totals"][w][key] for key in ("D", "s", "r")}
        for w in weights
    }
    result["classification"] = classify(
        {
            w: {"D": primary[w]["D"], "S": primary[w]["s"], "R": primary[w]["r"]}
            for w in weights
        }
    )
    result["historically_observed_endpoint_changes_before_diagnostic"] = {
        "exposure_weighted": 0.01705,
        "equal_input": 0.02310,
    }
    if publish:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUT / "reconstructed-path-hashes.npz",
            **{
                f"{arm}_sha256": np.asarray(hashes)
                for arm, hashes in reconstructed.items()
            },
        )
        (OUT / "step-ledger.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    return result


def _membership(root: Path) -> list[dict[str, Any]]:
    import gzip

    path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.publish), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
