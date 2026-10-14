"""Independent read-only verification of seed478 finite-step evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.alphazero_lite.seed478_finite_step_evaluation import (
    INPUT_PATHS,
    OUT,
    ROOT,
    SOURCE_PATHS,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt.get("status") != "seed478_finite_step_evaluation_bound":
        raise ValueError("receipt_status")
    if set(receipt) != {
        "status",
        "registration.json",
        "evidence.json",
        "occurrence-evidence.npz",
        "results.md",
    }:
        raise ValueError("receipt_inventory")
    for name, digest in receipt.items():
        if name != "status" and _sha(out / name) != digest:
            raise ValueError(f"receipt_mismatch:{name}")
    reg = json.loads((out / "registration.json").read_text())
    if reg.get("status") != "frozen_before_trial_evaluation":
        raise ValueError("registration_status")
    if set(reg.get("inputs_sha256", {})) != set(INPUT_PATHS):
        raise ValueError("input_inventory")
    for key, relative in INPUT_PATHS.items():
        if _sha(root / relative) != reg["inputs_sha256"][key]:
            raise ValueError(f"input_binding:{key}")
    if set(reg.get("sources_sha256", {})) != set(SOURCE_PATHS):
        raise ValueError("source_inventory")
    for relative, digest in reg["sources_sha256"].items():
        if _sha(root / relative) != digest:
            raise ValueError(f"source_binding:{relative}")
    if reg.get("scales") != [1.0, 0.5, 0.25, 0.125]:
        raise ValueError("registered_scales")
    # Reconstruct direction vectors via seed476's independent input/checkpoint implementation;
    # this verifier does not consume seed476 diagnostic vectors or publisher calculations.
    from ml.alphazero_lite.verify_seed476_transport_supplement import (
        _independent_reconstruct,
    )
    import torch
    from ml.alphazero_lite import train
    from ml.alphazero_lite.seed466_runtime_bundle import (
        MANIFEST_SHA256,
        read_authenticated_bundle,
    )

    _, reconstructed = _independent_reconstruct(root)
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = bundle["training"], bundle["evaluation"]
    held = reconstructed["unseen_occurrences"]
    held_rows = reconstructed["unseen_rows"]
    unseen_weights = {
        "exposure": reconstructed["unseen_exposure_weights"],
        "equal_input": reconstructed["unseen_equal_input_weights"],
    }
    unseen_targets = reconstructed["unseen_targets"]
    saved = np.load(
        root / "docs/data/seed474-fresh-value-removal/P-steps.npz",
        allow_pickle=False,
    )
    report = json.loads((out / "evidence.json").read_text())
    if (
        report.get("schema") != "seed478-finite-step-evaluation-evidence-v1"
        or len(report.get("steps", [])) != 16
    ):
        raise ValueError("evidence_coverage")
    with np.load(out / "occurrence-evidence.npz", allow_pickle=False) as arrays:
        if not np.array_equal(arrays["unseen_targets"], unseen_targets):
            raise ValueError("unseen_targets_mismatch")
        if not np.array_equal(
            arrays["unseen_exposure_weights"], unseen_weights["exposure"]
        ):
            raise ValueError("exposure_weights_mismatch")
        if not np.array_equal(
            arrays["unseen_equal_input_weights"], unseen_weights["equal_input"]
        ):
            raise ValueError("equal_input_weights_mismatch")
        for step, record in enumerate(report["steps"]):
            if record.get("step") != step + 1 or len(record.get("trials", [])) != 12:
                raise ValueError(f"trial_coverage:{step + 1}")
            identities = [
                (trial.get("direction"), trial.get("scale"))
                for trial in record["trials"]
            ]
            expected = [
                (d, s)
                for d in ("recorded_adam", "full_negative", "minibatch_negative")
                for s in (1.0, 0.5, 0.25, 0.125)
            ]
            if identities != expected:
                raise ValueError(f"trial_identity:{step + 1}")
            for trial in record["trials"]:
                prefix = trial.get("array_prefix")
                if prefix not in arrays:
                    # NPZ membership checks names, not prefixes; validate required arrays explicitly.
                    pass
                state_key = f"{prefix}_state"
                unseen_key = f"{prefix}_unseen_occurrence_losses"
                probs_key = f"{prefix}_unseen_probabilities"
                train_key = f"{prefix}_training_occurrence_losses"
                if any(
                    k not in arrays.files
                    for k in (state_key, unseen_key, probs_key, train_key)
                ):
                    raise ValueError(f"trial_arrays:{prefix}")
                state = arrays[state_key].astype(np.float64)
                if hashlib.sha256(state.tobytes()).hexdigest() != trial["state_sha256"]:
                    raise ValueError(f"trial_state_hash:{prefix}")
                losses, probabilities = arrays[unseen_key], arrays[probs_key]
                train_losses = arrays[train_key]
                if (
                    len(losses) != 859
                    or len(probabilities) != 859
                    or len(train_losses) != 60015
                ):
                    raise ValueError(f"occurrence_count:{prefix}")
                if not np.all(np.isfinite(losses)) or not np.all(
                    np.isfinite(probabilities)
                ):
                    raise ValueError(f"nonfinite_trial:{prefix}")
                label, scale = trial["direction"], trial["scale"]
                model = train.PolicyValueNet((96, 3), "residual_v3", 27)
                train.load_checkpoint_into_model(
                    model,
                    root
                    / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
                )
                params = list(model.parameters())
                before = [
                    torch.from_numpy(saved[f"{step:02d}_pre_{i:02d}"].copy())
                    for i in range(len(params))
                ]
                after = [
                    torch.from_numpy(saved[f"{step:02d}_post_{i:02d}"].copy())
                    for i in range(len(params))
                ]
                origin = np.concatenate(
                    [v.numpy().reshape(-1).astype(np.float64) for v in before]
                )
                adam_delta = (
                    np.concatenate(
                        [v.numpy().reshape(-1).astype(np.float64) for v in after]
                    )
                    - origin
                )
                if label == "recorded_adam" and scale == 1:
                    expected_state = np.concatenate(
                        [v.numpy().reshape(-1).astype(np.float64) for v in after]
                    )
                else:
                    gradient_key = f"{step:02d}_{'full_gradient' if label == 'full_negative' else 'minibatch_gradient'}"
                    raw = (
                        -reconstructed[gradient_key]
                        if label != "recorded_adam"
                        else adam_delta
                    )
                    norm = float(np.linalg.norm(adam_delta))
                    factor = (
                        1.0
                        if label == "recorded_adam"
                        else norm / float(np.linalg.norm(raw))
                    )
                    expected_parts = []
                    cursor = 0
                    for value in before:
                        size = value.numel()
                        proposal = (
                            (raw[cursor : cursor + size] * factor * scale)
                            .astype(np.float32)
                            .reshape(value.shape)
                        )
                        expected_parts.append(
                            (value + torch.from_numpy(proposal.copy()))
                            .numpy()
                            .reshape(-1)
                            .astype(np.float64)
                        )
                        cursor += size
                    expected_state = np.concatenate(expected_parts)
                if not np.allclose(state, expected_state, rtol=2e-6, atol=2e-8):
                    raise ValueError(f"direction_endpoint_mismatch:{prefix}")
                with torch.no_grad():
                    cursor = 0
                    for parameter in params:
                        size = parameter.numel()
                        parameter.copy_(
                            torch.from_numpy(
                                state[cursor : cursor + size]
                                .astype(np.float32)
                                .reshape(parameter.shape)
                            )
                        )
                        cursor += size
                model.eval()
                actual_probs, actual_losses = _predict(
                    model,
                    tr["inputs"][held_rows],
                    ev["policy_targets"][held],
                    tr["legal_masks"][held_rows],
                    128,
                )
                if not np.allclose(
                    probabilities, actual_probs, rtol=2e-6, atol=2e-8
                ) or not np.allclose(losses, actual_losses, rtol=2e-6, atol=2e-8):
                    raise ValueError(f"endpoint_prediction_mismatch:{prefix}")
                training_rows = reconstructed["training_occurrence_rows"]
                _, actual_training_losses = _predict(
                    model,
                    tr["inputs"][training_rows],
                    tr["policy_targets"][training_rows],
                    tr["legal_masks"][training_rows],
                    512,
                )
                if not np.allclose(
                    train_losses, actual_training_losses, rtol=2e-6, atol=2e-8
                ):
                    raise ValueError(f"training_endpoint_losses:{prefix}")
                training_weights = reconstructed["training_coefficients"].astype(
                    np.float64
                )
                training_weights /= training_weights.sum(dtype=np.float64)
                with torch.no_grad():
                    for parameter, value in zip(params, before, strict=True):
                        parameter.copy_(value)
                _, pre_training_losses = _predict(
                    model,
                    tr["inputs"][training_rows],
                    tr["policy_targets"][training_rows],
                    tr["legal_masks"][training_rows],
                    512,
                )
                actual_training_change = float(
                    np.dot(
                        actual_training_losses - pre_training_losses,
                        training_weights,
                    )
                )
                actual_training_first = float(
                    np.dot(reconstructed[f"{step:02d}_full_gradient"], state - origin)
                )
                if not np.isclose(
                    trial["training"]["change"],
                    actual_training_change,
                    rtol=2e-6,
                    atol=2e-8,
                ) or not np.isclose(
                    trial["training"]["first_order"],
                    actual_training_first,
                    rtol=2e-6,
                    atol=2e-8,
                ):
                    raise ValueError(f"training_accounting:{prefix}")
                for weighting in ("exposure", "equal_input"):
                    fields = trial["weightings"][weighting]
                    with torch.no_grad():
                        for parameter, value in zip(params, before, strict=True):
                            parameter.copy_(value)
                    _, pre_losses = _predict(
                        model,
                        tr["inputs"][held_rows],
                        ev["policy_targets"][held],
                        tr["legal_masks"][held_rows],
                        128,
                    )
                    weights = unseen_weights[weighting]
                    actual_change = float(np.dot(actual_losses - pre_losses, weights))
                    displacement = state - origin
                    actual_first = float(
                        np.dot(
                            reconstructed[f"{step:02d}_unseen_{weighting}_gradient"],
                            displacement,
                        )
                    )
                    if not np.isclose(
                        fields["change"], actual_change, rtol=2e-6, atol=2e-8
                    ) or not np.isclose(
                        fields["first_order"], actual_first, rtol=2e-6, atol=2e-8
                    ):
                        raise ValueError(
                            f"endpoint_accounting_mismatch:{prefix}:{weighting}"
                        )
                    if not np.isclose(
                        fields["change"],
                        fields["post_ce"] - fields["pre_ce"],
                        rtol=1e-12,
                        atol=1e-15,
                    ):
                        raise ValueError(f"loss_accounting:{prefix}:{weighting}")
                    if not np.isclose(
                        fields["remainder"],
                        fields["change"] - fields["first_order"],
                        rtol=1e-12,
                        atol=1e-15,
                    ):
                        raise ValueError(f"remainder:{prefix}:{weighting}")
                if not np.isclose(
                    trial["training"]["remainder"],
                    trial["training"]["change"] - trial["training"]["first_order"],
                    rtol=1e-12,
                    atol=1e-15,
                ):
                    raise ValueError(f"training_remainder:{prefix}")
        expected_curves = _independent_curves(report["steps"])
        if report.get("curves") != expected_curves:
            raise ValueError("curve_or_count_mismatch")
    if not reconstructed:
        raise ValueError("independent_direction_reconstruction_empty")


def _independent_curves(steps):
    curves = {}
    for direction in ("recorded_adam", "full_negative", "minibatch_negative"):
        for scale in (1.0, 0.5, 0.25, 0.125):
            for weighting in ("exposure", "equal_input"):
                values = [
                    t["weightings"][weighting]["change"]
                    for step in steps
                    for t in step["trials"]
                    if t["direction"] == direction and t["scale"] == scale
                ]
                curves[f"{direction}/{scale}/{weighting}"] = {
                    "improvement": sum(v < -1e-6 for v in values),
                    "regression": sum(v > 1e-6 for v in values),
                    "ambiguous": sum(abs(v) <= 1e-6 for v in values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                }
    return curves


def _predict(model, inputs, targets, legal, batch):
    import torch
    import torch.nn.functional as F

    probs, losses = [], []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch):
            sl = slice(start, start + batch)
            logits, _ = model(torch.from_numpy(inputs[sl].copy()))
            mask = torch.from_numpy(legal[sl].copy())
            logp = F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)
            target = torch.from_numpy(targets[sl].copy())
            probs.append(logp.exp().cpu().numpy().astype(np.float32))
            losses.append((-(target * logp).sum(1)).cpu().numpy().astype(np.float64))
    return np.concatenate(probs), np.concatenate(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    verify(args.root)
    print("seed478 independent verification passed")


if __name__ == "__main__":
    main()
