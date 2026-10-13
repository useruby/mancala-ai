"""Seed468 first-step optimizer-direction transport calculations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed468-first-step-optimizer-direction-transport"
BUNDLE = ROOT / "docs/data/seed466-replay-input-bundle"
INIT = (
    ROOT
    / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
MANIFEST_SHA = "db80f3bdfcc0a126032c1f5a16e78cda74b160d4fa2d80330cec8bed18085b34"
INIT_SHA = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(tensors: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [t.detach().cpu().numpy().reshape(-1) for t in tensors]
    ).astype(np.float32)


def weights(identities: list[str], equal: bool) -> np.ndarray:
    counts = {identity: identities.count(identity) for identity in set(identities)}
    if equal:
        return np.asarray(
            [1 / (len(counts) * counts[i]) for i in identities], dtype=np.float64
        )
    return np.full(len(identities), 1 / len(identities), dtype=np.float64)


def gradients(model: torch.nn.Module, objective: torch.Tensor) -> np.ndarray:
    params = list(model.parameters())
    got = torch.autograd.grad(objective, params, allow_unused=True)
    return flatten(
        [
            torch.zeros_like(p) if g is None else g
            for p, g in zip(params, got, strict=True)
        ]
    )


def adam_displacement(gradient: np.ndarray) -> tuple[np.ndarray, float, bool]:
    g = torch.from_numpy(np.asarray(gradient, dtype=np.float32).copy())
    norm = torch.linalg.vector_norm(g)
    clipped = bool(norm > 1.0)
    if clipped:
        g = g / norm
    # PyTorch Adam, step one, zero moments, coupled weight_decay=0.
    exp_avg = g * 0.1
    exp_avg_sq = g.square() * 0.001
    step_size = 0.001 / (1.0 - 0.9)
    delta = -step_size * exp_avg / (exp_avg_sq.sqrt() / (1.0 - 0.999) ** 0.5 + 1e-8)
    return delta.numpy().astype(np.float32), float(norm), clipped


def proposals(gradient: np.ndarray) -> dict[str, Any]:
    adam, norm, clipped = adam_displacement(gradient)
    plain = np.zeros_like(gradient, dtype=np.float32)
    gn = float(np.linalg.norm(gradient.astype(np.float64)))
    an = float(np.linalg.norm(adam.astype(np.float64)))
    if gn:
        plain = (-gradient * np.float32(an / gn)).astype(np.float32)
    return {
        "adam": adam,
        "negative_gradient": plain,
        "gradient_norm_preclip": norm,
        "adam_norm": an,
        "clipped": clipped,
    }


def geometry(gradient: np.ndarray, displacement: np.ndarray) -> dict[str, float | None]:
    g, d = gradient.astype(np.float64), displacement.astype(np.float64)
    dot = float(np.dot(g, d))
    gn, dn = float(np.linalg.norm(g)), float(np.linalg.norm(d))
    return {
        "dot": dot,
        "cosine": dot / (gn * dn) if gn and dn else None,
        "gradient_norm": gn,
        "displacement_norm": dn,
    }


def classify(records: list[dict[str, Any]]) -> str:
    def state(objective: str, direction: str, weighting: str) -> str:
        rows = [
            r
            for r in records
            if r["objective"] == objective
            and r["direction"] == direction
            and r["weighting"] == weighting
        ]
        cos = [r["cosine"] for r in rows]
        desc = sum(c is not None and c <= -1e-6 for c in cos)
        asc = sum(c is not None and c >= 1e-6 for c in cos)
        if desc >= 12:
            return "descent"
        if asc >= 12:
            return "ascent"
        return "mixed"

    def consistent(obj: str, direction: str, wanted: str) -> bool:
        return all(
            state(obj, direction, w) == wanted
            for w in ("exposure_weighted", "equal_input")
        )

    if consistent("P", "negative_gradient", "ascent"):
        return "minibatch_policy_transport_failure"
    if consistent("P", "negative_gradient", "descent") and consistent(
        "J", "negative_gradient", "ascent"
    ):
        return "fresh_value_transport_failure"
    for obj in ("P", "J"):
        if consistent(obj, "negative_gradient", "descent") and consistent(
            obj, "adam", "ascent"
        ):
            return "adam_direction_transport_failure"
    if all(
        consistent(obj, direction, "descent")
        for obj in ("P", "J")
        for direction in ("negative_gradient", "adam")
    ):
        return "first_order_descent_consistent"
    return "mixed_or_inconclusive_transport"


def calculate(root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    bundle = root / "docs/data/seed466-replay-input-bundle"
    init = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    if sha(bundle / "manifest.json") != MANIFEST_SHA or sha(init) != INIT_SHA:
        raise ValueError("frozen_input_identity_mismatch")
    freeze = json.loads(
        (
            root
            / "docs/data/seed468-first-step-optimizer-direction-transport/registration.json"
        ).read_text()
    )
    source_bindings = {
        freeze["inputs"]["diagnostic_source"]: freeze["inputs"][
            "diagnostic_source_sha256"
        ],
        freeze["inputs"]["verifier_source"]: freeze["inputs"]["verifier_source_sha256"],
        freeze["inputs"]["tests_source"]: freeze["inputs"]["tests_source_sha256"],
        freeze["inputs"]["authenticated_reader"]: freeze["inputs"][
            "reader_manifest_sha256"
        ],
        freeze["inputs"]["model_source"]: freeze["inputs"]["model_source_sha256"],
        "ml/alphazero_lite/seed466_replay_bundle.py": freeze["inputs"][
            "bundle_reader_sha256"
        ],
        "ml/alphazero_lite/verify_seed466_replay_bundle.py": freeze["inputs"][
            "independent_bundle_reconstruction_sha256"
        ],
        **freeze["project_source_closure_sha256"],
    }
    for relative, expected in source_bindings.items():
        if sha(root / relative) != expected:
            raise ValueError(f"frozen_source_identity_mismatch:{relative}")
    for filename, key in (
        ("training.npz", "training_npz_sha256"),
        ("evaluation.npz", "evaluation_npz_sha256"),
    ):
        if sha(bundle / filename) != freeze["frozen_bundle"][key]:
            raise ValueError(f"frozen_bundle_identity_mismatch:{filename}")
    frozen_evaluation = root / freeze["frozen_bundle"]["evaluation_manifest_path"]
    if sha(frozen_evaluation) != freeze["frozen_bundle"]["evaluation_manifest_sha256"]:
        raise ValueError("evaluation_manifest_identity_mismatch")
    from ml.alphazero_lite.verify_seed466_replay_bundle import verify as verify_bundle
    from ml.alphazero_lite.seed466_runtime_bundle import read_authenticated_bundle

    verify_bundle(bundle, root)
    read_authenticated_bundle(bundle, MANIFEST_SHA)
    with np.load(bundle / "training.npz", allow_pickle=False) as z:
        tr = {k: z[k] for k in z.files}
    if (
        hashlib.sha256(tr["first16_minibatch_positions"].tobytes()).hexdigest()
        != freeze["frozen_bundle"]["first16_minibatch_positions_sha256"]
    ):
        raise ValueError("minibatch_positions_identity_mismatch")
    with np.load(bundle / "evaluation.npz", allow_pickle=False) as z:
        ev = {k: z[k] for k in z.files}
    fresh_eval = ev["source_ids"] == 0
    ids = ev["input_identity"][fresh_eval].tolist()
    if len(ids) != 859 or len(set(ids)) != 851:
        raise ValueError("fresh_unseen_evaluation_population_mismatch")
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, init)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Fresh evaluation gradients at the fixed initializer, one independent graph per weighting.
    eval_x = (
        ev["inputs"][fresh_eval]
        if "inputs" in ev
        else tr["inputs"][ev["compact_rows"][fresh_eval]]
    )
    eval_targets = ev["policy_targets"][fresh_eval]
    eval_masks = tr["legal_masks"][ev["compact_rows"][fresh_eval]]
    eval_gradients = {}
    for weighting, w in (
        ("exposure_weighted", weights(ids, False)),
        ("equal_input", weights(ids, True)),
    ):
        x = torch.from_numpy(eval_x.copy())
        p = torch.from_numpy(eval_targets.copy())
        m = torch.from_numpy(eval_masks.copy())
        logits, _ = model(x)
        losses = -(p * F.log_softmax(logits.masked_fill(m == 0, -1e9), dim=1)).sum(
            dim=1
        )
        eval_gradients[weighting] = gradients(
            model, (losses * torch.from_numpy(w.astype(np.float32))).sum()
        )
    records, vectors = [], {}
    for weighting, vector in eval_gradients.items():
        vectors[f"unseen_gradient_{weighting}"] = vector
    for batch in range(16):
        pos = tr["first16_minibatch_positions"][batch * 512 : (batch + 1) * 512]
        compact = tr["expanded_replay_indices"][tr["train_positions"][pos]]
        fresh = tr["fresh_source_mask"][compact].astype(np.float32)
        c = tr["policy_coefficients"][compact] * fresh
        x = torch.from_numpy(tr["inputs"][compact].copy())
        p = torch.from_numpy(tr["policy_targets"][compact].copy())
        m = torch.from_numpy(tr["legal_masks"][compact].copy())
        v = torch.from_numpy(tr["value_targets"][compact].reshape(-1).copy())
        for objective in ("P", "J"):
            model.load_state_dict(original)
            logits, value = model(x)
            ce = -(p * F.log_softmax(logits.masked_fill(m == 0, -1e9), dim=1)).sum(
                dim=1
            )
            policy_mass = float(c.sum())
            if policy_mass <= 0:
                raise ValueError(f"empty_fresh_policy_mass:batch{batch}")
            policy = (ce * torch.from_numpy(c)).sum() / policy_mass
            loss = policy
            if objective == "J":
                huber = F.smooth_l1_loss(value.reshape(-1), v, reduction="none")
                loss = policy + 0.3 * (huber * torch.from_numpy(fresh)).sum() / 512
            grad = gradients(model, loss)
            prop = proposals(grad)
            vectors[f"batch{batch:02d}_{objective}_gradient"] = grad
            for direction in ("adam", "negative_gradient"):
                displacement = prop[direction].astype(np.float64)
                key = f"batch{batch:02d}_{objective}_{direction}"
                vectors[key] = prop[direction]
                for weighting, eg in eval_gradients.items():
                    result = geometry(eg, displacement)
                    records.append(
                        {
                            "batch": batch,
                            "objective": objective,
                            "direction": direction,
                            "weighting": weighting,
                            **result,
                            "fresh_count": int(fresh.sum()),
                            "policy_mass": float(c.sum()),
                            "batch_gradient_norm": float(
                                np.linalg.norm(grad.astype(np.float64))
                            ),
                            "clipped": prop["clipped"],
                        }
                    )
    if any(not torch.equal(model.state_dict()[k], v) for k, v in original.items()):
        raise ValueError("initializer_mutated")
    return records, vectors


def evaluation_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Bind the ordered occurrence cohort before any gradient execution."""
    bundle = root / "docs/data/seed466-replay-input-bundle"
    with np.load(bundle / "training.npz", allow_pickle=False) as z:
        training = {k: z[k] for k in z.files}
    with np.load(bundle / "evaluation.npz", allow_pickle=False) as z:
        evaluation = {k: z[k] for k in z.files}
    take = np.flatnonzero(evaluation["source_ids"] == 0)
    identities = evaluation["input_identity"][take].astype("U256")
    compact = evaluation["compact_rows"][take]
    legal = training["legal_masks"][compact]
    record = {
        "schema": "seed468-fresh-unseen-gt32-evaluation-v1",
        "membership_contract": "authenticated seed466 evaluation arrays are subset=unseen and active_stones>32",
        "selection_rule": "source_ids == 0; retain stored evaluation order and all occurrences",
        "selection_indices": take.astype(np.int64).tolist(),
        "identities": identities.tolist(),
        "source_ids": evaluation["source_ids"][take].astype(np.uint8).tolist(),
        "raw_lines": evaluation["raw_lines"][take].astype(np.int32).tolist(),
        "compact_rows": compact.astype(np.int32).tolist(),
        "policy_targets": evaluation["policy_targets"][take].tolist(),
        "legal_masks": legal.tolist(),
        "counts": {
            "occurrences": int(len(take)),
            "identities": int(len(set(identities.tolist()))),
        },
        "weighting": {
            "exposure_weighted": "1/859 per occurrence",
            "equal_input": "1/(851*k_i) for each occurrence of identity i",
        },
    }
    if record["counts"] != {"occurrences": 859, "identities": 851}:
        raise ValueError("fresh_unseen_evaluation_population_mismatch")
    return record


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--freeze-evaluation", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "docs/data/seed468-first-step-optimizer-direction-transport"
    out.mkdir(parents=True, exist_ok=True)
    if args.freeze_evaluation:
        from ml.alphazero_lite.verify_seed466_replay_bundle import (
            verify as verify_bundle,
        )

        verify_bundle(root / "docs/data/seed466-replay-input-bundle", root)
        manifest = evaluation_manifest(root)
        (out / "evaluation-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return
    records, vectors = calculate(root)
    np.savez_compressed(out / "vectors.npz", **vectors)
    counts = {}
    for objective in ("P", "J"):
        for direction in ("adam", "negative_gradient"):
            for weighting in ("exposure_weighted", "equal_input"):
                selected = [
                    r["cosine"]
                    for r in records
                    if r["objective"] == objective
                    and r["direction"] == direction
                    and r["weighting"] == weighting
                ]
                counts[f"{objective}/{direction}/{weighting}"] = {
                    "descent": sum(c is not None and c <= -1e-6 for c in selected),
                    "ascent": sum(c is not None and c >= 1e-6 for c in selected),
                    "ambiguous_or_zero_norm": sum(
                        c is None or -1e-6 < c < 1e-6 for c in selected
                    ),
                }
    report = {
        "schema": "seed468-first-step-optimizer-direction-transport-v1",
        "comparisons": records,
        "classification": classify(records),
        "classification_counts": counts,
        "interpretation": "first-order diagnostic only; not a causal explanation of a training trajectory or model-strength decision",
    }
    (out / "diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "registration_sha256": sha(out / "registration.json"),
        "evaluation_manifest_sha256": sha(out / "evaluation-manifest.json"),
        "diagnostic_sha256": sha(out / "diagnostic.json"),
        "vectors_sha256": sha(out / "vectors.npz"),
        "results_sha256": sha(out / "results.md"),
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
