"""Retrospective seed451 source-specific full-parameter gradient diagnostic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.verify_seed434_census_source import reconstruct_sources

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed451-source-policy-alignment")
CHECKPOINTS = {
    "initializer": (
        "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
    ),
    "T_final": (
        "docs/data/seed447-joint-output-cap/T-final.npz",
        "bcf3e76b12896cd4017aac5a8eda0b938088c9bee216b59d07cfde05c095f026",
    ),
}
SOURCES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
CHUNK = 512
VALUE_WEIGHT = 0.3
TOL = {"absolute": 2e-5, "relative": 2e-5}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inputs(root: Path) -> tuple[np.ndarray, ...]:
    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    scratch_root = root / ".tmp"
    scratch_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="seed451-lane-a-", dir=scratch_root
    ) as scratch:
        paths = []
        for source in SOURCES:
            source_path = (
                root
                / "docs/data/seed426-canonical-overlap/sources"
                / f"{source}.jsonl.gz"
            )
            target = Path(scratch) / f"{source}.jsonl"
            with gzip.open(source_path, "rb") as inp, target.open("wb") as out:
                while block := inp.read(1024 * 1024):
                    out.write(block)
            paths.append(target)
        x, p, v, replay, coeff = train.load_jsonl_replay(
            paths,
            [int(r["weight"]) for r in registration["replays"]],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[
                r["value_target_mode"] for r in registration["replays"]
            ],
            include_policy_loss_weights=True,
        )
    source_rows = list(reconstruct_sources(root, registration))
    split_path = root / registration["training"]["source_row_split"]["path"]
    with gzip.open(split_path, "rt", encoding="utf-8") as stream:
        split = json.load(stream)
    train_pos = np.asarray(split["train_positions"], dtype=np.int64)
    member = np.asarray([r["source"] for r in source_rows], dtype="U32")
    return x, p, v, replay, coeff, train_pos, member


def _gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    ids: np.ndarray,
    row_weight: np.ndarray,
    objective: str,
) -> np.ndarray:
    params = tuple(q for q in model.parameters() if q.requires_grad)
    total = np.zeros(sum(q.numel() for q in params), dtype=np.float64)
    for start in range(0, len(ids), CHUNK):
        selected = ids[start : start + CHUNK]
        weights = torch.from_numpy(row_weight[start : start + CHUNK].astype(np.float32))
        xb = torch.from_numpy(x[selected])
        logits, prediction = model(xb)
        if objective == "policy":
            mask = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[selected])
            )
            losses = train.compute_policy_cross_entropy(
                logits.masked_fill(mask <= 0, -1e9), torch.from_numpy(p[selected])
            )
        else:
            losses = train.compute_value_loss_vector(
                prediction,
                torch.from_numpy(v[selected]),
                value_loss="huber",
                huber_delta=1.0,
            )
        grads = torch.autograd.grad((losses * weights).sum(), params, allow_unused=True)
        total += np.concatenate(
            [
                np.zeros(q.numel(), dtype=np.float64)
                if g is None
                else g.detach().cpu().numpy().astype(np.float64).ravel()
                for q, g in zip(params, grads, strict=True)
            ]
        )
    return total


def _norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def _cos(a: np.ndarray, b: np.ndarray) -> float | None:
    den = _norm(a) * _norm(b)
    return None if den == 0 else float(np.dot(a, b) / den)


def _equal_identity_weights(identities: list[str]) -> np.ndarray:
    counts = {identity: identities.count(identity) for identity in set(identities)}
    return np.asarray(
        [1.0 / (len(counts) * counts[identity]) for identity in identities],
        dtype=np.float64,
    )


def classify_fresh_alignment(metrics: list[dict[str, Any]]) -> str:
    opposition = all(
        item["G_PH"]["cosine"] is not None
        and item["G_PH"]["cosine"] <= -0.10
        and item["G_PF"]["cosine"] is not None
        and item["G_PF"]["cosine"] >= 0.10
        for item in metrics
    )
    if opposition:
        return "historical_policy_opposition_present"
    if all(
        item["G_T"]["cosine"] is not None and item["G_T"]["cosine"] >= 0.10
        for item in metrics
    ):
        return "fresh_first_order_alignment_positive"
    return "mixed_or_checkpoint_dependent_alignment"


def _freeze(root: Path) -> None:
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    registration_path = out / "registration.json"
    if registration_path.exists():
        raise ValueError("registration_is_immutable")
    layout_manifest = json.loads(
        (
            root / "docs/data/seed438-seed437-gradient-correction/protocol.json"
        ).read_text()
    )
    source_paths = [
        "ml/alphazero_lite/seed451_source_policy_alignment.py",
        "ml/alphazero_lite/verify_seed451_source_policy_alignment.py",
        "ml/alphazero_lite/test_seed451_source_policy_alignment.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/verify_seed434_census_source.py",
    ]
    input_paths = (
        [
            "docs/data/seed416-policy-target-softening/registration-v3.json",
            "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
            "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
            "docs/data/seed442-kl-capped-adam-screen/registration.json",
            "docs/data/seed447-joint-output-cap/registration.json",
            "docs/data/seed449-policy-gain-localization/registration.json",
        ]
        + [value[0] for value in CHECKPOINTS.values()]
        + [
            f"docs/data/seed426-canonical-overlap/sources/{name}.jsonl.gz"
            for name in SOURCES
        ]
    )
    registration = {
        "schema": "seed451-registration-v1",
        "status": "frozen_before_gradient_computation",
        "analysis_design": "retrospective; seed447-450 outcomes are known",
        "checkpoints": {
            name: {"path": path, "sha256": expected}
            for name, (path, expected) in CHECKPOINTS.items()
        },
        "inputs_sha256": {path: sha(root / path) for path in input_paths},
        "source_sha256": {path: sha(root / path) for path in source_paths},
        "historical_seed438_parameter_layout": layout_manifest["bound_inputs"]["model"][
            "flat_parameter_layout"
        ],
        "sources": SOURCES,
        "population": "strict unseen >32; U_F fresh compact rows; U_H historical compact rows; preserve duplicate exposures and targets",
        "objective": "P_F and P_H normalized by respective policy coefficient mass; V mean Huber(delta=1); G_P=rho_F*G_PF+rho_H*G_PH; G_T=G_P+0.3*G_V",
        "value_coefficient_applied_once": True,
        "chunk_size": CHUNK,
        "tolerance": TOL,
        "classification": {
            "historical_opposition": "cos(U_F_equal,G_PH)<=-0.10 and cos(U_F_equal,G_PF)>=0.10 at both checkpoints",
            "joint_positive": "otherwise cos(U_F_equal,G_T)>=0.10 at both",
            "otherwise": "mixed_or_checkpoint_dependent_alignment",
        },
    }
    registration_path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    _freeze(root)
    x, p, v, replay, coeff, train_pos, row_source = _inputs(root)
    train_ids = replay[train_pos]
    sources = row_source[train_ids]
    expanded_coeff = coeff[train_ids].astype(np.float64)
    training_source_accounting = {
        source: {
            "training_exposures": int(np.count_nonzero(sources == source)),
            "policy_coefficient_mass": float(expanded_coeff[sources == source].sum()),
            "unique_compact_rows": int(len(np.unique(train_ids[sources == source]))),
        }
        for source in SOURCES
    }
    fresh = sources == "fresh"
    historical = ~fresh
    rho_f = float(expanded_coeff[fresh].sum() / expanded_coeff.sum())
    rho_h = float(expanded_coeff[historical].sum() / expanded_coeff.sum())
    membership_path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        unseen = []
        for line in stream:
            row = json.loads(line)
            if row["subset"] == "unseen" and row["active_stones"] > 32:
                unseen.append(row)
    # Above membership records are compact-row ordered; source attribution is from the
    # independently reconstructed source ledger, not replay multiplicity.
    hmask = np.asarray(
        [row_source[int(r["compact_row"])] != "fresh" for r in unseen], dtype=bool
    )
    fmask = ~hmask
    unseen_rows = np.asarray([int(r["compact_row"]) for r in unseen], dtype=np.int64)
    identities = [r["input_identity"] for r in unseen]
    groups = {"shared_trunk": [], "policy_head": [], "value_head": []}
    records: dict[str, Any] = {}
    vectors: dict[str, np.ndarray] = {}
    layout = None
    for label, (relative, expected) in CHECKPOINTS.items():
        checkpoint = root / relative
        if sha(checkpoint) != expected:
            raise ValueError(f"checkpoint_hash_mismatch:{label}")
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(model, checkpoint)
        model.eval()
        named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
        if layout is None:
            cursor = 0
            for name, param in named:
                group = (
                    "shared_trunk"
                    if name.startswith(("input_layer", "residual_layers"))
                    else "policy_head"
                    if name.startswith(("policy_hidden_layer", "policy_head"))
                    else "value_head"
                )
                groups[group].extend(range(cursor, cursor + param.numel()))
                layout = layout or []
                layout.append(
                    {
                        "name": name,
                        "shape": list(param.shape),
                        "start": cursor,
                        "stop": cursor + param.numel(),
                        "group": group,
                    }
                )
                cursor += param.numel()
        mass = expanded_coeff.sum()
        pf_weights = expanded_coeff * fresh / expanded_coeff[fresh].sum()
        ph_weights = expanded_coeff * historical / expanded_coeff[historical].sum()
        policy_w = expanded_coeff / mass
        value_w = np.ones(len(train_ids), dtype=np.float64) / len(train_ids)
        gpf = _gradient(model, x, p, v, train_ids, pf_weights, "policy")
        gph = _gradient(model, x, p, v, train_ids, ph_weights, "policy")
        gv = _gradient(model, x, p, v, train_ids, value_w, "value")
        gp = rho_f * gpf + rho_h * gph
        gt = gp + VALUE_WEIGHT * gv
        directp = _gradient(model, x, p, v, train_ids, policy_w, "policy")
        direct = directp + VALUE_WEIGHT * gv
        if not np.allclose(gp, directp, atol=TOL["absolute"], rtol=TOL["relative"]):
            raise ValueError("policy_mixture_direct_reconciliation_failed")
        records[label] = {
            "rho_F": rho_f,
            "rho_H": rho_h,
            "policy_mass": float(mass),
            "policy_denominator": float(mass),
            "value_denominator": int(len(train_ids)),
            "train_exposures": int(len(train_ids)),
            "fresh_train_exposures": int(fresh.sum()),
            "historical_train_exposures": int(historical.sum()),
            "direct_policy_difference_norm": _norm(gp - directp),
            "direct_T_difference_norm": _norm(gt - direct),
            "cohorts": {},
        }
        vectors.update(
            {
                f"{label}__{key}": val
                for key, val in (
                    ("G_PF", gpf),
                    ("G_PH", gph),
                    ("G_V", gv),
                    ("G_P", gp),
                    ("G_T", gt),
                )
            }
        )
        for cohort, mask in (("U_F", fmask), ("U_H", hmask)):
            ids = unseen_rows[mask]
            ids_identity = np.asarray(identities, dtype=object)[mask]
            for weighting in ("exposure", "equal_exact_input"):
                if weighting == "exposure":
                    weights = np.ones(len(ids), dtype=np.float64) / len(ids)
                else:
                    weights = _equal_identity_weights(ids_identity.tolist())
                gu = _gradient(model, x, p, v, ids, weights, "policy")
                key = f"{cohort}_{weighting}"
                vectors[f"{label}__{key}"] = gu
                contributions = {
                    "fresh": rho_f * float(np.dot(gu, gpf)),
                    "historical": rho_h * float(np.dot(gu, gph)),
                    "value": VALUE_WEIGHT * float(np.dot(gu, gv)),
                }
                group_accounting = {}
                for group, indexes in groups.items():
                    idx = np.asarray(indexes, dtype=np.int64)
                    group_accounting[group] = {
                        "fresh": rho_f * float(np.dot(gu[idx], gpf[idx])),
                        "historical": rho_h * float(np.dot(gu[idx], gph[idx])),
                        "value": VALUE_WEIGHT * float(np.dot(gu[idx], gv[idx])),
                        "joint": float(np.dot(gu[idx], gt[idx])),
                    }
                metrics = {
                    name: {
                        "norm": _norm(g),
                        "cosine": _cos(gu, g),
                        "dot": float(np.dot(gu, g)),
                        "unit_direction_loss_derivative": -float(np.dot(gu, g))
                        / _norm(g)
                        if _norm(g)
                        else None,
                    }
                    for name, g in (
                        ("G_PF", gpf),
                        ("G_PH", gph),
                        ("G_V", gv),
                        ("G_T", gt),
                    )
                }
                records[label]["cohorts"][key] = {
                    "exposures": int(len(ids)),
                    "identities": int(len(set(ids_identity))),
                    "metrics": metrics,
                    "signed_contributions": contributions,
                    "joint_dot": float(np.dot(gu, gt)),
                    "contribution_sum": sum(contributions.values()),
                    "group_accounting": group_accounting,
                }
    shared = len(
        set(identities[i] for i in range(len(identities)) if fmask[i])
        & set(identities[i] for i in range(len(identities)) if hmask[i])
    )
    # Requested fixed classifier: equal-input U_F only.
    alignment = [
        records[c]["cohorts"]["U_F_equal_exact_input"]["metrics"] for c in CHECKPOINTS
    ]
    classification = classify_fresh_alignment(alignment)
    opposition = classification == "historical_policy_opposition_present"
    positive = all(
        m["G_T"]["cosine"] is not None and m["G_T"]["cosine"] >= 0.10 for m in alignment
    )
    checkpoint_alignment = {
        name: {
            "historical_policy_opposition": metrics["G_PH"]["cosine"] is not None
            and metrics["G_PH"]["cosine"] <= -0.10
            and metrics["G_PF"]["cosine"] is not None
            and metrics["G_PF"]["cosine"] >= 0.10,
            "joint_first_order_alignment_positive": metrics["G_T"]["cosine"] is not None
            and metrics["G_T"]["cosine"] >= 0.10,
            "equal_input_U_F_cosines": {
                name: metrics[name]["cosine"] for name in ("G_PF", "G_PH", "G_T")
            },
        }
        for name, metrics in zip(CHECKPOINTS, alignment, strict=True)
    }
    out = root / OUT
    np.savez_compressed(out / "gradient-vectors.npz", **vectors)
    result = {
        "schema": "seed451-results-v1",
        "classification": classification,
        "historical_opposition_present": opposition,
        "joint_first_order_alignment_positive": positive,
        "checkpoint_alignment": checkpoint_alignment,
        "source_population": {
            "U_F_exposures": int(fmask.sum()),
            "U_F_identities": len(set(np.asarray(identities, dtype=object)[fmask])),
            "U_H_exposures": int(hmask.sum()),
            "U_H_identities": len(set(np.asarray(identities, dtype=object)[hmask])),
            "shared_identities": shared,
        },
        "training_source_accounting": training_source_accounting,
        "replay_multiplicity_by_source": {
            source: int(np.count_nonzero(row_source == source)) for source in SOURCES
        },
        "parameter_layout": layout,
        "parameter_groups": {
            k: {
                "indices": v,
                "parameter_count": len(v),
                "norms_by_checkpoint": {
                    c: {
                        cohort: _norm(vectors[f"{c}__{cohort}"][v])
                        for cohort in ("U_F_equal_exact_input", "U_H_equal_exact_input")
                    }
                    for c in CHECKPOINTS
                },
            }
            for k, v in groups.items()
        },
        "checkpoints": records,
        "limitations": [
            "Retrospective diagnostic; seed447–450 outcomes were known.",
            "First-order fixed-checkpoint alignment does not predict finite-step behavior or establish causality.",
            "No training, optimizer, search, games, target changes, replay-weight experiment, export, or promotion.",
        ],
    }
    (out / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (out / "results.md").write_text(
        "# Seed451 source-specific policy-gradient alignment\n\nRetrospective frozen-checkpoint diagnostic; seed447–450 outcomes were known.\n\n**Classification:** `"
        + classification
        + "`\n\nHistorical policy opposition: **"
        + str(opposition).lower()
        + "** across both checkpoints. Per-checkpoint U_F equal-input opposition/alignment: `"
        + json.dumps(checkpoint_alignment, sort_keys=True)
        + "`. Joint-objective positive first-order alignment at both checkpoints: **"
        + str(positive).lower()
        + "**.\n\nThis descriptive first-order diagnostic neither explains finite-step seed447 behavior causally nor authorizes replay-weight changes or follow-on training.\n"
    )
    published = (
        "registration.json",
        "gradient-vectors.npz",
        "results.json",
        "results.md",
    )
    (out / "receipt.json").write_text(
        json.dumps(
            {
                "schema": "seed451-publication-receipt-v1",
                "files_sha256": {name: sha(out / name) for name in published},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(run(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
