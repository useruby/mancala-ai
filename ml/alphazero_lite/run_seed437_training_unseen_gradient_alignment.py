"""Frozen full-parameter training/unseen gradient alignment diagnostic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.checkpoint_phase_selection import (
    active_pit_stones_from_encoded_state,
)
from ml.alphazero_lite.seed437_gradient_alignment import (
    equal_input_weights,
    geometry,
    normalized_exposure_weights,
    sum_vectors,
)
from ml.alphazero_lite.seed427_validation_subsets import construct, read_evidence
from ml.alphazero_lite.verify_seed434_census_source import (
    _positions,
    reconstruct_sources,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed437-training-unseen-gradient-alignment"
DATA = ROOT / "docs/data/seed426-canonical-overlap"
REGISTRATION = ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
INIT = (
    ROOT
    / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
ADAM = ROOT / "docs/data/seed435-adam-direction-screen/A-final.npz"
CHECKPOINTS = {
    "initializer": (
        INIT,
        "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
    ),
    "adam_a16": (
        ADAM,
        "afb164603b34d0449f463f3c8e475f9bb4408e7a704c031a6f000572e6c4e9c7",
    ),
}
BUCKETS = ("<=16", "17-21", "22-32", ">32")
SOURCES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
CHUNK_SIZE = 512
ATOL = 2e-5
RTOL = 2e-5
CLASSIFICATION = "persistent_training_objective_opposition"
NO_OPPOSITION = "no_persistent_training_objective_opposition"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def manifest_inputs() -> dict[str, Any]:
    reg = json.loads(REGISTRATION.read_text())
    rows = list(reconstruct_sources(ROOT, reg))
    if len(rows) != 87625:
        raise ValueError("compact_row_count_mismatch")
    split_spec = reg["training"]["source_row_split"]
    split_path = ROOT / split_spec["path"]
    source_counts = [sum(row["source"] == name for row in rows) for name in SOURCES]
    mult = [int(spec["weight"]) for spec in reg["replays"]]
    expanded_count = sum(n * m for n, m in zip(source_counts, mult, strict=True))
    if expanded_count != 149448:
        raise ValueError("replay_exposure_count_mismatch")
    _membership, train_n, val_n = _positions(
        ROOT, reg, len(rows), list(zip(source_counts, mult, strict=True))
    )
    if (train_n, val_n) != (134502, 14946):
        raise ValueError("split_exposure_counts_mismatch")
    membership_path = DATA / "seed427-validation-membership.jsonl.gz"
    membership_digest = (
        "554f3bd016a0f920ef253538d741a91cb27c9882d2dba94cf7509030af8ca385"
    )
    if sha(membership_path) != membership_digest:
        raise ValueError("seed427_membership_hash_mismatch")
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        membership = [json.loads(line) for line in stream]
    accounting_path = DATA / "row-accounting.jsonl.gz"
    accounting = read_evidence(accounting_path)
    rebuilt_membership, _rebuilt_counts = construct(accounting)
    if membership != rebuilt_membership:
        raise ValueError("seed427_membership_reconstruction_mismatch")
    selected = [
        row
        for row in membership
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    if (
        len(selected) != 2607
        or len({row["input_identity"] for row in selected}) != 1242
    ):
        raise ValueError("unseen_gt32_population_mismatch")
    checkpoint_bindings = {}
    for name, (checkpoint, expected) in CHECKPOINTS.items():
        if sha(checkpoint) != expected:
            raise ValueError(f"checkpoint_hash_mismatch:{name}")
        checkpoint_bindings[name] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": expected,
        }
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    layout = []
    offset = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
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
        layout.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "start": offset,
                "stop": offset + parameter.numel(),
                "group": group,
            }
        )
        offset += parameter.numel()
    return {
        "schema": "seed437-frozen-manifest-v1",
        "status": "frozen_before_gradient_execution",
        "interpretation": "retrospective observational diagnostic only; not causal or playing-strength evidence; authorizes no intervention",
        "checkpoints": checkpoint_bindings,
        "registration": {
            "path": str(REGISTRATION.relative_to(ROOT)),
            "sha256": sha(REGISTRATION),
        },
        "sources": [
            {
                "name": spec["name"],
                "weight": spec["weight"],
                "decompressed_sha256": spec["sha256"],
                "snapshot": f"docs/data/seed426-canonical-overlap/sources/{spec['name']}.jsonl.gz",
                "snapshot_sha256": sha(DATA / "sources" / f"{spec['name']}.jsonl.gz"),
            }
            for spec in reg["replays"]
        ],
        "split": {
            "path": split_spec["path"],
            "sha256": sha(split_path),
            "train_positions_sha256": split_spec["train_positions_sha256"],
            "validation_positions_sha256": split_spec["validation_positions_sha256"],
        },
        "membership": {
            "path": str(membership_path.relative_to(ROOT)),
            "sha256": membership_digest,
            "selected_exposures": len(selected),
            "exact_input_identities": 1242,
        },
        "counts": {
            "compact_rows": len(rows),
            "replay_exposures": expanded_count,
            "training_exposures": train_n,
            "validation_exposures": val_n,
        },
        "execution_sources": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                ROOT / "ml/alphazero_lite/train.py",
                ROOT / "ml/alphazero_lite/verify_seed434_census_source.py",
                ROOT / "ml/alphazero_lite/seed437_gradient_alignment.py",
                ROOT
                / "ml/alphazero_lite/verify_seed437_training_unseen_gradient_alignment.py",
                ROOT / "ml/alphazero_lite/test_seed437_gradient_alignment.py",
                Path(__file__).resolve(),
                ROOT / "ml/alphazero_lite/checkpoint_phase_selection.py",
                ROOT / "ml/alphazero_lite/seed427_validation_subsets.py",
                ROOT / "ml/alphazero_lite/exact_root_policy_targets.py",
                ROOT / "ml/alphazero_lite/seed429_policy_normalization.py",
                ROOT / "ml/alphazero_lite/seed432_policy_target_census.py",
                ROOT / "ml/alphazero_lite/seed433_census_correction.py",
            )
        },
        "reconstruction_evidence": {
            "row_accounting_path": str(accounting_path.relative_to(ROOT)),
            "row_accounting_sha256": sha(accounting_path),
            "membership_reconstructed_from_row_accounting": True,
        },
        "model": {
            "type": "residual_v3",
            "hidden_sizes": [96, 3],
            "input_features": 27,
            "parameter_groups": {
                "shared_trunk": ["input_layer.*", "residual_layers.*"],
                "policy_head": ["policy_hidden_layer.*", "policy_head.*"],
                "value_head": ["value_hidden_layer.*", "value_head.*"],
            },
            "flat_parameter_layout": layout,
            "parameter_count": offset,
        },
        "objectives": {
            "P": "sum(train exposure * row policy coefficient * legal-masked CE)/sum(the same weights)",
            "V": "sum(train exposure * smooth_l1(beta=1) value loss)/sum(train exposures)",
            "T": "P + 0.3*V",
            "U_exposure": "mean policy CE over unseen >32 validation exposures",
            "U_equal": "mean per exact-input identity of its within-identity exposure-mean policy CE",
        },
        "denominators": {
            "policy": "sum(compact row policy coefficient * training replay multiplicity)",
            "value": "134502 training exposures",
            "U_exposure": "2607 exposures",
            "U_equal": "1242 identities, inner denominator is each identity exposure count",
        },
        "execution": {
            "chunk_size": CHUNK_SIZE,
            "deterministic_order": "ascending compact-row index; selected repeated validation exposure records remain repeated",
            "accumulation": "float64 flat gradient sum of float32 autograd chunk gradients",
            "device": "CPU",
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
                "sys_executable": sys.executable,
            },
            "optimizer_steps": 0,
            "tolerance": {"absolute": ATOL, "relative": RTOL},
        },
        "decision_rule": "persistent_training_objective_opposition iff cosine(T,U) <= -0.05 for both U weightings at both checkpoints; otherwise no_persistent_training_objective_opposition",
    }


def prepare_arrays() -> tuple[Any, ...]:
    reg = json.loads(REGISTRATION.read_text())
    specs = reg["replays"]
    names = [spec["name"] for spec in specs]
    counts = [
        sum(1 for _ in gzip.open(DATA / "sources" / f"{name}.jsonl.gz", "rt"))
        for name in names
    ]
    with tempfile.TemporaryDirectory(prefix="seed437-", dir=ROOT / ".tmp") as scratch:
        paths = []
        for name in names:
            src = DATA / "sources" / f"{name}.jsonl.gz"
            target = Path(scratch) / f"{name}.jsonl"
            with gzip.open(src, "rb") as reader, target.open("wb") as writer:
                writer.write(reader.read())
            paths.append(target)
        x, p, v, replay, coeff = train.load_jsonl_replay(
            paths,
            [int(spec["weight"]) for spec in specs],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[spec["value_target_mode"] for spec in specs],
            include_policy_loss_weights=True,
        )
    with gzip.open(ROOT / reg["training"]["source_row_split"]["path"], "rt") as stream:
        split = json.load(stream)
    train_pos = np.asarray(split["train_positions"], dtype=np.int64)
    multiplicity = np.zeros(len(x), dtype=np.int64)
    np.add.at(multiplicity, replay[train_pos], 1)
    source_idx = np.repeat(np.arange(5), counts)
    stones = np.asarray([active_pit_stones_from_encoded_state(row) for row in x])
    bucket_idx = np.select(
        [stones <= 16, stones <= 21, stones <= 32], [0, 1, 2], default=3
    )
    with gzip.open(DATA / "seed427-validation-membership.jsonl.gz", "rt") as stream:
        members = [json.loads(line) for line in stream]
    selected = [
        r for r in members if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    compact = np.asarray([int(r["compact_row"]) for r in selected], dtype=np.int64)
    identities = [r["input_identity"] for r in selected]
    uweights = {
        "exposure": normalized_exposure_weights(len(selected)),
        "equal_input": equal_input_weights(identities),
    }
    return x, p, v, coeff, multiplicity, source_idx, bucket_idx, compact, uweights


def parameter_group_indexes(
    model: torch.nn.Module, params: tuple[torch.nn.Parameter, ...]
) -> dict[str, np.ndarray]:
    groups = {"shared_trunk": [], "policy_head": [], "value_head": []}
    offset = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = (
            "shared_trunk"
            if name.startswith(("input_layer.", "residual_layers."))
            else "policy_head"
            if name.startswith(("policy_hidden_layer.", "policy_head."))
            else "value_head"
            if name.startswith(("value_hidden_layer.", "value_head."))
            else None
        )
        if key is None:
            raise ValueError(f"unclassified_parameter:{name}")
        groups[key].extend(range(offset, offset + parameter.numel()))
        offset += parameter.numel()
    if offset != sum(p.numel() for p in params):
        raise ValueError("parameter_accounting_mismatch")
    return {key: np.asarray(indexes, dtype=np.int64) for key, indexes in groups.items()}


def flat_grad(
    model: torch.nn.Module,
    ids: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    objective: str,
    chunk: int,
) -> np.ndarray:
    params = tuple(param for param in model.parameters() if param.requires_grad)
    size = sum(param.numel() for param in params)
    total = np.zeros(size, dtype=np.float64)
    for start in range(0, len(ids), chunk):
        sel, row_weights = ids[start : start + chunk], weights[start : start + chunk]
        xb, pb, vb = (torch.from_numpy(a[sel]) for a in (x, p, v))
        mask = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[sel]))
        logits, prediction = model(xb)
        if objective == "value":
            losses = train.compute_value_loss_vector(
                prediction, vb, value_loss="huber", huber_delta=1.0
            )
            losses = losses * 0.3
        else:
            losses = train.compute_policy_cross_entropy(
                logits.masked_fill(mask <= 0, -1e9), pb
            )
        loss = (losses * torch.from_numpy(row_weights).to(losses.dtype)).sum()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        total += np.concatenate(
            [
                np.zeros(param.numel(), dtype=np.float64)
                if grad is None
                else grad.detach().cpu().numpy().astype(np.float64).ravel()
                for param, grad in zip(params, grads, strict=True)
            ]
        )
    return total


def accumulate(
    model: torch.nn.Module,
    ids: np.ndarray,
    perrow: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    objective: str,
    chunk: int,
) -> np.ndarray:
    return flat_grad(
        model, ids, perrow, arrays[0], arrays[1], arrays[2], objective, chunk
    )


def run_checkpoint(
    label: str, path: Path, expected: str, arrays: tuple[Any, ...], chunk: int
) -> dict[str, Any]:
    if sha(path) != expected:
        raise ValueError(f"checkpoint_hash_mismatch:{label}")
    x, p, v, coeff, mult, source_idx, bucket_idx, uids, uweights = arrays
    model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(model, path)
    model.eval()
    names = tuple(
        name
        for name, _ in model.named_parameters()
        if dict(model.named_parameters())[name].requires_grad
    )
    params = tuple(param for param in model.parameters() if param.requires_grad)
    group_indexes = parameter_group_indexes(model, params)
    size = sum(param.numel() for param in params)
    if len(names) != len(params) or sum(len(q) for q in group_indexes.values()) != size:
        raise ValueError("parameter_layout_accounting_failed")
    tr_ids = np.flatnonzero(mult)
    den_p = float(np.dot(coeff[tr_ids], mult[tr_ids]))
    den_v = float(mult.sum())
    source_names, bucket_names = SOURCES, BUCKETS
    result: dict[str, Any] = {
        "checkpoint_sha256": sha(path),
        "parameter_count": size,
        "parameter_names": list(names),
        "vectors": {},
        "decomposition": {},
    }
    for objective in ("P", "0.3V"):
        loss_kind = "policy" if objective == "P" else "value"
        base_weights = coeff * mult / den_p if objective == "P" else mult * 0.3 / den_v
        allvec = accumulate(
            model, tr_ids, base_weights[tr_ids], arrays, loss_kind, chunk
        )
        result["vectors"][objective] = allvec
        for dimension, labels, codes in (
            ("source", source_names, source_idx),
            ("bucket", bucket_names, bucket_idx),
        ):
            pieces = {}
            for i, part in enumerate(labels):
                ids = tr_ids[codes[tr_ids] == i]
                pieces[part] = accumulate(
                    model, ids, base_weights[ids], arrays, loss_kind, chunk
                )
            rebuilt = sum_vectors(pieces, size)
            if not np.allclose(rebuilt, allvec, atol=ATOL, rtol=RTOL):
                raise ValueError(
                    f"{dimension}_gradient_decomposition_failed:{label}:{objective}"
                )
            result["decomposition"].setdefault(objective, {})[dimension] = pieces
    result["vectors"]["T"] = result["vectors"]["P"] + result["vectors"]["0.3V"]
    result["decomposition"]["T"] = {
        axis: {
            key: result["decomposition"]["P"][axis][key]
            + result["decomposition"]["0.3V"][axis][key]
            for key in result["decomposition"]["P"][axis]
        }
        for axis in ("source", "bucket")
    }
    for weighting, uweight in uweights.items():
        uvec = accumulate(model, uids, uweight, arrays, "policy", chunk)
        result["vectors"][f"U_{weighting}"] = uvec
        bybucket = {}
        for i, bname in enumerate(bucket_names):
            chosen = uids[bucket_idx[uids] == i]
            w = uweight[bucket_idx[uids] == i]
            bybucket[bname] = accumulate(model, chosen, w, arrays, "policy", chunk)
        if not np.allclose(sum_vectors(bybucket, size), uvec, atol=ATOL, rtol=RTOL):
            raise ValueError(
                f"validation_bucket_decomposition_failed:{label}:{weighting}"
            )
        result["decomposition"].setdefault(f"U_{weighting}", {})["bucket"] = bybucket
    metric: dict[str, Any] = {}
    contributions: dict[str, Any] = {}
    for weighting in uweights:
        u = result["vectors"][f"U_{weighting}"]
        metric[weighting] = {}
        for objective in ("P", "0.3V", "T"):
            g = result["vectors"][objective]
            metric[weighting][objective] = geometry(g, u)
            contributions[f"{weighting}/{objective}"] = {
                "source": {
                    s: float(np.dot(u, z))
                    for s, z in result["decomposition"][objective]["source"].items()
                },
                "bucket": {
                    b: float(np.dot(u, z))
                    for b, z in result["decomposition"][objective]["bucket"].items()
                },
                "parameter_group": {
                    k: float(np.dot(u[ix], g[ix])) for k, ix in group_indexes.items()
                },
            }
            for axis, parts in contributions[f"{weighting}/{objective}"].items():
                if not np.isclose(
                    sum(parts.values()),
                    metric[weighting][objective]["dot"],
                    atol=ATOL,
                    rtol=RTOL,
                ):
                    raise ValueError(
                        f"dot_contribution_reconciliation_failed:{label}:{weighting}:{objective}:{axis}"
                    )
    result["metrics"] = metric
    result["dot_contributions"] = contributions
    result["parameter_group_sizes"] = {k: len(ix) for k, ix in group_indexes.items()}
    result["vector_evidence"] = {}
    vector_arrays: dict[str, np.ndarray] = {}
    for name, vector in result["vectors"].items():
        archive_key = f"{label}__{name}"
        vector_arrays[archive_key] = vector
        result["vector_evidence"][name] = {
            "sha256": hashlib.sha256(vector.astype("<f8").tobytes()).hexdigest(),
            "archive_key": archive_key,
        }
    for objective, partitions in result["decomposition"].items():
        for axis, pieces in partitions.items():
            for part, vector in pieces.items():
                key = f"{objective}__{axis}__{part}"
                archive_key = f"{label}__{key}"
                vector_arrays[archive_key] = vector
                result["vector_evidence"][key] = {
                    "sha256": hashlib.sha256(
                        vector.astype("<f8").tobytes()
                    ).hexdigest(),
                    "archive_key": archive_key,
                }
    result["parameter_group_indices"] = {
        key: indexes.tolist() for key, indexes in group_indexes.items()
    }
    result["_vector_arrays"] = vector_arrays
    result["decomposition"] = {
        objective: {axis: list(parts) for axis, parts in axes.items()}
        for objective, axes in result["decomposition"].items()
    }
    del result["vectors"]
    if sha(path) != expected:
        raise ValueError(f"checkpoint_mutated:{label}")
    return result


def execute(root: Path = ROOT, chunk: int = CHUNK_SIZE) -> dict[str, Any]:
    global ROOT, OUT, DATA, REGISTRATION, INIT, ADAM, CHECKPOINTS
    ROOT = root.resolve()
    OUT = ROOT / "docs/data/seed437-training-unseen-gradient-alignment"
    DATA = ROOT / "docs/data/seed426-canonical-overlap"
    REGISTRATION = (
        ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    INIT = (
        ROOT
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    ADAM = ROOT / "docs/data/seed435-adam-direction-screen/A-final.npz"
    CHECKPOINTS = {
        "initializer": (INIT, CHECKPOINTS["initializer"][1]),
        "adam_a16": (ADAM, CHECKPOINTS["adam_a16"][1]),
    }
    inputs = manifest_inputs()
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("manifest_not_frozen_run_register_first")
    frozen = json.loads(manifest_path.read_text())
    if frozen != inputs:
        raise ValueError("frozen_manifest_mismatch")
    before = {name: sha(path) for name, (path, _) in CHECKPOINTS.items()}
    arrays = prepare_arrays()
    checkpoints = {
        name: run_checkpoint(name, path, expected, arrays, chunk)
        for name, (path, expected) in CHECKPOINTS.items()
    }
    vector_arrays = {
        key: vector
        for checkpoint in checkpoints.values()
        for key, vector in checkpoint.pop("_vector_arrays").items()
    }
    vector_path = OUT / "gradient-vectors.npz"
    np.savez_compressed(vector_path, **vector_arrays)
    cosines = [
        checkpoints[cp]["metrics"][w]["T"]["cosine"]
        for cp in CHECKPOINTS
        for w in ("exposure", "equal_input")
    ]
    classification = (
        CLASSIFICATION
        if all(c is not None and c <= -0.05 for c in cosines)
        else NO_OPPOSITION
    )
    after = {name: sha(path) for name, (path, _) in CHECKPOINTS.items()}
    if before != after:
        raise ValueError("checkpoint_immutability_failed")
    results = {
        "schema": "seed437-results-v1",
        "manifest_sha256": sha(manifest_path),
        "classification": classification,
        "strength_claim": False,
        "checkpoint_immutability": {
            "before": before,
            "after": after,
            "unchanged": True,
        },
        "vector_archive": {
            "path": vector_path.name,
            "sha256": sha(vector_path),
            "arrays": len(vector_arrays),
            "encoding": "compressed NPZ; float64 arrays bound by per-vector digest",
        },
        "checkpoints": checkpoints,
    }
    out_path = OUT / "results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def register(root: Path = ROOT) -> dict[str, Any]:
    global ROOT, OUT, DATA, REGISTRATION, INIT, ADAM, CHECKPOINTS
    ROOT = root.resolve()
    OUT = ROOT / "docs/data/seed437-training-unseen-gradient-alignment"
    DATA = ROOT / "docs/data/seed426-canonical-overlap"
    REGISTRATION = (
        ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    INIT = (
        ROOT
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    ADAM = ROOT / "docs/data/seed435-adam-direction-screen/A-final.npz"
    CHECKPOINTS = {
        "initializer": (INIT, CHECKPOINTS["initializer"][1]),
        "adam_a16": (ADAM, CHECKPOINTS["adam_a16"][1]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "manifest.json"
    if target.exists():
        raise ValueError("manifest_is_immutable")
    manifest = manifest_inputs()
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    result = (
        register(args.root)
        if args.command == "register"
        else execute(args.root, args.chunk_size)
    )
    print(
        json.dumps(
            {key: result[key] for key in result if key != "checkpoints"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
