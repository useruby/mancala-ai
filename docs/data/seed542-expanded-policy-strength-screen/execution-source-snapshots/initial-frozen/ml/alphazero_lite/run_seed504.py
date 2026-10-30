"""Fixed four-epoch supervised diagnostic on the seed500 archived corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed500-selfplay-512"
OUT = ROOT / "docs/data/seed504"
INITIALIZER = ROOT / ".tmp/seed500-current-checkpoint.npz"
EXPECTED_INITIALIZER = (
    "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(shards: range) -> list[dict[str, Any]]:
    rows = []
    for shard in shards:
        path = CORPUS / f"shard-{shard}.jsonl"
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            row = json.loads(line)
            row["_source"] = path.name
            row["_line"] = line_no
            rows.append(row)
    return rows


def keys(row: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    x = np.asarray(row["state"], dtype=np.float32)
    # The first 15 production features encode the complete canonical board;
    # all 27 float32 features are the actual model input identity.
    return tuple(map(float, x[:15])), tuple(map(float, x))


def stones(row: dict[str, Any]) -> int:
    x = np.asarray(row["state"], dtype=np.float32)
    return int(round(float(x[:12].sum(dtype=np.float32)) * 48.0))


def bucket(n: int) -> str:
    return ">32" if n > 32 else "17-32" if n >= 17 else "<=16"


def split() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tr, dev0 = rows_for(range(503000, 503006)), rows_for(range(503006, 503008))
    canonical = {keys(r)[0] for r in tr}
    inputs = {keys(r)[1] for r in tr}
    dev = [r for r in dev0 if keys(r)[0] not in canonical and keys(r)[1] not in inputs]
    assert not ({keys(r)[0] for r in tr} & {keys(r)[0] for r in dev})
    assert not ({keys(r)[1] for r in tr} & {keys(r)[1] for r in dev})
    membership = {
        "training": [{"shard": r["_source"], "line": r["_line"]} for r in tr],
        "development_original": [
            {"shard": r["_source"], "line": r["_line"]} for r in dev0
        ],
        "development_filtered": [
            {"shard": r["_source"], "line": r["_line"]} for r in dev
        ],
    }
    counts = {}
    for name, group in (
        ("training", tr),
        ("development_original", dev0),
        ("development_filtered", dev),
    ):
        counts[name] = {
            "rows": len(group),
            "canonical_unique": len({keys(r)[0] for r in group}),
            "float32_unique": len({keys(r)[1] for r in group}),
            "active_stones": dict(Counter(bucket(stones(r)) for r in group)),
        }
    counts["excluded_development_rows"] = len(dev0) - len(dev)
    counts["remaining_overlap_canonical"] = len(canonical & {keys(r)[0] for r in dev})
    counts["remaining_overlap_float32"] = len(inputs & {keys(r)[1] for r in dev})
    return (
        tr,
        dev,
        {
            "counts": counts,
            "membership": membership,
            "membership_sha256": hashlib.sha256(
                json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )


def load_model(path: Path, device: torch.device) -> train.PolicyValueNet:
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, path)
    return model.to(device).eval()


def evaluate(
    model: train.PolicyValueNet, rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = np.asarray([r["state"] for r in rows], dtype=np.float32)
    p = np.asarray([r["policy"] for r in rows], dtype=np.float32)
    v = np.asarray([r["value"] for r in rows], dtype=np.float32)
    legal = train.legal_mask_matrix_for_encoded_states(x).astype(bool)
    with torch.no_grad():
        logits, pred_v = model(torch.from_numpy(x).to(next(model.parameters()).device))
        logits = logits.cpu().numpy()
        pred_v = pred_v.cpu().numpy().reshape(-1)
    logits[~legal] = -1e9
    log_probs = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
    ce = -(p * log_probs).sum(axis=1)
    mse = (pred_v - v) ** 2
    exact: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        exact[keys(row)[1]].append(i)
    groups = {
        "pooled": np.arange(len(rows)),
        **{
            b: np.asarray([i for i, r in enumerate(rows) if bucket(stones(r)) == b])
            for b in (">32", "17-32", "<=16")
        },
    }
    metrics = {}
    for group, ix in groups.items():
        if not len(ix):
            metrics[group] = {"rows": 0, "exposure": None, "equal_exact_input": None}
            continue
        identities = defaultdict(list)
        for i in ix:
            identities[keys(rows[int(i)])[1]].append(int(i))
        metrics[group] = {
            "rows": int(len(ix)),
            "exact_inputs": len(identities),
            "exposure": {
                "policy_ce": float(ce[ix].mean()),
                "actual_value_mse": float(mse[ix].mean()),
            },
            "equal_exact_input": {
                "policy_ce": float(
                    np.mean([ce[ids].mean() for ids in identities.values()])
                ),
                "actual_value_mse": float(
                    np.mean([mse[ids].mean() for ids in identities.values()])
                ),
            },
        }
    predictions = [
        {
            "source": r["_source"],
            "line": r["_line"],
            "input_sha256": hashlib.sha256(
                np.asarray(r["state"], dtype=np.float32).tobytes()
            ).hexdigest(),
            "policy_logits": logits[i].tolist(),
            "policy_probabilities": np.exp(log_probs[i]).tolist(),
            "value_prediction": float(pred_v[i]),
            "target_policy": p[i].tolist(),
            "target_value": float(v[i]),
        }
        for i, r in enumerate(rows)
    ]
    return metrics, predictions


def run(out: Path = OUT, *, do_train: bool = True) -> None:
    if sha(INITIALIZER) != EXPECTED_INITIALIZER:
        raise RuntimeError(f"initializer SHA mismatch: {sha(INITIALIZER)}")
    tr, dev, evidence = split()
    if evidence["counts"]["development_filtered"]["active_stones"].get(">32", 0) == 0:
        raise RuntimeError("filtered >32-stone development cohort is empty")
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    train_paths = [CORPUS / f"shard-{i}.jsonl" for i in range(503000, 503006)]
    dev_paths = [CORPUS / f"shard-{i}.jsonl" for i in range(503006, 503008)]
    x, p, v, replay, weights = train.load_jsonl_replay(
        train_paths,
        weights=[1] * 6,
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    if len(x) != len(tr):
        raise RuntimeError(f"loader target/multiplicity mismatch {len(x)} != {len(tr)}")
    generator = torch.Generator(device="cpu").manual_seed(504)
    expected_permutations = [
        torch.randperm(len(replay), generator=generator).tolist() for _ in range(4)
    ]
    (out / "permutations.json").write_text(
        json.dumps(expected_permutations, separators=(",", ":")) + "\n"
    )
    manifest = {
        "status": "prospective frozen before training",
        "protocol": {
            "epochs": 4,
            "batch_size": 512,
            "optimizer": "Adam",
            "lr": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0,
            "objective": "legal policy CE + 0.5 * Huber(value, delta=1)",
            "exact_root_policy_coefficient": 1,
            "gradient_clip_norm": 1,
            "seed": 504,
            "lr_scheduler": "none",
            "architecture": "residual_v3/kalah_v3",
        },
        "initializer": {
            "path": str(INITIALIZER.relative_to(ROOT)),
            "sha256": sha(INITIALIZER),
        },
        "sources": {
            "training": [
                {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
                for p in train_paths
            ],
            "development": [
                {"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in dev_paths
            ],
        },
        "split_sha256": sha(out / "split.json"),
        "permutations_sha256": sha(out / "permutations.json"),
        "runner_sha256": sha(Path(__file__)),
        "train_source_sha256": sha(Path(train.__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    device = torch.device("cpu")
    init = load_model(INITIALIZER, device)
    metrics, predictions = evaluate(init, dev)
    results = {
        "initializer": {
            "sha256": sha(INITIALIZER),
            "metrics": metrics,
            "predictions": predictions,
        }
    }
    if do_train:
        model = load_model(INITIALIZER, device)
        train.set_seed(504)
        perms: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []

        # Production train() uses a seeded torch generator for epoch orders and Adam defaults.
        def record_permutation(epoch: int | None, order: list[int]) -> None:
            perms.append({"epoch": epoch, "order": order})

        train.train(
            model,
            x,
            p,
            v,
            replay,
            policy_loss_weights=weights,
            epochs=4,
            batch_size=512,
            lr=0.0001,
            device=device,
            value_loss_weight=0.5,
            value_loss="huber",
            huber_delta=1.0,
            val_split=0.0,
            grad_clip=1.0,
            save_top_k=1,
            lr_scheduler="none",
            weight_decay=0.0,
            save_epochs={1, 2, 3, 4},
            save_epochs_dir=out,
            save_epochs_base="seed504",
            epoch_history=history,
            permutation_callback=record_permutation,
            primary_order_seed=504,
            final_checkpoint="final",
        )
        observed = [item["order"] for item in perms]
        if observed != expected_permutations:
            raise RuntimeError(
                "production epoch permutations differ from prospective manifest"
            )
        (out / "epochs.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n"
        )
        for epoch in range(1, 5):
            cp = out / f"seed504_epoch{epoch}.npz"
            loaded = load_model(cp, device)
            m, pred = evaluate(loaded, dev)
            results[f"E{epoch}"] = {
                "sha256": sha(cp),
                "metrics": m,
                "predictions": pred,
            }
    (out / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.out, do_train=not args.prepare_only)


if __name__ == "__main__":
    main()
