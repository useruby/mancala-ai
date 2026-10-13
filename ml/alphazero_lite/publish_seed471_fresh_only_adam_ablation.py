"""Publish seed471 endpoint predictions, metrics, effects, and decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)
from ml.alphazero_lite.seed471_fresh_only_adam_ablation import INIT, OUT, sha


def publish(root: Path) -> dict[str, Any]:
    root = root.resolve()
    torch.set_num_threads(1)
    out = root / OUT
    json.loads((out / "registration.json").read_text())
    if not all((out / f"{arm}-steps.npz").is_file() for arm in ("C", "T")):
        raise ValueError("both_arms_required")
    arrays = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = arrays["training"], arrays["evaluation"]
    x = tr["inputs"][ev["compact_rows"]]
    # The authenticated evaluation array contains its own occurrence targets and provenance.
    ids = ev["input_identity"].tolist()
    fresh = ev["source_ids"] == 0
    pooled = np.ones(len(ids), dtype=bool)
    if int(fresh.sum()) != 859 or len(ids) != 2607:
        raise ValueError("evaluation_cohort_mismatch")
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    target_p, target_v = ev["policy_targets"], ev["value_targets"].reshape(-1)
    compact = ev["compact_rows"]
    masks = tr["legal_masks"][compact]
    cohorts = {"fresh": np.flatnonzero(fresh), "pooled": np.flatnonzero(pooled)}
    report: dict[str, Any] = {
        "schema": "seed471-endpoint-evidence-v1",
        "arms": {},
        "effects": {},
        "gates": {},
    }
    for arm in ("initializer", "C", "T"):
        if arm == "initializer":
            train.load_checkpoint_into_model(model, root / INIT)
        else:
            with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as z:
                params = list(model.parameters())
                for i, param in enumerate(params):
                    param.data.copy_(torch.from_numpy(z[f"15_post_{i:02d}"].copy()))
        model.eval()
        with torch.inference_mode():
            logits, values = model(torch.from_numpy(x.copy()))
            logp = (
                F.log_softmax(
                    logits.masked_fill(torch.from_numpy(masks.copy()) == 0, -torch.inf),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            val = values.reshape(-1).cpu().numpy()
        arm_result: dict[str, Any] = {}
        for cohort, select in cohorts.items():
            occurrence_ce = (
                -(
                    target_p[select]
                    * np.where(np.isfinite(logp[select]), logp[select], 0)
                )
                .sum(axis=1)
                .astype(np.float64)
            )
            occurrence_mse = np.square(
                val[select].astype(np.float64) - target_v[select].astype(np.float64)
            )
            group_ids = [ids[i] for i in select]
            groups: dict[str, list[int]] = {}
            for index, identity in enumerate(group_ids):
                groups.setdefault(identity, []).append(index)
            arm_result[cohort] = {
                "occurrences": len(select),
                "identities": len(groups),
                "exposure_weighted": {
                    "policy_ce": float(occurrence_ce.mean()),
                    "value_mse": float(occurrence_mse.mean()),
                },
                "equal_input": {
                    metric: float(np.mean([np.mean(arr[ix]) for ix in groups.values()]))
                    for metric, arr in (
                        ("policy_ce", occurrence_ce),
                        ("value_mse", occurrence_mse),
                    )
                },
            }
            np.savez_compressed(
                out / f"{arm}-{cohort}-predictions.npz",
                prediction=logp[select].astype(np.float32),
                value=val[select],
                policy_target=target_p[select],
                value_target=target_v[select],
                input_identity=np.asarray(group_ids, dtype="U256"),
                legal_mask=masks[select],
                exposure_weight=np.full(
                    len(select), 1.0 / len(select), dtype=np.float64
                ),
                equal_input_weight=np.asarray(
                    [
                        1.0 / (len(groups) * len(groups[identity]))
                        for identity in group_ids
                    ],
                    dtype=np.float64,
                ),
            )
        # Training-occurrence objective over the full train occurrence vector.
        train_ids = tr["expanded_replay_indices"][tr["train_positions"]]
        losses, vpred = [], []
        with torch.inference_mode():
            for start in range(0, len(train_ids), 1024):
                rows = train_ids[start : start + 1024]
                lg, vp = model(torch.from_numpy(tr["inputs"][rows].copy()))
                legal = torch.from_numpy(tr["legal_masks"][rows].copy())
                ce = -(
                    torch.from_numpy(tr["policy_targets"][rows].copy())
                    * F.log_softmax(lg.masked_fill(legal == 0, -1e9), dim=1)
                ).sum(dim=1)
                losses.extend(ce.cpu().numpy().tolist())
                vpred.extend(vp.reshape(-1).cpu().numpy().tolist())
        fresh_train = tr["fresh_source_mask"][train_ids].astype(np.float64)
        coeff = tr["policy_coefficients"][train_ids].astype(np.float64)
        err = np.abs(
            np.asarray(vpred, dtype=np.float64)
            - tr["value_targets"][train_ids].reshape(-1).astype(np.float64)
        )
        huber = np.where(err < 1.0, 0.5 * err**2, err - 0.5)
        ce_all = np.asarray(losses, dtype=np.float64)
        total = float(
            np.sum(ce_all * coeff * fresh_train) / np.sum(coeff * fresh_train)
            + 0.3 * np.sum(huber * fresh_train) / len(train_ids)
        )
        arm_result["training_J"] = {
            "total": total,
            "policy_numerator": float(np.sum(ce_all * coeff * fresh_train)),
            "policy_denominator": float(np.sum(coeff * fresh_train)),
            "value_numerator": float(np.sum(huber * fresh_train)),
            "value_denominator": len(train_ids),
            "population_occurrences": len(train_ids),
        }
        report["arms"][arm] = arm_result
    for cohort in ("fresh", "pooled"):
        report["effects"][cohort] = {}
        for weight in ("exposure_weighted", "equal_input"):
            report["effects"][cohort][weight] = {
                arm: {
                    metric: report["arms"][arm][cohort][weight][metric]
                    - report["arms"]["initializer"][cohort][weight][metric]
                    for metric in ("policy_ce", "value_mse")
                }
                for arm in ("C", "T")
            }
            report["effects"][cohort][weight]["T_minus_C"] = {
                metric: report["arms"]["T"][cohort][weight][metric]
                - report["arms"]["C"][cohort][weight][metric]
                for metric in ("policy_ce", "value_mse")
            }
    for weight in ("exposure_weighted", "equal_input"):
        t, c, init = (report["arms"][arm] for arm in ("T", "C", "initializer"))
        report["gates"][f"fresh_ce_vs_initializer_{weight}"] = (
            t["fresh"][weight]["policy_ce"]
            <= init["fresh"][weight]["policy_ce"] - 0.001
        )
        report["gates"][f"fresh_ce_vs_C_{weight}"] = (
            t["fresh"][weight]["policy_ce"] <= c["fresh"][weight]["policy_ce"] - 0.001
        )
        report["gates"][f"pooled_ce_vs_initializer_{weight}"] = (
            t["pooled"][weight]["policy_ce"]
            <= init["pooled"][weight]["policy_ce"] + 0.005
        )
        report["gates"][f"pooled_mse_vs_initializer_{weight}"] = (
            t["pooled"][weight]["value_mse"]
            <= init["pooled"][weight]["value_mse"] + 0.01
        )
    report["gates"]["training_J_decreases"] = (
        report["arms"]["T"]["training_J"]["total"]
        < report["arms"]["initializer"]["training_J"]["total"]
    )
    report["decision"] = (
        "advance_to_fresh_confirmation"
        if all(report["gates"].values())
        else "close_small_step_fresh_only_branch"
    )
    report["decision_scope"] = (
        "authorizes only a separately designed fresh confirmation; not arena execution or promotion"
    )
    report["classification"] = report["decision"]
    (out / "endpoint-evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Seed471 fresh-only Adam learning-rate ablation",
        "",
        "Exploratory analysis informed by previously inspected evaluation results. The authenticated evaluation occurrences are not a new holdout or independent confirmation.",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "Gate outcomes (reported separately):",
        "",
    ]
    lines.extend(
        f"- `{name}`: **{'PASS' if passed else 'FAIL'}**"
        for name, passed in report["gates"].items()
    )
    lines.extend(
        [
            "",
            "| Arm | Cohort | Weighting | Policy CE | Value MSE |",
            "|---|---|---|---:|---:|",
        ]
    )
    for arm in ("initializer", "C", "T"):
        for cohort in ("fresh", "pooled"):
            for weighting in ("exposure_weighted", "equal_input"):
                metric = report["arms"][arm][cohort][weighting]
                lines.append(
                    f"| {arm} | {cohort} | {weighting} | {metric['policy_ce']:.9f} | {metric['value_mse']:.9f} |"
                )
    lines.extend(
        [
            "",
            "No arena execution, promotion, runtime export, or strength claim is authorized. A passing decision authorizes only a separately designed fresh confirmation.",
        ]
    )
    (out / "results.md").write_text("\n".join(lines) + "\n")
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "receipt.json")
    receipt = {p.name: sha(p) for p in files}
    receipt["status"] = "post_execution_bound_receipt"
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    publish(parser.parse_args().root)


if __name__ == "__main__":
    main()
