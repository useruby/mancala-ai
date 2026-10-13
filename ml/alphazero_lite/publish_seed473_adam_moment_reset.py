"""Publish seed473 occurrence predictions, metrics, effects, and decision."""

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
from ml.alphazero_lite.seed471_fresh_only_adam_ablation import INIT
from ml.alphazero_lite.seed473_adam_moment_reset import OUT, sha


def publish(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    gate = json.loads((out / "validity-gate.json").read_text())
    if gate.get("passed") is not True:
        raise ValueError("C_reproduction_gate_failed")
    if not all((out / f"{arm}-steps.npz").is_file() for arm in ("C", "R")):
        raise ValueError("both_arms_required")
    torch.set_num_threads(1)
    arrays = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = arrays["training"], arrays["evaluation"]
    x = tr["inputs"][ev["compact_rows"]]
    masks = tr["legal_masks"][ev["compact_rows"]]
    identities = ev["input_identity"].tolist()
    fresh = np.flatnonzero(ev["source_ids"] == 0)
    pooled = np.arange(len(identities))
    target_p = ev["policy_targets"]
    target_v = ev["value_targets"].reshape(-1)
    groups_by_cohort = {}
    for name, ix in (("fresh", fresh), ("pooled", pooled)):
        g: dict[str, list[int]] = {}
        for local, row in enumerate(ix):
            g.setdefault(identities[int(row)], []).append(local)
        groups_by_cohort[name] = (ix, g)
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    report: dict[str, Any] = {
        "schema": "seed473-endpoint-evidence-v1",
        "arms": {},
        "effects": {},
        "gates": {},
    }
    endpoint_models = {}
    for arm in ("initializer", "C", "R"):
        if arm == "initializer":
            train.load_checkpoint_into_model(model, root / INIT)
        else:
            with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as z:
                for i, param in enumerate(model.parameters()):
                    param.data.copy_(torch.from_numpy(z[f"15_post_{i:02d}"].copy()))
        endpoint_models[arm] = model
        model.eval()
        with torch.inference_mode():
            logits, value = model(torch.from_numpy(x.copy()))
            logp = (
                F.log_softmax(
                    logits.masked_fill(torch.from_numpy(masks.copy()) == 0, -torch.inf),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            value = value.reshape(-1).cpu().numpy()
        result: dict[str, Any] = {}
        for cohort, (ix, groups) in groups_by_cohort.items():
            ce = (
                -(target_p[ix] * np.where(np.isfinite(logp[ix]), logp[ix], 0))
                .sum(axis=1)
                .astype(np.float64)
            )
            mse = np.square(
                value[ix].astype(np.float64) - target_v[ix].astype(np.float64)
            )
            result[cohort] = {
                "occurrences": len(ix),
                "identities": len(groups),
                "exposure_weighted": {
                    "policy_ce": float(ce.mean()),
                    "value_mse": float(mse.mean()),
                },
                "equal_input": {
                    "policy_ce": float(
                        np.mean([ce[g].mean() for g in groups.values()])
                    ),
                    "value_mse": float(
                        np.mean([mse[g].mean() for g in groups.values()])
                    ),
                },
            }
            local_ids = [identities[int(i)] for i in ix]
            np.savez_compressed(
                out / f"{arm}-{cohort}-predictions.npz",
                prediction=logp[ix].astype(np.float32),
                value=value[ix],
                policy_target=target_p[ix],
                value_target=target_v[ix],
                input_identity=np.asarray(local_ids, dtype="U256"),
                legal_mask=masks[ix],
                exposure_weight=np.full(len(ix), 1 / len(ix), dtype=np.float64),
                equal_input_weight=np.asarray(
                    [1 / (len(groups) * len(groups[k])) for k in local_ids],
                    dtype=np.float64,
                ),
            )
        rows = tr["expanded_replay_indices"][tr["train_positions"]]
        ce_parts = []
        pred_parts = []
        for start in range(0, len(rows), 1024):
            rr = rows[start : start + 1024]
            with torch.inference_mode():
                lg, pv = model(torch.from_numpy(tr["inputs"][rr].copy()))
                legal = torch.from_numpy(tr["legal_masks"][rr].copy())
                ce_parts.append(
                    (
                        -(
                            torch.from_numpy(tr["policy_targets"][rr].copy())
                            * F.log_softmax(lg.masked_fill(legal == 0, -1e9), dim=1)
                        ).sum(1)
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                pred_parts.append(pv.reshape(-1).cpu().numpy().astype(np.float64))
        ce_all = np.concatenate(ce_parts)
        pred = np.concatenate(pred_parts)
        fr = tr["fresh_source_mask"][rows].astype(np.float64)
        co = tr["policy_coefficients"][rows].astype(np.float64)
        err = np.abs(pred - tr["value_targets"][rows].reshape(-1).astype(np.float64))
        huber = np.where(err < 1, 0.5 * err**2, err - 0.5)
        pden = float(np.sum(co * fr))
        pnum = float(np.sum(ce_all * co * fr))
        vnum = float(np.sum(huber * fr))
        result["training_J"] = {
            "total": pnum / pden + 0.3 * vnum / len(rows),
            "policy_numerator": pnum,
            "policy_denominator": pden,
            "value_numerator": vnum,
            "value_denominator": len(rows),
            "population_occurrences": len(rows),
        }
        report["arms"][arm] = result
    for cohort in ("fresh", "pooled"):
        report["effects"][cohort] = {}
        for weighting in ("exposure_weighted", "equal_input"):
            report["effects"][cohort][weighting] = {
                arm: {
                    m: report["arms"][arm][cohort][weighting][m]
                    - report["arms"]["initializer"][cohort][weighting][m]
                    for m in ("policy_ce", "value_mse")
                }
                for arm in ("C", "R")
            }
            report["effects"][cohort][weighting]["R_minus_C"] = {
                m: report["arms"]["R"][cohort][weighting][m]
                - report["arms"]["C"][cohort][weighting][m]
                for m in ("policy_ce", "value_mse")
            }
    for weighting in ("exposure_weighted", "equal_input"):
        r, c, init = (report["arms"][a] for a in ("R", "C", "initializer"))
        report["gates"][f"fresh_ce_vs_initializer_{weighting}"] = (
            r["fresh"][weighting]["policy_ce"]
            <= init["fresh"][weighting]["policy_ce"] - 0.001
        )
        report["gates"][f"fresh_ce_vs_C_{weighting}"] = (
            r["fresh"][weighting]["policy_ce"]
            <= c["fresh"][weighting]["policy_ce"] - 0.001
        )
        report["gates"][f"pooled_ce_vs_initializer_{weighting}"] = (
            r["pooled"][weighting]["policy_ce"]
            <= init["pooled"][weighting]["policy_ce"] + 0.005
        )
        report["gates"][f"pooled_mse_vs_initializer_{weighting}"] = (
            r["pooled"][weighting]["value_mse"]
            <= init["pooled"][weighting]["value_mse"] + 0.01
        )
    report["gates"]["training_J_decreases"] = (
        report["arms"]["R"]["training_J"]["total"]
        < report["arms"]["initializer"]["training_J"]["total"]
    )
    report["gates"]["exact_C_reproduction"] = True
    report["decision"] = (
        "advance_to_fresh_confirmation"
        if all(report["gates"].values())
        else "close_moment_reset_branch"
    )
    report["classification"] = report["decision"]
    report["decision_scope"] = (
        "exploratory reused cohort; passing authorizes only separately designed fresh confirmation"
    )
    (out / "endpoint-evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Seed473 Adam moment-reset ablation",
        "",
        "Exploratory analysis using a reused evaluation cohort; this is not independent confirmation.",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "Gates:",
    ]
    lines += [
        f"- `{k}`: **{'PASS' if v else 'FAIL'}**" for k, v in report["gates"].items()
    ]
    lines += [
        "",
        "No searches, games, runtime export, promotion, or strength claim is authorized.",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    receipt = {
        p.name: sha(p)
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "receipt.json"
    }
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
