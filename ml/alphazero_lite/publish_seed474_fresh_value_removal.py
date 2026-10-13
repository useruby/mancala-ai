"""Publish seed474 endpoint predictions, metrics, gates, and receipt."""

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
from ml.alphazero_lite.seed474_fresh_value_removal import OUT, sha


def publish(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    if json.loads((out / "validity-gate.json").read_text()).get("passed") is not True:
        raise ValueError("exact_C_reproduction_required")
    torch.set_num_threads(1)
    data = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = data["training"], data["evaluation"]
    x, masks = tr["inputs"][ev["compact_rows"]], tr["legal_masks"][ev["compact_rows"]]
    identities = ev["input_identity"].tolist()
    targets, value_targets = ev["policy_targets"], ev["value_targets"].reshape(-1)
    cohorts = {
        "fresh": np.flatnonzero(ev["source_ids"] == 0),
        "pooled": np.arange(len(identities)),
    }
    report: dict[str, Any] = {
        "schema": "seed474-endpoint-evidence-v1",
        "arms": {},
        "effects": {},
        "gates": {},
    }
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    for arm in ("initializer", "C", "P"):
        if arm == "initializer":
            train.load_checkpoint_into_model(model, root / INIT)
        else:
            with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as archive:
                for index, parameter in enumerate(model.parameters()):
                    parameter.data.copy_(
                        torch.from_numpy(archive[f"15_post_{index:02d}"].copy())
                    )
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
        arm_result: dict[str, Any] = {}
        for cohort, rows in cohorts.items():
            groups: dict[str, list[int]] = {}
            for local, row in enumerate(rows):
                groups.setdefault(identities[int(row)], []).append(local)
            ce = (
                -(targets[rows] * np.where(np.isfinite(logp[rows]), logp[rows], 0))
                .sum(1)
                .astype(np.float64)
            )
            mse = np.square(
                value[rows].astype(np.float64) - value_targets[rows].astype(np.float64)
            )
            arm_result[cohort] = {
                "occurrences": len(rows),
                "identities": len(groups),
                "exposure_weighted": {
                    "policy_ce": float(ce.mean()),
                    "value_mse": float(mse.mean()),
                },
                "equal_input": {
                    "policy_ce": float(
                        np.mean([ce[ix].mean() for ix in groups.values()])
                    ),
                    "value_mse": float(
                        np.mean([mse[ix].mean() for ix in groups.values()])
                    ),
                },
            }
            local_ids = [identities[int(i)] for i in rows]
            np.savez_compressed(
                out / f"{arm}-{cohort}-predictions.npz",
                prediction=logp[rows].astype(np.float32),
                value=value[rows],
                policy_target=targets[rows],
                value_target=value_targets[rows],
                input_identity=np.asarray(local_ids, dtype="U256"),
                legal_mask=masks[rows],
                exposure_weight=np.full(len(rows), 1 / len(rows), dtype=np.float64),
                equal_input_weight=np.asarray(
                    [1 / (len(groups) * len(groups[key])) for key in local_ids],
                    dtype=np.float64,
                ),
            )
        train_rows = tr["expanded_replay_indices"][tr["train_positions"]]
        ce_parts, value_parts = [], []
        for start in range(0, len(train_rows), 1024):
            part = train_rows[start : start + 1024]
            with torch.inference_mode():
                logits, pred = model(torch.from_numpy(tr["inputs"][part].copy()))
                legal = torch.from_numpy(tr["legal_masks"][part].copy())
                ce_parts.append(
                    (
                        -(
                            torch.from_numpy(tr["policy_targets"][part].copy())
                            * F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
                        ).sum(1)
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                value_parts.append(pred.reshape(-1).cpu().numpy().astype(np.float64))
        train_ce, train_value = np.concatenate(ce_parts), np.concatenate(value_parts)
        fresh = tr["fresh_source_mask"][train_rows].astype(np.float64)
        coeff = tr["policy_coefficients"][train_rows].astype(np.float64)
        pnum, pden = (
            float(np.sum(train_ce * fresh * coeff)),
            float(np.sum(fresh * coeff)),
        )
        err = np.abs(
            train_value - tr["value_targets"][train_rows].reshape(-1).astype(np.float64)
        )
        vnum = float(np.sum(np.where(err < 1, 0.5 * err**2, err - 0.5) * fresh))
        arm_result["training_policy_ce"] = {
            "numerator": pnum,
            "denominator": pden,
            "total": pnum / pden,
            "population_occurrences": len(train_rows),
        }
        arm_result["training_J"] = {
            "total": pnum / pden + 0.3 * vnum / len(train_rows),
            "value_numerator": vnum,
            "value_denominator": len(train_rows),
            "population_occurrences": len(train_rows),
        }
        report["arms"][arm] = arm_result
    for cohort in cohorts:
        report["effects"][cohort] = {}
        for weighting in ("exposure_weighted", "equal_input"):
            report["effects"][cohort][weighting] = {
                arm: {
                    metric: report["arms"][arm][cohort][weighting][metric]
                    - report["arms"]["initializer"][cohort][weighting][metric]
                    for metric in ("policy_ce", "value_mse")
                }
                for arm in ("C", "P")
            }
            report["effects"][cohort][weighting]["P_minus_C"] = {
                metric: report["arms"]["P"][cohort][weighting][metric]
                - report["arms"]["C"][cohort][weighting][metric]
                for metric in ("policy_ce", "value_mse")
            }
    for weighting in ("exposure_weighted", "equal_input"):
        p, c, init = (report["arms"][arm] for arm in ("P", "C", "initializer"))
        report["gates"][f"fresh_ce_vs_initializer_{weighting}"] = (
            p["fresh"][weighting]["policy_ce"]
            <= init["fresh"][weighting]["policy_ce"] - 0.001
        )
        report["gates"][f"fresh_ce_vs_C_{weighting}"] = (
            p["fresh"][weighting]["policy_ce"]
            <= c["fresh"][weighting]["policy_ce"] - 0.001
        )
        report["gates"][f"pooled_ce_vs_initializer_{weighting}"] = (
            p["pooled"][weighting]["policy_ce"]
            <= init["pooled"][weighting]["policy_ce"] + 0.005
        )
        report["gates"][f"pooled_mse_vs_initializer_{weighting}"] = (
            p["pooled"][weighting]["value_mse"]
            <= init["pooled"][weighting]["value_mse"] + 0.01
        )
    report["gates"]["training_policy_ce_decreases"] = (
        report["arms"]["P"]["training_policy_ce"]["total"]
        < report["arms"]["initializer"]["training_policy_ce"]["total"]
    )
    report["gates"]["exact_C_reproduction"] = True
    report["decision"] = (
        "advance_to_fresh_confirmation"
        if all(report["gates"].values())
        else "close_reset_adam_policy_only_branch"
    )
    report["classification"] = report["decision"]
    report["decision_scope"] = (
        "exploratory reused cohort; passing authorizes only separately designed fresh confirmation"
    )
    (out / "endpoint-evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Seed474 fresh-value removal",
        "",
        "Exploratory reused cohort; no independent-confirmation or strength claim.",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "Gates:",
    ]
    lines.extend(
        f"- `{key}`: **{'PASS' if passed else 'FAIL'}**"
        for key, passed in report["gates"].items()
    )
    (out / "results.md").write_text("\n".join(lines) + "\n")
    receipt = {
        path.name: sha(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "receipt.json"
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
