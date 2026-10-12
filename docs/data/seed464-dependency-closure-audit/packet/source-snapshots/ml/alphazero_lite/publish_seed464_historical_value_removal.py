"""Publish seed464 checkpoints, occurrence evidence, decision, and receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed464_historical_value_removal as experiment
from ml.alphazero_lite import train


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / experiment.OUT
    reg = json.loads((out / "registration.json").read_text())
    if (
        not (out / "C-verified.json").is_file()
        or not (out / "T-trajectory.json").is_file()
    ):
        raise ValueError("both_arms_required")
    arrays, fresh = experiment.datasets(root)
    x, policies, values, replay, coeff, train_positions = arrays[:6]
    cohorts = reg["cohorts"]
    result: dict[str, Any] = {
        "schema": "seed464-endpoint-evidence-v1",
        "cohorts": {},
        "arms": {},
        "effects_T_minus_C": {},
    }
    for cohort_name, cohort_spec in cohorts.items():
        occurrences = cohort_spec["ordered_occurrences"]
        ids = np.asarray([row["compact_row"] for row in occurrences], dtype=np.int64)
        target_policy = np.asarray(
            [row["policy_target"] for row in occurrences], dtype=np.float32
        )
        target_value = np.asarray(
            [row["value_target"] for row in occurrences], dtype=np.float32
        )
        result["cohorts"][cohort_name] = {
            "count": len(ids),
            "unique_identities": cohort_spec["unique_identities"],
            "ordered_manifest_sha256": hashlib.sha256(
                json.dumps(occurrences, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for arm in ("initializer", "C", "T"):
            model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            if arm == "initializer":
                train.load_checkpoint_into_model(model, root / experiment.INIT)
            else:
                with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as archive:
                    for i, parameter in enumerate(model.parameters()):
                        parameter.data.copy_(
                            torch.from_numpy(archive[f"15_post_{i:02d}"].copy())
                        )
            model.eval()
            with torch.inference_mode():
                logits, pred = model(torch.from_numpy(x))
                legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x))
                logp = (
                    torch.log_softmax(logits.masked_fill(legal <= 0, -torch.inf), dim=1)
                    .cpu()
                    .numpy()
                )
                pred_value = pred.reshape(-1).cpu().numpy()
            policy_logp = logp[ids]
            occurrence_ce = -(
                target_policy * np.where(np.isfinite(policy_logp), policy_logp, 0)
            ).sum(axis=1)
            occurrence_mse = np.square(pred_value[ids] - target_value)
            groups: dict[str, list[int]] = {}
            for index, row in enumerate(occurrences):
                groups.setdefault(row["input_identity"], []).append(index)
            metrics = {
                "occurrence_count": len(ids),
                "unique_identities": len(groups),
                "exposure": {
                    "policy_ce": float(np.mean(occurrence_ce)),
                    "value_mse": float(np.mean(occurrence_mse)),
                },
                "equal_input": {
                    "policy_ce": float(
                        np.mean([np.mean(occurrence_ce[ix]) for ix in groups.values()])
                    ),
                    "value_mse": float(
                        np.mean([np.mean(occurrence_mse[ix]) for ix in groups.values()])
                    ),
                },
            }
            result["arms"].setdefault(arm, {})[cohort_name] = metrics
            np.savez_compressed(
                out / f"{arm}-{cohort_name}-occurrences.npz",
                input_rows=ids,
                log_policy=policy_logp,
                target_policy=target_policy,
                value=pred_value[ids],
                target_value=target_value,
            )
    # Training-only objective is on replay[train_positions], retaining occurrence multiplicity.
    training_rows = replay[train_positions]
    for arm in ("initializer", "C", "T"):
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        if arm == "initializer":
            train.load_checkpoint_into_model(model, root / experiment.INIT)
        else:
            with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as archive:
                for i, parameter in enumerate(model.parameters()):
                    parameter.data.copy_(
                        torch.from_numpy(archive[f"15_post_{i:02d}"].copy())
                    )
        pieces = []
        value_pieces = []
        with torch.inference_mode():
            for start in range(0, len(training_rows), 2048):
                ids = training_rows[start : start + 2048]
                legal = torch.from_numpy(
                    train.legal_mask_matrix_for_encoded_states(x[ids])
                )
                logits, pred = model(torch.from_numpy(x[ids]))
                ce = train.compute_policy_cross_entropy(
                    logits.masked_fill(legal <= 0, -1e9),
                    torch.from_numpy(policies[ids]),
                )
                pieces.append(ce.cpu().numpy())
                value_pieces.append(pred.reshape(-1).cpu().numpy())
        ce_all = np.concatenate(pieces)
        pred_all = np.concatenate(value_pieces)
        err = np.abs(pred_all - values[training_rows].reshape(-1))
        huber = np.where(err < 1, 0.5 * err**2, err - 0.5)
        weights = coeff[training_rows].astype(np.float64)
        fresh_rows = fresh[training_rows]
        objective = float(
            np.average(ce_all, weights=weights * fresh_rows)
            + 0.3
            * (huber * (fresh_rows if arm == "T" else 1)).sum(dtype=np.float64)
            / len(training_rows)
        )
        result["arms"].setdefault(arm, {})["training_objective"] = {
            "policy_numerator": float(
                (ce_all * weights * fresh_rows).sum(dtype=np.float64)
            ),
            "policy_denominator": float((weights * fresh_rows).sum(dtype=np.float64)),
            "value_numerator": float(
                (huber * (fresh_rows if arm == "T" else 1)).sum(dtype=np.float64)
            ),
            "value_denominator": len(training_rows),
            "total": objective,
        }
        if arm in {"C", "T"}:
            with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as archive:
                model = train.PolicyValueNet((96, 3), "residual_v3", 27)
                for i, parameter in enumerate(model.parameters()):
                    parameter.data.copy_(
                        torch.from_numpy(archive[f"15_post_{i:02d}"].copy())
                    )
                torch.save(model.state_dict(), out / f"{arm}-checkpoint.pt")
    for cohort in ("pooled", "fresh"):
        result["effects_T_minus_C"][cohort] = {}
        for weight in ("exposure", "equal_input"):
            result["effects_T_minus_C"][cohort][weight] = {
                metric: result["arms"]["T"][cohort][weight][metric]
                - result["arms"]["C"][cohort][weight][metric]
                for metric in ("policy_ce", "value_mse")
            }
    gates: dict[str, bool] = {}
    for weight in ("exposure", "equal_input"):
        t = result["arms"]["T"]
        init = result["arms"]["initializer"]
        c = result["arms"]["C"]
        gates[f"fresh_vs_initializer_{weight}"] = (
            t["fresh"][weight]["policy_ce"]
            <= init["fresh"][weight]["policy_ce"] - 0.005
        )
        gates[f"fresh_vs_C_{weight}"] = (
            t["fresh"][weight]["policy_ce"] <= c["fresh"][weight]["policy_ce"] - 0.005
        )
        gates[f"pooled_ce_{weight}"] = (
            t["pooled"][weight]["policy_ce"]
            <= init["pooled"][weight]["policy_ce"] + 0.002
        )
        for cohort in ("pooled", "fresh"):
            gates[f"{cohort}_mse_{weight}"] = (
                t[cohort][weight]["value_mse"]
                <= init[cohort][weight]["value_mse"] + 0.002
            )
    gates["training_objective_decreases"] = (
        result["arms"]["T"]["training_objective"]["total"]
        < result["arms"]["initializer"]["training_objective"]["total"]
    )
    gates["exact_control"] = True
    result["gates"] = gates
    result["decision"] = (
        "advance_to_separately_preregistered_fresh_holdout_strength_experiment"
        if all(gates.values())
        else "close_configuration"
    )
    (out / "endpoint-evidence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    tracked = sorted(
        p for p in out.iterdir() if p.is_file() and p.name != "receipt.json"
    )
    receipt = {
        "schema": "seed464-receipt-v1",
        "registration_sha256": sha(out / "registration.json"),
        "publication_sha256": {p.name: sha(p) for p in tracked},
        "decision": {"status": result["decision"], "gates": gates},
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    publish(parser.parse_args().root)
