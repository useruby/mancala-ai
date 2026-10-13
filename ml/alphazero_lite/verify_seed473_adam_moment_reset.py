"""Independent, read-only reconstruction of the seed473 publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)
from ml.alphazero_lite.seed471_fresh_only_adam_ablation import (
    INIT,
    INPUTS,
    OUT as SEED471_OUT,
)
from ml.alphazero_lite.seed473_adam_moment_reset import OUT, SOURCES

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    originals = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in out.iterdir()
        if p.is_file()
    }
    try:
        reg = json.loads((out / "registration.json").read_text())
        if (
            reg.get("schema") != "seed473-adam-moment-reset-registration-v1"
            or reg.get("status") != "frozen_before_execution"
        ):
            raise ValueError("registration_state_mismatch")
        if set(reg["source_closure_sha256"]) != set(SOURCES):
            raise ValueError("source_inventory_mismatch")
        if reg["dependencies"] != {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }:
            raise ValueError("dependency_identity_mismatch")
        for path, expected in reg["source_closure_sha256"].items():
            if digest(root / path) != expected:
                raise ValueError(f"source_closure_mismatch:{path}")
        for key, expected in reg["inputs_sha256"].items():
            if digest(root / INPUTS[key]) != expected:
                raise ValueError(f"input_mismatch:{key}")
        receipt = json.loads((out / "receipt.json").read_text())
        if receipt.get("status") != "post_execution_bound_receipt":
            raise ValueError("receipt_status_mismatch")
        for name, expected in receipt.items():
            if name != "status" and digest(out / name) != expected:
                raise ValueError(f"receipt_binding_mismatch:{name}")
        gate = json.loads((out / "validity-gate.json").read_text())
        if gate.get("passed") is not True or gate.get(
            "seed471_T_steps_sha256"
        ) != digest(root / SEED471_OUT / "T-steps.npz"):
            raise ValueError("control_reproduction_gate_mismatch")
        # Control archive is independently compared state-by-state to historical T.
        with (
            np.load(out / "C-steps.npz", allow_pickle=False) as c,
            np.load(root / SEED471_OUT / "T-steps.npz", allow_pickle=False) as t,
        ):
            count = sum(k.startswith("00_post_") for k in t.files)
            for step in range(16):
                for i in range(count):
                    key = f"{step:02d}_post_{i:02d}"
                    if not np.array_equal(c[key], t[key]):
                        raise ValueError(f"control_state_mismatch:{step + 1}:{i}")
        arrays = read_authenticated_bundle(
            root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
        )
        tr = arrays["training"]
        positions = np.asarray(reg["minibatches"]["positions"], dtype=np.int64)
        if not np.array_equal(positions, tr["first16_minibatch_positions"][:8192]):
            raise ValueError("minibatch_order_mismatch")
        rows = tr["expanded_replay_indices"][tr["train_positions"][positions]]
        x = tr["inputs"][rows]
        targets = tr["policy_targets"][rows]
        masks = tr["legal_masks"][rows]
        coeff = tr["policy_coefficients"][rows]
        fresh = tr["fresh_source_mask"][rows].astype(np.float32)
        vals = tr["value_targets"][rows].reshape(-1)
        torch.set_num_threads(1)
        base = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(base, root / INIT)
        models = {"initializer": base}
        for arm in ("C", "R"):
            model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            model.load_state_dict(base.state_dict())
            params = list(model.parameters())
            optimizer = torch.optim.Adam(
                params, lr=0.0000625, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
            )
            with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as archive:
                saved = {k: archive[k] for k in archive.files}
            expected_keys = set()
            trajectory = json.loads((out / f"{arm}-trajectory.json").read_text())
            for step in range(16):
                if arm == "R":
                    optimizer.state.clear()
                sl = slice(step * 512, (step + 1) * 512)
                for i, p in enumerate(params):
                    expected_keys.update(
                        f"{step:02d}_{kind}_{i:02d}"
                        for kind in (
                            "pre",
                            "gradient",
                            "post",
                            "exp_avg",
                            "exp_avg_sq",
                            "displacement",
                        )
                    )
                    if not np.array_equal(
                        saved[f"{step:02d}_pre_{i:02d}"], p.detach().cpu().numpy()
                    ):
                        raise ValueError(f"pre_state_mismatch:{arm}:{step + 1}:{i}")
                optimizer.zero_grad(set_to_none=True)
                xb = torch.from_numpy(x[sl].copy())
                tb = torch.from_numpy(targets[sl].copy())
                mb = torch.from_numpy(masks[sl].copy())
                lg, pred = model(xb)
                ce = -(tb * F.log_softmax(lg.masked_fill(mb == 0, -1e9), dim=1)).sum(1)
                c = torch.from_numpy(coeff[sl].copy())
                f = torch.from_numpy(fresh[sl].copy())
                v = torch.from_numpy(vals[sl].copy())
                mass = (c * f).sum()
                policy = (ce * c * f).sum() / mass
                huber = F.smooth_l1_loss(pred.reshape(-1), v, reduction="none")
                value = (huber * f).sum() / 512
                objective = policy + 0.3 * value
                objective.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                grads = [p.grad.detach().clone() for p in params]
                optimizer.step()
                expected_record = {
                    "step": step + 1,
                    "objective": float(objective.detach()),
                    "policy_loss": float(policy.detach()),
                    "value_loss": float(value.detach()),
                    "policy_mass": float(mass),
                    "preclip_norm": float(norm),
                    "clipped": bool(float(norm) > 1),
                    "optimizer_step": 1 if arm == "R" else step + 1,
                    "reset_before_update": arm == "R",
                    "occurrence_indices": rows[sl].tolist(),
                }
                if (
                    trajectory.get("arm") != arm
                    or trajectory["steps"][step] != expected_record
                ):
                    raise ValueError(f"trajectory_mismatch:{arm}:{step + 1}")
                for i, (p, g) in enumerate(zip(params, grads, strict=True)):
                    state = optimizer.state[p]
                    pairs = {
                        "gradient": g,
                        "post": p.detach(),
                        "exp_avg": state["exp_avg"],
                        "exp_avg_sq": state["exp_avg_sq"],
                        "displacement": p.detach().double()
                        - torch.from_numpy(
                            saved[f"{step:02d}_pre_{i:02d}"].copy()
                        ).double(),
                    }
                    for kind, tensor in pairs.items():
                        actual = saved[f"{step:02d}_{kind}_{i:02d}"]
                        expected = tensor.detach().cpu().numpy()
                        if (
                            actual.dtype != expected.dtype
                            or actual.shape != expected.shape
                            or not np.allclose(actual, expected, rtol=2e-6, atol=2e-8)
                        ):
                            raise ValueError(
                                f"reconstruction_mismatch:{arm}:{step + 1}:{kind}:{i}"
                            )
            if set(saved) != expected_keys:
                raise ValueError(f"archive_inventory_mismatch:{arm}")
            models[arm] = model
        evidence = json.loads((out / "endpoint-evidence.json").read_text())
        ev = arrays["evaluation"]
        eval_rows = ev["compact_rows"]
        eval_x = tr["inputs"][eval_rows]
        eval_masks = tr["legal_masks"][eval_rows]
        identities = ev["input_identity"].tolist()
        expected_arms = {}
        for arm, model in models.items():
            model.eval()
            with torch.inference_mode():
                logits, predicted_value = model(torch.from_numpy(eval_x.copy()))
                logp = (
                    F.log_softmax(
                        logits.masked_fill(
                            torch.from_numpy(eval_masks.copy()) == 0, -torch.inf
                        ),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
                predicted_value = predicted_value.reshape(-1).cpu().numpy()
            expected_arms[arm] = {}
            for cohort, selected in (
                ("fresh", np.flatnonzero(ev["source_ids"] == 0)),
                ("pooled", np.arange(len(identities))),
            ):
                group_rows: dict[str, list[int]] = {}
                for local, occurrence in enumerate(selected):
                    group_rows.setdefault(identities[int(occurrence)], []).append(local)
                ce = (
                    -(
                        ev["policy_targets"][selected]
                        * np.where(np.isfinite(logp[selected]), logp[selected], 0)
                    )
                    .sum(axis=1)
                    .astype(np.float64)
                )
                mse = np.square(
                    predicted_value[selected].astype(np.float64)
                    - ev["value_targets"][selected].reshape(-1).astype(np.float64)
                )
                summary = {
                    "occurrences": len(selected),
                    "identities": len(group_rows),
                    "exposure_weighted": {
                        "policy_ce": float(ce.mean()),
                        "value_mse": float(mse.mean()),
                    },
                    "equal_input": {
                        "policy_ce": float(
                            np.mean([ce[ix].mean() for ix in group_rows.values()])
                        ),
                        "value_mse": float(
                            np.mean([mse[ix].mean() for ix in group_rows.values()])
                        ),
                    },
                }
                expected_arms[arm][cohort] = summary
                archive_path = out / f"{arm}-{cohort}-predictions.npz"
                with np.load(archive_path, allow_pickle=False) as archived:
                    local_ids = [identities[int(i)] for i in selected]
                    expected_predictions = {
                        "prediction": logp[selected].astype(np.float32),
                        "value": predicted_value[selected],
                        "policy_target": ev["policy_targets"][selected],
                        "value_target": ev["value_targets"].reshape(-1)[selected],
                        "input_identity": np.asarray(local_ids, dtype="U256"),
                        "legal_mask": eval_masks[selected],
                        "exposure_weight": np.full(
                            len(selected), 1 / len(selected), dtype=np.float64
                        ),
                        "equal_input_weight": np.asarray(
                            [
                                1 / (len(group_rows) * len(group_rows[item]))
                                for item in local_ids
                            ],
                            dtype=np.float64,
                        ),
                    }
                    if set(archived.files) != set(expected_predictions):
                        raise ValueError(
                            f"prediction_inventory_mismatch:{arm}:{cohort}"
                        )
                    for name, expected in expected_predictions.items():
                        actual = archived[name]
                        matches = (
                            np.array_equal(actual, expected)
                            if expected.dtype.kind in "US"
                            else np.allclose(actual, expected, rtol=2e-6, atol=2e-8)
                        )
                        if (
                            actual.dtype != expected.dtype
                            or actual.shape != expected.shape
                            or not matches
                        ):
                            raise ValueError(
                                f"prediction_mismatch:{arm}:{cohort}:{name}"
                            )
            train_rows = tr["expanded_replay_indices"][tr["train_positions"]]
            losses = []
            predicted_values = []
            for start in range(0, len(train_rows), 1024):
                part = train_rows[start : start + 1024]
                with torch.inference_mode():
                    lg, pv = model(torch.from_numpy(tr["inputs"][part].copy()))
                    legal = torch.from_numpy(tr["legal_masks"][part].copy())
                    lp = F.log_softmax(lg.masked_fill(legal == 0, -1e9), dim=1)
                    loss = -(
                        torch.from_numpy(tr["policy_targets"][part].copy()) * lp
                    ).sum(1)
                losses.append(loss.cpu().numpy().astype(np.float64))
                predicted_values.append(pv.reshape(-1).cpu().numpy().astype(np.float64))
            all_ce = np.concatenate(losses)
            all_values = np.concatenate(predicted_values)
            train_fresh = tr["fresh_source_mask"][train_rows].astype(np.float64)
            train_coeff = tr["policy_coefficients"][train_rows].astype(np.float64)
            err = np.abs(
                all_values
                - tr["value_targets"][train_rows].reshape(-1).astype(np.float64)
            )
            train_huber = np.where(err < 1, 0.5 * err**2, err - 0.5)
            pden = float(np.sum(train_coeff * train_fresh))
            pnum = float(np.sum(all_ce * train_coeff * train_fresh))
            vnum = float(np.sum(train_huber * train_fresh))
            expected_arms[arm]["training_J"] = {
                "total": pnum / pden + 0.3 * vnum / len(train_rows),
                "policy_numerator": pnum,
                "policy_denominator": pden,
                "value_numerator": vnum,
                "value_denominator": len(train_rows),
                "population_occurrences": len(train_rows),
            }
        if set(evidence["arms"]) != set(expected_arms):
            raise ValueError("endpoint_arm_inventory_mismatch")
        for arm, cohorts in expected_arms.items():
            for cohort, metrics in cohorts.items():
                for metric, expected in metrics.items():
                    actual = evidence["arms"][arm][cohort][metric]
                    if isinstance(expected, dict):
                        for key, value in expected.items():
                            if not np.isclose(actual[key], value, rtol=2e-6, atol=2e-8):
                                raise ValueError(
                                    f"endpoint_metric_mismatch:{arm}:{cohort}:{metric}:{key}"
                                )
                    elif actual != expected:
                        raise ValueError(
                            f"endpoint_count_mismatch:{arm}:{cohort}:{metric}"
                        )
        expected_effects = {}
        for cohort in ("fresh", "pooled"):
            expected_effects[cohort] = {}
            for weighting in ("exposure_weighted", "equal_input"):
                expected_effects[cohort][weighting] = {
                    arm: {
                        metric: expected_arms[arm][cohort][weighting][metric]
                        - expected_arms["initializer"][cohort][weighting][metric]
                        for metric in ("policy_ce", "value_mse")
                    }
                    for arm in ("C", "R")
                }
                expected_effects[cohort][weighting]["R_minus_C"] = {
                    metric: expected_arms["R"][cohort][weighting][metric]
                    - expected_arms["C"][cohort][weighting][metric]
                    for metric in ("policy_ce", "value_mse")
                }
        for cohort, weightings in expected_effects.items():
            for weighting, arms_effect in weightings.items():
                for arm, metrics in arms_effect.items():
                    for metric, expected in metrics.items():
                        actual = evidence["effects"][cohort][weighting][arm][metric]
                        if not np.isclose(actual, expected, rtol=2e-6, atol=2e-8):
                            raise ValueError(
                                f"effect_reconstruction_mismatch:{cohort}:{weighting}:{arm}:{metric}"
                            )
        gates = {}
        for weighting in ("exposure_weighted", "equal_input"):
            r, c, init = (expected_arms[name] for name in ("R", "C", "initializer"))
            gates[f"fresh_ce_vs_initializer_{weighting}"] = (
                r["fresh"][weighting]["policy_ce"]
                <= init["fresh"][weighting]["policy_ce"] - 0.001
            )
            gates[f"fresh_ce_vs_C_{weighting}"] = (
                r["fresh"][weighting]["policy_ce"]
                <= c["fresh"][weighting]["policy_ce"] - 0.001
            )
            gates[f"pooled_ce_vs_initializer_{weighting}"] = (
                r["pooled"][weighting]["policy_ce"]
                <= init["pooled"][weighting]["policy_ce"] + 0.005
            )
            gates[f"pooled_mse_vs_initializer_{weighting}"] = (
                r["pooled"][weighting]["value_mse"]
                <= init["pooled"][weighting]["value_mse"] + 0.01
            )
        gates["training_J_decreases"] = (
            expected_arms["R"]["training_J"]["total"]
            < expected_arms["initializer"]["training_J"]["total"]
        )
        gates["exact_C_reproduction"] = True
        decision = (
            "advance_to_fresh_confirmation"
            if all(gates.values())
            else "close_moment_reset_branch"
        )
        if evidence.get("gates") != gates:
            raise ValueError("gate_reconstruction_mismatch")
        if (
            evidence.get("decision") != decision
            or evidence.get("classification") != decision
        ):
            raise ValueError("decision_reconstruction_mismatch")
    finally:
        current = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in out.iterdir()
            if p.is_file()
        }
        if current != originals:
            raise ValueError("verification_mutated_publication")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    verify(parser.parse_args().root)
    print("seed473 independent verification passed")


if __name__ == "__main__":
    main()
