"""Independent read-only reconstruction of seed474 evidence."""

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
from ml.alphazero_lite.seed471_fresh_only_adam_ablation import INIT, INPUTS
from ml.alphazero_lite.seed473_adam_moment_reset import OUT as S473
from ml.alphazero_lite.seed474_fresh_value_removal import OUT, SOURCES

ROOT = Path(__file__).resolve().parents[2]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    original = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in out.iterdir()
        if p.is_file()
    }
    try:
        reg = json.loads((out / "registration.json").read_text())
        if (
            reg.get("schema") != "seed474-fresh-value-removal-registration-v1"
            or reg.get("status") != "frozen_before_execution"
        ):
            raise ValueError("registration_mismatch")
        if set(reg["source_closure_sha256"]) != set(SOURCES):
            raise ValueError("source_inventory_mismatch")
        if reg["dependencies"] != {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }:
            raise ValueError("dependency_identity_mismatch")
        for rel, expected in reg["source_closure_sha256"].items():
            if _hash(root / rel) != expected:
                raise ValueError(f"source_mismatch:{rel}")
        for key, expected in reg["inputs_sha256"].items():
            if _hash(root / INPUTS[key]) != expected:
                raise ValueError(f"input_mismatch:{key}")
        receipt = json.loads((out / "receipt.json").read_text())
        for name, expected in receipt.items():
            if name != "status" and _hash(out / name) != expected:
                raise ValueError(f"receipt_mismatch:{name}")
        gate = json.loads((out / "validity-gate.json").read_text())
        if gate.get("passed") is not True or gate.get(
            "seed473_R_steps_sha256"
        ) != _hash(root / S473 / "R-steps.npz"):
            raise ValueError("exact_C_gate_mismatch")
        data = read_authenticated_bundle(
            root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
        )
        tr = data["training"]
        positions = np.asarray(reg["minibatches"]["positions"], dtype=np.int64)
        if not np.array_equal(positions, tr["first16_minibatch_positions"][:8192]):
            raise ValueError("minibatch_order_mismatch")
        rows = tr["expanded_replay_indices"][tr["train_positions"][positions]]
        torch.set_num_threads(1)
        models = {}
        for arm in ("C", "P"):
            model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            train.load_checkpoint_into_model(model, root / INIT)
            parameters = list(model.parameters())
            optimizer = torch.optim.Adam(
                parameters, lr=0.0000625, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
            )
            with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as archive:
                saved = {key: archive[key] for key in archive.files}
            trajectory = json.loads((out / f"{arm}-trajectory.json").read_text())
            expected_keys = set()
            for step in range(16):
                optimizer.state.clear()
                sl = slice(step * 512, (step + 1) * 512)
                for i, parameter in enumerate(parameters):
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
                        saved[f"{step:02d}_pre_{i:02d}"], parameter.detach().numpy()
                    ):
                        raise ValueError(f"pre_state_mismatch:{arm}:{step + 1}:{i}")
                optimizer.zero_grad(set_to_none=True)
                xb = torch.from_numpy(tr["inputs"][rows[sl]].copy())
                target = torch.from_numpy(tr["policy_targets"][rows[sl]].copy())
                mask = torch.from_numpy(tr["legal_masks"][rows[sl]].copy())
                logits, prediction = model(xb)
                ce = -(
                    target * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)
                ).sum(1)
                coefficient = torch.from_numpy(
                    tr["policy_coefficients"][rows[sl]].copy()
                )
                fresh = torch.from_numpy(
                    tr["fresh_source_mask"][rows[sl]].astype(np.float32)
                )
                value_target = torch.from_numpy(
                    tr["value_targets"][rows[sl]].reshape(-1).copy()
                )
                mass = (coefficient * fresh).sum()
                policy = (ce * coefficient * fresh).sum() / mass
                value_loss = (
                    F.smooth_l1_loss(
                        prediction.reshape(-1), value_target, reduction="none"
                    )
                    * fresh
                ).sum() / 512
                scale = 0.3 if arm == "C" else 0.0
                objective = policy + scale * value_loss
                objective.backward()
                for parameter in parameters:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                gradients = [
                    parameter.grad.detach().clone() for parameter in parameters
                ]
                optimizer.step()
                record = {
                    "step": step + 1,
                    "objective": float(objective.detach()),
                    "policy_loss": float(policy.detach()),
                    "value_loss": float(value_loss.detach()),
                    "value_coefficient": scale,
                    "policy_mass": float(mass),
                    "preclip_norm": float(norm),
                    "clipped": bool(float(norm) > 1),
                    "optimizer_step": 1,
                    "reset_before_update": True,
                    "occurrence_indices": rows[sl].tolist(),
                }
                if trajectory.get("arm") != arm or trajectory["steps"][step] != record:
                    raise ValueError(f"trajectory_mismatch:{arm}:{step + 1}")
                for i, (parameter, gradient) in enumerate(
                    zip(parameters, gradients, strict=True)
                ):
                    state = optimizer.state[parameter]
                    pairs = {
                        "gradient": gradient,
                        "post": parameter.detach(),
                        "exp_avg": state["exp_avg"],
                        "exp_avg_sq": state["exp_avg_sq"],
                        "displacement": parameter.detach().double()
                        - torch.from_numpy(
                            saved[f"{step:02d}_pre_{i:02d}"].copy()
                        ).double(),
                    }
                    for kind, tensor in pairs.items():
                        actual, expected = (
                            saved[f"{step:02d}_{kind}_{i:02d}"],
                            tensor.cpu().numpy(),
                        )
                        if actual.dtype != expected.dtype or not np.array_equal(
                            actual, expected
                        ):
                            raise ValueError(
                                f"archive_mismatch:{arm}:{step + 1}:{kind}:{i}"
                            )
            if set(saved) != expected_keys:
                raise ValueError(f"archive_inventory_mismatch:{arm}")
            models[arm] = model
        initializer = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(initializer, root / INIT)
        models["initializer"] = initializer
        evidence = json.loads((out / "endpoint-evidence.json").read_text())
        ev = data["evaluation"]
        eval_rows = ev["compact_rows"]
        eval_mask = tr["legal_masks"][eval_rows]
        eval_x = torch.from_numpy(tr["inputs"][eval_rows].copy())
        identities = ev["input_identity"].tolist()
        reconstructed = {}
        for arm, model in models.items():
            with torch.inference_mode():
                logits, value = model(eval_x)
                logp = (
                    F.log_softmax(
                        logits.masked_fill(
                            torch.from_numpy(eval_mask.copy()) == 0, -torch.inf
                        ),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
                value = value.reshape(-1).cpu().numpy()
            reconstructed[arm] = {}
            for cohort, selected in (
                ("fresh", np.flatnonzero(ev["source_ids"] == 0)),
                ("pooled", np.arange(len(identities))),
            ):
                groups: dict[str, list[int]] = {}
                for local, row in enumerate(selected):
                    groups.setdefault(identities[int(row)], []).append(local)
                ce = -(
                    ev["policy_targets"][selected]
                    * np.where(np.isfinite(logp[selected]), logp[selected], 0)
                ).sum(1)
                mse = np.square(
                    value[selected].astype(np.float64)
                    - ev["value_targets"].reshape(-1)[selected].astype(np.float64)
                )
                reconstructed[arm][cohort] = {
                    "occurrences": len(selected),
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
                keys = [identities[int(i)] for i in selected]
                with np.load(
                    out / f"{arm}-{cohort}-predictions.npz", allow_pickle=False
                ) as archive:
                    expected = {
                        "prediction": logp[selected].astype(np.float32),
                        "value": value[selected],
                        "policy_target": ev["policy_targets"][selected],
                        "value_target": ev["value_targets"].reshape(-1)[selected],
                        "input_identity": np.asarray(keys, dtype="U256"),
                        "legal_mask": eval_mask[selected],
                        "exposure_weight": np.full(
                            len(selected), 1 / len(selected), dtype=np.float64
                        ),
                        "equal_input_weight": np.asarray(
                            [1 / (len(groups) * len(groups[k])) for k in keys],
                            dtype=np.float64,
                        ),
                    }
                    if set(archive.files) != set(expected):
                        raise ValueError(
                            f"prediction_inventory_mismatch:{arm}:{cohort}"
                        )
                    for name, wanted in expected.items():
                        actual = archive[name]
                        if (
                            actual.dtype != wanted.dtype
                            or actual.shape != wanted.shape
                            or not (
                                np.array_equal(actual, wanted)
                                if wanted.dtype.kind in "US"
                                else np.allclose(actual, wanted, rtol=2e-6, atol=2e-8)
                            )
                        ):
                            raise ValueError(
                                f"prediction_mismatch:{arm}:{cohort}:{name}"
                            )
        for arm, cohorts in reconstructed.items():
            for cohort, values in cohorts.items():
                for weight, metrics in values.items():
                    if weight in ("occurrences", "identities"):
                        if evidence["arms"][arm][cohort][weight] != metrics:
                            raise ValueError(
                                f"endpoint_count_mismatch:{arm}:{cohort}:{weight}"
                            )
                        continue
                    for metric, expected in metrics.items():
                        actual = evidence["arms"][arm][cohort][weight][metric]
                        if isinstance(expected, float) and not np.isclose(
                            actual, expected, rtol=2e-6, atol=2e-8
                        ):
                            raise ValueError(
                                f"endpoint_metric_mismatch:{arm}:{cohort}:{weight}:{metric}"
                            )
        gates = {"exact_C_reproduction": True}
        for weight in ("exposure_weighted", "equal_input"):
            p, c, init = (reconstructed[a] for a in ("P", "C", "initializer"))
            gates[f"fresh_ce_vs_initializer_{weight}"] = (
                p["fresh"][weight]["policy_ce"]
                <= init["fresh"][weight]["policy_ce"] - 0.001
            )
            gates[f"fresh_ce_vs_C_{weight}"] = (
                p["fresh"][weight]["policy_ce"]
                <= c["fresh"][weight]["policy_ce"] - 0.001
            )
            gates[f"pooled_ce_vs_initializer_{weight}"] = (
                p["pooled"][weight]["policy_ce"]
                <= init["pooled"][weight]["policy_ce"] + 0.005
            )
            gates[f"pooled_mse_vs_initializer_{weight}"] = (
                p["pooled"][weight]["value_mse"]
                <= init["pooled"][weight]["value_mse"] + 0.01
            )
        gates["training_policy_ce_decreases"] = (
            evidence["arms"]["P"]["training_policy_ce"]["total"]
            < evidence["arms"]["initializer"]["training_policy_ce"]["total"]
        )
        if evidence.get("gates") != gates or evidence.get("decision") != (
            "advance_to_fresh_confirmation"
            if all(gates.values())
            else "close_reset_adam_policy_only_branch"
        ):
            raise ValueError("gate_or_decision_mismatch")
        with (
            np.load(root / S473 / "R-steps.npz", allow_pickle=False) as historical,
            np.load(out / "C-steps.npz", allow_pickle=False) as control,
        ):
            for step in range(16):
                for index in range(
                    sum(k.startswith("00_post_") for k in historical.files)
                ):
                    key = f"{step:02d}_post_{index:02d}"
                    if not np.array_equal(historical[key], control[key]):
                        raise ValueError(f"exact_control_mismatch:{step + 1}:{index}")
    finally:
        current = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in out.iterdir()
            if p.is_file()
        }
        if current != original:
            raise ValueError("verification_mutated_publication")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    verify(parser.parse_args().root)
    print("seed474 independent reconstruction passed")


if __name__ == "__main__":
    main()
