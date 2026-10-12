"""Prospective seed464 historical-value-removal experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed435_adam_direction as loader
from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed464-historical-value-removal")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
OLD = Path("docs/data/seed462-historical-policy-removal")
MEMBERSHIP = Path(
    "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def datasets(root: Path) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    loader.ROOT = root
    loader.INIT = root / INIT
    arrays = loader._inputs()
    fresh = np.asarray([row["source"] == "fresh" for row in arrays[7]], dtype=bool)
    return arrays, fresh


def targets(root: Path, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    cache: dict[str, list[dict[str, Any]]] = {}
    policies: list[list[float]] = []
    values: list[float] = []
    for row in rows:
        source = row["source"]
        if source not in cache:
            with gzip.open(
                root / f"docs/data/seed426-canonical-overlap/sources/{source}.jsonl.gz",
                "rt",
                encoding="utf-8",
            ) as stream:
                cache[source] = [json.loads(line) for line in stream]
        record = cache[source][int(row["raw_line"]) - 1]
        policies.append(record["policy"])
        values.append(float(record["value"]))
    return np.asarray(policies, dtype=np.float32), np.asarray(values, dtype=np.float32)


def cohort_manifests(root: Path, x: np.ndarray) -> dict[str, Any]:
    with gzip.open(root / MEMBERSHIP, "rt", encoding="utf-8") as stream:
        membership = [json.loads(line) for line in stream]
    pooled = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    fresh = [r for r in pooled if r["source"] == "fresh"]
    manifests: dict[str, Any] = {}
    for name, rows in (("pooled", pooled), ("fresh", fresh)):
        policy, value = targets(root, rows)
        manifest = []
        for index, row in enumerate(rows):
            identity = (
                x[int(row["compact_row"])].astype("<f4", copy=False).tobytes().hex()
            )
            if identity != row["input_identity"]:
                raise ValueError(f"input_identity_mismatch:{name}:{index}")
            manifest.append(
                {
                    "order": index,
                    "input_identity": identity,
                    "compact_row": int(row["compact_row"]),
                    "source": row["source"],
                    "raw_line": int(row["raw_line"]),
                    "active_stones": int(row["active_stones"]),
                    "policy_target_sha256": hashlib.sha256(
                        policy[index].astype("<f4", copy=False).tobytes()
                    ).hexdigest(),
                    "value_target_sha256": hashlib.sha256(
                        value[index : index + 1].astype("<f4", copy=False).tobytes()
                    ).hexdigest(),
                    "policy_target": policy[index].tolist(),
                    "value_target": float(value[index]),
                }
            )
        manifests[name] = {
            "count": len(rows),
            "unique_identities": len({r["input_identity"] for r in rows}),
            "ordered_occurrences": manifest,
        }
    return manifests


def register(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    if out.exists():
        raise ValueError("seed464_output_exists")
    arrays, fresh = datasets(root)
    x, _, _, replay, coefficient, train_positions = arrays[:6]
    split_path = Path(
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz"
    )
    permutation_path = Path(
        "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz"
    )
    with gzip.open(root / permutation_path, "rt", encoding="utf-8") as stream:
        positions = [int(v) for v in json.load(stream)[0][:8192]]
    old_reg = json.loads((root / OLD / "registration.json").read_text())
    if positions != old_reg["first_16_minibatch_positions"]:
        raise ValueError("seed462_minibatch_mismatch")
    # This arm reproduces seed462 F: fresh-only policy objective and full value loss.
    control = json.loads((root / OLD / "F-trajectory.json").read_text())
    if len(control["steps"]) != 16:
        raise ValueError("seed462_F_trajectory_invalid")
    manifests = cohort_manifests(root, x)
    reg416 = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    paths = [
        str(INIT),
        str(split_path),
        str(permutation_path),
        str(MEMBERSHIP),
        str(OLD / "registration.json"),
        str(OLD / "F-trajectory.json"),
        str(OLD / "F-steps.npz"),
    ]
    paths.extend(
        f"docs/data/seed426-canonical-overlap/sources/{r['name']}.jsonl.gz"
        for r in reg416["replays"]
    )
    sources = [
        "ml/alphazero_lite/seed464_historical_value_removal.py",
        "ml/alphazero_lite/verify_seed464_historical_value_removal.py",
        "ml/alphazero_lite/publish_seed464_historical_value_removal.py",
        "ml/alphazero_lite/test_seed464_historical_value_removal.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/seed435_adam_direction.py",
    ]
    out.mkdir(parents=True)
    for source in sources:
        destination = out / "source-snapshots" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / source, destination)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    result = {
        "schema": "seed464-registration-v1",
        "status": "frozen_before_C",
        "hypothesis": "historical value supervision contributes to fresh-policy regression through the shared network",
        "initializer": {"path": str(INIT), "sha256": sha(root / INIT)},
        "source_sha256": {p: sha(root / p) for p in sources},
        "input_sha256": {p: sha(root / p) for p in paths},
        "environment": environment,
        "dependencies": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "architecture": {
            "name": "residual_v3",
            "hidden_sizes": [96, 3],
            "input_size": 27,
        },
        "optimizer": {
            "name": "Adam",
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "initial_moments": "zero",
            "gradient_clip_norm": 1.0,
            "updates": 16,
            "projection": None,
            "backtracking": None,
        },
        "minibatches": {
            "source": "seed462 first 16 epoch permutation minibatches",
            "batch_size": 512,
            "positions": positions,
        },
        "objective": {
            "policy_both": "sum(c_i * fresh_i * legal_CE_i) / sum(c_i * fresh_i)",
            "C_value": "0.3 * sum(Huber_i) / 512",
            "T_value": "0.3 * sum(fresh_i * Huber_i) / 512",
            "huber_delta": 1.0,
            "fixed_occurrence_denominator": 512,
        },
        "numeric_contract": {
            "inputs_targets_parameters": "float32",
            "source_coefficients": "float32",
            "fresh_mask": "boolean cast to float32 for multiplication",
            "policy_reduction": "float32 torch sum numerator and denominator then divide",
            "value_reduction": "float32 torch Huber vector sum divided by 512; T masks Huber before sum",
            "gradient_norm": "torch clip_grad_norm_ over trainable parameters",
            "metrics": "float64 numpy means after float32 inference and rowwise errors",
        },
        "cohorts": manifests,
        "cohort_selection": {
            "pooled": "membership occurrences in original order where subset == unseen and active_stones > 32",
            "fresh": "pooled occurrences additionally where source == fresh",
            "targets": "each occurrence uses its source JSONL record at raw_line; no compact-row target substitution",
        },
        "weighting": {
            "exposure": "arithmetic mean over ordered occurrences",
            "equal_input": "mean metric within exact input_identity groups, then arithmetic mean across identities",
        },
        "inference": {
            "batch_size": "all compact inputs in one model forward pass, preserving historical float32 log_softmax semantics; select cohort occurrences in manifest order",
            "legal_logits": "illegal logits masked to -inf before log_softmax",
        },
        "gates": {
            "fresh_ce_vs_initializer_and_C_max_improvement": 0.005,
            "pooled_ce_max_regression": 0.002,
            "pooled_and_fresh_mse_max_regression": 0.002,
            "training_objective": "T full training objective strictly below initializer",
            "exact_control": "C equals seed462 F updates and post-step parameters bitwise",
            "verification": "independent reconstruction required",
        },
    }
    result["registration_sha256_self_check"] = None
    (out / "registration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def policy_loss(
    ce: torch.Tensor, coefficient: torch.Tensor, fresh: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = coefficient * fresh.to(dtype=coefficient.dtype)
    denominator = weights.sum()
    if float(denominator) <= 0:
        raise ValueError("empty_policy_denominator")
    return (weights * ce).sum() / denominator, denominator


def value_loss(
    huber: torch.Tensor, fresh: torch.Tensor, arm: str, batch_size: int = 512
) -> torch.Tensor:
    """Apply historical value masking while keeping the original batch divisor."""
    if arm not in {"C", "T"}:
        raise ValueError(f"invalid_arm:{arm}")
    weighted = huber if arm == "C" else huber * fresh.to(dtype=huber.dtype)
    return weighted.sum() / batch_size


def run_arm(root: Path, arm: str) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    reg = json.loads((out / "registration.json").read_text())
    if reg["status"] != "frozen_before_C" or arm not in {"C", "T"}:
        raise ValueError("invalid_execution_state")
    if arm == "T" and not (out / "C-verified.json").is_file():
        raise ValueError("T_requires_verified_C")
    for rel, digest in reg["source_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"source_changed:{rel}")
    for rel, digest in reg["input_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"input_changed:{rel}")
    arrays, fresh = datasets(root)
    x, policies, values, replay, coeff, train_pos = arrays[:6]
    train_rows = replay[train_pos]
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INIT)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    archive: dict[str, np.ndarray] = {}
    records = []
    for step in range(16):
        pos = np.asarray(
            reg["minibatches"]["positions"][step * 512 : (step + 1) * 512],
            dtype=np.int64,
        )
        ids = train_rows[pos]
        before = [p.detach().clone() for p in params]
        optimizer.zero_grad(set_to_none=True)
        xb = torch.from_numpy(x[ids])
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ids]))
        logits, prediction = model(xb)
        ce = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policies[ids])
        )
        cb = torch.from_numpy(coeff[ids])
        mask = torch.from_numpy(fresh[ids])
        ploss, pmass = policy_loss(ce, cb, mask)
        huber = train.compute_value_loss_vector(
            prediction,
            torch.from_numpy(values[ids]),
            value_loss="huber",
            huber_delta=1.0,
        )
        vloss = value_loss(huber, mask, arm)
        objective = ploss + 0.3 * vloss
        objective.backward()
        unclipped = torch.nn.utils.clip_grad_norm_(params, 1.0)
        gradients = [p.grad.detach().clone() for p in params]
        optimizer.step()
        after = [p.detach().clone() for p in params]
        records.append(
            {
                "step": step + 1,
                "positions": pos.tolist(),
                "rows": ids.tolist(),
                "policy_mass": float(pmass),
                "policy_loss": float(ploss.detach()),
                "value_loss": float(vloss.detach()),
                "objective": float(objective.detach()),
                "preclip_norm": float(unclipped),
                "post_hash": _hash(after),
            }
        )
        for i, (p0, grad, p1) in enumerate(zip(before, gradients, after, strict=True)):
            for label, tensor in (
                ("pre", p0),
                ("gradient", grad),
                ("proposal", p1),
                ("post", p1),
                ("exp_avg", optimizer.state[params[i]]["exp_avg"]),
                ("exp_avg_sq", optimizer.state[params[i]]["exp_avg_sq"]),
            ):
                archive[f"{step:02d}_{label}_{i:02d}"] = (
                    tensor.detach().cpu().numpy().copy()
                )
    np.savez_compressed(out / f"{arm}-steps.npz", **archive)
    trajectory = {
        "arm": arm,
        "steps": records,
        "final_sha256": _hash([p.detach() for p in params]),
    }
    (out / f"{arm}-trajectory.json").write_text(
        json.dumps(trajectory, indent=2, sort_keys=True) + "\n"
    )
    if arm == "C":
        exact_seed462_control(root, out)
    return trajectory


def _hash(tensors: list[torch.Tensor]) -> str:
    return hashlib.sha256(
        b"".join(
            t.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
            for t in tensors
        )
    ).hexdigest()


def exact_seed462_control(root: Path, out: Path) -> dict[str, Any]:
    mismatches = []
    with (
        np.load(out / "C-steps.npz", allow_pickle=False) as ours,
        np.load(root / OLD / "F-steps.npz", allow_pickle=False) as old,
    ):
        for step in range(16):
            for i in range(sum(k.startswith(f"{step:02d}_post_") for k in ours.files)):
                for label in ("proposal", "post"):
                    if not np.array_equal(
                        ours[f"{step:02d}_{label}_{i:02d}"],
                        old[f"{step:02d}_{label}_{i:02d}"],
                    ):
                        mismatches.append(f"{label}:{step + 1}:{i}")
    if mismatches:
        (out / "C-invalid-attempt.json").write_text(
            json.dumps({"status": "invalid", "mismatches": mismatches}, indent=2) + "\n"
        )
        raise ValueError(f"seed462_F_control_mismatch:{mismatches[:8]}")
    result = {
        "status": "exact",
        "steps": 16,
        "comparisons": "all proposals and post-step parameter tensors",
        "mismatches": [],
    }
    (out / "C-verified.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "run-C", "run-T"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "register":
        register(args.root)
    else:
        run_arm(args.root, args.action[-1])


if __name__ == "__main__":
    main()
