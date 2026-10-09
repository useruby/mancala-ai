"""Matched-update optimizer-direction retrospective screen (seed435)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
    load_jsonl_replay,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed435-adam-direction-screen"
INIT = (
    ROOT
    / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
INIT_SHA = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
RECEIPTS = {
    "membership": (
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "554f3bd016a0f920ef253538d741a91cb27c9882d2dba94cf7509030af8ca385",
    ),
    "evaluation_receipt": (
        "docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json",
        "fbcaff1e183b2247673dd7f301c410e11524d13bbe47e067079ee8947aa60ee1",
    ),
    "supplemental_receipt": (
        "docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json",
        "1dd0f8e2442a32654b29c8969c1300cbbfecb83f84809d2a478290e0bda9209e",
    ),
}
TOL = 2e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]


def global_norm(values: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float(torch.sum(value.double() ** 2)) for value in values))


def first_moment_direction(
    moment: list[torch.Tensor], step: int, beta: float = 0.9
) -> list[torch.Tensor]:
    return [value / (1.0 - beta**step) for value in moment]


def matched_step(
    model: torch.nn.Module, moment: list[torch.Tensor], radius: float, step: int
) -> tuple[float, float]:
    direction = first_moment_direction(moment, step)
    norm = global_norm(direction)
    if norm == 0.0:
        if radius > 0.0:
            raise ValueError("positive_radius_with_zero_direction")
        return norm, 0.0
    scale = radius / norm
    actual: list[float] = []
    with torch.no_grad():
        for (_, parameter), vector in zip(parameters(model), direction, strict=True):
            delta = -scale * vector
            parameter.add_(delta)
            actual.append(float(torch.sum(delta.double() ** 2)))
    actual_norm = math.sqrt(sum(actual))
    if not math.isclose(actual_norm, radius, rel_tol=TOL, abs_tol=TOL):
        raise ValueError("matched_update_radius_mismatch")
    return norm, actual_norm


def _inputs() -> tuple[np.ndarray, ...]:
    reg = json.loads(
        (
            ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    if (
        sha(INIT) != INIT_SHA
        or sha(ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json")
        != "4839ed63d6a48945ea11ea4995570f11118adbd65c027f49c3c5f04066bbd066"
    ):
        raise ValueError("frozen_initializer_or_registration_identity_mismatch")
    from ml.alphazero_lite.verify_seed434_census_source import reconstruct_sources

    reconstructed = list(reconstruct_sources(ROOT, reg))
    if len(reconstructed) != 87625:
        raise ValueError("five_source_reconstruction_count_mismatch")
    for _, (relative, expected) in RECEIPTS.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"seed427_binding_mismatch:{relative}")
    paths = [
        Path(reg["derivatives"][item["name"]]["A"]["derivative"])
        for item in reg["replays"]
    ]
    weights = [int(item["weight"]) for item in reg["replays"]]
    x, p, v, replay, coeff = load_jsonl_replay(
        paths,
        weights,
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=[
            item["value_target_mode"] for item in reg["replays"]
        ],
        include_policy_loss_weights=True,
    )
    with gzip.open(ROOT / reg["training"]["source_row_split"]["path"], "rt") as stream:
        split = json.load(stream)
    train_positions = np.asarray(split["train_positions"], dtype=np.int64)
    val_positions = np.asarray(split["validation_positions"], dtype=np.int64)
    return x, p, v, replay, coeff, train_positions, val_positions, reconstructed


def batch_record(
    ids: list[int], replay: np.ndarray, reconstructed: list[dict[str, Any]]
) -> dict[str, Any]:
    row_ids = [int(replay[index]) for index in ids]
    return {
        "expanded_positions": ids,
        "compact_rows": row_ids,
        "identity_sha256": hashlib.sha256(
            json.dumps(row_ids, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def metrics(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    membership: list[dict[str, Any]],
) -> dict[str, Any]:
    x, p, v, replay, coefficients, train_positions, _val_positions, _source_rows = (
        arrays
    )
    model.eval()
    masks = legal_mask_matrix_for_encoded_states(x)
    with torch.inference_mode():
        logits, pred_value = model(torch.from_numpy(x))
        policy_rows = compute_policy_cross_entropy(
            logits.masked_fill(torch.from_numpy(masks) <= 0, -1e9), torch.from_numpy(p)
        ).numpy()
        value_rows = torch.square(pred_value - torch.from_numpy(v)).reshape(-1).numpy()
        value_huber = compute_value_loss_vector(
            pred_value, torch.from_numpy(v), value_loss="huber", huber_delta=1.0
        ).numpy()
    # Production training objective is evaluated over every replay exposure with the
    # original row coefficients and multiplicities, retaining its batch-loss form.
    all_rows = replay
    weights = coefficients[all_rows]
    train_obj = float(
        np.average(policy_rows[all_rows], weights=weights)
        + 0.3 * np.mean(value_huber[all_rows])
    )
    selected = [
        row
        for row in membership
        if row["active_stones"] > 32 and row["subset"] == "unseen"
    ]
    result: dict[str, Any] = {
        "full_training_objective": train_obj,
        "unseen_count": len(selected),
    }
    for definition in ("exposure_weighted", "equal_input"):
        identities: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            identities.setdefault(row["input_identity"], []).append(row)
        if definition == "exposure_weighted":
            row_ids = [int(row["compact_row"]) for row in selected]
            pol = float(np.mean([policy_rows[i] for i in row_ids]))
            mse = float(np.mean([value_rows[i] for i in row_ids]))
        else:
            grouped = [
                [int(row["compact_row"]) for row in group]
                for group in identities.values()
            ]
            pol = float(
                np.mean([np.mean([policy_rows[i] for i in group]) for group in grouped])
            )
            mse = float(
                np.mean([np.mean([value_rows[i] for i in group]) for group in grouped])
            )
        result[definition] = {"policy_ce": pol, "value_mse": mse}
    return result


def membership_rows() -> list[dict[str, Any]]:
    path = ROOT / RECEIPTS["membership"][0]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    return rows


def decide(results: dict[str, Any]) -> dict[str, Any]:
    base, a, b = (results[key] for key in ("initializer", "A", "B"))
    checks = {
        f"B_minus_A_{weighting}_policy": b[weighting]["policy_ce"]
        - a[weighting]["policy_ce"]
        <= -0.01
        for weighting in ("exposure_weighted", "equal_input")
    }
    checks.update(
        {
            f"B_minus_initializer_{weighting}_policy": b[weighting]["policy_ce"]
            - base[weighting]["policy_ce"]
            <= -0.005
            for weighting in ("exposure_weighted", "equal_input")
        }
    )
    checks["training_objective_decreases"] = (
        b["full_training_objective"] < base["full_training_objective"]
    )
    checks.update(
        {
            f"value_mse_guard_{weighting}": b[weighting]["value_mse"]
            <= base[weighting]["value_mse"] + 0.002
            for weighting in ("exposure_weighted", "equal_input")
        }
    )
    return {
        "classification": "advance_to_separately_preregistered_strength_experiment"
        if all(checks.values())
        else "close_optimizer_direction_branch",
        "checks": checks,
    }


def register() -> dict[str, Any]:
    arrays = _inputs()
    freeze = (
        ROOT
        / "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz"
    )
    with gzip.open(freeze, "rt") as stream:
        permutation = json.load(stream)[0]
    protocol = {
        "schema": "seed435-registration-v1",
        "status": "frozen_before_either_arm",
        "interpretation": "retrospective validation screen; cannot establish game-strength improvement or justify promotion",
        "initializer_sha256": INIT_SHA,
        "seed416_registration_sha256": "4839ed63d6a48945ea11ea4995570f11118adbd65c027f49c3c5f04066bbd066",
        "split_sha256": "f7b69ab8fd6c19128a29f9ae9a11a7ac42a5d26630368ff74e906f29ddb6be44",
        "permutation_sha256": "d868edebd9ae1876cbc5b9edca6fa0df9d3e6184b97246847f9012dfe2d293d1",
        "seed427_bindings": {
            key: {"path": val[0], "sha256": val[1]} for key, val in RECEIPTS.items()
        },
        "sources": [
            {"name": item["name"], "weight": item["weight"], "sha256": item["sha256"]}
            for item in json.loads(
                (
                    ROOT
                    / "docs/data/seed416-policy-target-softening/registration-v3.json"
                ).read_text()
            )["replays"]
        ],
        "configuration": {
            "model": "residual_v3",
            "hidden_sizes": [96, 3],
            "trainable": "all",
            "batch_size": 512,
            "updates_per_arm": 16,
            "lr_A": 0.001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0,
            "scheduler": "none",
            "clip": 1,
            "loss": "legal masked policy CE + 0.3*Huber(delta=1)",
        },
        "zero_moment": "reject positive requested radius; zero radius with zero direction is a no-op",
        "radius_tolerance": TOL,
        "decision": "B-A unseen policy CE <= -0.01 both weightings; B-initializer <= -0.005 both; B training objective decreases; B unseen value MSE <= initializer +0.002 both",
        "first_epoch_permutation_prefix": [
            int(value) for value in permutation[: 16 * 512]
        ],
        "train_positions": arrays[5][: 16 * 512].tolist(),
        "source_reconstruction_count": len(arrays[7]),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / "registration.json"
    if target.exists():
        raise ValueError("registration_is_immutable")
    target.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def run() -> dict[str, Any]:
    protocol_path = DATA / "registration.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["status"] != "frozen_before_either_arm":
        raise ValueError("registration_status_invalid")
    arrays = _inputs()
    x, p, v, replay, coeff, train_positions, _val, _sources = arrays
    train_replay = replay[train_positions]
    prefix = protocol["first_epoch_permutation_prefix"]
    if len(prefix) != 8192 or any(int(i) >= len(train_replay) for i in prefix):
        raise ValueError("registered_batch_prefix_invalid")
    membership = membership_rows()
    outcomes: dict[str, Any] = {"initializer": None}
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, INIT)
    outcomes["initializer"] = metrics(model, arrays, membership)
    init_payload = checkpoint_from_model(model)
    np.savez(DATA / "initializer.npz", **init_payload)
    radii: list[float] = []
    for arm in ("A", "B"):
        model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        load_checkpoint_into_model(model, INIT)
        adam = (
            torch.optim.Adam(
                model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
            if arm == "A"
            else None
        )
        moments = [torch.zeros_like(parameter) for _, parameter in parameters(model)]
        logs = []
        for step, start in enumerate(range(0, 8192, 512), 1):
            batch_ids = [int(i) for i in prefix[start : start + 512]]
            ids = train_replay[np.asarray(batch_ids, dtype=np.int64)]
            before = [value.detach().clone() for _, value in parameters(model)]
            if adam is not None:
                adam.zero_grad(set_to_none=True)
            else:
                for _, parameter in parameters(model):
                    parameter.grad = None
            xb = torch.from_numpy(x[ids])
            pb = torch.from_numpy(p[ids])
            vb = torch.from_numpy(v[ids])
            cb = torch.from_numpy(coeff[ids])
            mask = torch.from_numpy(legal_mask_matrix_for_encoded_states(x[ids]))
            logits, value_pred = model(xb)
            policy_loss = (
                compute_policy_cross_entropy(logits.masked_fill(mask <= 0, -1e9), pb)
                * cb
            ).sum() / cb.sum()
            loss = (
                policy_loss
                + 0.3
                * compute_value_loss_vector(
                    value_pred, vb, value_loss="huber", huber_delta=1
                ).mean()
            )
            loss.backward()
            raw = [
                parameter.grad.detach().clone() for _, parameter in parameters(model)
            ]
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in parameters(model)], 1.0
            )
            gradients = [
                parameter.grad.detach().clone() for _, parameter in parameters(model)
            ]
            gradient_norm = global_norm(gradients)
            if arm == "A":
                adam.step()
                deltas = [
                    parameter.detach() - old
                    for (_, parameter), old in zip(
                        parameters(model), before, strict=True
                    )
                ]
                radius = global_norm(deltas)
                radii.append(radius)
                moment_norm = None
                update_norm = radius
            else:
                moments = [
                    0.9 * old + 0.1 * grad
                    for old, grad in zip(moments, gradients, strict=True)
                ]
                direction = first_moment_direction(moments, step)
                dnorm = global_norm(direction)
                if dnorm == 0.0:
                    raise ValueError("positive_radius_with_zero_direction")
                radius = radii[step - 1]
                moment_norm, update_norm = matched_step(model, moments, radius, step)
            deltas_now = [
                parameter.detach() - old
                for (_, parameter), old in zip(parameters(model), before, strict=True)
            ]
            observed = global_norm(deltas_now)
            if not math.isclose(observed, radius, rel_tol=TOL, abs_tol=TOL):
                raise ValueError("stored_parameter_delta_radius_mismatch")
            logs.append(
                {
                    "step": step,
                    "batch": batch_record(batch_ids, train_replay, []),
                    "gradient_norm_preclip": global_norm(raw),
                    "gradient_norm_postclip": gradient_norm,
                    "first_moment_norm": moment_norm,
                    "adam_update_radius": radius if arm == "A" else None,
                    "requested_radius": radius,
                    "stored_delta_norm": observed,
                    "direction_cosine": None,
                    "parameter_group_update_norms": {
                        "trunk": global_norm(deltas_now[:8]),
                        "policy": global_norm(deltas_now[8:]),
                        "all": observed,
                    },
                    "batch_objective": float(loss.detach()),
                }
            )
        ckpt = DATA / f"{arm}-final.npz"
        np.savez(ckpt, **checkpoint_from_model(model))
        outcomes[arm] = metrics(model, arrays, membership)
        outcomes[arm]["checkpoint_sha256"] = sha(ckpt)
        (DATA / f"{arm}-updates.json").write_text(json.dumps(logs, indent=2) + "\n")
    results = {
        "schema": "seed435-results-v1",
        "registration_sha256": sha(protocol_path),
        "arms": outcomes,
        "decision": decide(outcomes),
    }
    (DATA / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    return results


def verify(root: Path = ROOT) -> dict[str, Any]:
    global ROOT, DATA, INIT
    ROOT = root.resolve()
    DATA = ROOT / "docs/data/seed435-adam-direction-screen"
    INIT = (
        ROOT
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    reg_path = DATA / "registration.json"
    reg = json.loads(reg_path.read_text())
    if sha(INIT) != INIT_SHA or reg["initializer_sha256"] != INIT_SHA:
        raise ValueError("initializer_binding_invalid")
    for arm in ("A", "B"):
        logs = json.loads((DATA / f"{arm}-updates.json").read_text())
        if len(logs) != 16 or [row["step"] for row in logs] != list(range(1, 17)):
            raise ValueError(f"update_accounting_invalid:{arm}")
        expected = reg["first_epoch_permutation_prefix"]
        flattened = [i for row in logs for i in row["batch"]["expanded_positions"]]
        if flattened != expected:
            raise ValueError(f"batch_order_invalid:{arm}")
        for row in logs:
            if not math.isclose(
                row["requested_radius"],
                row["stored_delta_norm"],
                rel_tol=reg["radius_tolerance"],
                abs_tol=reg["radius_tolerance"],
            ):
                raise ValueError(f"delta_radius_invalid:{arm}")
    result = json.loads((DATA / "results.json").read_text())
    if result["registration_sha256"] != sha(reg_path) or result["decision"] != decide(
        result["arms"]
    ):
        raise ValueError("results_or_decision_binding_invalid")
    return {
        "status": "valid",
        "classification": result["decision"]["classification"],
        "updates": {arm: 16 for arm in ("A", "B")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "run", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "register":
        print(json.dumps(register(), indent=2, sort_keys=True))
    elif args.command == "run":
        print(json.dumps(run(), indent=2, sort_keys=True))
    else:
        print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
