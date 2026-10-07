"""Post-execution deterministic telemetry recovery for seed435 (no new run)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml.alphazero_lite.seed435_adam_direction import (
    INIT,
    _inputs,
    batch_record,
    first_moment_direction,
    global_norm,
    load_checkpoint_into_model,
    parameters,
)
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
)


def _group(name: str) -> str:
    if name.startswith(("input_layer.", "residual_layers.")):
        return "shared_trunk"
    if name.startswith(("policy_hidden_layer.", "policy_head.")):
        return "policy_head"
    if name.startswith(("value_hidden_layer.", "value_head.")):
        return "value_head"
    raise ValueError(f"unassigned_parameter:{name}")


def recover(root: Path) -> dict:
    data = root / "docs/data/seed435-adam-direction-screen"
    arrays = _inputs()
    x, policy, value, replay, coefficients, train_positions, *_ = arrays
    train_replay = replay[train_positions]
    registration = json.loads((data / "registration.json").read_text())
    prefix = registration["first_epoch_permutation_prefix"]
    all_deltas: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    telemetry: dict[str, list[dict]] = {"A": [], "B": []}
    a_radii: list[float] = []
    for arm in ("A", "B"):
        model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        load_checkpoint_into_model(model, INIT)
        named = parameters(model)
        names = [name for name, _ in named]
        layout = []
        flat_offset = 0
        for name, parameter in named:
            size = parameter.numel()
            layout.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "start": flat_offset,
                    "stop": flat_offset + size,
                    "group": _group(name),
                }
            )
            flat_offset += size
        groups = {
            group: [i for i, name in enumerate(names) if _group(name) == group]
            for group in ("shared_trunk", "policy_head", "value_head")
        }
        if sorted(i for indices in groups.values() for i in indices) != list(
            range(len(names))
        ):
            raise ValueError("parameter_group_layout_not_exhaustive")
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
        moment = [torch.zeros_like(parameter) for _, parameter in named]
        for step in range(1, 17):
            batch_ids = [int(i) for i in prefix[(step - 1) * 512 : step * 512]]
            row_ids = train_replay[np.asarray(batch_ids, dtype=np.int64)]
            before = [parameter.detach().clone() for _, parameter in named]
            if adam is not None:
                adam.zero_grad(set_to_none=True)
            else:
                for _, parameter in named:
                    parameter.grad = None
            xb = torch.from_numpy(x[row_ids])
            pb = torch.from_numpy(policy[row_ids])
            vb = torch.from_numpy(value[row_ids])
            cb = torch.from_numpy(coefficients[row_ids])
            mask = torch.from_numpy(legal_mask_matrix_for_encoded_states(x[row_ids]))
            logits, predicted_value = model(xb)
            policy_loss = (
                compute_policy_cross_entropy(logits.masked_fill(mask <= 0, -1e9), pb)
                * cb
            ).sum() / cb.sum()
            loss = (
                policy_loss
                + 0.3
                * compute_value_loss_vector(
                    predicted_value, vb, value_loss="huber", huber_delta=1.0
                ).mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
            grads = [parameter.grad.detach().clone() for _, parameter in named]
            if arm == "A":
                adam.step()
            else:
                moment = [
                    0.9 * old + 0.1 * grad
                    for old, grad in zip(moment, grads, strict=True)
                ]
                direction = first_moment_direction(moment, step)
                radius = float(
                    json.loads((data / "A-updates.json").read_text())[step - 1][
                        "stored_delta_norm"
                    ]
                )
                norm = global_norm(direction)
                if norm == 0:
                    raise ValueError("zero_moment_during_recovery")
                scale = radius / norm
                with torch.no_grad():
                    for (_, parameter), vector in zip(named, direction, strict=True):
                        parameter.add_(-scale * vector)
            delta = [
                parameter.detach() - old
                for (_, parameter), old in zip(named, before, strict=True)
            ]
            vector = (
                torch.cat([item.reshape(-1).cpu() for item in delta])
                .numpy()
                .astype(np.float32)
            )
            all_deltas[arm].append(vector)
            norm = global_norm(delta)
            if arm == "A":
                a_radii.append(norm)
            record = {
                "step": step,
                "batch": batch_record(batch_ids, train_replay, []),
                "update_norm": norm,
                "group_norms": {
                    group: global_norm([delta[index] for index in indexes])
                    for group, indexes in groups.items()
                },
                "batch_objective": float(loss.detach()),
            }
            telemetry[arm].append(record)
        expected_checkpoint = data / f"{arm}-final.npz"
        recovered = checkpoint_from_model(model)
        with np.load(expected_checkpoint, allow_pickle=False) as expected:
            if set(expected.files) != set(recovered) or any(
                not np.array_equal(expected[key], recovered[key])
                for key in expected.files
            ):
                differences = {
                    key: float(np.max(np.abs(expected[key] - recovered[key])))
                    for key in expected.files
                    if key in recovered
                    and not np.array_equal(expected[key], recovered[key])
                }
                raise ValueError(
                    f"audit_reproduction_final_checkpoint_mismatch:{arm}:{differences}"
                )
        np.savez_compressed(
            data / f"{arm}-audit-deltas.npz", deltas=np.stack(all_deltas[arm])
        )
    for step, (a, b) in enumerate(
        zip(all_deltas["A"], all_deltas["B"], strict=True), 1
    ):
        na = float(np.linalg.norm(a.astype(np.float64)))
        nb = float(np.linalg.norm(b.astype(np.float64)))
        cosine = (
            float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (na * nb))
            if na and nb
            else None
        )
        for arm, vector in (("A", a), ("B", b)):
            telemetry[arm][step - 1]["direction_cosine_vs_paired_arm"] = cosine
            telemetry[arm][step - 1]["cosine_undefined_zero_vector"] = cosine is None
            telemetry[arm][step - 1]["delta_vector_norm"] = float(
                np.linalg.norm(vector.astype(np.float64))
            )
    np.savez_compressed(
        data / "A-B-audit-delta-cosines.npz",
        cosines=np.asarray(
            [
                row["direction_cosine_vs_paired_arm"]
                if row["direction_cosine_vs_paired_arm"] is not None
                else np.nan
                for row in telemetry["A"]
            ],
            dtype=np.float64,
        ),
    )
    (data / "post-execution-telemetry.json").write_text(
        json.dumps(
            {
                "interpretation": "deterministic audit reproduction to recover omitted telemetry; not independent experimental evidence",
                "final_parameter_arrays_match_original": True,
                "parameter_layout": layout,
                "group_assignment": {entry["name"]: entry["group"] for entry in layout},
                "arms": telemetry,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {"final_parameter_arrays_match_original": True, "updates": 32}


if __name__ == "__main__":
    print(
        json.dumps(
            recover(Path(__file__).resolve().parents[2]), indent=2, sort_keys=True
        )
    )
