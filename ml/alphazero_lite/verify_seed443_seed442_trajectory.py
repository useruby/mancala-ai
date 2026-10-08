"""Append-only semantic verification of the published seed442 trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite import train
from ml.alphazero_lite.verify_seed442_publication import verify as verify_publication


def _hash(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.asarray(value, dtype="<f4").tobytes())
    return digest.hexdigest()


def _norm(values: list[np.ndarray]) -> float:
    return math.sqrt(
        sum(float(np.sum(np.asarray(v, dtype=np.float64) ** 2)) for v in values)
    )


def _same(actual: float, recorded: float, label: str) -> None:
    if not math.isclose(actual, recorded, rel_tol=2e-6, abs_tol=2e-7):
        raise ValueError(f"{label}_mismatch")


def _validate_link(
    previous: list[np.ndarray], current: list[np.ndarray], label: str
) -> None:
    if len(previous) != len(current) or any(
        not np.array_equal(a, b) for a, b in zip(previous, current, strict=True)
    ):
        raise ValueError(f"{label}_invalid")


def _validate_trials(
    trials: list[dict[str, Any]],
    scales: list[float],
    cap: float,
    tolerance: float,
    selected: float | None,
    rejected: bool,
    label: str,
) -> None:
    if [float(t["scale"]) for t in trials] != [float(s) for s in scales[: len(trials)]]:
        raise ValueError(f"{label}_order_invalid")
    first_pass = next(
        (
            t
            for t in trials
            if t["batch_kl"] <= cap + tolerance and t["guard_kl"] <= cap + tolerance
        ),
        None,
    )
    if first_pass is None:
        if len(trials) != len(scales) or selected is not None or not rejected:
            raise ValueError(f"{label}_rejection_coverage_invalid")
    elif selected != first_pass["scale"] or rejected:
        raise ValueError(f"{label}_first_pass_invalid")


def validate_semantics(root: Path) -> dict[str, Any]:
    """Check trajectory invariants independently of the publication receipt."""
    root = root.resolve()
    seed442.configure_root(root)
    out = seed442.OUT
    registration = json.loads((out / "registration.json").read_text())
    evidence = json.loads((out / "evidence.json").read_text())
    arrays = seed442.load_inputs()
    train_positions, validation_positions = arrays[5], arrays[6]
    for name, positions, expected in (
        ("train", train_positions, registration["train_positions_sha256"]),
        (
            "validation",
            validation_positions,
            registration["validation_positions_sha256"],
        ),
    ):
        actual = hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}_positions_hash_mismatch")
    permutation_path = (
        root
        / "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz"
    )
    import gzip

    with gzip.open(permutation_path, "rt", encoding="utf-8") as stream:
        frozen_prefix = json.load(stream)[0][:8192]
    if [int(i) for i in frozen_prefix] != registration["first_16_minibatch_positions"]:
        raise ValueError("registered_position_prefix_mismatch")
    model = train.PolicyValueNet((96, 3), "residual_v3", arrays[0].shape[1])
    derived_steps: list[dict[str, Any]] = []
    parameters = list(model.named_parameters())
    if any(not p.requires_grad for _, p in parameters):
        raise ValueError("unexpected_non_trainable_parameter")
    expected_names = [name for name, _ in parameters]
    expected_shapes = [tuple(p.shape) for _, p in parameters]
    expected_dtypes = [p.detach().cpu().numpy().dtype for _, p in parameters]
    with (
        np.load(out / "step-tensors.npz", allow_pickle=False) as archive,
        np.load(
            out / "optimizer-state-reconstruction.npz", allow_pickle=False
        ) as reconstructed,
    ):
        keys = set(archive.files)
        expected_keys = {
            f"{arm}_{step:02d}_{kind}_{index:02d}"
            for arm in ("A", "B")
            for step in range(16)
            for kind in (
                "pre",
                "proposal",
                "post",
                "gradient",
                "exp_avg_pre",
                "exp_avg_sq_pre",
                "exp_avg",
                "exp_avg_sq",
            )
            for index in range(len(parameters))
        }
        if keys != expected_keys:
            raise ValueError("step_tensor_coverage_invalid")
        reconstructed_keys = {
            f"{arm}_{step:02d}_{kind}_{index:02d}"
            for arm in ("A", "B")
            for step in range(16)
            for kind in ("exp_avg_post", "exp_avg_sq_post")
            for index in range(len(parameters))
        }
        if set(reconstructed.files) != reconstructed_keys:
            raise ValueError("reconstructed_optimizer_coverage_invalid")
        initialized = train.PolicyValueNet((96, 3), "residual_v3", arrays[0].shape[1])
        train.load_checkpoint_into_model(initialized, seed442.seed435.INIT)
        initial = [p.detach().cpu().numpy().copy() for p in initialized.parameters()]
        for arm in ("A", "B"):
            steps = evidence["arms"][arm]["steps"]
            if len(steps) != 16:
                raise ValueError(f"step_count_invalid:{arm}")
            prior_post = initial
            previous_m = [np.zeros_like(p) for p in initial]
            previous_v = [np.zeros_like(p) for p in initial]
            for step, record in enumerate(steps):
                if record["step"] != step + 1 or record["optimizer_step"] != step + 1:
                    raise ValueError(f"step_index_invalid:{arm}:{step + 1}")
                if [record["step"] for record in steps].count(step + 1) != 1:
                    raise ValueError(f"duplicate_step:{arm}:{step + 1}")
                pre = [
                    archive[f"{arm}_{step:02d}_pre_{i:02d}"]
                    for i in range(len(parameters))
                ]
                proposal = [
                    archive[f"{arm}_{step:02d}_proposal_{i:02d}"]
                    for i in range(len(parameters))
                ]
                post = [
                    archive[f"{arm}_{step:02d}_post_{i:02d}"]
                    for i in range(len(parameters))
                ]
                grads = [
                    archive[f"{arm}_{step:02d}_gradient_{i:02d}"]
                    for i in range(len(parameters))
                ]
                for kind, tensors in (
                    ("pre", pre),
                    ("proposal", proposal),
                    ("post", post),
                    ("gradient", grads),
                ):
                    for i, tensor in enumerate(tensors):
                        if (
                            tensor.shape != expected_shapes[i]
                            or tensor.dtype != expected_dtypes[i]
                            or not np.isfinite(tensor).all()
                        ):
                            raise ValueError(
                                f"tensor_metadata_invalid:{arm}:{step + 1}:{kind}:{expected_names[i]}"
                            )
                _validate_link(
                    prior_post, pre, f"parameter_state_link:{arm}:{step + 1}"
                )
                if (
                    record["pre_hash"] != _hash(pre)
                    or record["proposal_hash"] != _hash(proposal)
                    or record["accepted_hash"] != _hash(post)
                ):
                    raise ValueError(f"parameter_hash_invalid:{arm}:{step + 1}")
                _same(
                    _norm(pre),
                    record["parameter_norm"],
                    f"parameter_norm:{arm}:{step + 1}",
                )
                _same(
                    _norm([b - a for a, b in zip(pre, proposal, strict=True)]),
                    record["proposal_norm"],
                    f"proposal_norm:{arm}:{step + 1}",
                )
                _same(
                    _norm([b - a for a, b in zip(pre, post, strict=True)]),
                    record["accepted_update_norm"],
                    f"accepted_norm:{arm}:{step + 1}",
                )
                m_next, v_next = [], []
                for i, gradient in enumerate(grads):
                    m_pre = archive[f"{arm}_{step:02d}_exp_avg_pre_{i:02d}"]
                    v_pre = archive[f"{arm}_{step:02d}_exp_avg_sq_pre_{i:02d}"]
                    if not np.array_equal(m_pre, previous_m[i]) or not np.array_equal(
                        v_pre, previous_v[i]
                    ):
                        raise ValueError(
                            f"optimizer_state_link_invalid:{arm}:{step + 1}:{i}"
                        )
                    m = (
                        torch.from_numpy(m_pre.copy()).lerp_(
                            torch.from_numpy(gradient.copy()), 0.1
                        )
                    ).numpy()
                    v_tensor = (
                        torch.from_numpy(v_pre.copy())
                        .mul_(0.999)
                        .addcmul_(
                            torch.from_numpy(gradient.copy()),
                            torch.from_numpy(gradient.copy()),
                            value=0.001,
                        )
                    )
                    v = v_tensor.numpy()
                    if not np.array_equal(
                        reconstructed[f"{arm}_{step:02d}_exp_avg_post_{i:02d}"], m
                    ) or not np.array_equal(
                        reconstructed[f"{arm}_{step:02d}_exp_avg_sq_post_{i:02d}"], v
                    ):
                        raise ValueError(
                            f"optimizer_reconstruction_mismatch:{arm}:{step + 1}:{i}"
                        )
                    m_next.append(m.copy())
                    v_next.append(v.copy())
                if (
                    _hash(
                        [x for pair in zip(m_next, v_next, strict=True) for x in pair]
                    )
                    != record["moment_post_hash"]
                ):
                    raise ValueError(f"moment_post_hash_invalid:{arm}:{step + 1}")
                if arm == "A" and any(
                    not np.array_equal(a, b)
                    for a, b in zip(post, proposal, strict=True)
                ):
                    raise ValueError(f"ordinary_adam_step_invalid:{step + 1}")
                if arm == "B":
                    trials = record["trials"]
                    scales = registration["kl"]["scales"]
                    first_pass = next(
                        (
                            t
                            for t in trials
                            if t["batch_kl"]
                            <= registration["kl"]["cap"]
                            + registration["kl"]["acceptance_tolerance"]
                            and t["guard_kl"]
                            <= registration["kl"]["cap"]
                            + registration["kl"]["acceptance_tolerance"]
                        ),
                        None,
                    )
                    selected = record["selected_scale"]
                    _validate_trials(
                        trials,
                        scales,
                        registration["kl"]["cap"],
                        registration["kl"]["acceptance_tolerance"],
                        selected,
                        record["rejected"],
                        f"trials:{step + 1}",
                    )
                    if first_pass is None:
                        expected_post = pre
                    else:
                        expected_post = [
                            b
                            if selected == 1.0
                            else np.asarray(a + (b - a) * selected, dtype=np.float32)
                            for a, b in zip(pre, proposal, strict=True)
                        ]
                    if any(
                        not np.array_equal(a, b)
                        for a, b in zip(expected_post, post, strict=True)
                    ):
                        raise ValueError(f"accepted_parameters_invalid:{step + 1}")
                    first_failed = not bool(
                        trials[0]["batch_kl"]
                        <= registration["kl"]["cap"]
                        + registration["kl"]["acceptance_tolerance"]
                        and trials[0]["guard_kl"]
                        <= registration["kl"]["cap"]
                        + registration["kl"]["acceptance_tolerance"]
                    )
                    derived_steps.append(
                        {
                            "proposal_norm": _norm(
                                [b - a for a, b in zip(pre, proposal, strict=True)]
                            ),
                            "selected_scale": selected,
                            "rejected": first_pass is None,
                            "cap_activated": first_failed,
                            "trials": trials,
                        }
                    )
                prior_post = post
                previous_m, previous_v = m_next, v_next
            final = train.PolicyValueNet((96, 3), "residual_v3", arrays[0].shape[1])
            train.load_checkpoint_into_model(final, out / f"{arm}-final.npz")
            final_values = [p.detach().cpu().numpy() for p in final.parameters()]
            if any(
                not np.array_equal(a, b)
                for a, b in zip(prior_post, final_values, strict=True)
            ):
                raise ValueError(f"endpoint_tensor_mismatch:{arm}")
    recomputed = dict(evidence)
    recomputed["arms"] = dict(evidence["arms"])
    recomputed["arms"]["B"] = dict(evidence["arms"]["B"])
    recomputed["arms"]["B"]["steps"] = derived_steps
    decision = seed442.decide(recomputed)
    if decision != evidence["decision"]:
        raise ValueError("verified_decision_reproduction_mismatch")
    return {
        "status": "valid",
        "arms": {"A": 16, "B": 16},
        "parameter_names": expected_names,
        "decision": decision,
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    published = verify_publication(root)
    semantic = validate_semantics(root)
    receipt_path = root / "docs/data/seed443-seed442-trajectory/receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for group in (
        "historical_inputs_sha256",
        "seed443_source_sha256",
        "seed443_artifact_sha256",
    ):
        for relative, expected in receipt[group].items():
            path = root / relative
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(f"seed443_receipt_hash_mismatch:{relative}")
    return {**published, "semantic": semantic}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
