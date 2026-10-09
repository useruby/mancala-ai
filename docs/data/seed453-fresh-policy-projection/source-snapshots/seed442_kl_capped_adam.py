"""Prospective train-only KL-capped Adam screen (seed442)."""

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

from ml.alphazero_lite import seed435_adam_direction as seed435
from ml.alphazero_lite import train
from ml.alphazero_lite.seed429_policy_normalization import (
    canonical_identity_from_encoded_state,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed442-kl-capped-adam-screen"
INIT_SHA = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
SCALES = tuple(2.0**-index for index in range(8))
CAP = 0.005
KL_TOLERANCE = 1e-10
WEIGHTINGS = ("exposure_weighted", "equal_input")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_hash(values: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def norm(values: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> float:
    return math.sqrt(sum(float(torch.sum(value.double() ** 2)) for value in values))


def configure_root(root: Path) -> None:
    global ROOT, OUT
    ROOT = root.resolve()
    OUT = ROOT / "docs/data/seed442-kl-capped-adam-screen"
    seed435.ROOT = ROOT
    seed435.DATA = ROOT / "docs/data/seed435-adam-direction-screen"
    seed435.INIT = (
        ROOT
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )


def load_inputs() -> tuple[np.ndarray, ...]:
    if sha(seed435.INIT) != INIT_SHA:
        raise ValueError("initializer_sha256_mismatch")
    return seed435._inputs()


def validate_registration_bindings(protocol: dict[str, Any]) -> None:
    for relative, expected in protocol["frozen_input_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"frozen_input_hash_mismatch:{relative}")
    for relative, expected in protocol["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"execution_source_hash_mismatch:{relative}")


def _active_stones(state: np.ndarray) -> int:
    return int(round(float(np.sum(state[:12], dtype=np.float64) * 48.0)))


def guard_cohort(root: Path, arrays: tuple[np.ndarray, ...]) -> list[dict[str, Any]]:
    x, _p, _v, replay, _coeff, train_positions, _val_positions, reconstructed = arrays
    training = replay[train_positions]
    first_by_identity: dict[str, int] = {}
    for compact_row in sorted(set(map(int, training))):
        state = x[compact_row]
        if _active_stones(state) <= 32:
            continue
        identity = state.astype("<f4", copy=False).tobytes().hex()
        first_by_identity.setdefault(identity, compact_row)
    membership_path = root / seed435.RECEIPTS["membership"][0]
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        unseen = [
            row
            for line in stream
            if (row := json.loads(line))["subset"] == "unseen"
            and row["active_stones"] > 32
        ]
    unseen_exact = {row["input_identity"] for row in unseen}
    unseen_canonical = {row["canonical_identity"] for row in unseen}
    candidates = []
    for identity, compact_row in first_by_identity.items():
        canonical = canonical_identity_from_encoded_state(x[compact_row].tolist())
        if identity in unseen_exact or canonical in unseen_canonical:
            continue
        digest = hashlib.sha256(f"seed442-guard:{identity}".encode()).hexdigest()
        source = reconstructed[compact_row]
        candidates.append(
            {
                "selection_sha256": digest,
                "exact_input_identity": identity,
                "canonical_identity": canonical,
                "compact_row": compact_row,
                "source": source.get("source"),
                "source_row": source.get("line"),
                "active_stones": _active_stones(x[compact_row]),
            }
        )
    candidates.sort(
        key=lambda item: (item["selection_sha256"], item["exact_input_identity"])
    )
    selected = candidates[:2048]
    if len(selected) != 2048:
        raise ValueError(f"guard_cohort_too_small:{len(selected)}")
    if {r["exact_input_identity"] for r in selected} & unseen_exact:
        raise ValueError("guard_exact_overlap")
    if {r["canonical_identity"] for r in selected} & unseen_canonical:
        raise ValueError("guard_canonical_overlap")
    return selected


def register(root: Path) -> dict[str, Any]:
    configure_root(root)
    arrays = load_inputs()
    x, _p, _v, _replay, _coeff, _train_pos, _val_pos, reconstructed = arrays
    cohort = guard_cohort(ROOT, arrays)
    freeze = ROOT / "docs/data/seed416-policy-target-softening/training-freeze-v3"
    registration = (
        ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    split = freeze / "source-row-split.json.gz"
    permutation = freeze / "epoch-permutations.json.gz"
    with gzip.open(permutation, "rt", encoding="utf-8") as stream:
        first_epoch = json.load(stream)[0]
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    cohort_path = out / "guard-cohort.json"
    reg_path = out / "registration.json"
    if reg_path.exists() or cohort_path.exists():
        raise ValueError("registration_is_immutable")
    cohort_path.write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_paths = [
        "ml/alphazero_lite/seed442_kl_capped_adam.py",
        "ml/alphazero_lite/seed442_analysis.py",
        "ml/alphazero_lite/verify_seed442_kl_capped_adam.py",
        "ml/alphazero_lite/test_seed442_kl_capped_adam.py",
        "ml/alphazero_lite/verify_seed441_receipt_hashes.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/seed435_adam_direction.py",
        "ml/alphazero_lite/seed429_policy_normalization.py",
        "ml/alphazero_lite/verify_seed434_census_source.py",
    ]
    frozen_files = [
        registration.relative_to(ROOT),
        split.relative_to(ROOT),
        permutation.relative_to(ROOT),
    ]
    missing_sources = [name for name in source_paths if not (ROOT / name).is_file()]
    if missing_sources:
        raise ValueError(f"registration_source_missing:{missing_sources}")
    source_hashes = {name: sha(ROOT / name) for name in source_paths}
    digest_map = {str(path): sha(ROOT / path) for path in frozen_files}
    digest_map[
        "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    ] = INIT_SHA
    for relative in (
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "docs/data/seed435-adam-direction-screen/A-final.npz",
        "docs/data/seed435-adam-direction-screen/supplemental-receipt.json",
    ):
        digest_map[relative] = sha(ROOT / relative)
    for replay in json.loads(registration.read_text(encoding="utf-8"))["replays"]:
        relative = (
            f"docs/data/seed426-canonical-overlap/sources/{replay['name']}.jsonl.gz"
        )
        digest_map[relative] = sha(ROOT / relative)
    protocol = {
        "schema": "seed442-prospective-registration-v1",
        "status": "frozen_before_either_arm",
        "interpretation": "short retrospective-population screen; not evidence of playing strength",
        "initializer_sha256": INIT_SHA,
        "frozen_input_sha256": digest_map,
        "source_sha256": source_hashes,
        "source_reconstruction_count": len(reconstructed),
        "train_positions_sha256": hashlib.sha256(
            arrays[5].astype("<i8").tobytes()
        ).hexdigest(),
        "validation_positions_sha256": hashlib.sha256(
            arrays[6].astype("<i8").tobytes()
        ).hexdigest(),
        "first_16_minibatch_positions": [int(i) for i in first_epoch[:8192]],
        "configuration": {
            "architecture": "residual_v3",
            "hidden_sizes": [96, 3],
            "batch_size": 512,
            "proposals_per_arm": 16,
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "schedule": "none",
            "gradient_clip_norm": 1.0,
            "value_coefficient": 0.3,
            "objective": "policy CE weighted by row coefficient over coefficient sum + 0.3 * mean Huber(delta=1)",
        },
        "guard": {
            "count": len(cohort),
            "selection": "lowest SHA256(seed442-guard:<exact-input-sha256>)",
            "weighting": "equal unique input",
            "cohort_sha256": sha(cohort_path),
        },
        "kl": {
            "cap": CAP,
            "scales": list(SCALES),
            "logits_dtype": "float32",
            "log_softmax_dtype": "float64",
            "acceptance_tolerance": KL_TOLERANCE,
            "direction": "KL(pre-step policy || trial policy), legal actions only",
        },
        "trial_arithmetic": "scale 1 uses Adam proposal tensors directly; smaller scales use float32 pre + float32(scale * (proposal - pre)); restore pre-step tensors before every trial",
        "rejection": "keep pre-step parameters after one Adam advance; continue; any rejected nonzero proposal disqualifies advancement",
        "decision": {
            "B_minus_A_policy_ce": -0.01,
            "B_minus_initializer_policy_ce": -0.005,
            "B_full_training_objective_decreases": True,
            "B_value_mse_margin": 0.002,
            "require_nonzero_scaled_acceptance": True,
            "reject_nonzero_proposals": False,
        },
    }
    reg_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return protocol


def _flat_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def _set_trial(
    pre: list[torch.Tensor], proposal: list[torch.Tensor], scale: float
) -> None:
    with torch.no_grad():
        for param, before, proposed in zip(
            _CURRENT_PARAMETERS, pre, proposal, strict=True
        ):
            if scale == 1.0:
                param.copy_(proposed)
            else:
                param.copy_(before + (proposed - before) * scale)


_CURRENT_PARAMETERS: list[torch.Tensor] = []


def _kl(
    model: torch.nn.Module,
    x: np.ndarray,
    ids: np.ndarray,
    weights: np.ndarray,
    pre_log_probs: torch.Tensor,
) -> float:
    total = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), 512):
            rows = ids[start : start + 512]
            logits, _ = model(torch.from_numpy(x[rows]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[rows])
            ).bool()
            z = logits.double().masked_fill(~legal, -torch.inf)
            logp = torch.log_softmax(z, dim=1)
            pre_log = pre_log_probs[rows].to(torch.float64)
            pre_prob = pre_log.exp().masked_fill(~legal, 0.0)
            kl_rows = (pre_prob * (pre_log - logp).masked_fill(~legal, 0.0)).sum(dim=1)
            total += float(
                (
                    kl_rows
                    * torch.from_numpy(weights[start : start + len(rows)]).double()
                ).sum()
            )
    return total


def _pre_policy(model: torch.nn.Module, x: np.ndarray, ids: np.ndarray) -> torch.Tensor:
    values = torch.full((len(x), train.POLICY_SIZE), -torch.inf, dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, len(ids), 512):
            rows = ids[start : start + 512]
            logits, _ = model(torch.from_numpy(x[rows]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[rows])
            ).bool()
            values[rows] = torch.log_softmax(
                logits.double().masked_fill(~legal, -torch.inf), dim=1
            )
    return values


def _write_archive(path: Path, tensors: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **tensors)


def select_largest_scale(
    trials: list[tuple[float, float, float]],
    *,
    cap: float = CAP,
    tolerance: float = KL_TOLERANCE,
) -> float | None:
    """Choose the first registered descending scale satisfying both caps."""
    for scale, batch_kl, guard_kl in trials:
        if batch_kl <= cap + tolerance and guard_kl <= cap + tolerance:
            return scale
    return None


def run(root: Path) -> dict[str, Any]:
    configure_root(root)
    protocol_path = OUT / "registration.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_either_arm":
        raise ValueError("registration_not_frozen")
    validate_registration_bindings(protocol)
    arrays = load_inputs()
    x, p, v, replay, coefficients, train_positions, _val_positions, _source_rows = (
        arrays
    )
    batch_order = np.asarray(protocol["first_16_minibatch_positions"], dtype=np.int64)
    train_replay = replay[train_positions]
    cohort = json.loads((OUT / "guard-cohort.json").read_text(encoding="utf-8"))
    guard_rows = np.asarray([row["compact_row"] for row in cohort], dtype=np.int64)
    membership = seed435.membership_rows()
    evidence: dict[str, Any] = {
        "schema": "seed442-evidence-v1",
        "registration_sha256": sha(protocol_path),
        "arms": {},
        "metrics": {},
    }
    tensor_archive: dict[str, np.ndarray] = {}
    template = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(template, seed435.INIT)
    init_model = template
    evidence["metrics"]["initializer"] = _endpoint_metrics(
        init_model, arrays, membership
    )
    for arm in ("A", "B"):
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(model, seed435.INIT)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        _CURRENT_PARAMETERS[:] = _flat_parameters(model)
        logs = []
        for step in range(16):
            expanded = batch_order[step * 512 : (step + 1) * 512]
            ids = train_replay[expanded]
            pre = [parameter.detach().clone() for parameter in _CURRENT_PARAMETERS]
            batch_pre = _pre_policy(model, x, ids) if arm == "B" else None
            guard_pre = _pre_policy(model, x, guard_rows) if arm == "B" else None
            moment_before = [
                (
                    optimizer.state[p]["exp_avg"].detach().clone(),
                    optimizer.state[p]["exp_avg_sq"].detach().clone(),
                )
                if "exp_avg" in optimizer.state[p]
                else (torch.zeros_like(p), torch.zeros_like(p))
                for p in _CURRENT_PARAMETERS
            ]
            optimizer.zero_grad(set_to_none=True)
            bx = torch.from_numpy(x[ids])
            legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ids]))
            logits, value_pred = model(bx)
            policy_loss = (
                train.compute_policy_cross_entropy(
                    logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[ids])
                )
                * torch.from_numpy(coefficients[ids])
            ).sum() / torch.from_numpy(coefficients[ids]).sum()
            value_loss = train.compute_value_loss_vector(
                value_pred,
                torch.from_numpy(v[ids]),
                value_loss="huber",
                huber_delta=1.0,
            ).mean()
            loss = policy_loss + 0.3 * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_CURRENT_PARAMETERS, 1.0)
            clipped_gradients = [p.grad.detach().clone() for p in _CURRENT_PARAMETERS]
            optimizer.step()
            proposal = [parameter.detach().clone() for parameter in _CURRENT_PARAMETERS]
            proposal_norm = norm(
                [new - old for new, old in zip(proposal, pre, strict=True)]
            )
            moment_proposed = [
                (
                    optimizer.state[p]["exp_avg"].detach().clone(),
                    optimizer.state[p]["exp_avg_sq"].detach().clone(),
                )
                for p in _CURRENT_PARAMETERS
            ]
            row: dict[str, Any] = {
                "step": step + 1,
                "expanded_positions": expanded.tolist(),
                "compact_rows": ids.tolist(),
                "batch_loss": float(loss.detach()),
                "proposal_norm": proposal_norm,
                "parameter_norm": norm(pre),
                "pre_hash": vector_hash(pre),
                "proposal_hash": vector_hash(proposal),
                "moment_pre_hash": vector_hash(
                    [item for pair in moment_before for item in pair]
                ),
                "moment_post_hash": vector_hash(
                    [item for pair in moment_proposed for item in pair]
                ),
                "optimizer_step": step + 1,
                "trials": [],
            }
            accepted_scale: float | None = None
            if arm == "A":
                _set_trial(pre, proposal, 1.0)
                accepted_scale = 1.0
                row["trials"].append(
                    {
                        "scale": 1.0,
                        "batch_kl": 0.0,
                        "guard_kl": 0.0,
                        "accepted": True,
                        "ordinary_adam": True,
                    }
                )
            else:
                assert batch_pre is not None and guard_pre is not None
                batch_weights = np.full(len(ids), 1.0 / len(ids), dtype=np.float64)
                tested: list[tuple[float, float, float]] = []
                for scale in SCALES:
                    _set_trial(pre, proposal, scale)
                    batch_kl = _kl(model, x, ids, batch_weights, batch_pre)
                    guard_kl = _kl(
                        model,
                        x,
                        guard_rows,
                        np.full(
                            len(guard_rows), 1.0 / len(guard_rows), dtype=np.float64
                        ),
                        guard_pre,
                    )
                    accepted = (
                        batch_kl <= CAP + KL_TOLERANCE
                        and guard_kl <= CAP + KL_TOLERANCE
                    )
                    tested.append((scale, batch_kl, guard_kl))
                    row["trials"].append(
                        {
                            "scale": scale,
                            "batch_kl": batch_kl,
                            "guard_kl": guard_kl,
                            "accepted": accepted,
                        }
                    )
                    if accepted:
                        accepted_scale = scale
                        break
                if accepted_scale != select_largest_scale(tested):
                    raise ValueError("largest_scale_selection_mismatch")
                if accepted_scale is None:
                    with torch.no_grad():
                        for parameter, before in zip(
                            _CURRENT_PARAMETERS, pre, strict=True
                        ):
                            parameter.copy_(before)
            post = [parameter.detach().clone() for parameter in _CURRENT_PARAMETERS]
            row.update(
                {
                    "selected_scale": accepted_scale,
                    "accepted_hash": vector_hash(post),
                    "accepted_update_norm": norm(
                        [new - old for new, old in zip(post, pre, strict=True)]
                    ),
                    "rejected": accepted_scale is None,
                }
            )
            logs.append(row)
            for prefix, values in (
                ("pre", pre),
                ("proposal", proposal),
                ("post", post),
            ):
                for index, value in enumerate(values):
                    tensor_archive[f"{arm}_{step:02d}_{prefix}_{index:02d}"] = (
                        value.cpu().numpy()
                    )
            for index, parameter in enumerate(_CURRENT_PARAMETERS):
                state = optimizer.state[parameter]
                tensor_archive[f"{arm}_{step:02d}_gradient_{index:02d}"] = (
                    clipped_gradients[index].cpu().numpy()
                )
                tensor_archive[f"{arm}_{step:02d}_exp_avg_pre_{index:02d}"] = (
                    moment_before[index][0].cpu().numpy()
                )
                tensor_archive[f"{arm}_{step:02d}_exp_avg_sq_pre_{index:02d}"] = (
                    moment_before[index][1].cpu().numpy()
                )
                tensor_archive[f"{arm}_{step:02d}_exp_avg_{index:02d}"] = (
                    state["exp_avg"].detach().cpu().numpy()
                )
                tensor_archive[f"{arm}_{step:02d}_exp_avg_sq_{index:02d}"] = (
                    state["exp_avg_sq"].detach().cpu().numpy()
                )
        if arm == "A":
            expected = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            train.load_checkpoint_into_model(expected, seed435.DATA / "A-final.npz")
            if any(
                not torch.equal(a, b)
                for a, b in zip(model.parameters(), expected.parameters(), strict=True)
            ):
                raise ValueError("A_final_parameter_mismatch_seed435")
        checkpoint = OUT / f"{arm}-final.npz"
        np.savez_compressed(checkpoint, **train.checkpoint_from_model(model))
        evidence["arms"][arm] = {
            "steps": logs,
            "final_parameter_sha256": vector_hash(_CURRENT_PARAMETERS),
            "checkpoint_sha256": sha(checkpoint),
        }
        evidence["metrics"][arm] = _endpoint_metrics(model, arrays, membership)
    evidence["decision"] = decide(evidence)
    _write_archive(OUT / "step-tensors.npz", tensor_archive)
    _publish_predictions(OUT, init_model, arrays, "initializer")
    for arm in ("A", "B"):
        endpoint = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(endpoint, OUT / f"{arm}-final.npz")
        _publish_predictions(OUT, endpoint, arrays, arm)
    (OUT / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def _endpoint_metrics(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    membership: list[dict[str, Any]],
) -> dict[str, Any]:
    x, policy, value, replay, coefficients, train_positions, _val, _sources = arrays
    model.eval()
    with torch.inference_mode():
        logits_parts, values_parts = [], []
        for start in range(0, len(x), 512):
            logits, predicted = model(torch.from_numpy(x[start : start + 512]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[start : start + 512])
            )
            logits_parts.append(logits.masked_fill(legal <= 0, -1e9))
            values_parts.append(predicted.reshape(-1))
        logits = torch.cat(logits_parts)
        predictions = torch.cat(values_parts)
        policy_rows = (
            train.compute_policy_cross_entropy(logits, torch.from_numpy(policy))
            .double()
            .numpy()
        )
        mse_rows = (predictions.double().numpy() - value[:, 0].astype(np.float64)) ** 2
        huber = (
            train.compute_value_loss_vector(
                predictions.reshape(-1, 1),
                torch.from_numpy(value),
                value_loss="huber",
                huber_delta=1.0,
            )
            .double()
            .numpy()
        )
    expanded_training = replay[train_positions]
    train_objective = float(
        np.average(
            policy_rows[expanded_training], weights=coefficients[expanded_training]
        )
        + 0.3 * np.mean(huber[expanded_training])
    )
    unseen = [
        row
        for row in membership
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    groups: dict[str, list[int]] = {}
    for row in unseen:
        groups.setdefault(row["input_identity"], []).append(int(row["compact_row"]))
    row_ids = [int(row["compact_row"]) for row in unseen]
    return {
        "full_training_objective": train_objective,
        "exposure_weighted": {
            "policy_ce": float(np.mean(policy_rows[row_ids])),
            "value_mse": float(np.mean(mse_rows[row_ids])),
        },
        "equal_input": {
            "policy_ce": float(
                np.mean([np.mean(policy_rows[rows]) for rows in groups.values()])
            ),
            "value_mse": float(
                np.mean([np.mean(mse_rows[rows]) for rows in groups.values()])
            ),
        },
    }


def _publish_predictions(
    out: Path, model: torch.nn.Module, arrays: tuple[np.ndarray, ...], label: str
) -> None:
    x = arrays[0]
    selected = [
        row
        for row in seed435.membership_rows()
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    ids = np.asarray([int(row["compact_row"]) for row in selected], dtype=np.int64)
    logits_parts, value_parts = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), 512):
            batch = ids[start : start + 512]
            logits, predicted = model(torch.from_numpy(x[batch]))
            logits_parts.append(logits.cpu().numpy())
            value_parts.append(predicted.reshape(-1).cpu().numpy())
    np.savez_compressed(
        out / f"{label}-predictions.npz",
        compact_rows=ids,
        input_identity=np.asarray([row["input_identity"] for row in selected]),
        policy_logits=np.concatenate(logits_parts),
        value_predictions=np.concatenate(value_parts),
    )


def decide(evidence: dict[str, Any]) -> dict[str, Any]:
    metrics = evidence["metrics"]
    init, a, b = (metrics[key] for key in ("initializer", "A", "B"))
    checks: dict[str, bool] = {}
    for weighting in WEIGHTINGS:
        checks[f"B_minus_A_{weighting}_policy"] = (
            b[weighting]["policy_ce"] - a[weighting]["policy_ce"] <= -0.01
        )
        checks[f"B_minus_initializer_{weighting}_policy"] = (
            b[weighting]["policy_ce"] - init[weighting]["policy_ce"] <= -0.005
        )
        checks[f"B_{weighting}_value_guard"] = (
            b[weighting]["value_mse"] <= init[weighting]["value_mse"] + 0.002
        )
    checks["B_training_objective_decreases"] = (
        b["full_training_objective"] < init["full_training_objective"]
    )
    steps = evidence.get("arms", {}).get("B", {}).get("steps", [])
    checks["scaled_nonzero_proposal_accepted"] = any(
        row["proposal_norm"] > 0 and row["selected_scale"] not in (None, 1.0)
        for row in steps
    )
    checks["no_nonzero_proposal_rejected"] = not any(
        row["proposal_norm"] > 0 and row["rejected"] for row in steps
    )
    cap_activated = any(
        row.get("proposal_norm", 0) > 0
        and row.get("trials")
        and not row["trials"][0]["accepted"]
        for row in steps
    )
    classification = (
        "cap_inactive_no_followup"
        if not cap_activated
        else "advance_to_separately_preregistered_strength_experiment"
        if all(checks.values())
        else "close_kl_capped_step_branch"
    )
    return {"classification": classification, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = register(args.root) if args.command == "register" else run(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
