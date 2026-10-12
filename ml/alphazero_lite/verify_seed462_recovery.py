"""Independent retrospective reconstruction of the archived seed462 attempt."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed435_adam_direction as data_loader
from ml.alphazero_lite import train

INIT = "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
ATTEMPT = "docs/data/seed462-historical-policy-removal"
CONTROL = "docs/data/seed442-kl-capped-adam-screen"
RECOVERY = "docs/data/seed462-historical-policy-removal-recovery"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten_hash(values: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return h.hexdigest()


def require_array_equal(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(actual, expected):
        raise ValueError(f"semantic_mismatch:{label}")


def validate_endpoint_semantics(endpoint: dict[str, Any]) -> None:
    for cohort in ("pooled", "fresh"):
        for weighting in ("exposure_weighted", "equal_input"):
            for metric in ("policy_ce", "value_mse"):
                expected = (
                    endpoint["arms"]["F"][cohort][weighting][metric]
                    - endpoint["arms"]["C"][cohort][weighting][metric]
                )
                observed = endpoint["effects_F_minus_C"][cohort][weighting][metric]
                if observed != expected:
                    raise ValueError(
                        f"semantic_mismatch:effect:{cohort}:{weighting}:{metric}"
                    )
    for weighting in ("exposure_weighted", "equal_input"):
        arms = endpoint["arms"]
        expected_gates = {
            f"fresh_vs_initializer_{weighting}": arms["F"]["fresh"][weighting][
                "policy_ce"
            ]
            <= arms["initializer"]["fresh"][weighting]["policy_ce"] - 0.005,
            f"fresh_vs_C_{weighting}": arms["F"]["fresh"][weighting]["policy_ce"]
            <= arms["C"]["fresh"][weighting]["policy_ce"] - 0.005,
            f"pooled_ce_{weighting}": arms["F"]["pooled"][weighting]["policy_ce"]
            <= arms["initializer"]["pooled"][weighting]["policy_ce"] + 0.002,
        }
        for cohort in ("pooled", "fresh"):
            expected_gates[f"{cohort}_mse_{weighting}"] = (
                arms["F"][cohort][weighting]["value_mse"]
                <= arms["initializer"][cohort][weighting]["value_mse"] + 0.002
            )
        for name, expected in expected_gates.items():
            if endpoint["gates"].get(name) is not expected:
                raise ValueError(f"semantic_mismatch:gate:{name}")
    if endpoint.get("decision_status") is not None:
        raise ValueError("recovery_endpoint_cannot_claim_decision")
    if "does not explicitly register" not in endpoint.get(
        "prospective_protocol_limitation", ""
    ):
        raise ValueError("protocol_limitation_missing")


def _fresh_mask(provenance: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([item["source"] == "fresh" for item in provenance], dtype=bool)


def _loss(
    model: torch.nn.Module,
    ids: np.ndarray,
    arm: str,
    arrays: tuple[Any, ...],
    fresh: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, policies, targets, _replay, coefficient = arrays[:5]
    legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ids]))
    logits, value_pred = model(torch.from_numpy(x[ids]))
    row_ce = train.compute_policy_cross_entropy(
        logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policies[ids])
    )
    coeff = torch.from_numpy(coefficient[ids])
    mask = torch.from_numpy(fresh[ids]).to(coeff.dtype)
    source_weights = coeff if arm == "C" else coeff * mask
    denominator = source_weights.sum()
    if float(denominator) <= 0:
        raise ValueError(f"empty_denominator:{arm}")
    policy_loss = (source_weights * row_ce).sum() / denominator
    value_loss = train.compute_value_loss_vector(
        value_pred, torch.from_numpy(targets[ids]), value_loss="huber", huber_delta=1.0
    ).mean()
    return policy_loss, value_loss, denominator


def _set(model: torch.nn.Module, values: list[np.ndarray]) -> None:
    with torch.no_grad():
        for parameter, value in zip(model.parameters(), values, strict=True):
            parameter.copy_(torch.from_numpy(np.asarray(value, dtype=np.float32)))


def _cohort_targets(
    root: Path, rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """Read each membership occurrence's own frozen source-record targets."""
    source_cache: dict[str, list[dict[str, Any]]] = {}
    policies: list[list[float]] = []
    values: list[float] = []
    for row in rows:
        source = str(row["source"])
        if source not in source_cache:
            path = (
                root / f"docs/data/seed426-canonical-overlap/sources/{source}.jsonl.gz"
            )
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                source_cache[source] = [json.loads(line) for line in stream]
        raw = source_cache[source][int(row["raw_line"]) - 1]
        policies.append(raw["policy"])
        values.append(float(raw["value"]))
    return np.asarray(policies, dtype=np.float32), np.asarray(values, dtype=np.float32)


def reconstruct(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / ATTEMPT
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    source_checks = {
        path: digest(root / path) == expected
        for path, expected in registration["source_sha256"].items()
    }
    input_checks = {
        path: digest(root / path) == expected
        for path, expected in registration["input_sha256"].items()
    }
    if not all(source_checks.values()) or not all(input_checks.values()):
        raise ValueError("registered_identity_mismatch")
    data_loader.ROOT = root
    data_loader.INIT = root / INIT
    arrays = data_loader._inputs()
    x, policies, values, replay, coeff, train_pos, _val_pos, provenance = arrays
    fresh = _fresh_mask(provenance)
    train_rows = replay[train_pos]
    positions = np.asarray(registration["first_16_minibatch_positions"], dtype=np.int64)
    trajectory_records: dict[str, Any] = {}
    reconstructed_models: dict[str, torch.nn.Module] = {}
    control_diffs: list[str] = []
    with (
        np.load(root / CONTROL / "step-tensors.npz", allow_pickle=False) as control,
    ):
        for arm in ("C", "F"):
            model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            train.load_checkpoint_into_model(model, root / INIT)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(
                params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
            )
            archive = np.load(out / f"{arm}-steps.npz", allow_pickle=False)
            recorded = json.loads((out / f"{arm}-trajectory.json").read_text())
            steps: list[dict[str, Any]] = []
            for step in range(16):
                expanded = positions[step * 512 : (step + 1) * 512]
                ids = train_rows[expanded]
                before = [p.detach().clone() for p in params]
                optimizer.zero_grad(set_to_none=True)
                policy_loss, value_loss, denominator = _loss(
                    model, ids, arm, arrays, fresh
                )
                total = policy_loss + 0.3 * value_loss
                total.backward()
                unclipped = torch.nn.utils.clip_grad_norm_(params, 1.0)
                gradients = [p.grad.detach().clone() for p in params]
                optimizer.step()
                after = [p.detach().clone() for p in params]
                rec = recorded["steps"][step]
                if rec["positions"] != expanded.tolist() or rec["rows"] != ids.tolist():
                    raise ValueError(f"minibatch_mismatch:{arm}:{step + 1}")
                if float(denominator) != rec["policy_mass"]:
                    raise ValueError(f"denominator_mismatch:{arm}:{step + 1}")
                if not np.isclose(
                    float(unclipped), rec["preclip_norm"], rtol=0, atol=1e-7
                ):
                    raise ValueError(f"clip_norm_mismatch:{arm}:{step + 1}")
                for j, (p0, grad, post) in enumerate(
                    zip(before, gradients, after, strict=True)
                ):
                    prefix = f"{step:02d}_"
                    expected_values = {
                        "pre": p0.detach().cpu().numpy(),
                        "gradient": grad.cpu().numpy(),
                        "proposal": post.detach().cpu().numpy(),
                        "post": post.detach().cpu().numpy(),
                        "exp_avg": optimizer.state[params[j]]["exp_avg"].cpu().numpy(),
                        "exp_avg_sq": optimizer.state[params[j]]["exp_avg_sq"]
                        .cpu()
                        .numpy(),
                    }
                    for key, expected in expected_values.items():
                        require_array_equal(
                            f"{key}:{arm}:{step + 1}:{j}",
                            archive[prefix + f"{key}_{j:02d}"],
                            expected,
                        )
                    if arm == "C":
                        # The ordinary Adam update is selected in full (scale 1).
                        if not np.array_equal(
                            expected_values["post"],
                            control[f"A_{step:02d}_post_{j:02d}"],
                        ):
                            control_diffs.append(f"post:{step + 1}:{j}")
                        if not np.array_equal(
                            expected_values["proposal"],
                            control[f"A_{step:02d}_proposal_{j:02d}"],
                        ):
                            control_diffs.append(f"update:{step + 1}:{j}")
                steps.append(
                    {
                        "step": step + 1,
                        "objective": float(total.detach()),
                        "policy": float(policy_loss.detach()),
                        "value": float(value_loss.detach()),
                        "policy_mass": float(denominator),
                        "preclip_norm": float(unclipped),
                        "clipped": bool(float(unclipped) > 1),
                        "post_hash": _flatten_hash(after),
                    }
                )
            archive.close()
            trajectory_records[arm] = steps
            reconstructed_models[arm] = model
    if control_diffs:
        raise ValueError(f"seed442_exact_control_mismatch:{control_diffs[:10]}")
    return {
        "status": "trajectory_reconstruction_valid",
        "source_checks": source_checks,
        "input_checks": input_checks,
        "step_count_per_arm": 16,
        "C_seed442_control": {
            "status": "exact",
            "steps": 16,
            "parameter_tensor_comparisons": 16 * len(params),
            "mismatches": [],
        },
        "steps": trajectory_records,
        "models": reconstructed_models,
        "arrays": arrays,
        "fresh": fresh,
        "registration": registration,
    }


def endpoint_evidence(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Retrospectively evaluate initializer and reconstructed endpoint states."""
    root = root.resolve()
    arrays = result["arrays"]
    x, policy, value, replay, coeff = arrays[:5]
    membership_path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        membership = [json.loads(line) for line in stream]
    pooled = [
        row
        for row in membership
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    fresh_cohort = [row for row in pooled if row["source"] == "fresh"]
    pooled_policy, pooled_value = _cohort_targets(root, pooled)
    fresh_policy, fresh_value = _cohort_targets(root, fresh_cohort)
    init = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(init, root / INIT)
    models = {
        "initializer": init,
        "C": result["models"]["C"],
        "F": result["models"]["F"],
    }
    endpoint: dict[str, Any] = {
        "schema": "seed462-retrospective-endpoints-v1",
        "timing": "post_execution_reconstruction",
        "cohorts_reused": True,
        "cohort_definition": "membership subset unseen and active_stones > 32; fresh subgroup source=fresh",
        "arms": {},
        "effects_F_minus_C": {},
        "cohorts": {
            "pooled": [
                {
                    key: row[key]
                    for key in ("input_identity", "compact_row", "source", "raw_line")
                }
                for row in pooled
            ],
            "fresh": [
                {
                    key: row[key]
                    for key in ("input_identity", "compact_row", "source", "raw_line")
                }
                for row in fresh_cohort
            ],
        },
    }
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.eval()
        with torch.inference_mode():
            pred_logits, pred_value = model(torch.from_numpy(x))
            legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x))
            logp = (
                torch.log_softmax(
                    pred_logits.masked_fill(legal <= 0, -torch.inf), dim=1
                )
                .cpu()
                .numpy()
            )
            values_pred = pred_value.reshape(-1).cpu().numpy()
        predictions[name] = logp
        metrics: dict[str, Any] = {}
        for cohort_name, cohort, cohort_policy, cohort_value in (
            ("pooled", pooled, pooled_policy, pooled_value),
            ("fresh", fresh_cohort, fresh_policy, fresh_value),
        ):
            ids = np.asarray([row["compact_row"] for row in cohort], dtype=np.int64)
            if any(
                row["input_identity"]
                != x[int(row["compact_row"])].astype("<f4", copy=False).tobytes().hex()
                for row in cohort
            ):
                raise ValueError(f"cohort_input_identity_mismatch:{cohort_name}")
            ce = -(cohort_policy * np.where(np.isfinite(logp[ids]), logp[ids], 0)).sum(
                axis=1
            )
            mse = np.square(values_pred[ids] - cohort_value)
            grouped: dict[str, list[int]] = {}
            for index, row in enumerate(cohort):
                grouped.setdefault(row["input_identity"], []).append(index)
            metrics[cohort_name] = {
                "occurrence_count": len(ids),
                "unique_identities": len(grouped),
                "exposure_weighted": {
                    "policy_ce": float(np.mean(ce)),
                    "value_mse": float(np.mean(mse)),
                },
                "equal_input": {
                    "policy_ce": float(
                        np.mean([np.mean(ce[ix]) for ix in grouped.values()])
                    ),
                    "value_mse": float(
                        np.mean([np.mean(mse[ix]) for ix in grouped.values()])
                    ),
                },
            }
        # Full replay occurrence objectives: original policy mixture and fresh policy,
        # each retaining the value term over every original occurrence.
        row_ce_parts = []
        for start in range(0, len(x), 2048):
            end = min(start + 2048, len(x))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[start:end])
            )
            with torch.inference_mode():
                logits, value_pred = model(torch.from_numpy(x[start:end]))
                row_ce_parts.append(
                    train.compute_policy_cross_entropy(
                        logits.masked_fill(legal <= 0, -1e9),
                        torch.from_numpy(policy[start:end]),
                    )
                    .cpu()
                    .numpy()
                )
        row_ce = np.concatenate(row_ce_parts)
        source_fresh = result["fresh"][replay]
        exp_coeff = coeff[replay]
        fresh_coeff = exp_coeff * source_fresh
        value_huber = np.abs(values_pred[replay] - value[replay].reshape(-1))
        value_huber = np.where(value_huber < 1, 0.5 * value_huber**2, value_huber - 0.5)
        metrics["full_training_objectives"] = {
            "original_mixture": float(
                np.average(row_ce[replay], weights=exp_coeff)
                + 0.3 * np.mean(value_huber)
            ),
            "fresh_policy": float(
                np.average(row_ce[replay], weights=fresh_coeff)
                + 0.3 * np.mean(value_huber)
            ),
        }
        endpoint["arms"][name] = metrics
        np.savez_compressed(
            root / RECOVERY / f"{name}-occurrence-predictions.npz",
            pooled_log_policy=logp[[r["compact_row"] for r in pooled]],
            pooled_target_policy=pooled_policy,
            pooled_value=values_pred[[r["compact_row"] for r in pooled]],
            pooled_target_value=pooled_value,
            fresh_log_policy=logp[[r["compact_row"] for r in fresh_cohort]],
            fresh_target_policy=fresh_policy,
            fresh_value=values_pred[[r["compact_row"] for r in fresh_cohort]],
            fresh_target_value=fresh_value,
        )
    for cohort in ("pooled", "fresh"):
        endpoint["effects_F_minus_C"][cohort] = {}
        for weighting in ("exposure_weighted", "equal_input"):
            endpoint["effects_F_minus_C"][cohort][weighting] = {
                metric: endpoint["arms"]["F"][cohort][weighting][metric]
                - endpoint["arms"]["C"][cohort][weighting][metric]
                for metric in ("policy_ce", "value_mse")
            }
    # Apply only the preregistered thresholds. Cohort selection rules are not explicit
    # in the original protocol; report gate arithmetic but mark decision unsupported.
    gates: dict[str, bool] = {}
    for weighting in ("exposure_weighted", "equal_input"):
        f = endpoint["arms"]["F"]
        gates[f"fresh_vs_initializer_{weighting}"] = (
            f["fresh"][weighting]["policy_ce"]
            <= endpoint["arms"]["initializer"]["fresh"][weighting]["policy_ce"] - 0.005
        )
        gates[f"fresh_vs_C_{weighting}"] = (
            f["fresh"][weighting]["policy_ce"]
            <= endpoint["arms"]["C"]["fresh"][weighting]["policy_ce"] - 0.005
        )
        gates[f"pooled_ce_{weighting}"] = (
            f["pooled"][weighting]["policy_ce"]
            <= endpoint["arms"]["initializer"]["pooled"][weighting]["policy_ce"] + 0.002
        )
        for cohort in ("pooled", "fresh"):
            gates[f"{cohort}_mse_{weighting}"] = (
                f[cohort][weighting]["value_mse"]
                <= endpoint["arms"]["initializer"][cohort][weighting]["value_mse"]
                + 0.002
            )
    gates["fresh_training_objective_decreases"] = (
        endpoint["arms"]["F"]["full_training_objectives"]["fresh_policy"]
        < endpoint["arms"]["initializer"]["full_training_objectives"]["fresh_policy"]
    )
    endpoint["gates"] = gates
    endpoint["prospective_protocol_limitation"] = (
        "The registration hash-binds the full validation membership file but does not explicitly register the pooled/fresh >32 cohort-selection rules or exposure-weighted/equal-input aggregation definitions. The numeric gate thresholds are present, but their exact estimands are not prospectively specified. No advancement/closure decision is supportable from these gates without retrofitting protocol fields."
    )
    validate_endpoint_semantics(endpoint)
    return endpoint


def verify_and_publish(root: Path) -> dict[str, Any]:
    root = root.resolve()
    recovery = root / RECOVERY
    trajectories = reconstruct(root)
    endpoint = endpoint_evidence(root, trajectories)
    (recovery / "trajectory-reconstruction.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in trajectories.items()
                if k not in {"models", "arrays", "fresh", "registration"}
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (recovery / "endpoint-metrics.json").write_text(
        json.dumps(endpoint, indent=2, sort_keys=True) + "\n"
    )
    return {
        "trajectory": "valid",
        "control": "exact",
        "endpoint_timing": "post_execution_reconstruction",
        "decision_status": "unsupported_due_to_unregistered_estimands",
        "all_numeric_gates": endpoint["gates"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(verify_and_publish(args.root), indent=2, sort_keys=True))
