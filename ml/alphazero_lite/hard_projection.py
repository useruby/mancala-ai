"""Parent-relative hard output-space trust region projection for policy-head updates."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

POLICY_KEYS = (
    "policy_hidden_layer.weight",
    "policy_hidden_layer.bias",
    "policy_head.weight",
    "policy_head.bias",
)


def compute_masked_policy_distribution(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute numerically stable masked softmax policy distribution."""
    masked_logits = torch.where(
        mask,
        logits,
        torch.tensor(-1e9, dtype=logits.dtype, device=logits.device),
    )
    shifted = masked_logits - torch.amax(masked_logits, dim=-1, keepdim=True)
    exp = torch.exp(shifted) * mask.float()
    denom = torch.sum(exp, dim=-1, keepdim=True)
    denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)
    return exp / denom


def compute_policy_divergences(
    cand_policy: torch.Tensor,
    ref_policy: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Compute L1, JS, and top-1 policy divergence metrics between candidate and reference."""
    # L1 divergence per state
    l1_per_state = torch.sum(torch.abs(cand_policy - ref_policy), dim=-1)
    l1_np = l1_per_state.detach().cpu().numpy().astype(np.float64)

    # JS divergence per state
    cand_safe = torch.clamp(cand_policy, min=1e-12)
    ref_safe = torch.clamp(ref_policy, min=1e-12)
    m = 0.5 * (cand_safe + ref_safe)
    kl_cand_m = torch.sum(
        torch.where(
            mask, cand_safe * torch.log(cand_safe / m), torch.zeros_like(cand_safe)
        ),
        dim=-1,
    )
    kl_ref_m = torch.sum(
        torch.where(
            mask, ref_safe * torch.log(ref_safe / m), torch.zeros_like(ref_safe)
        ),
        dim=-1,
    )
    js_per_state = 0.5 * (kl_cand_m + kl_ref_m)
    js_np = js_per_state.detach().cpu().numpy().astype(np.float64)

    # Top-1 change rate on legal actions
    cand_masked_scores = torch.where(
        mask, cand_policy, torch.tensor(-1e9, device=cand_policy.device)
    )
    ref_masked_scores = torch.where(
        mask, ref_policy, torch.tensor(-1e9, device=ref_policy.device)
    )
    cand_top1 = torch.argmax(cand_masked_scores, dim=-1)
    ref_top1 = torch.argmax(ref_masked_scores, dim=-1)
    top1_changed = (cand_top1 != ref_top1).float().detach().cpu().numpy()

    return {
        "mean_l1": float(np.mean(l1_np)),
        "max_l1": float(np.max(l1_np)),
        "p50_l1": float(np.percentile(l1_np, 50)),
        "p90_l1": float(np.percentile(l1_np, 90)),
        "p95_l1": float(np.percentile(l1_np, 95)),
        "p99_l1": float(np.percentile(l1_np, 99)),
        "mean_js": float(np.mean(js_np)),
        "max_js": float(np.max(js_np)),
        "top1_change": float(np.mean(top1_changed)),
    }


@dataclass
class TrustStateSet:
    """Frozen deterministic trust-state reference set extracted from replay data."""

    states: torch.Tensor
    masks: torch.Tensor
    ref_policies: torch.Tensor
    h_trunk: torch.Tensor
    state_indexes: list[int]
    state_set_hash: str
    total_replay_rows: int
    unique_state_count: int

    @classmethod
    def from_replay_rows(
        cls,
        rows: list[dict[str, Any]],
        p1_model: nn.Module,
        device: torch.device,
    ) -> TrustStateSet:
        """Construct deterministic trust-state set from first occurrence of unique replay states."""
        unique_indexes: list[int] = []
        seen: set[tuple[float, ...]] = set()

        for idx, row in enumerate(rows):
            st = tuple(float(x) for x in row["state"])
            if st not in seen:
                seen.add(st)
                unique_indexes.append(idx)

        states_np = np.asarray(
            [rows[i]["state"] for i in unique_indexes], dtype=np.float32
        )
        # Import legal_mask_matrix_for_encoded_states locally
        from ml.alphazero_lite.train import legal_mask_matrix_for_encoded_states

        masks_np = legal_mask_matrix_for_encoded_states(states_np).astype(bool)

        states_tensor = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        masks_tensor = torch.as_tensor(masks_np, dtype=torch.bool, device=device)

        p1_model.eval()
        with torch.no_grad():
            # Extract trunk embedding
            assert p1_model.input_layer is not None
            h = torch.relu(p1_model.input_layer(states_tensor))
            for l1, l2 in p1_model.residual_layers:
                res = h
                h = torch.relu(l1(h))
                h = torch.relu(l2(h) + res)
            h_trunk = h.detach()

            # Compute frozen reference policies
            assert p1_model.policy_hidden_layer is not None
            hp = torch.relu(p1_model.policy_hidden_layer(h_trunk))
            logits = p1_model.policy_head(hp)
            ref_policies = compute_masked_policy_distribution(
                logits, masks_tensor
            ).detach()

        # Compute hash of state set
        h_obj = hashlib.sha256()
        h_obj.update(states_np.tobytes())
        h_obj.update(masks_np.tobytes())
        state_set_hash = h_obj.hexdigest()

        return cls(
            states=states_tensor,
            masks=masks_tensor,
            ref_policies=ref_policies,
            h_trunk=h_trunk,
            state_indexes=unique_indexes,
            state_set_hash=state_set_hash,
            total_replay_rows=len(rows),
            unique_state_count=len(unique_indexes),
        )

    def evaluate_policy_head(
        self,
        policy_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate policy distribution using policy head parameters on precomputed trunk."""
        w_hidden = policy_params["policy_hidden_layer.weight"]
        b_hidden = policy_params["policy_hidden_layer.bias"]
        w_head = policy_params["policy_head.weight"]
        b_head = policy_params["policy_head.bias"]

        hp = torch.relu(torch.nn.functional.linear(self.h_trunk, w_hidden, b_hidden))
        logits = torch.nn.functional.linear(hp, w_head, b_head)
        return compute_masked_policy_distribution(logits, self.masks)

    def compute_mean_l1(
        self,
        policy_params: dict[str, torch.Tensor],
    ) -> float:
        """Compute mean legal-policy L1 divergence relative to frozen P1."""
        cand_pol = self.evaluate_policy_head(policy_params)
        l1 = torch.sum(torch.abs(cand_pol - self.ref_policies), dim=-1)
        return float(torch.mean(l1).item())

    def full_diagnostics(
        self,
        policy_params: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Compute complete divergence diagnostics relative to frozen P1."""
        cand_pol = self.evaluate_policy_head(policy_params)
        return compute_policy_divergences(cand_pol, self.ref_policies, self.masks)


def compute_parameter_delta_norm(
    theta_a: dict[str, torch.Tensor],
    theta_b: dict[str, torch.Tensor],
    keys: tuple[str, ...] = POLICY_KEYS,
) -> float:
    """Compute Frobenius L2 norm of parameter differences across given keys."""
    deltas = [
        (theta_a[k].detach().double() - theta_b[k].detach().double()).flatten()
        for k in keys
    ]
    return float(torch.linalg.vector_norm(torch.cat(deltas)).item())


def project_policy_head_step(
    model: nn.Module,
    theta_old: dict[str, torch.Tensor],
    theta_raw: dict[str, torch.Tensor],
    theta_p1: dict[str, torch.Tensor],
    trust_set: TrustStateSet,
    radius: float | None,
    max_bisection_steps: int = 30,
    tolerance: float = 1e-6,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Project raw proposed policy-head parameters onto parent-relative trust sphere.

    Given previous feasible parameters theta_old and proposed unconstrained step theta_raw,
    finds the maximum lambda in [0, 1] such that theta(lambda) = theta_old + lambda * (theta_raw - theta_old)
    satisfies mean_legal_L1(theta(lambda), P1) <= radius.
    """
    raw_diag = trust_set.full_diagnostics(theta_raw)
    raw_mean_l1 = raw_diag["mean_l1"]

    param_delta_raw_vs_old = compute_parameter_delta_norm(theta_raw, theta_old)

    # Unconstrained lane
    if radius is None or math.isinf(radius):
        accepted_theta = {k: v.clone() for k, v in theta_raw.items()}
        acc_diag = raw_diag
        lambda_accepted = 1.0
        projection_activated = False
    elif raw_mean_l1 <= radius:
        # Feasible step without projection
        accepted_theta = {k: v.clone() for k, v in theta_raw.items()}
        acc_diag = raw_diag
        lambda_accepted = 1.0
        projection_activated = False
    else:
        # Step violates trust region boundary - bisect along segment [theta_old, theta_raw]
        projection_activated = True
        low = 0.0
        high = 1.0

        for _ in range(max_bisection_steps):
            mid = (low + high) / 2.0
            theta_mid = {
                k: theta_old[k] + mid * (theta_raw[k] - theta_old[k])
                for k in POLICY_KEYS
            }
            mid_mean_l1 = trust_set.compute_mean_l1(theta_mid)
            if mid_mean_l1 <= radius:
                low = mid
            else:
                high = mid

        lambda_accepted = low
        accepted_theta = {
            k: theta_old[k] + lambda_accepted * (theta_raw[k] - theta_old[k])
            for k in POLICY_KEYS
        }
        acc_diag = trust_set.full_diagnostics(accepted_theta)

        # Assert projection invariant
        if acc_diag["mean_l1"] > radius + tolerance:
            raise RuntimeError(
                f"Projection invariant violated: accepted mean L1 {acc_diag['mean_l1']:.8f} > radius {radius:.8f} + tol {tolerance:.8f}"
            )

    # Load accepted parameters into model
    with torch.no_grad():
        current_dict = model.state_dict()
        for k in POLICY_KEYS:
            current_dict[k].copy_(accepted_theta[k])

    param_delta_acc_vs_old = compute_parameter_delta_norm(accepted_theta, theta_old)
    param_delta_acc_vs_p1 = compute_parameter_delta_norm(accepted_theta, theta_p1)

    telemetry = {
        "raw_mean_l1": raw_mean_l1,
        "accepted_mean_l1": acc_diag["mean_l1"],
        "raw_p50_l1": raw_diag["p50_l1"],
        "accepted_p50_l1": acc_diag["p50_l1"],
        "raw_p90_l1": raw_diag["p90_l1"],
        "accepted_p90_l1": acc_diag["p90_l1"],
        "raw_p95_l1": raw_diag["p95_l1"],
        "accepted_p95_l1": acc_diag["p95_l1"],
        "raw_p99_l1": raw_diag["p99_l1"],
        "accepted_p99_l1": acc_diag["p99_l1"],
        "raw_max_l1": raw_diag["max_l1"],
        "accepted_max_l1": acc_diag["max_l1"],
        "raw_mean_js": raw_diag["mean_js"],
        "accepted_mean_js": acc_diag["mean_js"],
        "raw_top1_change": raw_diag["top1_change"],
        "accepted_top1_change": acc_diag["top1_change"],
        "lambda_accepted": lambda_accepted,
        "projection_activated": projection_activated,
        "param_delta_raw_vs_old": param_delta_raw_vs_old,
        "param_delta_acc_vs_old": param_delta_acc_vs_old,
        "param_delta_acc_vs_p1": param_delta_acc_vs_p1,
    }

    return accepted_theta, telemetry
