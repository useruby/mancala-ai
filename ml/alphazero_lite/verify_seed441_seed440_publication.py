"""Independent structural and numeric verifier for seed441 completion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import torch

from ml.alphazero_lite import train
from ml.alphazero_lite import run_seed437_training_unseen_gradient_alignment as seed437
from ml.alphazero_lite.seed437_gradient_alignment import equal_input_weights

REL = Path("docs/data/seed441-seed440-publication-completion")
WEIGHTINGS = ("exposure_weighted", "equal_input")
GROUPS = ("shared_trunk", "policy_head", "value_head")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric_equal(left: Any, right: Any, atol: float, rtol: float, label: str) -> None:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"invalid_numeric_coverage:{label}")
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise ValueError(f"numeric_mismatch:{label}")


def _compare_tree(left: Any, right: Any, atol: float, rtol: float, path: str) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"ledger_keys_mismatch:{path}")
        for key in left:
            _compare_tree(left[key], right[key], atol, rtol, f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"ledger_length_mismatch:{path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare_tree(a, b, atol, rtol, f"{path}[{index}]")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        _numeric_equal([left], [right], atol, rtol, path)
    elif left != right:
        raise ValueError(f"ledger_value_mismatch:{path}")


def _layout(model: torch.nn.Module) -> list[dict[str, Any]]:
    result, offset = [], 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = (
            "shared_trunk"
            if name.startswith(("input_layer.", "residual_layers."))
            else "policy_head"
            if name.startswith(("policy_hidden_layer.", "policy_head."))
            else "value_head"
            if name.startswith(("value_hidden_layer.", "value_head."))
            else None
        )
        if group is None:
            raise ValueError(f"unclassified_parameter:{name}")
        stop = offset + parameter.numel()
        result.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "start": offset,
                "stop": stop,
                "group": group,
            }
        )
        offset = stop
    return result


def _add_delta(
    model: torch.nn.Module,
    delta: np.ndarray,
    layout: list[dict[str, Any]],
    scale: float,
) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for item in layout:
            part = (delta[item["start"] : item["stop"]] * scale).astype(np.float32)
            parameters[item["name"]].add_(torch.from_numpy(part.reshape(item["shape"])))


def _cross_entropy(
    model: torch.nn.Module,
    x: np.ndarray,
    policy: np.ndarray,
    ids: np.ndarray,
    weights: np.ndarray,
) -> float:
    total = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), 512):
            selected = ids[start : start + 512]
            row_weights = weights[start : start + 512]
            logits, _ = model(torch.from_numpy(x[selected]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[selected])
            )
            losses = train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[selected])
            )
            total += float(
                torch.sum(losses.double() * torch.from_numpy(row_weights).double())
            )
    return total


def _gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    policy: np.ndarray,
    ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    flat = np.zeros(
        sum(parameter.numel() for parameter in parameters), dtype=np.float64
    )
    for start in range(0, len(ids), 512):
        selected = ids[start : start + 512]
        row_weights = weights[start : start + 512]
        logits, _ = model(torch.from_numpy(x[selected]))
        legal = torch.from_numpy(
            train.legal_mask_matrix_for_encoded_states(x[selected])
        )
        losses = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[selected])
        )
        objective = (losses * torch.from_numpy(row_weights).to(losses.dtype)).sum()
        gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
        flat += np.concatenate(
            [
                np.zeros(parameter.numel(), dtype=np.float64)
                if gradient is None
                else gradient.detach().cpu().numpy().astype(np.float64).ravel()
                for parameter, gradient in zip(parameters, gradients, strict=True)
            ]
        )
    return flat


def _parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(
            parameter.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
        )
    return digest.hexdigest()


def _classify(totals: dict[str, dict[str, dict[str, float]]]) -> str:
    primary = totals["A"]
    if all(
        primary[w]["D"] > 0 and primary[w]["r"] >= 0.75 * primary[w]["D"]
        for w in WEIGHTINGS
    ):
        return "finite_step_residual_dominant"
    if all(
        primary[w]["D"] > 0 and primary[w]["s"] >= 0.75 * primary[w]["D"]
        for w in WEIGHTINGS
    ):
        return "first_order_direction_dominant"
    return "mixed_or_no_common_dominance"


def _recompute(root: Path) -> dict[str, Any]:
    """Independent attribution arithmetic; only model, loss and input loading are shared."""
    root = root.resolve()
    # prepare_arrays historically uses its own module globals. Bind every path it
    # consults before loading any arrays from this requested root.
    seed437.ROOT = root
    seed437.OUT = root / "docs/data/seed437-training-unseen-gradient-alignment"
    seed437.DATA = root / "docs/data/seed426-canonical-overlap"
    seed437.REGISTRATION = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    seed437.INIT = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    seed437.ADAM = root / "docs/data/seed435-adam-direction-screen/A-final.npz"
    x, policy, _value, _coeff, _mult, _source, _bucket, ids, _legacy_weights = (
        seed437.prepare_arrays()
    )
    import gzip

    membership_path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        membership = [json.loads(line) for line in stream]
    selected = [
        row
        for row in membership
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    identities = [row["input_identity"] for row in selected]
    if [int(row["compact_row"]) for row in selected] != ids.tolist():
        raise ValueError("membership_row_mismatch")
    weights = {
        "exposure_weighted": np.full(len(ids), 1.0 / len(ids), dtype=np.float64),
        "equal_input": equal_input_weights(identities),
    }
    initializer = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    template = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(template, initializer)
    layout = _layout(template)
    predictions: dict[str, np.ndarray] = {}
    gradients: dict[str, np.ndarray] = {}
    state_order: list[str] = []
    path_hashes: dict[str, list[str]] = {}

    def predict(model: torch.nn.Module, key: str) -> None:
        model.eval()
        logits_list, value_list = [], []
        with torch.inference_mode():
            for offset in range(0, len(ids), 512):
                logits, pred_value = model(
                    torch.from_numpy(x[ids[offset : offset + 512]])
                )
                logits_list.append(logits.cpu().numpy())
                value_list.append(pred_value.cpu().numpy())
        predictions[f"{key}_policy_logits"] = np.concatenate(logits_list)
        predictions[f"{key}_value"] = np.concatenate(value_list)
        state_order.append(key)

    result_arms: dict[str, Any] = {}
    for arm in ("A", "B"):
        delta_file = (
            root / f"docs/data/seed435-adam-direction-screen/{arm}-audit-deltas.npz"
        )
        with np.load(delta_file, allow_pickle=False) as archive:
            deltas = np.asarray(archive["deltas"], dtype=np.float64)
        if deltas.shape != (16, layout[-1]["stop"]) or not np.isfinite(deltas).all():
            raise ValueError(f"delta_archive_invalid:{arm}")
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(model, initializer)
        path_hashes[arm] = [_parameter_hash(model)]
        predict(model, f"{arm}_state_00")
        ledger = []
        for step, delta in enumerate(deltas, start=1):
            original = [parameter.detach().clone() for parameter in model.parameters()]
            pre, midpoint, post, dots, group_dots = {}, {}, {}, {}, {}
            for weighting in WEIGHTINGS:
                gradient = _gradient(model, x, policy, ids, weights[weighting])
                gradients[f"{arm}_step_{step:02d}_{weighting}"] = gradient
                pre[weighting] = _cross_entropy(
                    model, x, policy, ids, weights[weighting]
                )
                dots[weighting] = float(np.dot(gradient, delta))
                group_dots[weighting] = {}
                for group in GROUPS:
                    indexes = [
                        i
                        for item in layout
                        if item["group"] == group
                        for i in range(item["start"], item["stop"])
                    ]
                    group_dots[weighting][group] = float(
                        np.dot(gradient[indexes], delta[indexes])
                    )
            _add_delta(model, delta, layout, 0.5)
            midpoint_key = f"{arm}_midpoint_{step:02d}"
            predict(model, midpoint_key)
            for weighting in WEIGHTINGS:
                midpoint[weighting] = _cross_entropy(
                    model, x, policy, ids, weights[weighting]
                )
            with torch.no_grad():
                for parameter, before in zip(model.parameters(), original, strict=True):
                    parameter.copy_(before)
            _add_delta(model, delta, layout, 1.0)
            state_key = f"{arm}_state_{step:02d}"
            predict(model, state_key)
            for weighting in WEIGHTINGS:
                post[weighting] = _cross_entropy(
                    model, x, policy, ids, weights[weighting]
                )
            ledger.append(
                {
                    "step": step,
                    "delta_sha256": hashlib.sha256(
                        delta.astype("<f8").tobytes()
                    ).hexdigest(),
                    "pre_ce": pre,
                    "midpoint_ce": midpoint,
                    "post_ce": post,
                    "s": dots,
                    "group_s": group_dots,
                    "D": {w: post[w] - pre[w] for w in WEIGHTINGS},
                    "r": {w: post[w] - pre[w] - dots[w] for w in WEIGHTINGS},
                    "midpoint_linear_residual": {
                        w: midpoint[w] - pre[w] - 0.5 * dots[w] for w in WEIGHTINGS
                    },
                }
            )
            path_hashes[arm].append(_parameter_hash(model))
        endpoint = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        endpoint_file = (
            root / f"docs/data/seed435-adam-direction-screen/{arm}-final.npz"
        )
        train.load_checkpoint_into_model(endpoint, endpoint_file)
        if not all(
            torch.equal(left, right)
            for left, right in zip(
                model.parameters(), endpoint.parameters(), strict=True
            )
        ):
            raise ValueError(f"checkpoint_parameters_mismatch:{arm}")
        totals = {
            weighting: {
                term: float(sum(row[term][weighting] for row in ledger))
                for term in ("D", "s", "r")
            }
            for weighting in WEIGHTINGS
        }
        for weighting in WEIGHTINGS:
            totals[weighting]["telescoping_error"] = totals[weighting]["D"] - (
                ledger[-1]["post_ce"][weighting] - ledger[0]["pre_ce"][weighting]
            )
            totals[weighting]["identity_error"] = (
                totals[weighting]["D"] - totals[weighting]["s"] - totals[weighting]["r"]
            )
            totals[weighting]["group_reconciliation_error"] = totals[weighting][
                "s"
            ] - sum(row["group_s"][weighting][g] for row in ledger for g in GROUPS)
        result_arms[arm] = {
            "step_ledger": ledger,
            "path_parameter_sha256": path_hashes[arm],
            "totals": totals,
            "endpoint_sha256": sha(endpoint_file),
        }
    return {
        "_predictions": predictions,
        "_gradients": gradients,
        "_layout": layout,
        "_rows": ids.tolist(),
        "_identities": identities,
        "_state_order": state_order,
        "_ledger": result_arms,
        "_path_hashes": path_hashes,
        "_classification": _classify(
            {arm: value["totals"] for arm, value in result_arms.items()}
        ),
        "membership": {
            "selected_exposures": len(ids),
            "identities": len(set(identities)),
        },
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / REL
    completion = json.loads((out / "completion.json").read_text())
    bindings_file = (
        root / "docs/data/seed440-recorded-update-attribution/evidence-bindings.json"
    )
    bindings = json.loads(bindings_file.read_text())
    for section in ("seed435",):
        for item in bindings[section].values():
            if sha(root / item["path"]) != item["sha256"]:
                raise ValueError(f"historical_hash_mismatch:{item['path']}")
    for section in ("seed439_U_gradient_archive", "authoritative_membership"):
        item = bindings[section]
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError(f"historical_hash_mismatch:{item['path']}")
    protocol = json.loads(
        (
            root / "docs/data/seed440-recorded-update-attribution/protocol.json"
        ).read_text()
    )
    old_dir = root / "docs/data/seed440-recorded-update-attribution"
    for key, relative in (
        ("seed440_protocol_sha256", "protocol.json"),
        ("seed440_bindings_sha256", "evidence-bindings.json"),
        ("seed440_ledger_sha256", "step-ledger.json"),
    ):
        if completion.get(key) != sha(old_dir / relative):
            raise ValueError(f"completion_binding_mismatch:{key}")
    if (
        protocol["source_update_vectors"]["step_count"] != 16
        or protocol["classification"]["primary_arm"] != "A"
        or protocol["arithmetic"]["chunk_size"] != 512
        or protocol["membership"]["selected_exposures"] != 2607
        or protocol["membership"]["exact_input_identities"] != 1242
    ):
        raise ValueError("seed440_protocol_settings_mismatch")
    for arm in ("A", "B"):
        delta = root / f"docs/data/seed435-adam-direction-screen/{arm}-audit-deltas.npz"
        if sha(delta) != protocol["source_update_vectors"][f"{arm}_sha256"]:
            raise ValueError(f"protocol_delta_hash_mismatch:{arm}")
    supplemental_protocol = json.loads((out / "protocol.json").read_text())
    source_root = Path(__file__).resolve().parents[2]
    for source, digest in supplemental_protocol["source_sha256"].items():
        if sha(source_root / source) != digest:
            raise ValueError(f"supplemental_source_hash_mismatch:{source}")
    for dependency, digest in supplemental_protocol["dependency_sha256"].items():
        base = source_root if dependency.startswith("ml/") else root
        if sha(base / dependency) != digest:
            raise ValueError(f"dependency_hash_mismatch:{dependency}")
    if supplemental_protocol["classification"]["primary_arm"] != "A":
        raise ValueError("primary_arm_mismatch")
    if (
        supplemental_protocol["paths"]["arms"] != ["A", "B"]
        or supplemental_protocol["paths"]["updates_per_arm"] != 16
    ):
        raise ValueError("path_protocol_mismatch")
    if (
        supplemental_protocol["classification"]["finite_step_residual_dominant"]
        != protocol["classification"]["finite_step_residual_dominant"]
        or supplemental_protocol["classification"]["first_order_direction_dominant"]
        != protocol["classification"]["first_order_direction_dominant"]
    ):
        raise ValueError("classification_rule_mismatch")
    atol = protocol["arithmetic"]["tolerance"]["absolute"]
    rtol = protocol["arithmetic"]["tolerance"]["relative"]
    reproduced = _recompute(root)
    if completion["classification"] != reproduced["_classification"]:
        raise ValueError("completion_classification_mismatch")
    if completion["membership"] != {
        "selected_exposures": reproduced["membership"]["selected_exposures"],
        "unique_input_identities": reproduced["membership"]["identities"],
    }:
        raise ValueError("completion_membership_mismatch")
    rows = json.loads((out / "row-identities.json").read_text())
    if len(rows["compact_rows"]) != 2607 or len(rows["input_identities"]) != 2607:
        raise ValueError("row_identity_coverage_mismatch")
    expected_state_order = [
        f"{arm}_{kind}_{step:02d}"
        for arm in ("A", "B")
        for kind, first, last in (("state", 0, 16), ("midpoint", 1, 16))
        for step in range(first, last + 1)
    ]
    if set(rows["state_order"]) != set(expected_state_order):
        raise ValueError("prediction_state_coverage_mismatch")
    with np.load(out / "predictions.npz", allow_pickle=False) as predictions:
        expected_prediction_keys = {
            f"{state}_{suffix}"
            for state in expected_state_order
            for suffix in ("policy_logits", "value")
        }
        if set(predictions.files) != expected_prediction_keys:
            raise ValueError("prediction_archive_coverage_mismatch")
        for key in predictions.files:
            if not np.isfinite(predictions[key]).all():
                raise ValueError(f"nonfinite_prediction:{key}")
    expected_gradient_keys = {
        f"{arm}_step_{step:02d}_{weight}"
        for arm in ("A", "B")
        for step in range(1, 17)
        for weight in WEIGHTINGS
    }
    with np.load(out / "weighting-gradients.npz", allow_pickle=False) as gradients:
        if set(gradients.files) != expected_gradient_keys:
            raise ValueError("gradient_archive_coverage_mismatch")
        for key in gradients.files:
            if not np.isfinite(gradients[key]).all():
                raise ValueError(f"nonfinite_gradient:{key}")
    # Path and gradient evidence are reproduced directly; ledger comparison uses
    # registered numeric tolerances, never whole-dictionary float equality.
    with np.load(out / "weighting-gradients.npz", allow_pickle=False) as archived:
        for name in expected_gradient_keys:
            _numeric_equal(
                archived[name], reproduced["_gradients"][name], atol, rtol, name
            )
            if (
                archived[name].ndim != 1
                or archived[name].size != reproduced["_layout"][-1]["stop"]
            ):
                raise ValueError(f"gradient_layout_invalid:{name}")
    with np.load(out / "predictions.npz", allow_pickle=False) as archived:
        for name in archived.files:
            _numeric_equal(
                archived[name], reproduced["_predictions"][name], atol, rtol, name
            )
    if rows["state_order"] != reproduced["_state_order"]:
        raise ValueError("prediction_state_order_mismatch")
    layout = json.loads((out / "parameter-layout.json").read_text())
    if layout != reproduced["_layout"]:
        raise ValueError("parameter_layout_mismatch")
    if (
        rows["compact_rows"] != reproduced["_rows"]
        or rows["input_identities"] != reproduced["_identities"]
    ):
        raise ValueError("row_identity_mismatch")
    historical = json.loads(
        (
            root / "docs/data/seed440-recorded-update-attribution/step-ledger.json"
        ).read_text()
    )
    _compare_tree(historical["arms"], reproduced["_ledger"], atol, rtol, "arms")
    path_archive = old_dir / "reconstructed-path-hashes.npz"
    with np.load(path_archive, allow_pickle=False) as archive:
        for arm in ("A", "B"):
            if not np.array_equal(
                archive[f"{arm}_sha256"], reproduced["_path_hashes"][arm]
            ):
                raise ValueError(f"historical_path_hash_mismatch:{arm}")
    if (
        historical["classification"] != reproduced["_classification"]
        or completion["classification"] != historical["classification"]
    ):
        raise ValueError("classification_mismatch")
    _numeric_equal(
        [reproduced["membership"]["selected_exposures"]], [2607], 0, 0, "membership"
    )
    return {
        "status": "valid",
        "classification": completion["classification"],
        "states_per_arm": 17,
        "midpoints_per_arm": 16,
        "gradients_per_arm": 32,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
