"""Runner for retrospective attribution of seed453's archived C and B paths."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed452_fresh_policy_step_attribution as seed452
from ml.alphazero_lite import seed455_analysis as arithmetic
from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed455-fresh-projection-attribution")
CHUNK = 512
SOURCES = seed452.SOURCES


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(root: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Rebuild replay arrays using the production loader and frozen modes."""
    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    with tempfile.TemporaryDirectory(
        prefix="seed455-replay-", dir=root / ".tmp"
    ) as temp:
        paths = []
        for source in SOURCES:
            derivative = Path(temp) / f"{source}.jsonl"
            with (
                gzip.open(
                    root
                    / "docs/data/seed426-canonical-overlap/sources"
                    / f"{source}.jsonl.gz",
                    "rb",
                ) as compressed,
                derivative.open("wb") as output,
            ):
                while block := compressed.read(1024 * 1024):
                    output.write(block)
            paths.append(derivative)
        x, policy, _value, *_ = train.load_jsonl_replay(
            paths,
            [int(row["weight"]) for row in registration["replays"]],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[
                row["value_target_mode"] for row in registration["replays"]
            ],
            include_policy_loss_weights=True,
        )
    membership_path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        members = [json.loads(line) for line in stream]
    members = [
        row
        for row in members
        if row["subset"] == "unseen"
        and row["active_stones"] > 32
        and row["source"] == "fresh"
    ]
    return x, policy, members


def _layout(model: torch.nn.Module) -> list[dict[str, Any]]:
    layout, offset = [], 0
    for name, parameter in model.named_parameters():
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
        layout.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "start": offset,
                "stop": stop,
                "group": group,
            }
        )
        offset = stop
    if len(layout) != 22 or offset != 73159:
        raise ValueError("parameter_layout_size_invalid")
    return layout


def _states(root: Path, arm: str, model: torch.nn.Module) -> list[list[np.ndarray]]:
    params = tuple(model.parameters())
    path = root / f"docs/data/seed453-fresh-policy-projection/step-tensors-{arm}.npz"
    states = []
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            f"{arm}_{step:02d}_{i:02d}_{side}"
            for step in range(16)
            for i in range(22)
            for side in ("pre", "post")
        }
        if not expected.issubset(archive.files):
            raise ValueError(f"trajectory_tensor_coverage_invalid:{arm}")
        for step in range(16):
            pre = [archive[f"{arm}_{step:02d}_{i:02d}_pre"].copy() for i in range(22)]
            post = [archive[f"{arm}_{step:02d}_{i:02d}_post"].copy() for i in range(22)]
            if step and any(
                not np.array_equal(a, b) for a, b in zip(states[-1], pre, strict=True)
            ):
                raise ValueError(f"trajectory_link_invalid:{arm}:{step + 1}")
            if any(
                a.dtype != np.float32
                or not np.isfinite(a).all()
                or tuple(a.shape) != tuple(p.shape)
                for a, p in zip((*pre, *post), (*params, *params), strict=True)
            ):
                raise ValueError(f"trajectory_tensor_invalid:{arm}:{step + 1}")
            if not step:
                init = train.PolicyValueNet((96, 3), "residual_v3", 27)
                train.load_checkpoint_into_model(
                    init,
                    root
                    / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
                )
                if any(
                    not np.array_equal(a, p.detach().numpy())
                    for a, p in zip(pre, init.parameters(), strict=True)
                ):
                    raise ValueError(f"initializer_mismatch:{arm}")
            if step == 0:
                states.append([a.copy() for a in pre])
            states.append([a.copy() for a in post])
    final_path = root / "docs/data/seed453-fresh-policy-projection" / f"{arm}-final.npz"
    final = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(final, final_path)
    if any(
        not np.array_equal(a, p.detach().numpy())
        for a, p in zip(states[-1], final.parameters(), strict=True)
    ):
        raise ValueError(f"final_checkpoint_mismatch:{arm}")
    return states


def _loss_gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
    *,
    seed453_guard: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    params = tuple(model.parameters())
    accum = [np.zeros(tuple(p.shape), dtype=np.float64) for p in params]
    logits_rows, loss_rows = [], []
    denom = float(weights.sum(dtype=np.float64))
    for start in range(0, len(rows), CHUNK):
        ix = rows[start : start + CHUNK]
        logits, _ = model(torch.from_numpy(x[ix]))
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ix]))
        ce = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(target[ix])
        )
        logits_rows.append(logits.detach().cpu().numpy().copy())
        loss_rows.extend(ce.detach().double().cpu().numpy().tolist())
        weighted = (
            ce.sum() / len(rows)
            if seed453_guard
            else (ce.double() * torch.from_numpy(weights[start : start + CHUNK])).sum()
            / denom
        )
        grads = torch.autograd.grad(weighted, params, allow_unused=True)
        for i, grad in enumerate(grads):
            if grad is not None:
                accum[i] += grad.detach().cpu().numpy().astype(np.float64)
    if seed453_guard:
        accum = [value.astype(np.float32).astype(np.float64) for value in accum]
    flat = np.concatenate([a.ravel() for a in accum])
    return (
        float(np.dot(np.asarray(loss_rows, dtype=np.float64), weights) / denom),
        flat,
        np.concatenate(logits_rows),
    )


def run(root: Path, publish: bool = True) -> dict[str, Any]:
    """Compute 96 step/cohort records from fixed archived states only."""
    root = root.resolve()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((out / "freeze-v7.json").read_text())
    for rel, digest in frozen["sources"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_source_mismatch:{rel}")
    for rel, digest in frozen["inputs"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_input_mismatch:{rel}")
    x, target, unseen = inputs(root)
    guard = json.loads(
        (
            root / "docs/data/seed453-fresh-policy-projection/fresh-guard.json"
        ).read_text()
    )
    cohorts = {
        "training_guard": (
            [int(r["compact_row"]) for r in guard],
            [r["exact_input_identity"] for r in guard],
            "equal_input",
        ),
        "fresh_unseen_exposures": (
            [int(r["compact_row"]) for r in unseen],
            [r["input_identity"] for r in unseen],
            "exposure_weighted",
        ),
        "fresh_unseen_equal_input": (
            [int(r["compact_row"]) for r in unseen],
            [r["input_identity"] for r in unseen],
            "equal_input",
        ),
    }
    if (
        len(cohorts["training_guard"][0]) != 1399
        or len(unseen) != 859
        or len(set(cohorts["fresh_unseen_equal_input"][1])) != 851
    ):
        raise ValueError("fixed_cohort_population_mismatch")
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    layout = _layout(model)
    result: dict[str, Any] = {
        "schema": "seed455-results-v1",
        "status": "retrospective_frozen_trajectory_attribution",
        "cohorts": {},
        "parameter_count": 73159,
        "path_states": 17,
        "step_count": 16,
        "limitations": [
            "Retrospective descriptive attribution; does not establish causality.",
            "R contains finite-step nonlinear and numerical effects and is not pure curvature.",
            "A protected first-order training-guard slope does not guarantee nonincreasing finite-step training loss or transfer to unseen inputs.",
            "No training, strength evaluation, alternative trajectory, export, or promotion is authorized.",
        ],
    }
    gradient_archive: dict[str, np.ndarray] = {}
    prediction_archive: dict[str, np.ndarray] = {"ordered_targets": target.copy()}
    for name, (row_ids, identities, _weighting) in cohorts.items():
        indices = np.asarray(row_ids, dtype=np.int64)
        prediction_archive[f"cohort_{name}_compact_rows"] = indices
        prediction_archive[f"cohort_{name}_identities"] = np.asarray(
            identities, dtype="U"
        )
        prediction_archive[f"cohort_{name}_targets"] = target[indices].copy()
    for arm in ("C", "B"):
        states = _states(root, arm, model)
        arm_result = result["cohorts"].setdefault(arm, {})
        for cohort, (rows_list, identities, weighting) in cohorts.items():
            rows = np.asarray(rows_list, dtype=np.int64)
            w = arithmetic.row_weights(identities, weighting)
            losses, gradients = [], []
            for si, state in enumerate(states):
                with torch.no_grad():
                    for p, value in zip(model.parameters(), state, strict=True):
                        p.copy_(torch.from_numpy(value.copy()))
                loss, grad, logits = _loss_gradient(
                    model,
                    x,
                    target,
                    rows,
                    w,
                    seed453_guard=cohort == "training_guard",
                )
                losses.append(loss)
                gradients.append(grad)
                prediction_archive[f"{arm}_{cohort}_{si:02d}"] = logits.copy()
            ledger = []
            for step in range(16):
                delta = np.concatenate(
                    [
                        (b.astype(np.float64) - a.astype(np.float64)).ravel()
                        for a, b in zip(states[step], states[step + 1], strict=True)
                    ]
                )
                parts = arithmetic.decompose(
                    losses[step], losses[step + 1], gradients[step], delta, layout
                )
                record = {
                    "arm": arm,
                    "step": step + 1,
                    "cohort": cohort,
                    "pre_ce": losses[step],
                    "post_ce": losses[step + 1],
                    **parts,
                }
                ledger.append(record)
                gradient_archive[f"{arm}_{cohort}_{step:02d}"] = gradients[step]
            arm_result[cohort] = {
                "ledger": ledger,
                "totals": {k.upper(): v for k, v in arithmetic.totals(ledger).items()},
                "endpoint_change": losses[-1] - losses[0],
            }
    for arm in ("C", "B"):
        for cohort in ("fresh_unseen_exposures", "fresh_unseen_equal_input"):
            for step in range(16):
                guard_gradient = gradient_archive[f"{arm}_training_guard_{step:02d}"]
                unseen_gradient = gradient_archive[f"{arm}_{cohort}_{step:02d}"]
                denominator = float(
                    np.linalg.norm(guard_gradient) * np.linalg.norm(unseen_gradient)
                )
                cosine = (
                    None
                    if denominator == 0.0
                    else float(np.dot(guard_gradient, unseen_gradient) / denominator)
                )
                result["cohorts"][arm][cohort]["ledger"][step][
                    "guard_unseen_gradient_cosine"
                ] = cosine
                result["cohorts"][arm]["training_guard"]["ledger"][step].setdefault(
                    "guard_unseen_gradient_cosines", {}
                )[cohort] = cosine
    unseen_totals = {
        "exposure_weighted": result["cohorts"]["B"]["fresh_unseen_exposures"]["totals"],
        "equal_input": result["cohorts"]["B"]["fresh_unseen_equal_input"]["totals"],
    }
    result["classification"] = arithmetic.classify(*unseen_totals.values())
    result["row_count"] = sum(
        len(v["ledger"]) for a in result["cohorts"].values() for v in a.values()
    )
    result["blocks"] = {}
    for arm, arm_data in result["cohorts"].items():
        result["blocks"][arm] = {}
        for cohort, record in arm_data.items():
            result["blocks"][arm][cohort] = {}
            for name, lo, hi in (
                ("steps_1_3", 1, 3),
                ("steps_4_16", 4, 16),
                ("all_16", 1, 16),
            ):
                subset = [r for r in record["ledger"] if lo <= r["step"] <= hi]
                result["blocks"][arm][cohort][name] = {
                    k.upper(): float(sum(r[k] for r in subset)) for k in ("d", "s", "r")
                }
    if publish:
        (out / "parameter-layout.json").write_text(
            json.dumps(layout, indent=2, sort_keys=True) + "\n"
        )
        (out / "results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        (out / "predictions-targets.npz").write_bytes(_npz_bytes(prediction_archive))
        (out / "gradients.npz").write_bytes(_npz_bytes(gradient_archive))
        (out / "results.md").write_text(_markdown(result))
        inventory = {
            str(path.relative_to(root)): sha(path)
            for path in sorted(out.rglob("*"))
            if path.is_file() and path.name != "receipt.json"
        }
        (out / "receipt.json").write_text(
            json.dumps(
                {"schema": "seed455-receipt-v1", "files_sha256": inventory},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return result


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    import io

    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Seed455 — seed453 fresh-policy projection attribution",
        "",
        f"**Classification:** `{result['classification']}`",
        "",
        "Retrospective descriptive diagnostic. Seed453 and seed454 historical decisions are unchanged.",
        "",
        "| Arm | Cohort | D | S | R | Actual endpoint CE change |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm, cohorts in result["cohorts"].items():
        for cohort, record in cohorts.items():
            t = record["totals"]
            lines.append(
                f"| {arm} | {cohort} | {t['D']:.12g} | {t['S']:.12g} | {t['R']:.12g} | {record['endpoint_change']:.12g} |"
            )
    lines.extend(
        [
            "",
            "A protected first-order fresh-training guard slope does not guarantee nonincreasing finite-step guard loss or transfer to fresh unseen inputs. The remainder includes nonlinear and numerical effects; this analysis is not causal and authorizes no training or strength evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def freeze(root: Path) -> dict[str, Any]:
    """Bind diagnostic code and archived inputs before any diagnostic forward pass."""
    root = root.resolve()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / "freeze-v7.json"
    if path.exists():
        raise ValueError("seed455_freeze_is_immutable")
    sources = [
        "ml/alphazero_lite/seed455_analysis.py",
        "ml/alphazero_lite/seed455_fresh_projection_attribution.py",
        "ml/alphazero_lite/verify_seed455_fresh_projection_attribution.py",
        "ml/alphazero_lite/test_seed455_fresh_projection_attribution.py",
        "ml/alphazero_lite/test_seed455_physical_copy.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/seed452_fresh_policy_step_attribution.py",
    ]
    inputs = [
        "docs/data/seed453-fresh-policy-projection/registration.json",
        "docs/data/seed453-fresh-policy-projection/evidence.json",
        "docs/data/seed453-fresh-policy-projection/receipt.json",
        "docs/data/seed453-fresh-policy-projection/fresh-guard.json",
        "docs/data/seed453-fresh-policy-projection/step-tensors-B.npz",
        "docs/data/seed453-fresh-policy-projection/step-tensors-C.npz",
        "docs/data/seed453-fresh-policy-projection/B-final.npz",
        "docs/data/seed453-fresh-policy-projection/C-final.npz",
        "docs/data/seed453-fresh-policy-projection/initializer-predictions.npz",
        "docs/data/seed453-fresh-policy-projection/B-predictions.npz",
        "docs/data/seed453-fresh-policy-projection/C-predictions.npz",
        "docs/data/seed453-fresh-policy-projection/verifier-correction.json",
        "docs/data/seed453-fresh-policy-projection/verifier-correction-receipt.json",
        "docs/data/seed454-full-guard-compliance/superseding-v7/audit-registration.json",
        "docs/data/seed454-full-guard-compliance/superseding-v7/audit.json",
        "docs/data/seed452-fresh-policy-step-attribution/results.json",
        "docs/data/seed452-fresh-policy-step-attribution/receipt.json",
        "docs/data/seed452-fresh-policy-step-attribution/ordered-predictions-targets.npz",
        "docs/data/seed452-fresh-policy-step-attribution/gradients.npz",
        "docs/data/seed452-fresh-policy-step-attribution/parameter-layout.json",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json",
        "docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json",
        "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        "docs/data/seed455-fresh-projection-attribution/superseded-attempt-v1.json",
        "docs/data/seed455-fresh-projection-attribution/superseded-attempt-v2.json",
        "docs/data/seed455-fresh-projection-attribution/superseded-attempt-v3.json",
    ] + [
        f"docs/data/seed426-canonical-overlap/sources/{name}.jsonl.gz"
        for name in SOURCES
    ]
    protocol = {
        "schema": "seed455-freeze-v1",
        "supersedes": [
            "freeze.json",
            "freeze-v2.json",
            "freeze-v3.json",
            "freeze-v4.json",
            "freeze-v5.json",
            "freeze-v6.json",
        ],
        "temporal_scope": "retrospective diagnostic; seed453 execution is complete",
        "freeze_timing": "after seed453 execution and before seed455 diagnostic execution",
        "sources": {rel: sha(root / rel) for rel in sources},
        "inputs": {rel: sha(root / rel) for rel in inputs},
        "trajectory": "only archived C and B trajectories; 17 states and 16 accepted displacements",
        "cohorts": {
            "training_guard": 1399,
            "fresh_unseen_exposures": 859,
            "fresh_unseen_exact_identities": 851,
        },
        "chunk_rows": CHUNK,
    }
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.root) if args.command == "freeze" else run(args.root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
