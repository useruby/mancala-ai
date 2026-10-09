"""Retrospective per-step policy attribution on seed447's frozen T path."""

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

from ml.alphazero_lite import train
from ml.alphazero_lite.seed452_arithmetic import classify, decompose, weights

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed452-fresh-policy-step-attribution")
CHUNK = 512
ATOL, RTOL = 2e-6, 2e-6
SOURCES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
SOURCE_SHA = {
    "fresh": "5f5062416ae620e4e3491989f19c166d26840b3fead8074323f23eb01211746b",
    "generic_bootstrap": "51132b09c453712b8f48ab748e5025b25338681f9f89dd65a1fe2598ca266330",
    "random_teacher": "fa6f83e3fc6bb3c93f87e9b640b8d58213f526115e3fdf582c96d9b1b7b7d1d7",
    "opening_disagreement": "5d73b65f57cf3e4199608c5cfbd80fa9656d5f267b5a823b12c7940a80824bdc",
    "stability": "77de3df0a93f928f8e131566c1f18f83b7d0100874248e2d29a6cb37f6e45b36",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    tmp = root / ".tmp"
    tmp.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="seed452-lane-a-", dir=tmp) as directory:
        paths = []
        for source in SOURCES:
            compressed = (
                root
                / "docs/data/seed426-canonical-overlap/sources"
                / f"{source}.jsonl.gz"
            )
            if sha(compressed) != SOURCE_SHA[source]:
                raise ValueError(f"source_snapshot_hash_mismatch:{source}")
            derivative = Path(directory) / f"{source}.jsonl"
            with gzip.open(compressed, "rb") as inp, derivative.open("wb") as out:
                while block := inp.read(1024 * 1024):
                    out.write(block)
            paths.append(derivative)
        x, p, v, _replay, _coeff = train.load_jsonl_replay(
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
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    # Membership's source tag is the canonical raw-row provenance; fresh is its own cohort.
    return x, p, v, members


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
    return layout


def _states(root: Path, model: torch.nn.Module) -> list[list[np.ndarray]]:
    params = tuple(model.parameters())
    init_path = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    train.load_checkpoint_into_model(model, init_path)
    states = [[p.detach().cpu().numpy().copy() for p in params]]
    archive_path = root / "docs/data/seed447-joint-output-cap/step-tensors.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        expected = {
            f"T_{step:02d}_{side}_{index:02d}"
            for step in range(16)
            for side in ("pre", "post")
            for index in range(len(params))
        }
        if not expected.issubset(archive.files):
            raise ValueError("trajectory_tensor_coverage_invalid")
        for step in range(16):
            pre = [archive[f"T_{step:02d}_pre_{i:02d}"] for i in range(len(params))]
            post = [archive[f"T_{step:02d}_post_{i:02d}"] for i in range(len(params))]
            if any(
                not np.array_equal(a, b) for a, b in zip(states[-1], pre, strict=True)
            ):
                raise ValueError(f"trajectory_link_invalid:{step + 1}")
            if any(
                a.dtype != np.float32 or not np.isfinite(a).all() for a in (*pre, *post)
            ):
                raise ValueError(f"trajectory_tensor_invalid:{step + 1}")
            states.append([a.copy() for a in post])
    endpoint = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        endpoint, root / "docs/data/seed447-joint-output-cap/T-final.npz"
    )
    if any(
        not np.array_equal(a, b.detach().cpu().numpy())
        for a, b in zip(states[-1], endpoint.parameters(), strict=True)
    ):
        raise ValueError("trajectory_endpoint_invalid")
    return states


def _row_losses(
    model: torch.nn.Module, x: np.ndarray, p: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    logits, losses = [], []
    with torch.inference_mode():
        for start in range(0, len(ids), CHUNK):
            ix = ids[start : start + CHUNK]
            output, _value = model(torch.from_numpy(x[ix]))
            legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ix]))
            ce = train.compute_policy_cross_entropy(
                output.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[ix])
            )
            logits.append(output.cpu().numpy())
            losses.extend(ce.double().cpu().numpy().tolist())
    return np.concatenate(logits), np.asarray(losses, dtype=np.float64)


def _gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    p: np.ndarray,
    ids: np.ndarray,
    row_weights: np.ndarray,
) -> np.ndarray:
    params = tuple(model.parameters())
    accum = [np.zeros(tuple(q.shape), dtype=np.float64) for q in params]
    denominator = float(np.sum(row_weights, dtype=np.float64))
    for start in range(0, len(ids), CHUNK):
        ix = ids[start : start + CHUNK]
        logits, _ = model(torch.from_numpy(x[ix]))
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ix]))
        losses = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[ix])
        )
        weight = torch.from_numpy(row_weights[start : start + CHUNK]).double()
        loss = torch.sum(losses.double() * weight) / denominator
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if grad is not None:
                accum[index] += grad.detach().cpu().numpy().astype(np.float64)
    return np.concatenate([item.ravel() for item in accum])


def _freeze(root: Path, layout: list[dict[str, Any]]) -> dict[str, Any]:
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    if path.exists():
        raise ValueError("seed452_protocol_is_immutable")
    execution = [
        "ml/alphazero_lite/seed452_arithmetic.py",
        "ml/alphazero_lite/seed452_fresh_policy_step_attribution.py",
        "ml/alphazero_lite/verify_seed452_fresh_policy_step_attribution.py",
        "ml/alphazero_lite/test_seed452_arithmetic.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/verify_seed450_seed449_correction.py",
    ]
    inputs = [
        "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        "docs/data/seed447-joint-output-cap/T-final.npz",
        "docs/data/seed447-joint-output-cap/step-tensors.npz",
        "docs/data/seed447-joint-output-cap/registration.json",
        "docs/data/seed447-joint-output-cap/receipt.json",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "docs/data/seed449-policy-gain-localization/ordered-row-ledger.jsonl",
        "docs/data/seed451-source-policy-alignment/gradient-vectors.npz",
        "docs/data/seed451-source-policy-alignment/receipt.json",
    ]
    inputs += [
        f"docs/data/seed426-canonical-overlap/sources/{s}.jsonl.gz" for s in SOURCES
    ]
    protocol = {
        "schema": "seed452-frozen-protocol-v1",
        "status": "frozen_before_diagnostic_gradients",
        "temporal_scope": "retrospective; seed447 endpoint outcomes are already known",
        "population": {
            "fresh": {"exposures": 859, "identities": 851},
            "historical": {"exposures": 1748, "identities": 394},
            "shared_identities": 3,
        },
        "weights": "exposure uniform; equal exact identity mass while retaining all exposures and row targets",
        "trajectory": "17 states from registered initializer and 16 archived T pre/post connections; final equals registered T-final",
        "objective": "production legal-action policy cross entropy",
        "chunks": {"forward": CHUNK, "gradient": CHUNK},
        "tolerances": {"absolute": ATOL, "relative": RTOL},
        "parameter_layout": layout,
        "grouping": "shared trunk, policy head, value head; residual is ungrouped",
        "classification": "D>0 and S>=.75D recorded_direction_harms_fresh_policy; else D>0 and R>=.75D fresh_policy_finite_step_residual_dominant; else mixed_or_no_fresh_policy_regression, separately per weighting",
        "execution_sha256": {rel: sha(root / rel) for rel in execution},
        "input_sha256": {rel: sha(root / rel) for rel in inputs},
        "source_snapshots_sha256": {s: SOURCE_SHA[s] for s in SOURCES},
        "interpretation": "signed descriptive retrospective attribution; residual contains nonlinear and numerical effects and is not causal mechanism; no follow-on experiment or promotion authorized",
    }
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def run(root: Path, publish: bool = False) -> dict[str, Any]:
    root = root.resolve()
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    layout = _layout(model)
    x, p, _v, members = _inputs(root)
    target_archive = np.load(
        root / "docs/data/seed447-joint-output-cap/ordered-targets.npz",
        allow_pickle=False,
    )
    if len(members) != 2607 or not np.array_equal(
        np.asarray([r["compact_row"] for r in members]), target_archive["compact_rows"]
    ):
        raise ValueError("ordered_population_mismatch")
    identities = [r["input_identity"] for r in members]
    # Source attribution is the immutable membership field, not identity grouping.
    cohort_masks = {
        "fresh": np.asarray([r["source"] == "fresh" for r in members]),
        "historical": np.asarray([r["source"] != "fresh" for r in members]),
    }
    for name, mask in cohort_masks.items():
        if (int(mask.sum()), len(set(np.asarray(identities, dtype=object)[mask]))) != (
            (859, 851) if name == "fresh" else (1748, 394)
        ):
            raise ValueError(f"source_population_invalid:{name}")
    states = _states(root, model)
    protocol = (
        _freeze(root, layout)
        if publish
        else json.loads((root / OUT / "protocol.json").read_text())
    )
    for rel, digest in {
        **protocol["execution_sha256"],
        **protocol["input_sha256"],
    }.items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_binding_mismatch:{rel}")
    cohort_results: dict[str, Any] = {}
    gradients: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {"identities": np.asarray(identities)}
    all_rows = []
    for cohort, mask in cohort_masks.items():
        ids = np.asarray(
            [int(members[i]["compact_row"]) for i in np.flatnonzero(mask)],
            dtype=np.int64,
        )
        ident = list(np.asarray(identities, dtype=object)[mask])
        weighting_vectors = weights(ident)
        cohort_results[cohort] = {}
        for weighting, row_weights in weighting_vectors.items():
            loss_states = []
            for si, state in enumerate(states):
                with torch.no_grad():
                    for param, value in zip(model.parameters(), state, strict=True):
                        param.copy_(torch.from_numpy(value.copy()))
                logits, raw = _row_losses(model, x, p, ids)
                loss = float(
                    np.sum(raw * row_weights, dtype=np.float64)
                    / np.sum(row_weights, dtype=np.float64)
                )
                loss_states.append(loss)
                predictions[f"{cohort}_{weighting}_{si:02d}_logits"] = logits
            ledger = []
            for step in range(16):
                with torch.no_grad():
                    for param, value in zip(
                        model.parameters(), states[step], strict=True
                    ):
                        param.copy_(torch.from_numpy(value.copy()))
                gradient = _gradient(model, x, p, ids, row_weights)
                delta = np.concatenate(
                    [
                        (post.astype(np.float64) - pre.astype(np.float64)).ravel()
                        for pre, post in zip(
                            states[step], states[step + 1], strict=True
                        )
                    ]
                )
                pieces = decompose(
                    loss_states[step], loss_states[step + 1], gradient, delta, layout
                )
                row = {
                    "step": step + 1,
                    "cohort": cohort,
                    "weighting": weighting,
                    "pre": loss_states[step],
                    "post": loss_states[step + 1],
                    **pieces,
                }
                ledger.append(row)
                gradients[f"{cohort}_{weighting}_{step:02d}"] = gradient
            cohort_results[cohort][weighting] = {
                "ledger": ledger,
                "totals": {
                    key.upper(): float(sum(row[key] for row in ledger))
                    for key in ("d", "s", "r")
                },
                "endpoint_change": loss_states[-1] - loss_states[0],
            }
            all_rows.extend(ledger)
    fresh_totals = {
        weighting: cohort_results["fresh"][weighting]["totals"]
        for weighting in weights(
            list(np.asarray(identities, dtype=object)[cohort_masks["fresh"]])
        )
    }
    decision = classify(fresh_totals)
    result = {
        "schema": "seed452-results-v1",
        "status": "retrospective_frozen_trajectory_attribution",
        "classification": decision,
        "cohorts": cohort_results,
        "population": protocol["population"],
        "path_states": 17,
        "step_count": 16,
        "row_count": len(all_rows),
        "limitations": [
            "Seed447 endpoint outcomes were known before this retrospective diagnostic.",
            "Residual includes nonlinear and numerical effects and is not a causal mechanism or pure curvature.",
            "No replay-weight change, optimizer change, training, strength evaluation, export, or promotion is authorized.",
        ],
    }
    if publish:
        out = root / OUT
        np.savez_compressed(
            out / "ordered-predictions-targets.npz",
            **predictions,
            targets=p[target_archive["compact_rows"]],
            compact_rows=target_archive["compact_rows"],
        )
        np.savez_compressed(out / "gradients.npz", **gradients)
        (out / "parameter-layout.json").write_text(
            json.dumps(layout, indent=2, sort_keys=True) + "\n"
        )
        (out / "step-ledger.json").write_text(
            json.dumps({"rows": all_rows}, indent=2, sort_keys=True) + "\n"
        )
        (out / "results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        blocks = {}
        for cohort in cohort_results:
            blocks[cohort] = {}
            for weighting, record in cohort_results[cohort].items():
                blocks[cohort][weighting] = {}
                for name, lo, hi in (("1-4", 1, 4), ("5-8", 5, 8), ("9-16", 9, 16)):
                    subset = [r for r in record["ledger"] if lo <= r["step"] <= hi]
                    blocks[cohort][weighting][name] = {
                        key.upper(): float(sum(r[key] for r in subset))
                        for key in ("d", "s", "r")
                    }
        (out / "blocks.json").write_text(
            json.dumps(blocks, indent=2, sort_keys=True) + "\n"
        )
        lines = [
            "# Seed452 — fresh-policy finite-step attribution",
            "",
            "Retrospective diagnostic; seed447 endpoint outcomes were already known.",
            "",
            f"**Fresh classification:** `{decision['overall']}`",
            "",
            "| Cohort | Weighting | D | S | R | Endpoint change |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for cohort, wrows in cohort_results.items():
            for weighting, record in wrows.items():
                t = record["totals"]
                lines.append(
                    f"| {cohort} | {weighting} | {t['D']:.12g} | {t['S']:.12g} | {t['R']:.12g} | {record['endpoint_change']:.12g} |"
                )
        lines += [
            "",
            "Group contributions are first-order terms only. Residuals are not assigned to groups and include nonlinear and numerical effects; they are not causal mechanisms.",
            "",
        ]
        (out / "results.md").write_text("\n".join(lines))
        names = sorted(
            path.name
            for path in out.iterdir()
            if path.is_file() and path.name != "receipt.json"
        )
        receipt = {
            "schema": "seed452-receipt-v1",
            "files_sha256": {name: sha(out / name) for name in names},
        }
        (out / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.publish), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
