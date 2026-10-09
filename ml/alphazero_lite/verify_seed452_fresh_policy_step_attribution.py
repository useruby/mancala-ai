"""Independent read-only semantic verification of seed452 evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train

OUT = Path("docs/data/seed452-fresh-policy-step-attribution")
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layout(model: torch.nn.Module) -> list[dict[str, Any]]:
    result, offset = [], 0
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
            raise ValueError("parameter_group_invalid")
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


def _load_inputs(root: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    scratch_root = root / ".tmp"
    scratch_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="seed452-verify-lane-a-", dir=scratch_root
    ) as tmp:
        derivatives = []
        for source in SOURCES:
            compressed = (
                root
                / "docs/data/seed426-canonical-overlap/sources"
                / f"{source}.jsonl.gz"
            )
            if _sha(compressed) != SOURCE_SHA[source]:
                raise ValueError(f"source_snapshot_hash_mismatch:{source}")
            path = Path(tmp) / f"{source}.jsonl"
            with gzip.open(compressed, "rb") as inp, path.open("wb") as out:
                while chunk := inp.read(1024 * 1024):
                    out.write(chunk)
            derivatives.append(path)
        x, policy, _value, _replay, _coefficient = train.load_jsonl_replay(
            derivatives,
            [int(item["weight"]) for item in registration["replays"]],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[
                item["value_target_mode"] for item in registration["replays"]
            ],
            include_policy_loss_weights=True,
        )
    with gzip.open(
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        members = [json.loads(line) for line in stream]
    members = [
        row
        for row in members
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    targets = np.load(
        root / "docs/data/seed447-joint-output-cap/ordered-targets.npz",
        allow_pickle=False,
    )
    rows = targets["compact_rows"]
    if [r["compact_row"] for r in members] != rows.tolist() or not np.array_equal(
        policy[rows], targets["policy_targets"]
    ):
        raise ValueError("ordered_input_reconstruction_mismatch")
    return x, policy, members


def _states(root: Path, model: torch.nn.Module) -> list[list[np.ndarray]]:
    initializer = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    train.load_checkpoint_into_model(model, initializer)
    states = [[p.detach().cpu().numpy().copy() for p in model.parameters()]]
    with np.load(
        root / "docs/data/seed447-joint-output-cap/step-tensors.npz", allow_pickle=False
    ) as z:
        expected = {
            f"T_{k:02d}_{side}_{i:02d}"
            for k in range(16)
            for side in ("pre", "post")
            for i in range(len(states[0]))
        }
        if not expected.issubset(z.files):
            raise ValueError("state_archive_coverage_mismatch")
        for step in range(16):
            before = [z[f"T_{step:02d}_pre_{i:02d}"] for i in range(len(states[0]))]
            after = [z[f"T_{step:02d}_post_{i:02d}"] for i in range(len(states[0]))]
            if any(
                not np.array_equal(a, b)
                for a, b in zip(states[-1], before, strict=True)
            ):
                raise ValueError(f"state_connection_mismatch:{step + 1}")
            states.append([v.copy() for v in after])
    end = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        end, root / "docs/data/seed447-joint-output-cap/T-final.npz"
    )
    if any(
        not np.array_equal(a, b.detach().cpu().numpy())
        for a, b in zip(states[-1], end.parameters(), strict=True)
    ):
        raise ValueError("endpoint_checkpoint_mismatch")
    return states


def _losses(
    model: torch.nn.Module, x: np.ndarray, p: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    outputs, rowloss = [], []
    with torch.no_grad():
        for start in range(0, len(ids), 512):
            index = ids[start : start + 512]
            logits, _ = model(torch.from_numpy(x[index]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[index])
            )
            ce = train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[index])
            )
            outputs.append(logits.cpu().numpy())
            rowloss.extend(ce.double().cpu().numpy().tolist())
    return np.concatenate(outputs), np.asarray(rowloss, dtype=np.float64)


def _gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    p: np.ndarray,
    ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    params = tuple(model.parameters())
    accum = [np.zeros(tuple(param.shape), dtype=np.float64) for param in params]
    denom = float(np.sum(weights, dtype=np.float64))
    for start in range(0, len(ids), 512):
        ix = ids[start : start + 512]
        logits, _ = model(torch.from_numpy(x[ix]))
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ix]))
        losses = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[ix])
        )
        loss = (
            torch.sum(
                losses.double()
                * torch.from_numpy(weights[start : start + 512]).double()
            )
            / denom
        )
        derivatives = torch.autograd.grad(loss, params, allow_unused=True)
        for j, (param, derivative) in enumerate(zip(params, derivatives, strict=True)):
            if derivative is not None:
                accum[j] += derivative.detach().cpu().numpy().astype(np.float64)
    return np.concatenate([part.ravel() for part in accum])


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    protocol = json.loads((out / "protocol.json").read_text())
    receipt = json.loads((out / "receipt.json").read_text())
    for name, digest in receipt["files_sha256"].items():
        if _sha(out / name) != digest:
            raise ValueError(f"artifact_hash_mismatch:{name}")
    for section in ("execution_sha256", "input_sha256"):
        for name, digest in protocol[section].items():
            actual = _sha(root / name)
            amendment_path = out / "verifier-amendment.json"
            amended = False
            if (
                amendment_path.is_file()
                and name
                == "ml/alphazero_lite/verify_seed452_fresh_policy_step_attribution.py"
            ):
                amendment = json.loads(amendment_path.read_text())
                amended = amendment == {
                    "schema": "seed452-verifier-amendment-v1",
                    "protocol_sha256": _sha(out / "protocol.json"),
                    "original_verifier_sha256": digest,
                    "corrected_verifier_sha256": actual,
                    "arithmetic_protocol_changed": False,
                }
            if actual != digest and not amended:
                raise ValueError(f"frozen_binding_mismatch:{name}")
    result = json.loads((out / "results.json").read_text())
    ledger = json.loads((out / "step-ledger.json").read_text())["rows"]
    layout = _layout(train.PolicyValueNet((96, 3), "residual_v3", 27))
    if (
        layout != json.loads((out / "parameter-layout.json").read_text())
        or layout != protocol["parameter_layout"]
    ):
        raise ValueError("layout_semantics_mismatch")
    x, p, members = _load_inputs(root)
    populations = {
        "fresh": [i for i, row in enumerate(members) if row["source"] == "fresh"],
        "historical": [i for i, row in enumerate(members) if row["source"] != "fresh"],
    }
    if (
        len(populations["fresh"]),
        len({members[i]["input_identity"] for i in populations["fresh"]}),
        len(populations["historical"]),
        len({members[i]["input_identity"] for i in populations["historical"]}),
    ) != (859, 851, 1748, 394):
        raise ValueError("cohort_population_mismatch")
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    states = _states(root, model)
    archive = np.load(out / "gradients.npz", allow_pickle=False)
    pred = np.load(out / "ordered-predictions-targets.npz", allow_pickle=False)
    source_targets = np.load(
        root / "docs/data/seed447-joint-output-cap/ordered-targets.npz",
        allow_pickle=False,
    )
    if not np.array_equal(pred["compact_rows"], source_targets["compact_rows"]):
        raise ValueError("published_compact_rows_mismatch")
    if not np.array_equal(pred["targets"], source_targets["policy_targets"]):
        raise ValueError("published_targets_mismatch")
    if pred["identities"].tolist() != [row["input_identity"] for row in members]:
        raise ValueError("published_identity_order_mismatch")
    recomputed = {}
    for cohort, positions in populations.items():
        ids = np.asarray([members[i]["compact_row"] for i in positions], dtype=np.int64)
        names = [members[i]["input_identity"] for i in positions]
        from collections import Counter

        counts = Counter(names)
        wts = {
            "exposure_weighted": np.full(len(ids), 1 / len(ids), np.float64),
            "equal_exact_input": np.asarray(
                [1 / (len(counts) * counts[name]) for name in names], np.float64
            ),
        }
        recomputed[cohort] = {}
        for weighting, rowweights in wts.items():
            losses = []
            for state_num, state in enumerate(states):
                with torch.no_grad():
                    for param, value in zip(model.parameters(), state, strict=True):
                        param.copy_(torch.from_numpy(value.copy()))
                logits, raw = _losses(model, x, p, ids)
                if not np.allclose(
                    pred[f"{cohort}_{weighting}_{state_num:02d}_logits"],
                    logits,
                    atol=2e-6,
                    rtol=2e-6,
                ):
                    raise ValueError("prediction_semantics_mismatch")
                losses.append(
                    float(
                        np.sum(raw * rowweights, dtype=np.float64)
                        / np.sum(rowweights, dtype=np.float64)
                    )
                )
            rows = []
            for step in range(16):
                with torch.no_grad():
                    for param, value in zip(
                        model.parameters(), states[step], strict=True
                    ):
                        param.copy_(torch.from_numpy(value.copy()))
                g = _gradient(model, x, p, ids, rowweights)
                if not np.array_equal(archive[f"{cohort}_{weighting}_{step:02d}"], g):
                    raise ValueError("gradient_vector_mismatch")
                delta = np.concatenate(
                    [
                        (b.astype(np.float64) - a.astype(np.float64)).ravel()
                        for a, b in zip(states[step], states[step + 1], strict=True)
                    ]
                )
                d = losses[step + 1] - losses[step]
                s = float(np.dot(g, delta))
                row = {
                    "step": step + 1,
                    "cohort": cohort,
                    "weighting": weighting,
                    "pre": losses[step],
                    "post": losses[step + 1],
                    "d": d,
                    "s": s,
                    "r": d - s,
                }
                row["group_s"] = {
                    group: float(
                        sum(
                            np.dot(
                                g[e["start"] : e["stop"]], delta[e["start"] : e["stop"]]
                            )
                            for e in layout
                            if e["group"] == group
                        )
                    )
                    for group in ("shared_trunk", "policy_head", "value_head")
                }
                rows.append(row)
            actual = [
                row
                for row in ledger
                if row["cohort"] == cohort and row["weighting"] == weighting
            ]
            if len(actual) != 16:
                raise ValueError("ledger_row_count_mismatch")
            for got, want in zip(actual, rows, strict=True):
                for key in ("step", "cohort", "weighting"):
                    if got[key] != want[key]:
                        raise ValueError("ledger_identity_mismatch")
                for key in ("pre", "post", "d", "s", "r"):
                    if not math.isclose(
                        got[key], want[key], abs_tol=2e-6, rel_tol=2e-6
                    ):
                        raise ValueError(f"ledger_{key}_mismatch")
                for group, value in want["group_s"].items():
                    if not math.isclose(
                        got["group_s"][group], value, abs_tol=2e-6, rel_tol=2e-6
                    ):
                        raise ValueError("group_contribution_mismatch")
            sums = {
                key.upper(): float(sum(row[key] for row in rows))
                for key in ("d", "s", "r")
            }
            endpoint = losses[-1] - losses[0]
            if not math.isclose(
                sums["D"], endpoint, abs_tol=2e-6, rel_tol=2e-6
            ) or not math.isclose(
                sums["D"], sums["S"] + sums["R"], abs_tol=2e-6, rel_tol=2e-6
            ):
                raise ValueError("endpoint_or_residual_identity_mismatch")
            published = result["cohorts"][cohort][weighting]
            for key in sums:
                if not math.isclose(
                    published["totals"][key], sums[key], abs_tol=2e-6, rel_tol=2e-6
                ):
                    raise ValueError("published_total_mismatch")
            if not math.isclose(
                published["endpoint_change"], endpoint, abs_tol=2e-6, rel_tol=2e-6
            ):
                raise ValueError("endpoint_change_mismatch")
            blocks = json.loads((out / "blocks.json").read_text())[cohort][weighting]
            for label, lower, upper in (("1-4", 1, 4), ("5-8", 5, 8), ("9-16", 9, 16)):
                selected = [row for row in rows if lower <= row["step"] <= upper]
                for key in ("d", "s", "r"):
                    if not math.isclose(
                        blocks[label][key.upper()],
                        sum(row[key] for row in selected),
                        abs_tol=2e-6,
                        rel_tol=2e-6,
                    ):
                        raise ValueError("block_total_mismatch")
            recomputed[cohort][weighting] = sums

    prior_rows = []
    with (
        root / "docs/data/seed449-policy-gain-localization/ordered-row-ledger.jsonl"
    ).open(encoding="utf-8") as stream:
        prior_rows = [json.loads(line) for line in stream]
    if len(prior_rows) != len(members) or any(
        a["input_identity"] != b["input_identity"]
        for a, b in zip(prior_rows, members, strict=True)
    ):
        raise ValueError("seed449_ordered_rows_mismatch")
    endpoint_discrepancies = {}
    for cohort, positions in populations.items():
        identities = [members[i]["input_identity"] for i in positions]
        counts = {identity: identities.count(identity) for identity in set(identities)}
        for weighting in ("exposure_weighted", "equal_exact_input"):
            ww = (
                np.full(len(positions), 1 / len(positions), np.float64)
                if weighting == "exposure_weighted"
                else np.asarray(
                    [1 / (len(counts) * counts[item]) for item in identities],
                    np.float64,
                )
            )
            for state_name, seed449_name, state_index in (
                ("initializer", "initializer", 0),
                ("T_final", "T", 16),
            ):
                recorded = float(
                    sum(
                        prior_rows[i]["losses"][seed449_name] * weight
                        for i, weight in zip(positions, ww, strict=True)
                    )
                )
                measured = (
                    result["cohorts"][cohort][weighting]["ledger"][0]["pre"]
                    if state_index == 0
                    else result["cohorts"][cohort][weighting]["ledger"][-1]["post"]
                )
                endpoint_discrepancies[f"{cohort}/{weighting}/{state_name}"] = (
                    measured - recorded
                )
                if not math.isclose(measured, recorded, abs_tol=2e-6, rel_tol=2e-6):
                    raise ValueError("seed449_endpoint_reconciliation_mismatch")

    # Seed451's independently archived fresh gradients are a checkpoint-level cross-check.
    seed451 = np.load(
        root / "docs/data/seed451-source-policy-alignment/gradient-vectors.npz",
        allow_pickle=False,
    )
    for state_num, checkpoint in ((0, "initializer"), (16, "T_final")):
        ids = np.asarray(
            [members[i]["compact_row"] for i in populations["fresh"]], dtype=np.int64
        )
        names = [members[i]["input_identity"] for i in populations["fresh"]]
        counts = {identity: names.count(identity) for identity in set(names)}
        for weighting, rowweights, suffix in (
            (
                "exposure_weighted",
                np.full(len(ids), 1 / len(ids), np.float64),
                "exposure",
            ),
            (
                "equal_exact_input",
                np.asarray(
                    [1 / (len(counts) * counts[item]) for item in names], np.float64
                ),
                "equal_exact_input",
            ),
        ):
            with torch.no_grad():
                for param, value in zip(
                    model.parameters(), states[state_num], strict=True
                ):
                    param.copy_(torch.from_numpy(value.copy()))
            calculated = _gradient(model, x, p, ids, rowweights)
            key = f"{checkpoint}__U_F_{suffix}"
            if not np.allclose(calculated, seed451[key], atol=2e-5, rtol=2e-5):
                raise ValueError("seed451_gradient_reconciliation_mismatch")
    fresh = recomputed["fresh"]
    labels = {}
    for weighting, value in fresh.items():
        if value["D"] > 0 and value["S"] >= 0.75 * value["D"]:
            label = "recorded_direction_harms_fresh_policy"
        elif value["D"] > 0 and value["R"] >= 0.75 * value["D"]:
            label = "fresh_policy_finite_step_residual_dominant"
        else:
            label = "mixed_or_no_fresh_policy_regression"
        labels[weighting] = label
    overall = (
        next(iter(set(labels.values())))
        if len(set(labels.values())) == 1
        else "weighting_dependent_fresh_attribution"
    )
    if result["classification"] != {"by_weighting": labels, "overall": overall}:
        raise ValueError("classification_mismatch")
    return {
        "status": "valid",
        "classification": overall,
        "rows": len(ledger),
        "states": len(states),
        "seed449_endpoint_discrepancies": endpoint_discrepancies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
