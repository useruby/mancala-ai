"""Retrospective prediction-space decomposition of seed445 value-MSE residuals."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed444_minibatch_denominator import _load_frozen_inputs

ROOT = Path(__file__).resolve().parents[2]
OUT = "docs/data/seed446-prediction-space-decomposition"
WEIGHTINGS = ("exposure_weighted", "equal_input")
ARMS = ("A", "B")
ATOL = 2e-7
RTOL = 2e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights(identities: list[str]) -> dict[str, np.ndarray]:
    counts = {identity: identities.count(identity) for identity in set(identities)}
    return {
        "exposure_weighted": np.full(len(identities), 1.0 / len(identities)),
        "equal_input": np.asarray(
            [1.0 / (len(counts) * counts[key]) for key in identities],
            dtype=np.float64,
        ),
    }


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    return float(
        np.sum(np.asarray(values, dtype=np.float64) * weight, dtype=np.float64)
        / np.sum(weight, dtype=np.float64)
    )


def classify(q: float, n: float) -> str:
    residual = q + n
    if residual > 0 and q >= 0.75 * residual:
        return "squared_prediction_movement_dominant"
    if residual > 0 and n >= 0.75 * residual:
        return "output_response_remainder_dominant"
    return "mixed_or_no_common_residual_component"


def decompose(
    before: np.ndarray,
    after: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    gradient: np.ndarray,
    displacement: np.ndarray,
) -> dict[str, float]:
    """Compute c, q and n, with independent parameter-space first-order dot."""
    pre = np.asarray(before, dtype=np.float64)
    post = np.asarray(after, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    delta = post - pre
    c = 2.0 * weighted_mean((pre - y) * delta, weight)
    q = weighted_mean(delta * delta, weight)
    s = float(np.dot(np.asarray(gradient, dtype=np.float64), displacement))
    n = c - s
    d = weighted_mean((post - y) ** 2, weight) - weighted_mean((pre - y) ** 2, weight)
    return {"c": c, "q": q, "s": s, "n": n, "d": d}


def reconstruct_targets(root: Path) -> tuple[np.ndarray, list[int], list[str]]:
    registration_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    registration = json.loads(registration_path.read_text())
    arrays = _load_frozen_inputs(root, registration)
    source_targets = np.asarray(arrays[2], dtype=np.float32)
    with gzip.open(
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        members = [json.loads(line) for line in stream]
    selected = [
        row
        for row in members
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    rows = [int(row["compact_row"]) for row in selected]
    identities = [str(row["input_identity"]) for row in selected]
    target = source_targets[np.asarray(rows, dtype=np.int64)].reshape(-1)
    archived = json.loads(
        (
            root / "docs/data/seed445-finite-step-attribution/row-identities.json"
        ).read_text()
    )
    if archived["compact_row"] != rows or archived["input_identity"] != identities:
        raise ValueError("target_source_row_correspondence_mismatch")
    return target, rows, identities


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    receipt = json.loads((out / "receipt.json").read_text())
    for relative, expected in receipt["bound_inputs_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"bound_input_hash_mismatch:{relative}")
    targets, rows, identities = reconstruct_targets(root)
    target_doc = json.loads((out / "ordered-targets.json").read_text())
    if target_doc["compact_row"] != rows or target_doc["input_identity"] != identities:
        raise ValueError("ordered_target_identity_mismatch")
    if not np.array_equal(
        np.asarray(target_doc["value_target"], dtype=np.float32), targets
    ):
        raise ValueError("ordered_target_values_mismatch")
    ids_doc = json.loads(
        (
            root / "docs/data/seed445-finite-step-attribution/row-identities.json"
        ).read_text()
    )
    if ids_doc["input_identity"] != identities or ids_doc["compact_row"] != rows:
        raise ValueError("seed445_identity_mismatch")
    weights_by_kind = weights(identities)
    predictions = np.load(
        root / "docs/data/seed445-finite-step-attribution/predictions.npz",
        allow_pickle=False,
    )
    gradients = np.load(
        root / "docs/data/seed445-finite-step-attribution/gradients.npz",
        allow_pickle=False,
    )
    states_file = np.load(
        root / "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz",
        allow_pickle=False,
    )
    ledger = json.loads(
        (
            root / "docs/data/seed445-finite-step-attribution/step-ledger.json"
        ).read_text()
    )
    rows_out = []
    totals: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        totals[arm] = {}
        for weighting in WEIGHTINGS:
            sums = {key: 0.0 for key in ("c", "q", "n", "s", "d")}
            for step in range(16):
                before = predictions[f"{arm}_{step:02d}_values"]
                after = predictions[f"{arm}_{step + 1:02d}_values"]
                # State vectors follow the canonical #445 parameter layout.
                preparts = [
                    states_file[f"{arm}_{step:02d}_pre_{i:02d}"] for i in range(22)
                ]
                postparts = [
                    states_file[f"{arm}_{step:02d}_post_{i:02d}"] for i in range(22)
                ]
                displacement = np.concatenate(
                    [
                        (b.astype(np.float64) - a.astype(np.float64)).ravel()
                        for a, b in zip(preparts, postparts, strict=True)
                    ]
                )
                grad = gradients[f"{arm}_value_mse_{weighting}_{step:02d}"]
                item = decompose(
                    before,
                    after,
                    targets,
                    weights_by_kind[weighting],
                    grad,
                    displacement,
                )
                record = next(
                    r
                    for r in ledger["arms"][arm]["step_ledger"]
                    if r["step"] == step + 1
                    and r["objective"] == "value_mse"
                    and r["weighting"] == weighting
                )
                if not np.isclose(
                    item["d"], item["c"] + item["q"], atol=ATOL, rtol=RTOL
                ):
                    raise ValueError("prediction_identity_d_equals_c_plus_q")
                if not np.isclose(
                    item["d"] - item["s"], item["q"] + item["n"], atol=ATOL, rtol=RTOL
                ):
                    raise ValueError("prediction_identity_r_equals_q_plus_n")
                for key, value in (("d", item["d"]), ("s", item["s"])):
                    if not np.isclose(value, record[key], atol=ATOL, rtol=RTOL):
                        raise ValueError(f"seed445_ledger_{key}_mismatch")
                rows_out.append(
                    {"arm": arm, "weighting": weighting, "step": step + 1, **item}
                )
                for key in sums:
                    sums[key] += item[key]
            if not np.isclose(
                sums["d"], sums["s"] + sums["q"] + sums["n"], atol=ATOL, rtol=RTOL
            ):
                raise ValueError("total_reconciliation_failed")
            totals[arm][weighting] = sums
    expected = json.loads((out / "step-ledger.json").read_text())
    if expected["rows"] != rows_out or expected["totals"] != totals:
        raise ValueError("decomposition_ledger_mismatch")
    result = {
        weight: classify(totals["B"][weight]["q"], totals["B"][weight]["n"])
        for weight in WEIGHTINGS
    }
    if len(set(result.values())) != 1:
        classification = "mixed_or_no_common_residual_component"
    else:
        classification = next(iter(result.values()))
    if expected["classification"] != classification:
        raise ValueError("classification_mismatch")
    return {
        "status": "valid",
        "classification": classification,
        "weighting_classifications": result,
        "exposures": len(rows),
    }


def publish(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    targets, rows, identities = reconstruct_targets(root)
    (out / "ordered-targets.json").write_text(
        json.dumps(
            {
                "schema": "seed446-ordered-targets-v1",
                "compact_row": rows,
                "input_identity": identities,
                "value_target": targets.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    weights_by_kind = weights(identities)
    predictions = np.load(
        root / "docs/data/seed445-finite-step-attribution/predictions.npz",
        allow_pickle=False,
    )
    gradients = np.load(
        root / "docs/data/seed445-finite-step-attribution/gradients.npz",
        allow_pickle=False,
    )
    archive = np.load(
        root / "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz",
        allow_pickle=False,
    )
    previous = json.loads(
        (
            root / "docs/data/seed445-finite-step-attribution/step-ledger.json"
        ).read_text()
    )
    ledger_rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        totals[arm] = {}
        for weighting in WEIGHTINGS:
            sums = {key: 0.0 for key in ("c", "q", "n", "s", "d")}
            for step in range(16):
                before = predictions[f"{arm}_{step:02d}_values"]
                after = predictions[f"{arm}_{step + 1:02d}_values"]
                preparts = [archive[f"{arm}_{step:02d}_pre_{i:02d}"] for i in range(22)]
                postparts = [
                    archive[f"{arm}_{step:02d}_post_{i:02d}"] for i in range(22)
                ]
                displacement = np.concatenate(
                    [
                        (b.astype(np.float64) - a.astype(np.float64)).ravel()
                        for a, b in zip(preparts, postparts, strict=True)
                    ]
                )
                gradient = gradients[f"{arm}_value_mse_{weighting}_{step:02d}"]
                item = decompose(
                    before,
                    after,
                    targets,
                    weights_by_kind[weighting],
                    gradient,
                    displacement,
                )
                baseline = next(
                    r
                    for r in previous["arms"][arm]["step_ledger"]
                    if r["step"] == step + 1
                    and r["objective"] == "value_mse"
                    and r["weighting"] == weighting
                )
                if not np.isclose(
                    item["d"], baseline["d"], atol=ATOL, rtol=RTOL
                ) or not np.isclose(item["s"], baseline["s"], atol=ATOL, rtol=RTOL):
                    raise ValueError("seed445_reconciliation_failed")
                ledger_rows.append(
                    {"arm": arm, "weighting": weighting, "step": step + 1, **item}
                )
                for key in sums:
                    sums[key] += item[key]
            totals[arm][weighting] = sums
    b_classifications = {
        weighting: classify(totals["B"][weighting]["q"], totals["B"][weighting]["n"])
        for weighting in WEIGHTINGS
    }
    classification = (
        next(iter(b_classifications.values()))
        if len(set(b_classifications.values())) == 1
        else "mixed_or_no_common_residual_component"
    )
    decomposition = {
        "schema": "seed446-step-ledger-v1",
        "rows": ledger_rows,
        "totals": totals,
        "weighting_classifications": b_classifications,
        "classification": classification,
    }
    (out / "step-ledger.json").write_text(
        json.dumps(decomposition, indent=2, sort_keys=True) + "\n"
    )
    (out / "analysis.md").write_text(
        "# seed446 — prediction-space decomposition\n\n"
        "Retrospective decomposition of seed445's already-known value-MSE residual, using archived float32 predictions and targets with float64 arithmetic. No models were evaluated and no gradients were computed.\n\n"
        f"**B classification under both weightings: `{classification}`.**\n\n"
        "| Weighting | Q = sum(q) | N = sum(n) | R = Q + N | Q/R | N/R |\n|---|---:|---:|---:|---:|---:|\n"
        + "".join(
            f"| {w} | {totals['B'][w]['q']:.12g} | {totals['B'][w]['n']:.12g} | {totals['B'][w]['q'] + totals['B'][w]['n']:.12g} | {totals['B'][w]['q'] / (totals['B'][w]['q'] + totals['B'][w]['n']):.8g} | {totals['B'][w]['n'] / (totals['B'][w]['q'] + totals['B'][w]['n']):.8g} |\n"
            for w in WEIGHTINGS
        )
        + "\n`q` is nonnegative squared prediction movement. `n = c - s` is the signed output-response remainder relative to the independently recomputed parameter-space first-order dot; it includes nonlinear and numerical effects and is not pure network curvature or a causal intervention. This retrospective result proposes no training constraint and authorizes no training, gate change, strength claim, or promotion. The historical `finite_step_value_residual_dominant` and `close_kl_capped_step_branch` findings are preserved.\n"
    )
    protocol = {
        "schema": "seed446-frozen-protocol-v1",
        "status": "retrospective_archived_input_decomposition",
        "bound_seed445": "docs/data/seed445-finite-step-attribution/protocol-v2.json; receipt.json; verifier-amendment-1.json; predictions.npz; gradients.npz; step-ledger.json; parameter-layout.json; row-identities.json",
        "authoritative_trajectory": "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz",
        "target_reconstruction": "seed416 registration-v3 loader semantics over bound compressed source snapshots; unseen and active_stones > 32 membership order; float32 loader value targets selected by compact_row",
        "population": {"exposures": 2607, "identities": 1242},
        "weightings": {
            "exposure_weighted": "uniform across all exposure rows",
            "equal_input": "equal identity mass, retaining within-identity exposures",
        },
        "arithmetic": "float64 from archived float32 predictions/targets; c=2*weighted_mean((pre-target)*delta); q=weighted_mean(delta^2); s=dot(archived full gradient,float64 adjacent-state difference); n=c-s; d=c+q",
        "tolerances": {"absolute": ATOL, "relative": RTOL},
        "classification": "B per weighting: R=Q+N; if R>0 and Q>=.75R squared_prediction_movement_dominant; else if R>0 and N>=.75R output_response_remainder_dominant; else mixed_or_no_common_residual_component",
        "interpretation": "descriptive, signed, retrospective; n includes nonlinear and numerical effects; no causal claim or authorization",
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    bound = {
        "docs/data/seed445-finite-step-attribution/protocol-v2.json",
        "docs/data/seed445-finite-step-attribution/receipt.json",
        "docs/data/seed445-finite-step-attribution/verifier-amendment-1.json",
        "docs/data/seed445-finite-step-attribution/predictions.npz",
        "docs/data/seed445-finite-step-attribution/gradients.npz",
        "docs/data/seed445-finite-step-attribution/step-ledger.json",
        "docs/data/seed445-finite-step-attribution/parameter-layout.json",
        "docs/data/seed445-finite-step-attribution/row-identities.json",
        "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
    }
    receipt = {
        "schema": "seed446-publication-receipt-v1",
        "status": "published_retrospective_decomposition",
        "bound_inputs_sha256": {path: sha(root / path) for path in sorted(bound)},
        "artifacts_sha256": {
            name: sha(out / name)
            for name in (
                "protocol.json",
                "ordered-targets.json",
                "step-ledger.json",
                "analysis.md",
            )
        },
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def bind_supplemental(root: Path) -> None:
    """Bind verifier/tests after the already-observed decomposition execution."""
    root = root.resolve()
    out = root / OUT
    receipt445 = json.loads(
        (root / "docs/data/seed445-finite-step-attribution/receipt.json").read_text()
    )
    seed445_protocol = json.loads(
        (
            root / "docs/data/seed445-finite-step-attribution/protocol-v2.json"
        ).read_text()
    )
    bound = (
        set(receipt445["artifacts_sha256"])
        | set(receipt445["historical_seed442_sha256"])
        | set(seed445_protocol["authoritative_input_sha256"])
        | set(seed445_protocol["execution_source_sha256"])
    )
    bound.update(
        {
            "docs/data/seed445-finite-step-attribution/receipt.json",
            "docs/data/seed442-kl-capped-adam-screen/evidence.json",
            "ml/alphazero_lite/seed446_prediction_space_decomposition.py",
            "ml/alphazero_lite/verify_seed446_prediction_space_decomposition.py",
            "ml/alphazero_lite/test_seed446_prediction_space_decomposition.py",
            "docs/data/seed446-prediction-space-decomposition/protocol.json",
            "docs/data/seed446-prediction-space-decomposition/ordered-targets.json",
            "docs/data/seed446-prediction-space-decomposition/step-ledger.json",
            "docs/data/seed446-prediction-space-decomposition/analysis.md",
            "docs/data/seed446-prediction-space-decomposition/receipt.json",
            "docs/data/seed446-prediction-space-decomposition/verification-report.json",
        }
    )
    supplemental = {
        "schema": "seed446-post-execution-supplemental-receipt-v1",
        "status": "post_execution_verifier_and_test_binding",
        "observed_outcome_before_verifier_binding": True,
        "relationship_to_seed445": {
            "receipt": "docs/data/seed445-finite-step-attribution/receipt.json",
            "amendment": "docs/data/seed445-finite-step-attribution/verifier-amendment-1.json",
            "interpretation": "#445's original verified execution and post-execution batching amendment remain intact; this supplemental receipt binds the subsequently implemented seed446 verifier/tests and does not imply they predated the result.",
        },
        "bound_sha256": {path: sha(root / path) for path in sorted(bound)},
    }
    (out / "supplemental-receipt.json").write_text(
        json.dumps(supplemental, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bind-supplemental", action="store_true")
    args = parser.parse_args()
    if args.bind_supplemental:
        bind_supplemental(args.root)
    elif args.verify:
        print(json.dumps(verify(args.root), indent=2, sort_keys=True))
    else:
        publish(args.root)


if __name__ == "__main__":
    main()
