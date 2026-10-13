"""Post-execution independent metric reconstruction for the frozen seed474 attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)
from ml.alphazero_lite.seed471_fresh_only_adam_ablation import INIT
from ml.alphazero_lite.seed473_adam_moment_reset import OUT as SEED473_OUT
from ml.alphazero_lite.seed474_fresh_value_removal import OUT

ROOT = Path(__file__).resolve().parents[2]
REPORT = "verification-supplement.json"
RECEIPT = "verification-supplement-receipt.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(actual: Any, expected: Any, location: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"inventory_mismatch:{location}")
        for key, value in expected.items():
            _close(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=2e-6, atol=2e-8):
            raise ValueError(f"value_mismatch:{location}")
    elif actual != expected:
        raise ValueError(f"value_mismatch:{location}")


def reconstruct(root: Path, out: Path) -> dict[str, Any]:
    """Recompute metrics from authenticated inputs and independently checked endpoints."""
    root, out = root.resolve(), out.resolve()
    # The frozen verifier reconstructs every gradient, reset update, moment, counter,
    # parameter state, exact C-vs-seed473-R comparison, and endpoint prediction.
    from ml.alphazero_lite import (
        verify_seed474_fresh_value_removal as trajectory_verifier,
    )

    previous = trajectory_verifier.OUT
    try:
        trajectory_verifier.OUT = out
        trajectory_verifier.verify(root)
    finally:
        trajectory_verifier.OUT = previous

    data = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = data["training"], data["evaluation"]
    torch.set_num_threads(1)
    models = {"initializer": train.PolicyValueNet((96, 3), "residual_v3", 27)}
    train.load_checkpoint_into_model(models["initializer"], root / INIT)
    for arm in ("C", "P"):
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        model.load_state_dict(models["initializer"].state_dict())
        with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as archive:
            with torch.no_grad():
                for index, parameter in enumerate(model.parameters()):
                    parameter.copy_(
                        torch.from_numpy(archive[f"15_post_{index:02d}"].copy())
                    )
        models[arm] = model

    eval_rows = ev["compact_rows"]
    eval_x = torch.from_numpy(tr["inputs"][eval_rows].copy())
    legal_eval = tr["legal_masks"][eval_rows]
    identities = ev["input_identity"].tolist()
    cohorts = {
        "fresh": np.flatnonzero(ev["source_ids"] == 0),
        "pooled": np.arange(len(identities)),
    }
    arms: dict[str, Any] = {}
    for arm, model in models.items():
        model.eval()
        with torch.inference_mode():
            logits, value = model(eval_x)
            logp = (
                F.log_softmax(
                    logits.masked_fill(
                        torch.from_numpy(legal_eval.copy()) == 0, -torch.inf
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            values = value.reshape(-1).cpu().numpy()
        arms[arm] = {}
        for cohort, selected in cohorts.items():
            groups: dict[str, list[int]] = {}
            for local, occurrence in enumerate(selected):
                groups.setdefault(identities[int(occurrence)], []).append(local)
            row_ce = (
                -(
                    ev["policy_targets"][selected]
                    * np.where(np.isfinite(logp[selected]), logp[selected], 0)
                )
                .sum(axis=1)
                .astype(np.float64)
            )
            row_mse = np.square(
                values[selected].astype(np.float64)
                - ev["value_targets"].reshape(-1)[selected].astype(np.float64)
            )
            arms[arm][cohort] = {
                "occurrences": int(len(selected)),
                "identities": len(groups),
                "exposure_weighted": {
                    "policy_ce": float(np.mean(row_ce)),
                    "value_mse": float(np.mean(row_mse)),
                },
                "equal_input": {
                    "policy_ce": float(
                        np.mean([row_ce[idx].mean() for idx in groups.values()])
                    ),
                    "value_mse": float(
                        np.mean([row_mse[idx].mean() for idx in groups.values()])
                    ),
                },
            }
            # Recheck the original occurrence archive, targets, legal masks, and weights.
            ids = [identities[int(i)] for i in selected]
            expected = {
                "prediction": logp[selected].astype(np.float32),
                "value": values[selected],
                "policy_target": ev["policy_targets"][selected],
                "value_target": ev["value_targets"].reshape(-1)[selected],
                "input_identity": np.asarray(ids, dtype="U256"),
                "legal_mask": legal_eval[selected],
                "exposure_weight": np.full(
                    len(selected), 1 / len(selected), dtype=np.float64
                ),
                "equal_input_weight": np.asarray(
                    [1 / (len(groups) * len(groups[item])) for item in ids],
                    dtype=np.float64,
                ),
            }
            with np.load(
                out / f"{arm}-{cohort}-predictions.npz", allow_pickle=False
            ) as archive:
                if set(archive.files) != set(expected):
                    raise ValueError(f"prediction_inventory_mismatch:{arm}:{cohort}")
                for name, expected_array in expected.items():
                    actual = archive[name]
                    matches = (
                        np.array_equal(actual, expected_array)
                        if expected_array.dtype.kind in "US"
                        else np.allclose(actual, expected_array, rtol=2e-6, atol=2e-8)
                    )
                    if (
                        actual.dtype != expected_array.dtype
                        or actual.shape != expected_array.shape
                        or not matches
                    ):
                        raise ValueError(
                            f"prediction_or_weight_mismatch:{arm}:{cohort}:{name}"
                        )

    population = tr["expanded_replay_indices"][tr["train_positions"]]
    training: dict[str, Any] = {}
    for arm, model in models.items():
        ce_chunks, value_chunks = [], []
        model.eval()
        for start in range(0, len(population), 1024):
            rows = population[start : start + 1024]
            with torch.inference_mode():
                logits, predictions = model(torch.from_numpy(tr["inputs"][rows].copy()))
                legal = torch.from_numpy(tr["legal_masks"][rows].copy())
                ce = -(
                    torch.from_numpy(tr["policy_targets"][rows].copy())
                    * F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
                ).sum(1)
            ce_chunks.append(ce.cpu().numpy().astype(np.float64))
            value_chunks.append(
                predictions.reshape(-1).cpu().numpy().astype(np.float64)
            )
        ce = np.concatenate(ce_chunks)
        predictions = np.concatenate(value_chunks)
        fresh = tr["fresh_source_mask"][population].astype(np.float64)
        coefficient = tr["policy_coefficients"][population].astype(np.float64)
        policy_den = float(np.sum(coefficient * fresh, dtype=np.float64))
        policy_num = float(np.sum(ce * coefficient * fresh, dtype=np.float64))
        error = np.abs(
            predictions - tr["value_targets"][population].reshape(-1).astype(np.float64)
        )
        huber = np.where(error < 1, 0.5 * error**2, error - 0.5)
        value_num = float(np.sum(huber * fresh, dtype=np.float64))
        training[arm] = {
            "occurrences": int(len(population)),
            "policy_numerator": policy_num,
            "policy_denominator": policy_den,
            "policy_ce": policy_num / policy_den,
            "value_huber_numerator": value_num,
            "value_denominator": int(len(population)),
            "J": policy_num / policy_den + 0.3 * value_num / len(population),
        }
        published = json.loads((out / "endpoint-evidence.json").read_text())["arms"][
            arm
        ]
        if not np.isclose(
            training[arm]["policy_ce"],
            published["training_policy_ce"]["total"],
            rtol=2e-6,
            atol=2e-8,
        ):
            raise ValueError(f"training_policy_ce_mismatch:{arm}")
        for actual, expected_value, label in (
            (published["training_policy_ce"]["numerator"], policy_num, "policy_num"),
            (published["training_policy_ce"]["denominator"], policy_den, "policy_den"),
            (published["training_J"]["value_numerator"], value_num, "value_num"),
            (
                published["training_J"]["value_denominator"],
                len(population),
                "value_den",
            ),
            (published["training_J"]["total"], training[arm]["J"], "J"),
        ):
            if not np.isclose(actual, expected_value, rtol=2e-6, atol=2e-8):
                raise ValueError(
                    f"training_objective_accounting_mismatch:{arm}:{label}"
                )
        arms[arm]["training_policy_ce"] = {
            k: training[arm][k]
            for k in (
                "occurrences",
                "policy_numerator",
                "policy_denominator",
                "policy_ce",
            )
        }
        arms[arm]["training_J"] = {
            k: training[arm][k]
            for k in ("value_huber_numerator", "value_denominator", "J")
        }

    effects: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for cohort in cohorts:
        effects[cohort] = {}
        for weighting in ("exposure_weighted", "equal_input"):
            effects[cohort][weighting] = {
                arm: {
                    metric: arms[arm][cohort][weighting][metric]
                    - arms["initializer"][cohort][weighting][metric]
                    for metric in ("policy_ce", "value_mse")
                }
                for arm in ("C", "P")
            }
            effects[cohort][weighting]["P_minus_C"] = {
                metric: arms["P"][cohort][weighting][metric]
                - arms["C"][cohort][weighting][metric]
                for metric in ("policy_ce", "value_mse")
            }
    for weighting in ("exposure_weighted", "equal_input"):
        gates[f"fresh_ce_vs_initializer_{weighting}"] = (
            arms["P"]["fresh"][weighting]["policy_ce"]
            <= arms["initializer"]["fresh"][weighting]["policy_ce"] - 0.001
        )
        gates[f"fresh_ce_vs_C_{weighting}"] = (
            arms["P"]["fresh"][weighting]["policy_ce"]
            <= arms["C"]["fresh"][weighting]["policy_ce"] - 0.001
        )
        gates[f"pooled_ce_vs_initializer_{weighting}"] = (
            arms["P"]["pooled"][weighting]["policy_ce"]
            <= arms["initializer"]["pooled"][weighting]["policy_ce"] + 0.005
        )
        gates[f"pooled_mse_vs_initializer_{weighting}"] = (
            arms["P"]["pooled"][weighting]["value_mse"]
            <= arms["initializer"]["pooled"][weighting]["value_mse"] + 0.01
        )
    gates["training_policy_ce_decreases"] = (
        training["P"]["policy_ce"] < training["initializer"]["policy_ce"]
    )
    validity = json.loads((out / "validity-gate.json").read_text())
    gates["exact_C_reproduction"] = validity.get("passed") is True and _sha(
        root / SEED473_OUT / "R-steps.npz"
    ) == validity.get("seed473_R_steps_sha256")
    decision = (
        "advance_to_fresh_confirmation"
        if all(gates.values())
        else "close_reset_adam_policy_only_branch"
    )
    published = json.loads((out / "endpoint-evidence.json").read_text())
    _close(
        published["arms"],
        {
            arm: {
                **{c: arms[arm][c] for c in cohorts},
                "training_policy_ce": published["arms"][arm]["training_policy_ce"],
                "training_J": published["arms"][arm]["training_J"],
            }
            for arm in arms
        },
        "arms",
    )
    _close(published["effects"], effects, "effects")
    if (
        published["gates"] != gates
        or published["decision"] != decision
        or published["classification"] != decision
    ):
        raise ValueError("gate_or_classification_mismatch")

    reg = json.loads((out / "registration.json").read_text())
    files = {
        p.name: _sha(p)
        for p in sorted(out.iterdir())
        if p.is_file() and p.name not in (REPORT, RECEIPT)
    }
    return {
        "schema": "seed474-post-execution-verification-supplement-v1",
        "verification": "independent_reconstruction_passed",
        "original_receipt_sha256": _sha(out / "receipt.json"),
        "registration_sha256": _sha(out / "registration.json"),
        "registered_source_sha256": reg["source_closure_sha256"],
        "authenticated_input_sha256": reg["inputs_sha256"],
        "seed473_R_steps_sha256": _sha(root / SEED473_OUT / "R-steps.npz"),
        "original_publication_inventory_sha256": files,
        "arms": arms,
        "effects": effects,
        "gates": gates,
        "decision": decision,
        "scope": "exploratory reused evaluation cohort; no independent-confirmation or strength claim",
        "historical_preservation_incident": {
            "description": "An earlier seed472 receipt-rebound regression invocation rewrote evidence.npz and receipt.json while restoring temporary mutations by NPZ reserialization.",
            "recovery": "Restored both files from the committed seed472 revision; verified their SHA256 identities and reran the seed472 independent verifier. Subsequent seed472 tamper tests run against a temporary publication copy.",
            "evidence_sha256_committed_and_current": "bf2632c89b7dbed6a3f19f2a3296494dd811e0d599450a6b4a50860c1c9ee29e",
            "evidence_git_blob_identity": "0b8b7080c6b2fb04bb788aaeb3bd1226ebd4c489",
            "receipt_sha256_committed_and_current": "86cf7eeb8676dbcde1243081b67dfc30fee88f87a392788d3646c13f8da2b90c",
            "receipt_git_blob_identity": "961c6af11abaae6dac0c524e97f1141601c570a5",
            "mtime_note": "The original pre-incident mtimes were not recorded. Current verification records and preserves the restored files' before/after mtimes.",
        },
    }


def publish(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    report = reconstruct(root, out)
    report_path = out / REPORT
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "seed474-post-execution-supplement-receipt-v1",
        "supplement_source_sha256": _sha(Path(__file__).resolve()),
        "report_sha256": _sha(report_path),
        "original_receipt_sha256": _sha(out / "receipt.json"),
        "status": "post_execution_supplement_bound",
    }
    (out / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return report


def verify(root: Path, out: Path | None = None) -> None:
    root = root.resolve()
    out = (out or (root / OUT)).resolve()
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in out.iterdir()
        if p.is_file()
    }
    try:
        expected = reconstruct(root, out)
        actual = json.loads((out / REPORT).read_text())
        _close(actual, expected, "supplement")
        receipt = json.loads((out / RECEIPT).read_text())
        if receipt != {
            "schema": "seed474-post-execution-supplement-receipt-v1",
            "supplement_source_sha256": _sha(Path(__file__).resolve()),
            "report_sha256": _sha(out / REPORT),
            "original_receipt_sha256": _sha(out / "receipt.json"),
            "status": "post_execution_supplement_bound",
        }:
            raise ValueError("supplement_receipt_mismatch")
    finally:
        after = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in out.iterdir()
            if p.is_file()
        }
        if before != after:
            raise ValueError("supplement_verification_mutated_evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "publish":
        report = publish(args.root)
        print(f"seed474 supplement published: {report['decision']}")
    else:
        verify(args.root)
        print("seed474 post-execution supplement verification passed")


if __name__ == "__main__":
    main()
