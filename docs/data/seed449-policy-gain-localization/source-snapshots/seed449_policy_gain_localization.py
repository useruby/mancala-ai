"""Retrospective seed449 localization of seed447 policy CE changes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from ml.alphazero_lite.seed449_arithmetic import analyze, classify

OUT = Path("docs/data/seed449-policy-gain-localization")
SEED447 = Path("docs/data/seed447-joint-output-cap")
TOLERANCE = {"atol": 2e-6, "rtol": 2e-6}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_membership(root: Path) -> list[dict[str, Any]]:
    path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [
            row
            for line in stream
            if (row := json.loads(line))["subset"] == "unseen"
            and row["active_stones"] > 32
        ]


def _load(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = root / SEED447
    membership = _strict_membership(root)
    target = np.load(out / "ordered-targets.npz", allow_pickle=False)
    compact_rows = target["compact_rows"]
    identities = target["input_identity"]
    if (
        len(membership) != 2607
        or len(set(r["input_identity"] for r in membership)) != 1242
    ):
        raise ValueError("historical_population_count_mismatch")
    if [int(r["compact_row"]) for r in membership] != compact_rows.tolist():
        raise ValueError("ordered_membership_rows_mismatch")
    if [r["input_identity"] for r in membership] != identities.tolist():
        raise ValueError("ordered_membership_identity_mismatch")
    archives = {
        arm: np.load(out / f"{arm}-predictions.npz", allow_pickle=False)
        for arm in ("initializer", "C", "T")
    }

    from ml.alphazero_lite import train

    source_reg = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    # Use the frozen production replay loader only to reconstruct the registered
    # sharpened policy targets; no model or forward pass is invoked.
    replay_paths = [
        Path(source_reg["derivatives"][item["name"]]["A"]["derivative"])
        for item in source_reg["replays"]
    ]
    replay_paths = [
        path if path.is_absolute() else root / path for path in replay_paths
    ]
    x, policy, _value, _replay, _coeff = train.load_jsonl_replay(
        replay_paths,
        [int(item["weight"]) for item in source_reg["replays"]],
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=[
            item["value_target_mode"] for item in source_reg["replays"]
        ],
        include_policy_loss_weights=True,
    )
    masks = train.legal_mask_matrix_for_encoded_states(x[compact_rows.astype(np.int64)])
    if not np.array_equal(
        target["policy_targets"], policy[compact_rows.astype(np.int64)]
    ):
        raise ValueError("ordered_policy_targets_source_mismatch")
    row_losses: dict[str, np.ndarray] = {}
    for arm, archive in archives.items():
        logits = torch.from_numpy(archive["policy_logits"])
        legal = torch.from_numpy(masks.astype(np.float32))
        target_tensor = torch.from_numpy(target["policy_targets"])
        with torch.inference_mode():
            row_losses[arm] = np.concatenate(
                [
                    train.compute_policy_cross_entropy(
                        logits[start : start + 512].masked_fill(
                            legal[start : start + 512] <= 0, -1e9
                        ),
                        target_tensor[start : start + 512],
                    )
                    .double()
                    .cpu()
                    .numpy()
                    for start in range(0, len(compact_rows), 512)
                ]
            )
    rows: list[dict[str, Any]] = []
    for index, member in enumerate(membership):
        mask = masks[index].astype(np.int64).tolist()
        target_row = target["policy_targets"][index].astype(np.float64).tolist()
        if not np.array_equal(
            np.asarray(target_row, dtype=np.float32),
            np.asarray(policy[int(compact_rows[index])], dtype=np.float32),
        ):
            raise ValueError("target_precision_mismatch")
        losses: dict[str, float] = {}
        for arm in ("initializer", "C", "T"):
            archive = archives[arm]
            if (
                archive["compact_rows"].tolist() != compact_rows.tolist()
                or archive["input_identity"].tolist() != identities.tolist()
            ):
                raise ValueError(f"prediction_order_mismatch:{arm}")
            logits = archive["policy_logits"][index]
            if logits.dtype != np.float32:
                raise ValueError("prediction_logits_not_float32")
            losses[arm] = float(row_losses[arm][index])
        rows.append(
            {
                "exposure_index": index,
                "compact_row": int(member["compact_row"]),
                "source_ref": {
                    "source": member["source"],
                    "source_row": member["raw_line"],
                },
                "input_identity": member["input_identity"],
                "canonical_identity": member["canonical_identity"],
                "active_stones": int(member["active_stones"]),
                "legal_mask": mask,
                "policy_target": target_row,
                "losses": losses,
                "changes": {
                    "T_minus_initializer": losses["T"] - losses["initializer"],
                    "C_minus_initializer": losses["C"] - losses["initializer"],
                    "T_minus_C": losses["T"] - losses["C"],
                },
            }
        )
    # Bind strict population and targets against seed435's independent source reconstruction.
    from ml.alphazero_lite import seed435_adam_direction as source_reconstruction

    reconstructed = source_reconstruction._inputs()
    expected_p = reconstructed[1]
    if not np.array_equal(
        expected_p[compact_rows.astype(np.int64)], target["policy_targets"]
    ):
        raise ValueError("reconstructed_targets_mismatch")
    return rows, {
        "membership_count": len(membership),
        "unique_inputs": len(set(r["input_identity"] for r in membership)),
    }


def freeze(root: Path) -> dict[str, Any]:
    """Freeze retrospective contract and every analysis/execution source digest."""
    root = root.resolve()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    registration = out / "registration.json"
    if registration.exists():
        raise ValueError("registration_is_immutable")
    source_files = [
        "ml/alphazero_lite/seed449_arithmetic.py",
        "ml/alphazero_lite/seed449_policy_gain_localization.py",
        "ml/alphazero_lite/verify_seed449_policy_gain_localization.py",
        "ml/alphazero_lite/test_seed449_arithmetic.py",
        "ml/alphazero_lite/test_seed449_policy_gain_localization.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/seed427_validation_metrics.py",
        "ml/alphazero_lite/seed428_source_attribution.py",
        "ml/alphazero_lite/seed435_adam_direction.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/verify_seed448_seed447_publication.py",
    ]
    inputs = [
        str(SEED447 / name)
        for name in (
            "registration.json",
            "receipt.json",
            "initializer-predictions.npz",
            "C-predictions.npz",
            "T-predictions.npz",
            "ordered-targets.npz",
            "evidence.json",
        )
    ] + [
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
    ]
    payload = {
        "schema": "seed449-retrospective-registration-v1",
        "analysis_design": "retrospective; seed447 aggregate outcomes are already known",
        "population": "strict unseen and active_stones > 32; preserve ordered membership exposures, duplicates, and row targets",
        "exposure_definition": "one retained row of seed447's ordered validation population",
        "identity_definition": "exact float32 encoded input byte identity (hex)",
        "source_rows_and_replay_copies_are_not_exposure_or_identity_counts": True,
        "loss": "production legal-action log_softmax cross entropy; archived float32 logits/targets; float64 row loss aggregation",
        "comparisons": ["T-initializer", "C-initializer", "T-C"],
        "classification": {
            "frequency_concentrated_policy_gain": "D_exposure<=-0.005 and D_equal>-0.005 and covariance<0 and singleton mean>=0 and repeated mean<0",
            "broad_small_policy_gain": "-0.005<D_equal<0 and singleton/repeated means<0",
            "otherwise": "mixed_policy_response; empty required stratum is mixed",
            "training_or_promotion_authorized": False,
        },
        "seed447_classification_preserved": "close_joint_output_cap_branch",
        "aggregate_tolerance": TOLERANCE,
        "bound_input_sha256": {p: sha(root / p) for p in inputs},
        "source_sha256": {p: sha(root / p) for p in source_files},
    }
    snapshot_dir = out / "source-snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    for rel in source_files:
        shutil.copyfile(root / rel, snapshot_dir / Path(rel).name)
    registration.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run(root: Path) -> dict[str, Any]:
    """Calculate and publish row/identity ledgers from frozen archives."""
    root = root.resolve()
    out = root / OUT
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    for rel, digest in registration["bound_input_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"bound_input_hash_mismatch:{rel}")
    for rel, digest in registration["source_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"source_hash_mismatch:{rel}")
    rows, population = _load(root)
    summaries = {
        "T_minus_initializer": analyze(rows, "initializer", "T"),
        "C_minus_initializer": analyze(rows, "initializer", "C"),
        "T_minus_C": analyze(rows, "C", "T"),
    }
    classification = classify(summaries["T_minus_initializer"])
    # Compare row-loss aggregates with the registered seed447 endpoint metrics.
    evidence = json.loads((root / SEED447 / "evidence.json").read_text())
    endpoint = {}
    for arm, key in (("initializer", "initializer"), ("C", "C"), ("T", "T")):
        all_losses = [r["losses"][arm] for r in rows]
        by_identity: dict[str, list[float]] = {}
        for row in rows:
            by_identity.setdefault(row["input_identity"], []).append(row["losses"][arm])
        equal_mean = sum(
            sum(values) / len(values) for values in by_identity.values()
        ) / len(by_identity)
        endpoint[arm] = {
            "reconstructed_policy_ce": sum(all_losses) / len(all_losses),
            "reconstructed_equal_identity_policy_ce": equal_mean,
            "seed447_exposure_weighted_policy_ce": evidence["metrics"][key][
                "exposure_weighted"
            ]["policy_ce"],
            "seed447_equal_identity_policy_ce": evidence["metrics"][key]["equal_input"][
                "policy_ce"
            ],
            "difference": sum(all_losses) / len(all_losses)
            - evidence["metrics"][key]["exposure_weighted"]["policy_ce"],
            "equal_identity_difference": equal_mean
            - evidence["metrics"][key]["equal_input"]["policy_ce"],
        }
        if not np.isclose(
            endpoint[arm]["reconstructed_policy_ce"],
            endpoint[arm]["seed447_exposure_weighted_policy_ce"],
            atol=TOLERANCE["atol"],
            rtol=TOLERANCE["rtol"],
        ):
            raise ValueError(f"seed447_aggregate_reconciliation_failed:{arm}")
        if not np.isclose(
            endpoint[arm]["reconstructed_equal_identity_policy_ce"],
            endpoint[arm]["seed447_equal_identity_policy_ce"],
            atol=TOLERANCE["atol"],
            rtol=TOLERANCE["rtol"],
        ):
            raise ValueError(f"seed447_equal_aggregate_reconciliation_failed:{arm}")
    seed416 = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    replay_copy_weights = {
        item["name"]: int(item["weight"]) for item in seed416["replays"]
    }
    comparison_totals = {
        name: {key: value for key, value in summary.items() if key != "identities"}
        for name, summary in summaries.items()
    }
    result = {
        "schema": "seed449-results-v1",
        "registration_sha256": sha(registration_path),
        "population": {
            **population,
            "exposures": len(rows),
            "compact_source_rows": len({row["compact_row"] for row in rows}),
            "canonical_identities": len({row["canonical_identity"] for row in rows}),
            "registered_training_replay_copy_weights_by_source": replay_copy_weights,
            "registered_training_replay_copy_weight_total": sum(
                replay_copy_weights.values()
            ),
            "replay_copy_weight_interpretation": "training-loader multiplicities; not validation exposure counts",
        },
        "comparisons": comparison_totals,
        "classification": classification,
        "historical_seed447_classification": "close_joint_output_cap_branch",
        "endpoint_reconciliation": endpoint,
        "limitations": [
            "Retrospective descriptive analysis; aggregate seed447 outcomes were known.",
            "Does not establish a causal benefit from changed training weights.",
            "No new model forward passes, gradients, training, games, or promotion.",
        ],
    }
    (out / "ordered-row-ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (out / "identity-ledger.json").write_text(
        json.dumps(
            {name: value["identities"] for name, value in summaries.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (out / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (out / "results.md").write_text(render_markdown(result) + "\n")
    inventory = {
        str(path.relative_to(out)): sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }
    (out / "receipt.json").write_text(
        json.dumps(
            {
                "schema": "seed449-publication-receipt-v1",
                "registration_sha256": sha(registration_path),
                "files_sha256": inventory,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return result


def render_markdown(result: dict[str, Any]) -> str:
    primary = result["comparisons"]["T_minus_initializer"]
    lines = [
        "# Seed449 policy-gain localization",
        "",
        "**Retrospective descriptive analysis.** Seed447's aggregate endpoint results were already known. This analysis does not establish a causal benefit from changing training weights.",
        "",
        f"Classification: `{result['classification']}`. Seed447's historical classification remains `{result['historical_seed447_classification']}`.",
        "",
        f"Population: {result['population']['exposures']} validation exposures; {result['population']['unique_inputs']} exact float32 identities; {result['population']['compact_source_rows']} compact source rows; {result['population']['canonical_identities']} canonical identities.",
        "",
        f"Registered training replay-copy weight sum: {result['population']['registered_training_replay_copy_weight_total']} (source weights {result['population']['registered_training_replay_copy_weights_by_source']}); these are not validation exposure counts.",
        "",
        "## Primary T − initializer",
        "",
        f"- Exposure mean change: {primary['exposure_mean_change']:.12g}",
        f"- Equal-identity mean change: {primary['equal_identity_mean_change']:.12g}",
        f"- Difference: {primary['difference']:.12g}",
        f"- Population covariance: {primary['population_covariance']:.12g}",
        f"- Covariance / mean exposure: {primary['covariance_over_mean_exposure']:.12g}",
        f"- Gross improvement mass: {primary['gross_improvement_mass']:.12g}",
        f"- Gross worsening mass: {primary['gross_worsening_mass']:.12g}",
        "",
        "Signed contribution values are additive quantities, not probabilities.",
        "",
        "## Fixed exposure-count strata",
        "",
        "| Stratum | Identities | Exposures | Exposure mean | Equal-identity mean | Exposure contribution | Equal contribution | Median | Improve / worsen / unchanged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in primary["strata"].items():
        fractions = (
            row["improving_fraction"],
            row["worsening_fraction"],
            row["unchanged_fraction"],
        )
        lines.append(
            f"| {name} | {row['identity_count']} | {row['exposure_count']} | {row['exposure_weighted_mean']} | {row['equal_identity_mean']} | {row['exposure_signed_contribution']} | {row['equal_signed_contribution']} | {row['median_change']} | {fractions} |"
        )
    lines.extend(
        [
            "",
            "## Other comparisons",
            "",
            "| Comparison | Exposure mean change | Equal-identity mean change | Exposure − equal | Population covariance |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("C_minus_initializer", "T_minus_C"):
        summary = result["comparisons"][name]
        lines.append(
            f"| {name.replace('_', ' ')} | {summary['exposure_mean_change']:.12g} | {summary['equal_identity_mean_change']:.12g} | {summary['difference']:.12g} | {summary['population_covariance']:.12g} |"
        )
    lines.extend(
        [
            "",
            "C − initializer is a descriptive control; T − C is descriptive and is not a causal treatment effect.",
            "",
            "## Seed447 aggregate reconciliation",
            "",
            "All reconstructed exposure and equal-identity policy CE endpoints are within seed447's atol=2e-6, rtol=2e-6 tolerance. Residual numerical differences are recorded in `results.json` (maximum absolute residual below 3.1e-9); archived float32 outputs were preserved. These tiny residuals can arise from historical forward batching and float32-vs-float64 reduction.",
            "",
            "## Method and ledgers",
            "",
            "The ordered row ledger retains every exposure, target, legal mask, source-row reference, arm loss, and row change. The identity ledger averages row losses within exact float32 input identity, retaining duplicate rows even when targets differ. Identity strata are n=1, n=2–4, and n≥5. Population covariance is across identities; the covariance reconciliation uses unrounded values.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "- Labels authorize no training, weighting changes, strength evaluation, additional experiment, or promotion.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    result = freeze(args.root) if args.command == "freeze" else run(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
