"""Independent read-only verifier for the seed449 retrospective publication."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

OUT = Path("docs/data/seed449-policy-gain-localization")
SEED447 = Path("docs/data/seed447-joint-output-cap")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=2e-6, abs_tol=2e-6)


def _require_equal(actual: Any, expected: Any, code: str) -> None:
    if actual != expected:
        raise ValueError(code)


def _validate_rows(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> None:
    if len(expected) != len(actual):
        raise ValueError("seed449_row_coverage_invalid")
    for expected_row, actual_row in zip(expected, actual, strict=True):
        if expected_row != actual_row:
            raise ValueError("seed449_row_semantics_invalid")


def _validate_identity_ledger(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    if actual != expected:
        raise ValueError("seed449_identity_ledger_semantics_invalid")


def _validate_summary(
    name: str, published: dict[str, Any], expected: dict[str, Any]
) -> None:
    for key in (
        "initial",
        "comparison",
        "exposure_count",
        "identity_count",
        "strata",
    ):
        if published[key] != expected[key]:
            raise ValueError(f"seed449_semantic_field_invalid:{name}:{key}")
    for key in (
        "exposure_mean_change",
        "equal_identity_mean_change",
        "difference",
        "population_covariance",
        "covariance_over_mean_exposure",
        "gross_improvement_mass",
        "gross_worsening_mass",
        "net_signed_mass",
    ):
        if not _close(published[key], expected[key]):
            raise ValueError(f"seed449_semantic_field_invalid:{name}:{key}")
    if not math.isclose(
        expected["difference"],
        expected["covariance_over_mean_exposure"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("seed449_covariance_identity_failed")


def _validate_classification(published: str, summaries: dict[str, Any]) -> None:
    if published != _classification(summaries["T_minus_initializer"]):
        raise ValueError("seed449_classification_invalid")


def _ce(logits: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    legal = np.flatnonzero(mask)
    if len(legal) == 0 or np.any(target[mask == 0] != 0):
        raise ValueError("independent_mask_or_target_invalid")
    values = logits.astype(np.float64)
    maximum = float(np.max(values[legal]))
    log_z = maximum + math.log(sum(math.exp(float(values[i]) - maximum) for i in legal))
    return -sum(float(target[i]) * (float(values[i]) - log_z) for i in legal)


def _read_inputs(root: Path, reg: dict[str, Any]) -> list[dict[str, Any]]:
    from ml.alphazero_lite import train

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
    target_path = root / SEED447 / "ordered-targets.npz"
    targets = np.load(target_path, allow_pickle=False)
    expected_rows = targets["compact_rows"]
    expected_ids = targets["input_identity"].tolist()
    if len(members) != 2607 or len(set(expected_ids)) != 1242:
        raise ValueError("independent_population_count_invalid")
    if [row["compact_row"] for row in members] != expected_rows.tolist():
        raise ValueError("independent_ordered_compact_rows_invalid")
    if [row["input_identity"] for row in members] != expected_ids:
        raise ValueError("independent_ordered_identity_invalid")
    # Reconstruct row target evidence from seed416 via the registered production loader.
    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    paths = [
        Path(registration["derivatives"][item["name"]]["A"]["derivative"])
        for item in registration["replays"]
    ]
    paths = [path if path.is_absolute() else root / path for path in paths]
    _x, policy, _value, _replay, _coeff = train.load_jsonl_replay(
        paths,
        [int(row["weight"]) for row in registration["replays"]],
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=[
            row["value_target_mode"] for row in registration["replays"]
        ],
        include_policy_loss_weights=True,
    )
    if not np.array_equal(policy[expected_rows], targets["policy_targets"]):
        raise ValueError("independent_ordered_targets_source_mismatch")
    masks = train.legal_mask_matrix_for_encoded_states(_x[expected_rows]).astype(
        np.int64
    )
    archives = {
        arm: np.load(root / SEED447 / f"{arm}-predictions.npz", allow_pickle=False)
        for arm in ("initializer", "C", "T")
    }
    rows = []
    losses_by_arm: dict[str, np.ndarray] = {}
    for arm, archive in archives.items():
        logits = torch.from_numpy(archive["policy_logits"])
        legal = torch.from_numpy(masks.astype(np.float32))
        target_tensor = torch.from_numpy(targets["policy_targets"])
        with torch.inference_mode():
            losses_by_arm[arm] = np.concatenate(
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
                    for start in range(0, len(expected_rows), 512)
                ]
            )
    for index, member in enumerate(members):
        mask = masks[index]
        target = targets["policy_targets"][index]
        losses = {}
        for arm, archive in archives.items():
            if (
                not np.array_equal(archive["compact_rows"], expected_rows)
                or archive["input_identity"].tolist() != expected_ids
            ):
                raise ValueError(f"independent_prediction_order_invalid:{arm}")
            logits = archive["policy_logits"][index]
            if logits.dtype != np.float32:
                raise ValueError("independent_logits_dtype_invalid")
            losses[arm] = float(losses_by_arm[arm][index])
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
                "active_stones": member["active_stones"],
                "legal_mask": mask.tolist(),
                "policy_target": target.astype(np.float64).tolist(),
                "losses": losses,
                "changes": {
                    "T_minus_initializer": losses["T"] - losses["initializer"],
                    "C_minus_initializer": losses["C"] - losses["initializer"],
                    "T_minus_C": losses["T"] - losses["C"],
                },
            }
        )
    return rows


def _independent_summary(
    rows: list[dict[str, Any]], initial: str, comparison: str
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["input_identity"], []).append(row)
    identities = []
    for key, group in groups.items():
        delta = [row["losses"][comparison] - row["losses"][initial] for row in group]
        identities.append(
            {
                "input_identity": key,
                "canonical_identity": group[0]["canonical_identity"],
                "exposure_count": len(group),
                "compact_rows": [r["compact_row"] for r in group],
                "source_refs": [r["source_ref"] for r in group],
                "initial_mean_loss": sum(r["losses"][initial] for r in group)
                / len(group),
                "comparison_mean_loss": sum(r["losses"][comparison] for r in group)
                / len(group),
                "mean_change": sum(delta) / len(group),
                "row_changes": delta,
            }
        )
    m, n = len(identities), len(rows)
    d_exposure = sum(r["losses"][comparison] - r["losses"][initial] for r in rows) / n
    d_equal = sum(i["mean_change"] for i in identities) / m
    mean_n = n / m
    cov = (
        sum(
            (i["exposure_count"] - mean_n) * (i["mean_change"] - d_equal)
            for i in identities
        )
        / m
    )
    strata = {}
    for label, predicate in (
        ("1", lambda v: v == 1),
        ("2-4", lambda v: 2 <= v <= 4),
        (">=5", lambda v: v >= 5),
    ):
        selected = [i for i in identities if predicate(i["exposure_count"])]
        values = [v for i in selected for v in i["row_changes"]]
        ids_count = len(selected)
        exp_count = len(values)
        sorted_values = sorted(values)
        median = (
            ((sorted_values[(exp_count - 1) // 2] + sorted_values[exp_count // 2]) / 2)
            if exp_count
            else None
        )
        strata[label] = {
            "identity_count": ids_count,
            "exposure_count": exp_count,
            "exposure_weighted_mean": sum(values) / exp_count if exp_count else None,
            "equal_identity_mean": sum(i["mean_change"] for i in selected) / ids_count
            if ids_count
            else None,
            "exposure_signed_contribution": sum(
                i["mean_change"] * i["exposure_count"] for i in selected
            )
            / n,
            "equal_signed_contribution": sum(i["mean_change"] for i in selected) / m,
            "median_change": median,
            "improving_fraction": sum(v < 0 for v in values) / exp_count
            if exp_count
            else None,
            "worsening_fraction": sum(v > 0 for v in values) / exp_count
            if exp_count
            else None,
            "unchanged_fraction": sum(v == 0 for v in values) / exp_count
            if exp_count
            else None,
        }
    changes = [r["losses"][comparison] - r["losses"][initial] for r in rows]
    return {
        "initial": initial,
        "comparison": comparison,
        "exposure_count": n,
        "identity_count": m,
        "exposure_mean_change": d_exposure,
        "equal_identity_mean_change": d_equal,
        "difference": d_exposure - d_equal,
        "population_covariance": cov,
        "covariance_over_mean_exposure": cov / mean_n,
        "gross_improvement_mass": sum(v for v in changes if v < 0),
        "gross_worsening_mass": sum(v for v in changes if v > 0),
        "net_signed_mass": sum(changes),
        "strata": strata,
        "identities": identities,
    }


def _classification(summary: dict[str, Any]) -> str:
    strata = summary["strata"]
    singles = strata["1"]["exposure_weighted_mean"]
    repeated_count = strata["2-4"]["exposure_count"] + strata[">=5"]["exposure_count"]
    if singles is None or not repeated_count:
        return "mixed_policy_response"
    repeated = (
        strata["2-4"]["exposure_signed_contribution"]
        + strata[">=5"]["exposure_signed_contribution"]
    ) / (repeated_count / summary["exposure_count"])
    exposure, equal = (
        summary["exposure_mean_change"],
        summary["equal_identity_mean_change"],
    )
    if (
        exposure <= -0.005
        and equal > -0.005
        and summary["population_covariance"] < 0
        and singles >= 0
        and repeated < 0
    ):
        return "frequency_concentrated_policy_gain"
    if -0.005 < equal < 0 and singles < 0 and repeated < 0:
        return "broad_small_policy_gain"
    return "mixed_policy_response"


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    reg_path = out / "registration.json"
    reg = json.loads(reg_path.read_text())
    for relative, digest in reg["bound_input_sha256"].items():
        if _sha(root / relative) != digest:
            raise ValueError(f"seed449_bound_input_hash_invalid:{relative}")
    for relative, digest in reg["source_sha256"].items():
        if (
            _sha(root / relative) != digest
            or _sha(out / "source-snapshots" / Path(relative).name) != digest
        ):
            raise ValueError(f"seed449_source_binding_invalid:{relative}")
    rows = _read_inputs(root, reg)
    ledger = [
        json.loads(line)
        for line in (out / "ordered-row-ledger.jsonl").read_text().splitlines()
    ]
    _validate_rows(rows, ledger)
    computed = {
        "T_minus_initializer": _independent_summary(rows, "initializer", "T"),
        "C_minus_initializer": _independent_summary(rows, "initializer", "C"),
        "T_minus_C": _independent_summary(rows, "C", "T"),
    }
    result = json.loads((out / "results.json").read_text())
    _require_equal(
        result["registration_sha256"],
        _sha(reg_path),
        "seed449_registration_result_binding_invalid",
    )
    receipt = json.loads((out / "receipt.json").read_text())
    _require_equal(
        receipt["registration_sha256"],
        _sha(reg_path),
        "seed449_receipt_registration_invalid",
    )
    actual_inventory = {
        str(path.relative_to(out)): _sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }
    _require_equal(
        receipt["files_sha256"],
        actual_inventory,
        "seed449_publication_inventory_invalid",
    )
    population = result["population"]
    canonical_count = len({r["canonical_identity"] for r in rows})
    if (
        population["membership_count"] != len(rows)
        or population["unique_inputs"] != len({r["input_identity"] for r in rows})
        or population["exposures"] != len(rows)
        or population["compact_source_rows"] != len({r["compact_row"] for r in rows})
        or population["canonical_identities"] != canonical_count
    ):
        raise ValueError("seed449_population_total_invalid")
    published = result["comparisons"]
    identity_ledger = json.loads((out / "identity-ledger.json").read_text())
    for name, summary in computed.items():
        other = published[name]
        _validate_summary(name, other, summary)
        _validate_identity_ledger(identity_ledger[name], summary["identities"])
    expected_class = _classification(computed["T_minus_initializer"])
    _validate_classification(result["classification"], computed)
    if result["historical_seed447_classification"] != "close_joint_output_cap_branch":
        raise ValueError("seed449_historical_classification_changed")
    seed447 = json.loads((root / SEED447 / "evidence.json").read_text())
    for arm in ("initializer", "C", "T"):
        values = [row["losses"][arm] for row in rows]
        by_input: dict[str, list[float]] = {}
        for row in rows:
            by_input.setdefault(row["input_identity"], []).append(row["losses"][arm])
        exp_mean = sum(values) / len(values)
        equal_mean = sum(sum(group) / len(group) for group in by_input.values()) / len(
            by_input
        )
        published_endpoint = result["endpoint_reconciliation"][arm]
        if not _close(
            exp_mean, seed447["metrics"][arm]["exposure_weighted"]["policy_ce"]
        ):
            raise ValueError(f"seed449_seed447_exposure_aggregate_invalid:{arm}")
        if not _close(equal_mean, seed447["metrics"][arm]["equal_input"]["policy_ce"]):
            raise ValueError(f"seed449_seed447_equal_aggregate_invalid:{arm}")
        if not _close(
            published_endpoint["reconstructed_policy_ce"], exp_mean
        ) or not _close(
            published_endpoint["reconstructed_equal_identity_policy_ce"], equal_mean
        ):
            raise ValueError(f"seed449_endpoint_totals_invalid:{arm}")
    return {
        "status": "valid",
        "classification": expected_class,
        "exposures": len(rows),
        "identities": len({r["input_identity"] for r in rows}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))
