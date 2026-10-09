"""Read-only semantic verifier for seed455's published attribution."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ml.alphazero_lite import train

OUT = Path("docs/data/seed455-fresh-projection-attribution")
SOURCES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
CHUNK = 512


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(f"semantic_verification_failed:{label}")


def _rebuild_inputs(
    root: Path,
) -> tuple[
    np.ndarray, np.ndarray, list[dict[str, Any]], set[int], list[tuple[str, int]]
]:
    """Reconstruct compressed replay shards with production-loader semantics."""
    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    with tempfile.TemporaryDirectory(
        prefix="seed455-verify-replay-", dir=root / ".tmp"
    ) as directory:
        shards = []
        for source_name in SOURCES:
            path = Path(directory) / f"{source_name}.jsonl"
            with (
                gzip.open(
                    root
                    / "docs/data/seed426-canonical-overlap/sources"
                    / f"{source_name}.jsonl.gz",
                    "rb",
                ) as compressed,
                path.open("wb") as output,
            ):
                while block := compressed.read(1024 * 1024):
                    output.write(block)
            shards.append(path)
        x, p, _v, replay, _coeff = train.load_jsonl_replay(
            shards,
            [int(item["weight"]) for item in registration["replays"]],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[
                item["value_target_mode"] for item in registration["replays"]
            ],
            include_policy_loss_weights=True,
        )
    split_path = (
        root
        / "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz"
    )
    with gzip.open(split_path, "rt", encoding="utf-8") as split_stream:
        split = json.load(split_stream)
    training_compact = set(
        map(int, replay[np.asarray(split["train_positions"], dtype=np.int64)])
    )
    compact_provenance: list[tuple[str, int]] = []
    for source_name in SOURCES:
        with gzip.open(
            root
            / "docs/data/seed426-canonical-overlap/sources"
            / f"{source_name}.jsonl.gz",
            "rt",
            encoding="utf-8",
        ) as source_stream:
            for source_row, _line in enumerate(source_stream, start=1):
                compact_provenance.append((source_name, source_row))
    with gzip.open(
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        population = [json.loads(line) for line in stream]
    unseen = [
        item
        for item in population
        if item["subset"] == "unseen"
        and item["source"] == "fresh"
        and item["active_stones"] > 32
    ]
    return x, p, unseen, training_compact, compact_provenance


def _read_states(
    root: Path, arm: str, model: torch.nn.Module
) -> list[list[np.ndarray]]:
    """Read and independently link 17 states from the immutable seed453 shard."""
    parameters = tuple(model.parameters())
    checkpoint_path = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    initializer = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(initializer, checkpoint_path)
    states = [[p.detach().cpu().numpy().copy() for p in initializer.parameters()]]
    archive_path = (
        root / "docs/data/seed453-fresh-policy-projection" / f"step-tensors-{arm}.npz"
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        for step in range(16):
            before = [
                archive[f"{arm}_{step:02d}_{index:02d}_pre"].copy()
                for index in range(len(parameters))
            ]
            after = [
                archive[f"{arm}_{step:02d}_{index:02d}_post"].copy()
                for index in range(len(parameters))
            ]
            _require(
                len(before) == 22 and len(after) == 22, f"tensor_count:{arm}:{step}"
            )
            _check_trajectory_link(states[-1], before, f"{arm}:{step + 1}")
            for value, parameter in zip(
                (*before, *after), (*parameters, *parameters), strict=True
            ):
                _require(
                    value.dtype == np.float32
                    and value.shape == tuple(parameter.shape)
                    and bool(np.isfinite(value).all()),
                    f"state_tensor:{arm}:{step + 1}",
                )
            states.append(after)
    final = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        final, root / "docs/data/seed453-fresh-policy-projection" / f"{arm}-final.npz"
    )
    _require(
        all(
            np.array_equal(value, parameter.detach().cpu().numpy())
            for value, parameter in zip(states[-1], final.parameters(), strict=True)
        ),
        f"final_checkpoint:{arm}",
    )
    return states


def _layout(model: torch.nn.Module) -> list[dict[str, Any]]:
    output, start = [], 0
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
        _require(group is not None, f"parameter_group:{name}")
        stop = start + parameter.numel()
        output.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "start": start,
                "stop": stop,
                "group": group,
            }
        )
        start = stop
    _require(len(output) == 22 and start == 73159, "model_layout_dimensions")
    return output


def _classification(exposure: dict[str, float], equal: dict[str, float]) -> str:
    totals = (exposure, equal)
    if all(t["D"] > 2e-6 and t["S"] >= 0.5 * t["D"] for t in totals):
        return "fresh_unseen_direction_dominant"
    if all(t["D"] > 2e-6 and t["R"] >= 0.5 * t["D"] for t in totals):
        return "fresh_unseen_remainder_dominant"
    return "mixed_or_weighting_dependent_attribution"


def _check_cohort_archive(
    evidence: Any,
    cohort: str,
    compact_rows: np.ndarray,
    identities: list[str],
    targets: np.ndarray,
) -> None:
    """Semantically bind ordered membership, provenance identities, and all row targets."""
    _require(
        np.array_equal(evidence[f"cohort_{cohort}_compact_rows"], compact_rows),
        f"cohort_order:{cohort}",
    )
    _require(
        np.array_equal(
            evidence[f"cohort_{cohort}_identities"], np.asarray(identities, dtype="U")
        ),
        f"cohort_identity:{cohort}",
    )
    _require(
        np.array_equal(evidence[f"cohort_{cohort}_targets"], targets[compact_rows]),
        f"cohort_targets:{cohort}",
    )


def _check_prediction(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    _require(np.array_equal(actual, expected), f"prediction:{label}")


def _check_gradient(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    _require(np.allclose(actual, expected, atol=1e-10, rtol=1e-9), f"gradient:{label}")


def _check_attribution(
    row: dict[str, Any],
    d: float,
    slope: float,
    group_slopes: dict[str, float],
    label: str,
) -> None:
    remainder = d - slope
    _require(np.isclose(row["d"], d, atol=2e-10, rtol=0), f"D:{label}")
    _require(np.isclose(row["s"], slope, atol=2e-10, rtol=0), f"S:{label}")
    _require(np.isclose(row["r"], remainder, atol=2e-10, rtol=0), f"R:{label}")
    _require(set(row["group_s"]) == set(group_slopes), f"group_names:{label}")
    _require(
        all(
            np.isclose(row["group_s"][key], val, atol=2e-10, rtol=0)
            for key, val in group_slopes.items()
        ),
        f"group_contribution:{label}",
    )


def _check_totals(
    record: dict[str, Any], expected: Mapping[str, float], label: str
) -> None:
    for key, value in expected.items():
        _require(
            np.isclose(record["totals"][key], value, atol=2e-10, rtol=0),
            f"total:{label}:{key}",
        )


def _check_classification(
    actual: str, exposure: dict[str, float], equal: dict[str, float]
) -> None:
    _require(actual == _classification(exposure, equal), "classification")


def _check_trajectory_link(
    previous: list[np.ndarray], current: list[np.ndarray], label: str
) -> None:
    _require(
        len(previous) == len(current)
        and all(np.array_equal(a, b) for a, b in zip(previous, current, strict=True)),
        f"trajectory_link:{label}",
    )


def _cohort_weights(identities: list[str], weighting: str) -> np.ndarray:
    """Verifier-local weighting, retaining each exposure and row target."""
    if not identities:
        raise ValueError("empty_cohort")
    if weighting == "exposure_weighted":
        return np.full(len(identities), 1.0 / len(identities), dtype=np.float64)
    if weighting != "equal_input":
        raise ValueError(f"invalid_weighting:{weighting}")
    counts = {identity: identities.count(identity) for identity in set(identities)}
    return np.asarray(
        [1.0 / (len(counts) * counts[identity]) for identity in identities],
        dtype=np.float64,
    )


def _check_inventory(root: Path, inventory: dict[str, str]) -> None:
    for relative, digest in inventory.items():
        _require(
            (root / relative).is_file() and sha(root / relative) == digest,
            f"publication_inventory:{relative}",
        )


def verify(root: Path) -> dict[str, Any]:
    """Independently rebuild archive-based objectives and gradients, read-only."""
    root = root.resolve()
    out = root / OUT
    frozen = json.loads((out / "freeze-v7.json").read_text())
    for category in ("sources", "inputs"):
        for relative, digest in frozen[category].items():
            _require(sha(root / relative) == digest, f"frozen_{category}:{relative}")
    inventory = json.loads((out / "receipt.json").read_text())
    _check_inventory(root, inventory["files_sha256"])
    result = json.loads((out / "results.json").read_text())
    layout = json.loads((out / "parameter-layout.json").read_text())
    arrays = np.load(out / "gradients.npz", allow_pickle=False)
    evidence = np.load(out / "predictions-targets.npz", allow_pickle=False)
    x, targets, unseen, training_compact, compact_provenance = _rebuild_inputs(root)
    guard = json.loads(
        (
            root / "docs/data/seed453-fresh-policy-projection/fresh-guard.json"
        ).read_text()
    )
    registration453 = json.loads(
        (
            root / "docs/data/seed453-fresh-policy-projection/registration.json"
        ).read_text()
    )
    _require(guard == registration453["guard"], "registered_guard_identity_and_order")
    expected_cohorts = {
        "training_guard": (
            [int(r["compact_row"]) for r in guard],
            [r["exact_input_identity"] for r in guard],
        ),
        "fresh_unseen_exposures": (
            [int(r["compact_row"]) for r in unseen],
            [r["input_identity"] for r in unseen],
        ),
        "fresh_unseen_equal_input": (
            [int(r["compact_row"]) for r in unseen],
            [r["input_identity"] for r in unseen],
        ),
    }
    evaluation_inputs = {item["input_identity"] for item in unseen}
    evaluation_canonical = {item["canonical_identity"] for item in unseen}
    guard_inputs = {item["exact_input_identity"] for item in guard}
    guard_canonical = {item["canonical_identity"] for item in guard}
    _require(
        len(unseen) == 859 and len({r["input_identity"] for r in unseen}) == 851,
        "unseen_membership",
    )
    _require(len(guard) == 1399, "training_guard_membership")
    _require(
        len({item["exact_input_identity"] for item in guard}) == 1399,
        "guard_identity_uniqueness",
    )
    for item in guard:
        compact = int(item["compact_row"])
        _require(compact in training_compact, f"guard_not_training:{compact}")
        _require(
            compact_provenance[compact] == (item["source"], int(item["source_row"])),
            f"guard_source_provenance:{compact}",
        )
    _require(
        not guard_inputs.intersection(evaluation_inputs),
        "training_evaluation_input_exclusion",
    )
    _require(
        not guard_canonical.intersection(evaluation_canonical),
        "training_evaluation_canonical_exclusion",
    )
    _require(
        all(
            item["subset"] == "unseen"
            and item["source"] == "fresh"
            and item["active_stones"] > 32
            and not item["input_seen_in_train"]
            and not item["canonical_seen_in_train"]
            for item in unseen
        ),
        "unseen_provenance",
    )
    actual_layout = _layout(train.PolicyValueNet((96, 3), "residual_v3", 27))
    _require(
        layout == actual_layout and len(layout) == 22 and layout[-1]["stop"] == 73159,
        "parameter_layout",
    )
    prediction_targets = evidence["ordered_targets"]
    _require(bool(np.array_equal(prediction_targets, targets)), "targets_reconstructed")
    for cohort, (indices, identities) in expected_cohorts.items():
        compact = np.asarray(indices, dtype=np.int64)
        _check_cohort_archive(evidence, cohort, compact, identities, targets)
        expected_ids = (
            [item["input_identity"] for item in unseen]
            if cohort.startswith("fresh_unseen")
            else [item["exact_input_identity"] for item in guard]
        )
        for row, identity in zip(compact, expected_ids, strict=True):
            actual_identity = x[int(row)].astype("<f4", copy=False).tobytes().hex()
            _require(actual_identity == identity, f"input_identity:{cohort}:{row}")
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    trajectory = {}
    for arm in ("C", "B"):
        states = _read_states(root, arm, model)
        trajectory[arm] = states
    required_rows = 96
    _require(result.get("row_count") == required_rows, "row_count")
    _require(len(result["cohorts"]) == 2, "arms")
    recomputed_gradients: dict[str, dict[str, list[np.ndarray]]] = {"C": {}, "B": {}}
    for arm in ("C", "B"):
        for cohort, (indices, identities) in expected_cohorts.items():
            record = result["cohorts"][arm][cohort]
            rows = np.asarray(indices, dtype=np.int64)
            weighting = (
                "equal_input"
                if cohort in ("training_guard", "fresh_unseen_equal_input")
                else "exposure_weighted"
            )
            weights = _cohort_weights(identities, weighting)
            loss_values = []
            independent_gradients = []
            for state_index, state in enumerate(trajectory[arm]):
                with torch.no_grad():
                    for parameter, value in zip(model.parameters(), state, strict=True):
                        parameter.copy_(torch.from_numpy(value.copy()))
                loss_accum = np.float64(0)
                grad_accum = [
                    np.zeros(tuple(p.shape), dtype=np.float64)
                    for p in model.parameters()
                ]
                normalizer = weights.sum(dtype=np.float64)
                for begin in range(0, len(rows), 512):
                    selected = rows[begin : begin + 512]
                    logits, _ = model(torch.from_numpy(x[selected]))
                    legal = torch.from_numpy(
                        train.legal_mask_matrix_for_encoded_states(x[selected])
                    )
                    ce = train.compute_policy_cross_entropy(
                        logits.masked_fill(legal <= 0, -1e9),
                        torch.from_numpy(targets[selected]),
                    )
                    local_w = weights[begin : begin + len(selected)]
                    loss_accum += (
                        np.sum(
                            ce.detach().double().cpu().numpy() * local_w,
                            dtype=np.float64,
                        )
                        / normalizer
                    )
                    if cohort == "training_guard":
                        gradient_loss = ce.sum() / len(rows)
                    else:
                        gradient_loss = (
                            ce.double() * torch.from_numpy(local_w)
                        ).sum() / normalizer
                    grads = torch.autograd.grad(
                        gradient_loss, tuple(model.parameters()), allow_unused=True
                    )
                    for i, grad in enumerate(grads):
                        if grad is not None:
                            grad_accum[i] += (
                                grad.detach().cpu().numpy().astype(np.float64)
                            )
                    stored = evidence[f"{arm}_{cohort}_{state_index:02d}"]
                    _check_prediction(
                        logits.detach().cpu().numpy(),
                        stored[begin : begin + len(selected)],
                        f"{arm}:{cohort}:{state_index}:{begin}",
                    )
                loss_values.append(float(loss_accum))
                independent_gradients.append(
                    np.concatenate([g.ravel() for g in grad_accum])
                )
                if cohort == "training_guard":
                    independent_gradients[-1] = (
                        independent_gradients[-1].astype(np.float32).astype(np.float64)
                    )
            for step, ledger in enumerate(record["ledger"]):
                _require(
                    ledger["step"] == step + 1
                    and ledger["arm"] == arm
                    and ledger["cohort"] == cohort,
                    "ledger_identity",
                )
                delta = np.concatenate(
                    [
                        (post.astype(np.float64) - pre.astype(np.float64)).ravel()
                        for pre, post in zip(
                            trajectory[arm][step],
                            trajectory[arm][step + 1],
                            strict=True,
                        )
                    ]
                )
                gradient = independent_gradients[step]
                d = loss_values[step + 1] - loss_values[step]
                slope = float(np.dot(gradient, delta))
                key = f"{arm}_{cohort}_{step:02d}"
                _check_gradient(arrays[key], gradient, key)
                value_entries = [
                    item for item in layout if item["group"] == "value_head"
                ]
                _require(
                    all(
                        not np.any(gradient[item["start"] : item["stop"]])
                        for item in value_entries
                    ),
                    f"unused_value_parameters:{key}",
                )
                if arm == "B" and cohort == "training_guard":
                    with np.load(
                        root
                        / "docs/data/seed453-fresh-policy-projection/step-tensors-B.npz",
                        allow_pickle=False,
                    ) as path_archive:
                        archived = np.concatenate(
                            [
                                path_archive[f"B_{step:02d}_{index:02d}_fresh_gradient"]
                                .ravel()
                                .astype(np.float64)
                                for index in range(22)
                            ]
                        )
                    _require(
                        np.array_equal(
                            gradient.astype(np.float32), archived.astype(np.float32)
                        ),
                        f"seed453_guard_gradient:{step + 1}",
                    )
                    evidence453 = json.loads(
                        (
                            root
                            / "docs/data/seed453-fresh-policy-projection/evidence.json"
                        ).read_text()
                    )
                    accepted = evidence453["arms"]["B"]["steps"][step]
                    selected = accepted["selected_scale"]
                    trial = next(
                        item for item in accepted["trials"] if item["scale"] == selected
                    )
                    _require(
                        np.isclose(
                            slope, trial["realized_fresh_dot"], atol=2e-10, rtol=0
                        ),
                        f"seed453_accepted_slope:{step + 1}",
                    )
                _require(
                    np.isclose(ledger["pre_ce"], loss_values[step], atol=2e-10, rtol=0),
                    f"pre_ce:{key}",
                )
                _require(
                    np.isclose(
                        ledger["post_ce"], loss_values[step + 1], atol=2e-10, rtol=0
                    ),
                    f"post_ce:{key}",
                )
                recomputed_gradients[arm][cohort] = independent_gradients
                calculated_groups = {
                    name: sum(
                        float(
                            np.dot(
                                gradient[item["start"] : item["stop"]],
                                delta[item["start"] : item["stop"]],
                            )
                        )
                        for item in layout
                        if item["group"] == name
                    )
                    for name in ("shared_trunk", "policy_head", "value_head")
                }
                _check_attribution(ledger, d, slope, calculated_groups, key)
            totals = {
                k: sum(row[k.lower()] for row in record["ledger"])
                for k in ("D", "S", "R")
            }
            _check_totals(record, totals, f"{arm}:{cohort}")
            _require(
                np.isclose(
                    totals["D"], loss_values[-1] - loss_values[0], atol=3e-9, rtol=0
                ),
                f"telescoping:{arm}:{cohort}",
            )
            _require(
                np.isclose(
                    record["endpoint_change"],
                    loss_values[-1] - loss_values[0],
                    atol=3e-9,
                    rtol=0,
                ),
                f"endpoint:{arm}:{cohort}",
            )
            for block_name, lower, upper in (
                ("steps_1_3", 1, 3),
                ("steps_4_16", 4, 16),
                ("all_16", 1, 16),
            ):
                subset = [
                    row for row in record["ledger"] if lower <= row["step"] <= upper
                ]
                for key in ("d", "s", "r"):
                    _require(
                        np.isclose(
                            result["blocks"][arm][cohort][block_name][key.upper()],
                            sum(row[key] for row in subset),
                            atol=2e-10,
                            rtol=0,
                        ),
                        f"block:{arm}:{cohort}:{block_name}:{key}",
                    )
            _require(
                all(
                    np.isclose(row["d"], row["s"] + row["r"], atol=2e-12, rtol=0)
                    for row in record["ledger"]
                ),
                f"D_equals_S_plus_R:{arm}:{cohort}",
            )
    exposure = result["cohorts"]["B"]["fresh_unseen_exposures"]["totals"]
    equal = result["cohorts"]["B"]["fresh_unseen_equal_input"]["totals"]
    _check_classification(result["classification"], exposure, equal)
    for arm in ("C", "B"):
        guard_gradients = recomputed_gradients[arm]["training_guard"]
        for cohort in ("fresh_unseen_exposures", "fresh_unseen_equal_input"):
            for step, guard_gradient in enumerate(guard_gradients[:16]):
                unseen_gradient = recomputed_gradients[arm][cohort][step]
                norm_product = float(
                    np.linalg.norm(guard_gradient) * np.linalg.norm(unseen_gradient)
                )
                expected = (
                    None
                    if norm_product == 0.0
                    else float(np.dot(guard_gradient, unseen_gradient) / norm_product)
                )
                actual = result["cohorts"][arm][cohort]["ledger"][step][
                    "guard_unseen_gradient_cosine"
                ]
                _require(
                    (actual is None and expected is None)
                    or (
                        actual is not None
                        and expected is not None
                        and np.isclose(actual, expected, atol=2e-10, rtol=0)
                    ),
                    f"gradient_cosine:{arm}:{cohort}:{step + 1}",
                )
    _require(
        all(
            row["s"] <= 1e-7
            for row in result["cohorts"]["B"]["training_guard"]["ledger"]
        ),
        "training_guard_slope_compliance",
    )
    _require(
        all(
            row["guard_unseen_gradient_cosines"][name] is None
            or -1.00000001 <= row["guard_unseen_gradient_cosines"][name] <= 1.00000001
            for row in result["cohorts"]["B"]["training_guard"]["ledger"]
            for name in ("fresh_unseen_exposures", "fresh_unseen_equal_input")
        ),
        "cosine_bounds",
    )
    historical452 = json.loads(
        (
            root / "docs/data/seed452-fresh-policy-step-attribution/results.json"
        ).read_text()
    )
    for cohort, weighting in (
        ("fresh_unseen_exposures", "exposure_weighted"),
        ("fresh_unseen_equal_input", "equal_exact_input"),
    ):
        expected = historical452["cohorts"]["fresh"][weighting]["totals"]
        for term, actual in result["cohorts"]["C"][cohort]["totals"].items():
            _require(
                np.isclose(actual, expected[term], atol=3e-9, rtol=0),
                f"seed452_reconciliation:{cohort}:{term}",
            )
    evidence453 = json.loads(
        (root / "docs/data/seed453-fresh-policy-projection/evidence.json").read_text()
    )
    for arm in ("B", "C"):
        for cohort, weighting in (
            ("fresh_unseen_exposures", "exposure_weighted"),
            ("fresh_unseen_equal_input", "equal_input"),
        ):
            endpoint = result["cohorts"][arm][cohort]["ledger"][-1]["post_ce"]
            expected = evidence453["metrics"][arm]["fresh"][weighting]["policy_ce"]
            _require(
                np.isclose(endpoint, expected, atol=5e-9, rtol=0),
                f"seed453_endpoint:{arm}:{cohort}",
            )
    return {
        "status": "valid",
        "classification": result["classification"],
        "row_count": result["row_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
