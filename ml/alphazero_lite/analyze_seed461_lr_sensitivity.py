"""Validate provenance and analyze the preregistered seed461 LR arena."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".tmp/seed461-lr-sensitivity"
DATA = ROOT / "docs/data"
SUITE = DATA / "seed461-lr-sensitivity-openings-v2.jsonl"
REGISTRATION = DATA / "seed461-lr-sensitivity-registration-v2.json"
TRAINING_AMENDMENT = DATA / "seed461-lr-sensitivity-training-amendment.json"
BINDING = DATA / "seed461-lr-sensitivity-evaluation-binding.json"
ORDERS = tuple(range(38411, 38416))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _validate_report(
    report: dict[str, Any],
    *,
    run: str,
    candidate: dict[str, Any],
    opponent: dict[str, Any],
    expected_suite_hash: str,
    expected_runtime: dict[str, Any],
) -> None:
    notes = report.get("notes", {})
    if report.get("games_played") != 512 or report.get("score") is None:
        raise ValueError(f"game_count_or_score_invalid:{run}")
    expected = {
        "suite_sha256": expected_suite_hash,
        "challenger_path": candidate["artifact"],
        "current_path": opponent["artifact"],
        "challenger_simulations": 384,
        "current_simulations": 384,
        "seed": 384,
    }
    for field, value in expected.items():
        if notes.get(field) != value:
            raise ValueError(f"report_identity_mismatch:{run}:{field}")
    if notes.get("search_profile_hash") != notes.get("search_profile", {}).get("hash"):
        raise ValueError(f"runtime_profile_hash_mismatch:{run}")
    profile = notes.get("search_profile", {})
    if profile.get("c_puct") != 1.25 or profile.get("simulations") != 384:
        raise ValueError(f"runtime_search_contract_mismatch:{run}")
    for key, value in expected_runtime.items():
        if notes.get(key) != value:
            raise ValueError(f"runtime_identity_mismatch:{run}:{key}")


def validate_games(
    rows: list[dict[str, Any]], openings: list[dict[str, Any]]
) -> np.ndarray:
    if len(rows) != 512:
        raise ValueError("arena_game_count_mismatch")
    expected_prefixes = {
        index: row["prefix_moves"] for index, row in enumerate(openings)
    }
    by_opening: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        index = int(row["opening_index"])
        if (
            index not in expected_prefixes
            or row.get("opening_prefix_moves") != expected_prefixes[index]
        ):
            raise ValueError("registered_opening_identity_mismatch")
        if row.get("winner") not in {"challenger", "current", "draw"}:
            raise ValueError("unknown_winner")
        by_opening.setdefault(index, []).append(row)
    if set(by_opening) != set(expected_prefixes):
        raise ValueError("missing_registered_opening")
    scores = np.zeros(256, dtype=np.float64)
    for index, games in by_opening.items():
        seats = [int(row["challenger_player"]) for row in games]
        if len(games) != 2 or sorted(seats) != [0, 1]:
            raise ValueError(f"missing_or_duplicate_seat:{index}")
        scores[index] = (
            sum(
                1.0
                if row["winner"] == "challenger"
                else 0.5
                if row["winner"] == "draw"
                else 0.0
                for row in games
            )
            / 2
        )
    return scores


def reproducibility_audit(registration: dict[str, Any]) -> dict[str, str]:
    spec = registration["training"]
    train.set_seed(int(spec["seed"]))
    paths = [Path(row["path"]) for row in spec["replays"]]
    x, policy, value, replay_indexes, _ = train.load_jsonl_replay(
        paths,
        [row["weight"] for row in spec["replays"]],
        policy_target_mode=spec["policy_target_mode"],
        value_target_mode="default",
        replay_value_target_modes=[row["value_target_mode"] for row in spec["replays"]],
        include_policy_loss_weights=True,
    )
    train_positions, val_positions = train.split_replay_positions_by_source_row(
        replay_indexes, val_split=float(spec["validation_split"])
    )
    split_hash = hashlib.sha256(
        train_positions.tobytes() + val_positions.tobytes()
    ).hexdigest()
    replay_material = json.dumps(
        {
            "replays": spec["replays"],
            "multiplicities": spec["weights"],
            "source_row_count": int(x.shape[0]),
            "replay_index_count": int(replay_indexes.shape[0]),
            "policy_shape": list(policy.shape),
            "value_shape": list(value.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    replay_hash = hashlib.sha256(replay_material + replay_indexes.tobytes()).hexdigest()
    return {
        "initialization_sha256": spec["parent_sha256"],
        "split_membership_sha256": split_hash,
        "replay_multiplicity_sha256": replay_hash,
        "train_split_count": str(len(train_positions)),
        "validation_count": str(len(val_positions)),
    }


def main() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    amendment = json.loads(TRAINING_AMENDMENT.read_text(encoding="utf-8"))
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    training_path = Path(amendment["training_record"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if sha(REGISTRATION) != amendment["registration_sha256"]:
        raise ValueError("training_registration_hash_mismatch")
    if (
        sha(training_path) != amendment["training_record_sha256"]
        or sha(training_path) != binding["training_sha256"]
    ):
        raise ValueError("training_record_hash_mismatch")
    if training.get("registration_sha256") != sha(REGISTRATION):
        raise ValueError("completed_training_registration_hash_mismatch")
    if sha(TRAINING_AMENDMENT) != binding["training_amendment_sha256"]:
        raise ValueError("training_amendment_hash_mismatch")
    if (
        sha(SUITE) != registration["evaluation"]["suite_sha256"]
        or sha(SUITE) != binding["suite_sha256"]
    ):
        raise ValueError("registered_suite_hash_mismatch")
    if (
        sha(Path(registration["training"]["parent"]))
        != registration["training"]["parent_sha256"]
    ):
        raise ValueError("frozen_parent_hash_mismatch")
    openings = read_jsonl(SUITE)
    if (
        len(openings) != 256
        or len({json.dumps(row["state"], sort_keys=True) for row in openings}) != 256
    ):
        raise ValueError("registered_suite_membership_invalid")
    pair_invariants = reproducibility_audit(registration)

    expected_arena_suite_hash = sha(SUITE)
    opponent = binding["opponent"]
    opponent_dir = Path(opponent["artifact"])
    for filename, key in (
        ("weights.json", "weights_sha256"),
        ("metadata.json", "metadata_sha256"),
        ("search_policy.json", "search_policy_sha256"),
    ):
        if sha(opponent_dir / filename) != opponent[key]:
            raise ValueError(f"opponent_artifact_hash_mismatch:{filename}")

    vectors: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in ORDERS}
    provenance: dict[str, Any] = {}
    for seed in ORDERS:
        for arm in ("A", "B"):
            run = f"order_{seed}_{arm}"
            candidate = binding["candidates"][run]
            train_result = training["trajectories"][run]
            expected_epochs = amendment["checkpoints"][run]["epochs"]
            if (
                train_result["epochs"] != expected_epochs
                or amendment["checkpoints"][run]["selected_sha256"]
                != train_result["selected_sha256"]
            ):
                raise ValueError(f"checkpoint_epoch_hash_mismatch:{run}")
            selected_epoch = min(
                train_result["history"], key=lambda item: item["validation_total_loss"]
            )["epoch"]
            selected_hash = train_result["epochs"][f"E{selected_epoch}"]
            if (
                selected_hash != train_result["selected_sha256"]
                or selected_hash != candidate["checkpoint_sha256"]
                or candidate["selected_epoch"] != f"E{selected_epoch}"
            ):
                raise ValueError(f"selected_checkpoint_binding_mismatch:{run}")
            if run not in amendment["checkpoints"]:
                raise ValueError(f"training_amendment_missing_run:{run}")
            artifact = Path(candidate["artifact"])
            for filename, key in (
                ("model.npz", "model_sha256"),
                ("weights.json", "weights_sha256"),
                ("metadata.json", "metadata_sha256"),
                ("search_policy.json", "search_policy_sha256"),
            ):
                if sha(artifact / filename) != candidate[key]:
                    raise ValueError(
                        f"candidate_artifact_hash_mismatch:{run}:{filename}"
                    )
            report_path = WORK / "arena" / f"{run}.json"
            games_path = WORK / "arena" / f"{run}-games.jsonl"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_binding = binding["reports"][run]
            if (
                sha(report_path) != report_binding["report_sha256"]
                or sha(games_path) != report_binding["games_sha256"]
            ):
                raise ValueError(f"actual_evidence_file_hash_mismatch:{run}")
            runtime = registration["evaluation"]["runtime_contract"]
            runtime_identity = runtime
            _validate_report(
                report,
                run=run,
                candidate=candidate,
                opponent=opponent,
                expected_suite_hash=expected_arena_suite_hash,
                expected_runtime=runtime_identity,
            )
            vectors[seed][arm] = validate_games(read_jsonl(games_path), openings)
            if not np.isclose(float(vectors[seed][arm].mean()), float(report["score"])):
                raise ValueError(f"report_score_does_not_match_games:{run}")
            provenance[run] = {
                "report_sha256": sha(report_path),
                "games_sha256": sha(games_path),
                "checkpoint_sha256": selected_hash,
                "selected_epoch": candidate["selected_epoch"],
                "candidate_weights_sha256": candidate["weights_sha256"],
                "runtime_profile_hash": report["notes"]["search_profile_hash"],
                "absolute_score": float(vectors[seed][arm].mean()),
                "validation_total_loss_by_epoch": [
                    float(row["validation_total_loss"])
                    for row in train_result["history"]
                ],
            }

    per_opening = np.stack([vectors[seed]["B"] - vectors[seed]["A"] for seed in ORDERS])
    for seed in ORDERS:
        a_train = training["trajectories"][f"order_{seed}_A"]
        b_train = training["trajectories"][f"order_{seed}_B"]
        if a_train["permutation_sha256"] != b_train["permutation_sha256"]:
            raise ValueError(f"paired_epoch_permutation_mismatch:{seed}")
        if (
            a_train["metrics"]["train_split_count"]
            != b_train["metrics"]["train_split_count"]
            or a_train["metrics"]["validation_count"]
            != b_train["metrics"]["validation_count"]
        ):
            raise ValueError(f"paired_split_membership_contract_mismatch:{seed}")
        if a_train["metrics"]["train_split_count"] != int(
            pair_invariants["train_split_count"]
        ) or a_train["metrics"]["validation_count"] != int(
            pair_invariants["validation_count"]
        ):
            raise ValueError(f"derived_split_membership_count_mismatch:{seed}")
    order_effects = per_opening.mean(axis=1)
    rng = np.random.default_rng(384)
    indices = rng.integers(0, 256, size=(10_000, 256))
    bootstrap = per_opening[:, indices].mean(axis=2).mean(axis=0)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    a_scores = np.asarray([vectors[seed]["A"].mean() for seed in ORDERS])
    b_scores = np.asarray([vectors[seed]["B"].mean() for seed in ORDERS])
    mean = float(order_effects.mean())
    nonnegative = int(np.count_nonzero(order_effects >= 0))
    worst = float(order_effects.min())
    criteria = mean >= 0.03 and float(lower) > 0 and nonnegative >= 4 and worst >= -0.05
    if criteria:
        decision = "advance_lr_0.0005_as_research_candidate"
    elif mean < 0 or float(upper) < 0 or worst < -0.05:
        decision = "reject_lr_0.0005_retain_lr_0.001"
    else:
        decision = "inconclusive_retain_lr_0.001"
    result = {
        "schema": "seed461-lr-sensitivity-results-v1",
        "status": "completed_fixed_budget",
        "game_count": 5120,
        "pair_invariants": pair_invariants,
        "evidence_identity": {
            "registration_sha256": sha(REGISTRATION),
            "suite_sha256": sha(SUITE),
            "training_record_sha256": sha(training_path),
            "training_amendment_sha256": sha(TRAINING_AMENDMENT),
            "evaluation_binding_sha256": sha(BINDING),
            "opponent_weights_sha256": opponent["weights_sha256"],
            "opponent_metadata_sha256": opponent["metadata_sha256"],
            "opponent_search_policy_sha256": opponent["search_policy_sha256"],
        },
        "primary_endpoint": "mean of five order-paired B-minus-A opening-score differences",
        "primary_mean": mean,
        "paired_order_effects": {
            str(seed): float(value)
            for seed, value in zip(ORDERS, order_effects, strict=True)
        },
        "bootstrap": {
            "samples": 10000,
            "seed": 384,
            "cluster": "opening, shared across all order pairs",
            "confidence": 0.95,
            "lower": float(lower),
            "upper": float(upper),
        },
        "absolute_scores_vs_seed455": {
            str(seed): {
                "A_lr_0.001": float(vectors[seed]["A"].mean()),
                "B_lr_0.0005": float(vectors[seed]["B"].mean()),
            }
            for seed in ORDERS
        },
        "between_order_spread": {
            "definition": "range of five absolute opening scores",
            "A_range": float(np.ptp(a_scores)),
            "B_range": float(np.ptp(b_scores)),
            "decreased": bool(np.ptp(b_scores) < np.ptp(a_scores)),
        },
        "selected_epochs": {
            run: item["selected_epoch"] for run, item in provenance.items()
        },
        "decision": decision,
        "decision_reason": (
            "A paired order regressed by more than 0.05 and between-order spread increased"
            if worst < -0.05 and np.ptp(b_scores) >= np.ptp(a_scores)
            else "Predefined advance criteria were not all met"
        ),
        "decision_gates": {
            "mean_at_least_0.03": mean >= 0.03,
            "lower_bound_above_zero": bool(lower > 0),
            "at_least_four_nonnegative_pairs": nonnegative >= 4,
            "no_pair_below_minus_0.05": worst >= -0.05,
            "nonnegative_pairs": nonnegative,
            "worst_pair": worst,
        },
        "stability_claim": "reduced between-order score spread"
        if np.ptp(b_scores) < np.ptp(a_scores)
        else "not supported: between-order spread did not decrease",
        "inference_scope": "conditional on these five order seeds and this frozen dataset",
        "another_dataset_generation_replication_required_before_production_adoption": True,
        "provenance": provenance,
    }
    out = DATA / "seed461-lr-sensitivity-results.json"
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Seed461 lower-learning-rate sensitivity",
        "",
        f"**Decision:** {decision}. {result['decision_reason']}",
        "",
        f"Primary mean B−A opening-score difference: **{mean:.4f}** "
        f"(95% shared-opening cluster bootstrap interval {float(lower):.4f} to {float(upper):.4f}).",
        "",
        "| Order seed | A score (LR 0.001) | B score (LR 0.0005) | B−A | A/B selected epoch |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for seed, effect in zip(ORDERS, order_effects, strict=True):
        report_lines.append(
            f"| {seed} | {a_scores[list(ORDERS).index(seed)]:.4f} | "
            f"{b_scores[list(ORDERS).index(seed)]:.4f} | {effect:+.4f} | "
            f"{provenance[f'order_{seed}_A']['selected_epoch']}/"
            f"{provenance[f'order_{seed}_B']['selected_epoch']} |"
        )
    report_lines.extend(
        [
            "",
            f"Between-order score range: A **{np.ptp(a_scores):.4f}**, "
            f"B **{np.ptp(b_scores):.4f}**. Stability improvement: "
            f"{'yes' if np.ptp(b_scores) < np.ptp(a_scores) else 'no'}.",
            "",
            "Validation total-loss trajectories (epochs 1–4):",
            "",
            "| Order | Arm | Selected | Validation total loss by epoch |",
            "|---:|:---:|:---:|:---|",
        ]
    )
    for seed in ORDERS:
        for arm in ("A", "B"):
            run = f"order_{seed}_{arm}"
            trajectory = training["trajectories"][run]["history"]
            losses = ", ".join(
                f"{row['validation_total_loss']:.4f}" for row in trajectory
            )
            report_lines.append(
                f"| {seed} | {arm} | {provenance[run]['selected_epoch']} | {losses} |"
            )
    report_lines.extend(
        [
            "",
            "Inference is conditional on these five order seeds and this frozen dataset. "
            "A favorable result would require another dataset/generation replication before production adoption.",
            "Detailed hash-bound provenance and per-game file hashes are in "
            "`seed461-lr-sensitivity-results.json`; raw artifacts are under `.tmp/seed461-lr-sensitivity/`.",
        ]
    )
    (DATA / "seed461-lr-sensitivity-results.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "primary_mean",
                    "bootstrap",
                    "paired_order_effects",
                    "decision",
                    "between_order_spread",
                    "stability_claim",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
