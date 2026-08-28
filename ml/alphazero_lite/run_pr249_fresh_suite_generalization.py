#!/usr/bin/env python3
"""Evaluate PR #248/#249 fixed-target candidates on sealed fresh opening suites.

This runner never generates self-play or trains.  It only consumes retained
step-16 artifacts, creates three preregistered evaluation suites, and runs the
ordinary paired PUCT arena used by the source experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import build_opening_suite as suites  # noqa: E402
from ml.alphazero_lite import (  # noqa: E402
    run_fresh_p1_onpolicy_shadow_replay as replay_train,
)
from ml.alphazero_lite import run_pr241_policy_target_noise_isolation as isolation  # noqa: E402
from ml.alphazero_lite import run_pr242_target_entropy_factorization as pr244  # noqa: E402
from ml.alphazero_lite.evaluation_metrics import (  # noqa: E402
    paired_effect_difference,
    paired_opening_candidate_effect,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.self_play import encode_state  # noqa: E402

CANONICAL_SUITE = Path("/tmp/azlite_opening_suite/medium_eval.jsonl")
CANONICAL_SHA = "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04"
P1_ARTIFACT = replay_train.P1_CHECKPOINT.parent / "artifact"
CONTEXT = "1200:1200"
WORKERS = 24
SUITE_SEEDS = {"A": 1042, "B": 2042, "C": 3042}
CANDIDATES = {
    "seed45_fixed768_positive": {
        "artifact": Path(
            "/tmp/azlite_pr247_fixed_target_budget/train/fixed768/step_0016/artifact"
        ),
        "checkpoint": Path(
            "/tmp/azlite_pr247_fixed_target_budget/train/fixed768/step_0016.pt"
        ),
        "replay": "245a452f80970485dd9d07dad560e35f04bbccc16f6147e36c98598b7426f106",
        "batch": "a4e859f1340078e21c9ad2b4e0c04bac5c69c7463d8b8b859f83f79f915382fe",
        "historical_effect": 0.041015625,
    },
    "seed45_fixed1024_negative": {
        "artifact": Path(
            "/tmp/azlite_pr247_fixed_target_budget/train/fixed1024/step_0016/artifact"
        ),
        "checkpoint": Path(
            "/tmp/azlite_pr247_fixed_target_budget/train/fixed1024/step_0016.pt"
        ),
        "replay": "245a452f80970485dd9d07dad560e35f04bbccc16f6147e36c98598b7426f106",
        "batch": "a4e859f1340078e21c9ad2b4e0c04bac5c69c7463d8b8b859f83f79f915382fe",
        "historical_effect": -0.01953125,
    },
    "seed46_fresh1024_positive": {
        "artifact": Path(
            "/tmp/azlite_pr248_prospective_fixed_target_budget/train/fresh1024/step_0016/artifact"
        ),
        "checkpoint": Path(
            "/tmp/azlite_pr248_prospective_fixed_target_budget/train/fresh1024/step_0016.pt"
        ),
        "replay": "007a72d5c07c15f353ef244cab627198cb7be6b73c43b7a03ea2feb767b1fbb4",
        "batch": "671ac065c7b82c4a04b9201a4cb60b0e0d2ee4c73c390566c800704252a2e748",
        "historical_effect": 0.041015625,
    },
    "seed46_fresh768_negative": {
        "artifact": Path(
            "/tmp/azlite_pr248_prospective_fixed_target_budget/train/fresh768/step_0016/artifact"
        ),
        "checkpoint": Path(
            "/tmp/azlite_pr248_prospective_fixed_target_budget/train/fresh768/step_0016.pt"
        ),
        "replay": "83afa719a260083902540419b72b30b6a96d7bc63015be2482af442ab7d4baa9",
        "batch": "671ac065c7b82c4a04b9201a4cb60b0e0d2ee4c73c390566c800704252a2e748",
        "historical_effect": -0.01953125,
    },
}
CONTRASTS = {
    "seed45_fixed768_minus_fixed1024": (
        "seed45_fixed768_positive",
        "seed45_fixed1024_negative",
    ),
    "seed46_fresh1024_minus_fresh768": (
        "seed46_fresh1024_positive",
        "seed46_fresh768_negative",
    ),
}


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def artifact_sha(path: Path) -> str:
    return sha256_file(path / "weights.json")


def checkpoint_hashes(path: Path) -> dict[str, str]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if set(saved) != {"model", "optimizer"}:
        fail(f"invalid checkpoint: {path}")
    return {
        "full_model_sha256": hashlib.sha256(
            json.dumps(
                {
                    k: v.detach().cpu().numpy().tolist()
                    for k, v in saved["model"].items()
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "optimizer_sha256": replay_train.optimizer_state_sha256(saved["optimizer"]),
    }


def candidate_manifest() -> dict[str, Any]:
    result = {}
    for name, candidate in CANDIDATES.items():
        artifact, checkpoint = candidate["artifact"], candidate["checkpoint"]
        if not artifact.is_dir() or not checkpoint.is_file():
            fail(f"missing retained candidate artifact: {name}")
        hashes = checkpoint_hashes(checkpoint)
        result[name] = {
            **hashes,
            "adapter_sha256": artifact_sha(artifact),
            "checkpoint_file_sha256": sha256_file(checkpoint),
            "source_replay_sha256": candidate["replay"],
            "source_batch_plan_sha256": candidate["batch"],
            "historical_canonical_effect": candidate["historical_effect"],
        }
    return result


def all_openings() -> list[dict[str, Any]]:
    prefixes = []
    for ply in (2, 4, 6):
        prefixes.extend(suites.enumerate_legal_prefixes(ply))
    unique, _, _ = suites.deduplicate_openings(prefixes)
    return suites.stratify_openings(unique)


def suite_keys(entries: list[dict[str, Any]]) -> set[str]:
    return {suites.canonical_key(entry["state"]) for entry in entries}


def fresh_suites(output: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """Use the canonical selector with deterministic rejection for disjoint suites."""
    canonical = suites.load_suite_jsonl(str(CANONICAL_SUITE))
    used = suite_keys(canonical)
    training_states = set().union(*replay_states().values())
    universe = [
        entry
        for entry in all_openings()
        if tuple(encode_state(entry["state"], input_encoding="kalah_v3"))
        not in training_states
    ]
    paths, manifest = (
        {},
        {
            "canonical_sha256": sha256_file(CANONICAL_SUITE),
            "selection_pool_excludes_training_replay_states": True,
            "suites": {},
        },
    )
    for label, seed in SUITE_SEEDS.items():
        # The canonical selector has singleton strata, so independent selection
        # would necessarily repeat those openings.  Remove prior states first,
        # then apply its unchanged stratification and selection procedure.
        available = [
            entry
            for entry in universe
            if suites.canonical_key(entry["state"]) not in used
        ]
        selected = suites.select_diverse(available, 128, seed)
        keys = suite_keys(selected)
        if len(keys) != 128 or keys & used:
            fail(f"cannot construct disjoint suite {label}")
        path = output / "suites" / f"suite_{label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        paths[label] = path
        manifest["suites"][label] = {
            "preregistered_seed": seed,
            "selector_seed": seed,
            "selection_pool_excludes_prior_suites": True,
            "sha256": sha256_file(path),
            "openings": 128,
            "consumed": True,
        }
        used |= keys
    return paths, manifest


def replay_states() -> dict[str, set[tuple[int, ...]]]:
    return {
        "seed45": {
            tuple(row["state"])
            for row in read_jsonl(
                Path("/tmp/azlite_pr247_fixed_target_budget/derived/fixed768.jsonl")
            )
        },
        "seed46": {
            tuple(row["state"])
            for row in read_jsonl(
                Path(
                    "/tmp/azlite_pr248_prospective_fixed_target_budget/generated/fresh768.jsonl"
                )
            )
        },
    }


def overlap_report(paths: dict[str, Path]) -> dict[str, Any]:
    canonical = suites.load_suite_jsonl(str(CANONICAL_SUITE))
    replays = replay_states()
    all_keys, report = (
        set(),
        {
            "within_suite_duplicates": {},
            "cross_suite_duplicates": 0,
            "canonical_duplicates": {},
            "replay_state_duplicates": {},
        },
    )
    for label, path in paths.items():
        entries = suites.load_suite_jsonl(str(path))
        keys = suite_keys(entries)
        encoded = {
            tuple(encode_state(entry["state"], input_encoding="kalah_v3"))
            for entry in entries
        }
        report["within_suite_duplicates"][label] = len(entries) - len(keys)
        report["canonical_duplicates"][label] = len(keys & suite_keys(canonical))
        report["replay_state_duplicates"][label] = {
            seed: len(encoded & states) for seed, states in replays.items()
        }
        report["cross_suite_duplicates"] += len(keys & all_keys)
        all_keys |= keys
    if (
        any(report["within_suite_duplicates"].values())
        or report["cross_suite_duplicates"]
        or any(report["canonical_duplicates"].values())
        or any(any(v.values()) for v in report["replay_state_duplicates"].values())
    ):
        fail("fresh suite overlap")
    report["passed"] = True
    return report


def arena(path: Path, candidate: Path, suite: Path, role: str) -> list[dict[str, Any]]:
    return isolation.arena_records(
        path, candidate, P1_ARTIFACT, CONTEXT, role, WORKERS, suite
    )


def wdl(records: list[dict[str, Any]]) -> dict[str, int]:
    return pr244.win_draw_loss(records)


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def cached_records(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 512:
        return None
    return records


def evaluate(
    paths: dict[str, Path], output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls, result = {}, {}
    for label, suite in paths.items():
        control_path = output / "records" / label / "p1_control.json"
        controls[label] = cached_records(control_path) or arena(
            output / "arena" / label, P1_ARTIFACT, suite, "p1_control"
        )
        write_records(control_path, controls[label])
        result[label] = {"control_wdl": wdl(controls[label]), "candidates": {}}
        for name, candidate in CANDIDATES.items():
            record_path = output / "records" / label / f"{name}.json"
            records = cached_records(record_path) or arena(
                output / "arena" / label, candidate["artifact"], suite, name
            )
            write_records(record_path, records)
            effect = paired_opening_candidate_effect(records, controls[label])
            result[label]["candidates"][name] = {
                "effect": effect,
                "win_draw_loss": wdl(records),
                "records": records,
            }
    return result, controls


def record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["opening_index"]),
        int(record["challenger_player"]),
        int(record["game_within_opening"]),
    )


def equivalence(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    by_left = {record_key(row): row for row in left}
    by_right = {record_key(row): row for row in right}
    if set(by_left) != set(by_right):
        fail("historical record key mismatch")
    trajectory_equal = [
        by_left[key]["trajectory"] == by_right[key]["trajectory"] for key in by_left
    ]
    outcome_equal = [
        by_left[key]["winner"] == by_right[key]["winner"] for key in by_left
    ]
    return {
        "completely_game_record_identical": all(
            by_left[key] == by_right[key] for key in by_left
        ),
        "trajectory_identical": all(trajectory_equal),
        "outcome_identical": all(outcome_equal),
        "outcome_identical_but_trajectory_different": all(outcome_equal)
        and not all(trajectory_equal),
    }


def divergence(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    by_left = {record_key(row): row for row in left}
    by_right = {record_key(row): row for row in right}
    divergent, plies, seats, changed = 0, [], Counter(), 0
    for key, left_row in by_left.items():
        right_row = by_right[key]
        left_moves = left_row["trajectory"].split(",")
        right_moves = right_row["trajectory"].split(",")
        first = next(
            (
                i
                for i, pair in enumerate(zip(left_moves, right_moves))
                if pair[0] != pair[1]
            ),
            None,
        )
        if first is not None or len(left_moves) != len(right_moves):
            divergent += 1
            plies.append(
                first if first is not None else min(len(left_moves), len(right_moves))
            )
            seats[str(left_row["challenger_player"])] += 1
        changed += left_row["winner"] != right_row["winner"]
    return {
        "games": len(by_left),
        "fraction_with_move_divergence": divergent / len(by_left),
        "first_divergent_ply_distribution": dict(sorted(Counter(plies).items())),
        "divergent_challenger_seat": dict(sorted(seats.items())),
        "final_outcome_changed_fraction": changed / len(by_left),
    }


def canonical_audit(output: Path) -> dict[str, Any]:
    control_path = output / "canonical" / "records" / "p1_control.json"
    control = cached_records(control_path) or arena(
        output / "canonical" / "arena", P1_ARTIFACT, CANONICAL_SUITE, "p1_control"
    )
    write_records(control_path, control)
    records = {}
    for name, candidate in CANDIDATES.items():
        record_path = output / "canonical" / "records" / f"{name}.json"
        records[name] = cached_records(record_path) or arena(
            output / "canonical" / "arena", candidate["artifact"], CANONICAL_SUITE, name
        )
        write_records(record_path, records[name])
        effect = paired_opening_candidate_effect(records[name], control)[
            "paired_candidate_effect"
        ]
        if effect != candidate["historical_effect"]:
            fail(f"canonical reproduction mismatch: {name}={effect}")
    return {
        "canonical_suite_sha256": sha256_file(CANONICAL_SUITE),
        "reproduced_effects": {
            name: candidate["historical_effect"]
            for name, candidate in CANDIDATES.items()
        },
        "positive_models": equivalence(
            records["seed45_fixed768_positive"], records["seed46_fresh1024_positive"]
        ),
        "negative_models": equivalence(
            records["seed45_fixed1024_negative"], records["seed46_fresh768_negative"]
        ),
    }


def bootstrap(data: np.ndarray, seed: int = 42) -> dict[str, float | int]:
    draws = data[
        np.random.default_rng(seed).integers(0, len(data), size=(10_000, len(data)))
    ].mean(axis=1)
    return {
        "effect": float(data.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "samples": 10_000,
    }


def analyses(evaluations: dict[str, Any]) -> dict[str, Any]:
    result = {"per_suite_contrasts": {}, "pooled": {}, "hierarchical_bootstrap": {}}
    for contrast, (positive, negative) in CONTRASTS.items():
        pooled, by_suite = [], []
        result["per_suite_contrasts"][contrast] = {}
        for label, data in evaluations.items():
            left, right = (
                data["candidates"][positive]["effect"],
                data["candidates"][negative]["effect"],
            )
            diff = paired_effect_difference(left, right)
            result["per_suite_contrasts"][contrast][label] = diff
            values = np.asarray(list(diff["per_opening_effect"].values()), dtype=float)
            pooled.extend(values)
            by_suite.append(values)
        result["pooled"][contrast] = {
            **bootstrap(np.asarray(pooled)),
            "positive_suites": sum(float(v.mean()) > 0 for v in by_suite),
            "suite_effects": [float(v.mean()) for v in by_suite],
        }
        rng = np.random.default_rng(42)
        draws = np.asarray(
            [
                rng.choice(by_suite[rng.integers(0, 3)], 128, replace=True).mean()
                for _ in range(10_000)
            ]
        )
        result["hierarchical_bootstrap"][contrast] = {
            "effect": float(np.mean(pooled)),
            "lower_95": float(np.quantile(draws, 0.025)),
            "upper_95": float(np.quantile(draws, 0.975)),
            "samples": 10_000,
        }
    return result


def classification(analysis: dict[str, Any]) -> str:
    passes = []
    for contrast in CONTRASTS:
        item = analysis["pooled"][contrast]
        passes.append(item["lower_95"] > 0 and item["positive_suites"] >= 2)
    if all(passes):
        return "canonical_strength_signal_generalizes"
    if sum(passes) == 1:
        return "one_positive_candidate_generalizes"
    if all(analysis["pooled"][contrast]["upper_95"] < 0 for contrast in CONTRASTS):
        return "canonical_suite_overfit"
    return "fresh_suite_results_inconclusive"


def concise_evaluations(evaluations: dict[str, Any]) -> dict[str, Any]:
    return {
        suite: {
            "control_wdl": data["control_wdl"],
            "candidates": {
                candidate: {
                    key: value for key, value in item.items() if key != "records"
                }
                for candidate, item in data["candidates"].items()
            },
        }
        for suite, data in evaluations.items()
    }


def fresh_telemetry(evaluations: dict[str, Any]) -> dict[str, Any]:
    return {
        label: {
            contrast: divergence(
                data["candidates"][positive]["records"],
                data["candidates"][negative]["records"],
            )
            for contrast, (positive, negative) in CONTRASTS.items()
        }
        for label, data in evaluations.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr249_fresh_suite_generalization"),
    )
    parser.add_argument("--generate-suites-only", action="store_true")
    args = parser.parse_args()
    if sha256_file(CANONICAL_SUITE) != CANONICAL_SHA:
        fail("canonical suite SHA")
    manifest = candidate_manifest()
    args.workdir.mkdir(parents=True, exist_ok=True)
    paths, suite_manifest = fresh_suites(args.workdir)
    overlap = overlap_report(paths)
    frozen = {
        "candidate_manifest": manifest,
        "suite_manifest": suite_manifest,
        "overlap_report": overlap,
    }
    (args.workdir / "frozen_manifest.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.generate_suites_only:
        return
    canonical = canonical_audit(args.workdir)
    evaluations, _controls = evaluate(paths, args.workdir)
    analysis = analyses(evaluations)
    summary = {
        **frozen,
        "canonical_invariant_and_audit": canonical,
        "evaluation_context": {
            "budget": CONTEXT,
            "ordinary_puct": True,
            "workers": WORKERS,
        },
        "per_suite": concise_evaluations(evaluations),
        "analysis": analysis,
        "first_divergence_telemetry": fresh_telemetry(evaluations),
        "classification": classification(analysis),
    }
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary["classification"])


if __name__ == "__main__":
    main()
