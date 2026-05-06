#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite


ARTIFACT_SCHEMA = "azlite_forensic_references_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy-simulations", type=int, default=1200)
    parser.add_argument("--value-simulations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-seed",
        type=int,
        action="append",
        dest="sample_seeds",
        default=None,
        help="Optional explicit seeds to sample; defaults to --seed when omitted.",
    )
    return parser.parse_args()


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _aggregate_teacher_value(seed_samples: list[dict[str, Any]]) -> float | None:
    teacher_values = [float(sample["teacher_value"]) for sample in seed_samples if sample.get("teacher_value") is not None]
    if not teacher_values:
        return None
    return _round_optional(sum(teacher_values) / len(teacher_values))


def finalize_reference_row(
    *,
    row_id: str,
    canonical_state: str,
    state: dict[str, Any],
    seed_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_reference_moves = sorted({int(sample["reference_move"]) for sample in seed_samples})
    reference_unstable = len(observed_reference_moves) > 1
    stable_sample = seed_samples[0]
    row = {
        "id": row_id,
        "canonical_state": canonical_state,
        "state": state,
        "reference_move": None if reference_unstable else int(stable_sample["reference_move"]),
        "teacher_value": None if reference_unstable else _aggregate_teacher_value(seed_samples),
        "reference_unstable": reference_unstable,
        "observed_reference_moves": observed_reference_moves,
        "seed_samples": [
            {
                "seed": int(sample["seed"]),
                "reference_move": int(sample["reference_move"]),
                "teacher_value": _round_optional(sample.get("teacher_value")),
            }
            for sample in seed_samples
        ],
    }
    if not reference_unstable and "child_stats" in stable_sample:
        row["child_stats"] = stable_sample["child_stats"]
    return row


def finalize_reference_rows(evaluated_positions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_canonical_state: dict[str, dict[str, Any]] = {}
    for position in evaluated_positions:
        canonical_state = str(position["canonical_state"])
        if canonical_state in rows_by_canonical_state:
            continue
        rows_by_canonical_state[canonical_state] = finalize_reference_row(
            row_id=str(position["row_id"]),
            canonical_state=canonical_state,
            state=position["state"],
            seed_samples=position["seed_samples"],
        )
    return list(rows_by_canonical_state.values())


def build_reference_artifact(
    *,
    suite_path: str | Path,
    out_path: str | Path,
    policy_simulations: int,
    value_simulations: int,
    seed: int,
    sample_seeds: Iterable[int] | None = None,
    reference_runner: Callable[[dict[str, Any], int, int, int, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    suite = load_suite(suite_path)
    artifact_path = Path(out_path)
    value_reference_simulations = int(value_simulations) if int(value_simulations) > 0 else int(policy_simulations)
    sampled_seeds = [int(value) for value in sample_seeds] if sample_seeds is not None else [int(seed)]
    if not sampled_seeds:
        raise ValueError("sample_seeds must not be empty")

    if reference_runner is None:
        from ml.alphazero_lite.run_forensic_suite import run_reference

        reference_runner = run_reference

    unique_positions_by_canonical_state: dict[str, dict[str, Any]] = {}
    for index, position in enumerate(suite):
        canonical_state = canonical_state_key(position.state)
        if canonical_state in unique_positions_by_canonical_state:
            continue

        samples = []
        for sample_seed in sampled_seeds:
            reference = reference_runner(
                position.state,
                int(policy_simulations),
                value_reference_simulations,
                int(sample_seed),
                index,
            )
            samples.append(
                {
                    "seed": int(sample_seed),
                    "reference_move": int(reference["selected_move"]),
                    "teacher_value": reference.get("teacher_value"),
                    **({"child_stats": reference["child_stats"]} if "child_stats" in reference else {}),
                }
            )

        unique_positions_by_canonical_state[canonical_state] = {
            "row_id": position.id,
            "canonical_state": canonical_state,
            "state": position.state,
            "seed_samples": samples,
        }

    rows = finalize_reference_rows(unique_positions_by_canonical_state.values())

    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "suite_path": str(Path(suite_path)),
        "reference": {
            "policy_simulations": int(policy_simulations),
            "value_simulations": value_reference_simulations,
            "sample_seeds": sampled_seeds,
        },
        "rows": rows,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    args = parse_args()
    rows = build_reference_artifact(
        suite_path=args.suite,
        out_path=args.out,
        policy_simulations=args.policy_simulations,
        value_simulations=args.value_simulations,
        seed=args.seed,
        sample_seeds=args.sample_seeds,
    )
    print(f"wrote {len(rows)} forensic reference rows to {args.out}")


if __name__ == "__main__":
    main()
