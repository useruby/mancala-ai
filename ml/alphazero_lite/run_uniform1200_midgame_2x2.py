#!/usr/bin/env python3
"""Run PR #289's locked midgame replay-exposure x target-shape ablation.

This runner never generates self-play, changes search, or promotes an artifact.
It derives all dynamic replay lanes from PR #288's committed row-matched rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_table  # noqa: E402
from ml.alphazero_lite.self_play import value_target_bucket_for_move_index  # noqa: E402

SCHEMA = "azlite_uniform1200_midgame_2x2_v1"
SEEDS = (44, 45, 46)
LANES = (
    "uniform_exposure__sharpened",
    "control_like_exposure__sharpened",
    "uniform_exposure__unsharpened",
    "control_like_exposure__unsharpened",
)
EXPOSURE_REDUCTION_REQUIRED = 0.75
MIN_SHARED_STATES = 200
PHASES = ("early", "mid", "late")
FALLBACKS = (
    ("legal_move_count", "extra_turn_available", "capture_available"),
    ("legal_move_count", "extra_turn_available"),
    ("legal_move_count",),
)
AUDIT_PATH = (
    REPO_ROOT / "docs/data/alphazero-lite-uniform1200-replay-distribution-audit.json"
)
ROWMATCHED_ROOT = Path("/home/alex/Mancala/rowmatched-work")
CONFIRMATION_ROOT = Path("/tmp/uniform1200-seed-confirmation")
SELECTED_REPLAY = Path(
    "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl"
)
CONTROLS_REPLAY = Path(
    "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"
)
EXPECTED_PARENT_SHA256 = (
    "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
)
EXPECTED_FIXED_SHAS = {
    "selected": "787cca93c7db454c2daebe1fdab001e93df0b9f80a1e7005598a0324774c427b",
    "controls": "ffac24dc8fb773fddd4dc14f397317d7c5e0a973631ca71adb216b5e68b444a5",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def stable_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def phase(row: dict[str, Any]) -> str:
    return value_target_bucket_for_move_index(int(row.get("move_index", 0)))


def stratum(row: dict[str, Any]) -> tuple[int, int, bool, bool]:
    """Derive the registered pre-action descriptor via the Kalah rule helpers."""
    state = decode_state(list(row["state"]))
    game = KalahGame.from_state(state)
    consequences = move_consequence_table(state)
    return (
        game.current_player,
        len(game.possible_moves()),
        any(item["gives_extra_turn"] for item in consequences if item["legal"]),
        any(item["produces_capture"] for item in consequences if item["legal"]),
    )


def coarsen(
    key: tuple[int, int, bool, bool], fields: tuple[str, ...]
) -> tuple[Any, ...]:
    values = dict(
        zip(
            (
                "current_player",
                "legal_move_count",
                "extra_turn_available",
                "capture_available",
            ),
            key,
            strict=True,
        )
    )
    return tuple(values[field] for field in fields)


def _quota(counts: Counter[Any], total: int) -> dict[Any, int]:
    exact = {key: value * total / sum(counts.values()) for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda item: (-(exact[item] - result[item]), item))[
        : total - sum(result.values())
    ]:
        result[key] += 1
    return result


def _tv(left: Counter[Any], right: Counter[Any]) -> float:
    left_total, right_total = sum(left.values()), sum(right.values())
    return 0.5 * sum(
        abs(left[key] / left_total - right[key] / right_total)
        for key in set(left) | set(right)
    )


def control_like_mid_rows(
    uniform_mid: list[dict[str, Any]], control_mid: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match control descriptor proportions using only uniform rows.

    Unsupported fine strata are allocated under the first registered coarsening
    that has uniform availability; sampling is SHA-stable and replacement is a
    last resort.
    """
    target = Counter(stratum(row) for row in control_mid)
    available: dict[tuple[int, int, bool, bool], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in uniform_mid:
        available[stratum(row)].append(row)
    desired = _quota(target, len(uniform_mid))
    selected: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    for key in sorted(desired):
        quota = desired[key]
        candidates = available.get(key, [])
        fields = (
            "current_player",
            "legal_move_count",
            "extra_turn_available",
            "capture_available",
        )
        if not candidates:
            for fallback in FALLBACKS:
                candidates = [
                    row
                    for candidate, rows in available.items()
                    if coarsen(candidate, fallback) == coarsen(key, fallback)
                    for row in rows
                ]
                if candidates:
                    fields = fallback
                    fallbacks.append(
                        {
                            "target_stratum": list(key),
                            "coarsened_fields": list(fields),
                            "quota": quota,
                        }
                    )
                    break
        if not candidates:
            raise ValueError(
                f"unsupported control stratum has no registered fallback: {key}"
            )
        ordered = sorted(candidates, key=stable_digest)
        if quota > len(ordered):
            fallbacks.append(
                {
                    "target_stratum": list(key),
                    "coarsened_fields": list(fields),
                    "quota": quota,
                    "replacement": quota - len(ordered),
                }
            )
        selected.extend(ordered[index % len(ordered)] for index in range(quota))
    if len(selected) != len(uniform_mid):
        raise AssertionError("quota allocation changed selected-phase row count")
    selected_counts = Counter(stratum(row) for row in selected)
    original_counts = Counter(stratum(row) for row in uniform_mid)
    target_counts = Counter(stratum(row) for row in control_mid)
    original_tv, selected_tv = (
        _tv(original_counts, target_counts),
        _tv(selected_counts, target_counts),
    )
    if (
        original_tv
        and (original_tv - selected_tv) / original_tv < EXPOSURE_REDUCTION_REQUIRED
    ):
        raise ValueError("control-like exposure did not reduce stratum TV by 75%")
    return selected, {
        "target_control_proportions": proportions(target_counts),
        "original_uniform_proportions": proportions(original_counts),
        "control_like_proportions": proportions(selected_counts),
        "original_tv": original_tv,
        "control_like_tv": selected_tv,
        "tv_reduction": 0.0
        if not original_tv
        else (original_tv - selected_tv) / original_tv,
        "maximum_percentage_point_error": 100
        * max(
            (
                abs(
                    selected_counts[key] / len(selected)
                    - target_counts[key] / len(control_mid)
                )
                for key in set(selected_counts) | set(target_counts)
            ),
            default=0.0,
        ),
        "fallbacks": fallbacks,
    }


def proportions(counts: Counter[Any]) -> dict[str, float]:
    total = sum(counts.values())
    return {
        "|".join(map(str, key)) if isinstance(key, tuple) else str(key): value / total
        for key, value in sorted(counts.items())
    }


def unsharpen(policy: list[float]) -> list[float]:
    roots = [math.sqrt(value) if value > 0.0 else 0.0 for value in policy]
    total = sum(roots)
    if total <= 0.0:
        raise ValueError("policy has no legal probability mass")
    return [value / total for value in roots]


def sharpen(policy: list[float]) -> list[float]:
    squares = [value * value if value > 0.0 else 0.0 for value in policy]
    total = sum(squares)
    return [value / total for value in squares]


def transform_rows(
    rows: list[dict[str, Any]],
    selected_phase: str,
    *,
    exposure: str,
    target: str,
    control_mid: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_mid = [row for row in rows if phase(row) == selected_phase]
    replacement, matching = (
        (base_mid, {})
        if exposure == "uniform_exposure"
        else control_like_mid_rows(base_mid, control_mid or [])
    )
    iterator = iter(replacement)
    output = []
    changed = 0
    for row in rows:
        result = (
            copy.deepcopy(next(iterator))
            if phase(row) == selected_phase
            else copy.deepcopy(row)
        )
        if target == "unsharpened" and phase(result) == selected_phase:
            original = list(result["policy"])
            result["policy"] = unsharpen(original)
            result["experiment_target_provenance"] = (
                "reconstructed_pre_square_temperature_adjusted_policy"
            )
            changed += 1
        output.append(result)
    if len(output) != len(rows):
        raise AssertionError("dynamic row count changed")
    return output, {
        "selected_phase_rows": len(base_mid),
        "target_rows_changed": changed,
        "matching": matching,
    }


def phase_selection(audit: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, list[float]] = {item: [] for item in PHASES}
    support: dict[str, list[int]] = {item: [] for item in PHASES}
    for seed in SEEDS:
        aggregate = audit["seeds"][str(seed)]["comparisons"]["rowmatched_uniform1200"][
            "matched_targets"
        ]["aggregate"]
        for item in PHASES:
            record = aggregate[f"phase={item}"]
            support[item].append(record["states"])
            if record["states"] < MIN_SHARED_STATES:
                raise ValueError(
                    f"seed {seed} phase {item} has insufficient shared-state support"
                )
            values[item].append(record["tv"])
    medians = {item: statistics.median(value) for item, value in values.items()}
    selected = max(PHASES, key=lambda item: (medians[item], item))
    return {
        "rule": "largest median matched-state policy TV across seeds 44/45/46 with >=200 shared states per seed",
        "per_phase_tv": values,
        "per_phase_shared_states": support,
        "median_tv": medians,
        "selected_phase": selected,
        "move_index_semantics": {"early": "0-9", "mid": "10-29", "late": "30+"},
    }


def dynamic_path(seed: int) -> Path:
    return ROWMATCHED_ROOT / "replays" / f"seed{seed}-uniform1200-rowmatched.jsonl"


def control_path(seed: int) -> Path:
    return (
        CONFIRMATION_ROOT
        / "runs"
        / f"seed{seed}"
        / "control_384_192"
        / f"uniform1200-confirm-seed{seed}-control_384_192-iter1"
        / "self_play.jsonl"
    )


def exposure_report(dynamic: Path) -> dict[str, Any]:
    sources = (
        ("dynamic_current_self_play", dynamic, 1),
        ("selected_fixed_replay", SELECTED_REPLAY, 1),
        ("guard_control_fixed_replay", CONTROLS_REPLAY, 2),
    )
    entries = [
        {"source": name, "rows": len(load_rows(path)), "weight": weight}
        for name, path, weight in sources
    ]
    total = sum(item["rows"] * item["weight"] for item in entries)
    return {
        "dynamic_raw_rows": entries[0]["rows"],
        "fixed_replay_rows": sum(item["rows"] for item in entries[1:]),
        "replay_weights": [item["weight"] for item in entries],
        "effective_sampled_index_counts": [
            item["rows"] * item["weight"] for item in entries
        ],
        "total_optimizer_examples_per_epoch": total,
    }


def preflight() -> dict[str, Any]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit["classification"] != "uniform1200_distribution_target_interaction":
        raise ValueError("PR #289 classification differs from the registered premise")
    parent = REPO_ROOT / "storage/ai/alphazero_lite/current/weights.json"
    if sha256_file(parent) != EXPECTED_PARENT_SHA256:
        raise ValueError("incumbent parent SHA mismatch")
    fixed = {
        name: sha256_file(path)
        for name, path in (("selected", SELECTED_REPLAY), ("controls", CONTROLS_REPLAY))
    }
    if fixed != EXPECTED_FIXED_SHAS:
        raise ValueError("fixed replay SHA mismatch")
    inherited = audit["artifact_sha256"]
    for seed in SEEDS:
        if (
            sha256_file(dynamic_path(seed))
            != inherited[str(seed)]["rowmatched"]["replay"]
        ):
            raise ValueError(f"seed {seed} PR #288 rowmatched replay SHA mismatch")
        if sha256_file(control_path(seed)) != inherited[str(seed)]["control"]["replay"]:
            raise ValueError(f"seed {seed} control replay SHA mismatch")
    return {
        "promotion": {"performed": False},
        "self_play": {"generated": False},
        "phase_selection": phase_selection(audit),
        "inherited_artifact_sha256": inherited,
        "parent_weights_sha256": sha256_file(parent),
        "fixed_replays": fixed,
    }


def train(replay: Path, seed: int, lane: str, workdir: Path) -> Path:
    out = (
        workdir / "runs" / f"seed{seed}" / lane / f"midgame-2x2-seed{seed}-{lane}-iter1"
    )
    out.mkdir(parents=True, exist_ok=True)
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        "ml/alphazero_lite/train.py",
        "--data",
        str(replay),
        "--data-files",
        f"{replay},{SELECTED_REPLAY},{CONTROLS_REPLAY}",
        "--replay-weights",
        "1,1,2",
        "--init-checkpoint",
        str(control_path(seed).parent / "parent_init_checkpoint.npz"),
        "--out",
        str(out / "checkpoint.npz"),
        "--epochs",
        "4",
        "--batch-size",
        "512",
        "--device",
        "auto",
        "--lr-scheduler",
        "none",
        "--hidden-sizes",
        "96,3",
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--value-loss",
        "huber",
        "--huber-delta",
        "1.0",
        "--value-loss-weight",
        "0.3",
        "--val-split",
        "0.1",
        "--grad-clip",
        "1.0",
        "--save-top-k",
        "3",
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--seed",
        str(seed),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "ml/alphazero_lite/export_artifact.py",
            "--checkpoint",
            str(out / "checkpoint.npz"),
            "--out-dir",
            str(out),
            "--version",
            out.name,
            "--model-type",
            "residual_v3",
            "--rules-version",
            "kalah_v1",
            "--input-encoding",
            "kalah_v3",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)
    result: dict[str, Any] = {"schema": SCHEMA, "preflight": preflight(), "seeds": {}}
    selected_phase = result["preflight"]["phase_selection"]["selected_phase"]
    for seed in SEEDS:
        base, control = load_rows(dynamic_path(seed)), load_rows(control_path(seed))
        seed_result: dict[str, Any] = {"lanes": {}}
        expected_exposure = None
        for lane in LANES:
            exposure, target = lane.split("__", 1)
            rows, diagnostics = transform_rows(
                base,
                selected_phase,
                exposure=exposure,
                target=target,
                control_mid=[row for row in control if phase(row) == selected_phase],
            )
            replay = (
                dynamic_path(seed)
                if lane == LANES[0]
                else args.workdir / "replays" / f"seed{seed}-{lane}.jsonl"
            )
            if lane != LANES[0]:
                write_rows(replay, rows)
            report = exposure_report(replay)
            if expected_exposure is None:
                expected_exposure = report
            elif report != expected_exposure:
                raise ValueError(
                    f"seed {seed} effective exposure differs across factorial cells"
                )
            entry: dict[str, Any] = {
                "replay": str(replay),
                "replay_sha256": sha256_file(replay),
                "diagnostics": diagnostics,
                "effective_training_exposure": report,
                "training": {
                    "status": "reused_pr288" if lane == LANES[0] else "not_requested"
                },
            }
            if args.train and lane != LANES[0]:
                out = train(replay, seed, lane, args.workdir)
                entry["training"] = {
                    "status": "completed",
                    "artifact": str(out),
                    "weights_sha256": sha256_file(out / "weights.json"),
                }
            seed_result["lanes"][lane] = entry
        result["seeds"][str(seed)] = seed_result
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
