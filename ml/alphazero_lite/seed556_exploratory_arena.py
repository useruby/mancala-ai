"""Exploratory T512 versus frozen serving-P screen protocol."""

from __future__ import annotations

from typing import Any

from ml.alphazero_lite import seed552_c108_serving_p as seed552
from ml.alphazero_lite import seed555_policy_strength_screen as seed555

ROOT = seed555.ROOT
OUT = "docs/data/seed556-exploratory-arena/screen-0005"
PAIR_CONCURRENCY = 18
SEED554 = seed555.SEED554
SEED511 = seed555.SEED511
OPENINGS_PATH = seed555.OPENINGS_PATH
OPENINGS_SHA256 = seed555.OPENINGS_SHA256
OPENINGS = seed555.OPENINGS
GAMES = seed555.GAMES
SEED = seed555.SEED
SIMULATIONS = seed555.SIMULATIONS
BOOTSTRAP_RESAMPLES = seed555.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = seed555.BOOTSTRAP_SEED
T512_CHECKPOINT = seed555.T512_CHECKPOINT
P_CHECKPOINT = seed552.P_CHECKPOINT
INITIALIZER = seed555.INITIALIZER
T512_SHA256 = seed555.T512_SHA256
P_SHA256 = seed552.P_SHA256
INITIALIZER_SHA256 = seed555.INITIALIZER_SHA256
TABLEBASE_PATH = seed555.TABLEBASE_PATH
TABLEBASE_SHA256 = seed555.TABLEBASE_SHA256
TABLEBASE_SIZE = seed555.TABLEBASE_SIZE
TABLEBASE_POINTER_SHA256 = seed555.TABLEBASE_POINTER_SHA256
TABLEBASE_POINTER_COMMIT = seed555.TABLEBASE_POINTER_COMMIT
NATIVE_PROBE = seed555.NATIVE_PROBE
SEARCH_OPTIONS = seed555.SEARCH_OPTIONS
ASSIGNMENTS = {
    "challenger": "#554 T512 update-108 policy and own trunk",
    "challenger_policy_artifact": "T512",
    "challenger_policy_trunk": "T512",
    "challenger_value_artifact": "initializer",
    "challenger_value_trunk": "initializer",
    "opponent": "#552 frozen serving-P seed504 E4 policy and own trunk",
    "opponent_policy_artifact": "P-E4",
    "opponent_policy_trunk": "P-E4",
    "opponent_value_artifact": "initializer",
    "opponent_value_trunk": "initializer",
}


def __getattr__(name: str) -> Any:
    """Share unchanged frozen protocol helpers with authenticated predecessors."""
    if hasattr(seed555, name):
        return getattr(seed555, name)
    if hasattr(seed552, name):
        return getattr(seed552, name)
    raise AttributeError(name)


def overlap_report(root, openings):
    """Recompute T512, P-training, and development overlap for this matchup."""
    t512 = seed555.overlap_report(root, openings)
    p = seed552.opening_overlaps(root, openings)
    rows = []
    counts = {
        key: {"canonical": 0, "production_float32": 0}
        for key in ("T512_training", "P_policy_training", "unchanged_development")
    }
    for a, b in zip(t512["per_opening"], p["per_opening"], strict=True):
        row = {
            "opening_index": a["opening_index"],
            "canonical_identity": a["canonical_identity"],
            "production_float32_identity": a["production_float32_identity"],
            "T512_training_canonical_overlap": a["T512_training_canonical_overlap"],
            "T512_training_production_float32_overlap": a[
                "T512_training_production_float32_overlap"
            ],
            "P_policy_training_canonical_overlap": b[
                "P_policy_training_canonical_overlap"
            ],
            "P_policy_training_production_float32_overlap": b[
                "P_policy_training_production_float32_overlap"
            ],
            "unchanged_development_canonical_overlap": a[
                "unchanged_development_canonical_overlap"
            ],
            "unchanged_development_production_float32_overlap": a[
                "unchanged_development_production_float32_overlap"
            ],
        }
        for key, fields in {
            "T512_training": (
                "T512_training_canonical_overlap",
                "T512_training_production_float32_overlap",
            ),
            "P_policy_training": (
                "P_policy_training_canonical_overlap",
                "P_policy_training_production_float32_overlap",
            ),
            "unchanged_development": (
                "unchanged_development_canonical_overlap",
                "unchanged_development_production_float32_overlap",
            ),
        }.items():
            counts[key]["canonical"] += bool(row[fields[0]])
            counts[key]["production_float32"] += bool(row[fields[1]])
        rows.append(row)
    return {
        "schema": "seed556_training_population_overlap_v1",
        "opening_count": OPENINGS,
        "overlap_opening_counts": counts,
        "T512_training_rows": t512["training_rows"]["T512"],
        "P_policy_training_rows": p["population_rows"]["P_policy_training"],
        "unchanged_development_members": t512["unchanged_development_members"],
        "seed461_continuation_overlap_unresolved": True,
        "per_opening": rows,
    }


def analyze(ledger):
    """Score the challenger using the registered #555 cluster bootstrap."""
    mapped = [
        {
            **row,
            "outcome": {"P": "C128", "T512": "T512"}.get(
                row["outcome"], row["outcome"]
            ),
        }
        for row in ledger
    ]
    result, scores = seed555.analyze(mapped)
    result["terminal_losses"] = result.pop("terminal_losses")
    result["decision"] = (
        "propose_T512_vs_serving_P_1200_screen"
        if result["decision"] == "propose_T512_vs_serving_P_screen"
        else "do_not_advance_T512"
        if result["strength_decision_valid"]
        else "no_valid_advancement_decision"
    )
    result["opponent"] = "frozen serving P (seed504 E4)"
    return result, scores
