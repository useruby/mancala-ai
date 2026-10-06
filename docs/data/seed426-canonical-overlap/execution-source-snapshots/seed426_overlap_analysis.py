"""Pure identity and overlap accounting for the retrospective seed426 audit."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter, defaultdict
from typing import Any

IDENTITY_DEFINITIONS = {
    "canonical": "ordered JSON integer tuple (player_pits[6], opponent_pits[6], player_store, opponent_store, current_player); all fields retained; byte-string JSON encoding with compact separators",
    "input": "exact loaded input vector serialized as 27 consecutive little-endian IEEE-754 binary32 values; equality is byte equality; group membership reported independently from canonical identity",
}
ACCOUNTING_DEFINITIONS = {
    "raw_line": "1-based line number within each registered source file",
    "compact_row": "global index assigned in registered source order and raw-line order to each parsed, loader-eligible row; the frozen call has exclude_buckets=None, so expected skipped raw lines are explicitly empty",
    "loaded_targets": "actual frozen train.load_jsonl_replay called with policy_target_mode=sharpened, value_target_mode=default, and registered per-source replay_value_target_modes; per-source target arrays are hash-bound; source feature arrays compared with loaded x",
    "weighted_position_order": "source order; within each source the complete compact index sequence repeated weight times, matching np.tile(np.arange(source_start, source_end), weight); copy is not an independent source row",
    "split": "ordered freeze train_positions and validation_positions index the expanded positions; both ordered int64 arrays checked by exact equality and registered SHA256; source-row sets are checked disjoint and complete",
    "denominator_unique_identity": "validation unique identities; numerator is identities also present in training",
    "denominator_eligible_source_rows": "distinct validation compact rows; numerator is distinct validation compact rows whose identity occurs in any training row",
    "denominator_weighted_positions": "expanded validation positions; numerator is expanded validation positions whose identity occurs in any training row; weighted copies are exposure accounting only, not independent observations",
    "source_attribution": "additive membership-pattern table partitions shared identities by exact train/validation source-membership sets; source-pair exposures are pairwise and overlap when identities occur in multiple sources",
}
DECISION_RULES = {
    "canonical": "any canonical intersection means canonical_state_overlap_detected; zero intersection means canonical_state_disjoint",
    "metric_correction": "if >=5% of >32 weighted validation positions share a canonical identity with training, recommend a separately authorized validation-metric correction using an identity-disjoint holdout",
    "otherwise": "close the diagnostic without authorizing training or replay changes",
}


def state_identity(state: dict[str, Any]) -> bytes:
    """Canonical identity includes pits, stores, and the side to move."""
    fields = (
        tuple(int(value) for value in state["player_pits"]),
        tuple(int(value) for value in state["opponent_pits"]),
        int(state["player_store"]),
        int(state["opponent_store"]),
        int(state["current_player"]),
    )
    return json.dumps(fields, separators=(",", ":")).encode("ascii")


def float32_identity(encoded: list[float]) -> bytes:
    """Little-endian IEEE-754 float32 bytes, with shape supplied by manifest."""
    return struct.pack("<" + "f" * len(encoded), *encoded)


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize train/validation identities from expanded row-accounting entries."""
    result: dict[str, Any] = {}

    def summarize(selected: list[dict[str, Any]], key: str) -> dict[str, Any]:
        train_rows = [row for row in selected if row["partition"] == "train"]
        valid_rows = [row for row in selected if row["partition"] == "validation"]
        train = Counter(row[key] for row in train_rows)
        valid = Counter(row[key] for row in valid_rows)
        shared = train.keys() & valid.keys()
        shared_validation = [row for row in valid_rows if row[key] in train]
        by_source: dict[str, Counter] = defaultdict(Counter)
        for row in selected:
            by_source[row["source"]][(row["partition"], row[key])] += 1
        within = sum(
            bool(by_source[source][("train", identity)])
            and bool(by_source[source][("validation", identity)])
            for source in by_source
            for identity in {item for _, item in by_source[source]}
        )
        total_validation = len(valid_rows)
        shared_rows = len(shared_validation)
        train_members: dict[Any, set[str]] = defaultdict(set)
        valid_members: dict[Any, set[str]] = defaultdict(set)
        for row in train_rows:
            train_members[row[key]].add(row["source"])
        for row in valid_rows:
            valid_members[row[key]].add(row["source"])
        patterns: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
        pairwise_sources: Counter[str] = Counter()
        additive_source: Counter[tuple[str, str]] = Counter()
        identity_exposures = [valid[item] for item in shared]
        for identity in shared:
            train_sources = tuple(sorted(train_members[identity]))
            valid_sources = tuple(sorted(valid_members[identity]))
            patterns[(train_sources, valid_sources)] += valid[identity]
            for train_source in train_sources:
                for valid_source in valid_sources:
                    pairwise_sources[
                        "within_source"
                        if train_source == valid_source
                        else "cross_source"
                    ] += valid[identity]
            has_within = bool(set(train_sources) & set(valid_sources))
            has_cross = any(
                train_source != valid_source
                for train_source in train_sources
                for valid_source in valid_sources
            )
            attribution = (
                "within_and_cross"
                if has_within and has_cross
                else "within_source_only"
                if has_within
                else "cross_source_only"
            )
            additive_source[(attribution, "identities")] += 1
            additive_source[(attribution, "validation_positions")] += valid[identity]
        total_shared_exposure = sum(identity_exposures)
        top_exposure = sorted(identity_exposures, reverse=True)
        return {
            "train_unique": len(train),
            "validation_unique": len(valid),
            "intersection_unique": len(shared),
            "validation_unseen_unique": len(valid.keys() - train.keys()),
            "validation_overlap_denominators": {
                "unique_identity": {
                    "numerator": len(shared),
                    "denominator": len(valid),
                    "fraction": len(shared) / len(valid) if valid else 0.0,
                },
                "eligible_source_rows": {
                    "numerator": len({row["compact_row"] for row in shared_validation}),
                    "denominator": len({row["compact_row"] for row in valid_rows}),
                    "fraction": len({row["compact_row"] for row in shared_validation})
                    / len({row["compact_row"] for row in valid_rows})
                    if valid_rows
                    else 0.0,
                },
                "weighted_positions": {
                    "numerator": shared_rows,
                    "denominator": total_validation,
                    "fraction": shared_rows / total_validation
                    if total_validation
                    else 0.0,
                },
            },
            "shared_group_sizes": {
                str(size): count
                for size, count in sorted(
                    Counter(valid[item] for item in shared).items()
                )
            },
            "within_source_shared_identity_pairs": within,
            "source_pairwise_validation_exposure": dict(pairwise_sources),
            "additive_source_attribution": {
                category: {
                    metric: additive_source[(category, metric)]
                    for metric in ("identities", "validation_positions")
                }
                for category in (
                    "within_source_only",
                    "cross_source_only",
                    "within_and_cross",
                )
            },
            "additive_membership_patterns": [
                {
                    "training_sources": list(train_sources),
                    "validation_sources": list(valid_sources),
                    "shared_validation_positions": count,
                }
                for (train_sources, valid_sources), count in sorted(patterns.items())
            ],
            "exposure_concentration": {
                "shared_validation_positions": total_shared_exposure,
                "top_1_fraction": top_exposure[0] / total_shared_exposure
                if total_shared_exposure
                else 0.0,
                "top_10_fraction": sum(top_exposure[:10]) / total_shared_exposure
                if total_shared_exposure
                else 0.0,
                "identity_hhi": sum(
                    (count / total_shared_exposure) ** 2 for count in identity_exposures
                )
                if total_shared_exposure
                else 0.0,
            },
            "shared_validation_by_source": dict(
                sorted(Counter(row["source"] for row in shared_validation).items())
            ),
            "shared_groups": sorted(
                (
                    {"train_rows": train[item], "validation_rows": valid[item]}
                    for item in shared
                ),
                key=lambda group: (-group["validation_rows"], -group["train_rows"]),
            ),
        }

    for identity_name in ("canonical", "input"):
        key = f"{identity_name}_identity"
        result[identity_name] = summarize(rows, key)
        result[identity_name]["buckets"] = {
            label: summarize(
                [row for row in rows if predicate(row["active_stones"])], key
            )
            for label, predicate in (
                (">32", lambda value: value > 32),
                ("17-32", lambda value: 17 <= value <= 32),
                ("<=16", lambda value: value <= 16),
            )
        }
    return result


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
