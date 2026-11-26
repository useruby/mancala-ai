"""Independent verification of Seed 639 per-state policy evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

import numpy as np


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _entropy(probabilities: list[float]) -> float:
    return -math.fsum(p * math.log2(p) for p in probabilities if p > 0.0)


def verify_policy_records(
    population_rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Rebuild identities, legal masks, priors, TV, and action classes."""
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.self_play import encode_state

    targets = {17: 103, 18: 103, 19: 102, 20: 102, 21: 102}
    if len(population_rows) != 512 or len(records) != 512:
        raise ValueError("seed639_record_coverage_not_512")
    canonical_seen: set[str] = set()
    float_seen: set[str] = set()
    by_bin: Counter[int] = Counter()
    for index, (source, record) in enumerate(
        zip(population_rows, records, strict=True)
    ):
        state = source.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"seed639_state_missing:{index}")
        canonical_id = _sha256(_canonical(state))
        encoded = np.asarray(
            encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
        )
        float_id = _sha256(encoded.tobytes())
        if canonical_id != source.get(
            "canonical_identity"
        ) or canonical_id != record.get("canonical_identity"):
            raise ValueError(f"seed639_canonical_identity_mismatch:{index}")
        if float_id != source.get(
            "production_float32_identity"
        ) or float_id != record.get("production_float32_identity"):
            raise ValueError(f"seed639_float32_identity_mismatch:{index}")
        if canonical_id in canonical_seen or float_id in float_seen:
            raise ValueError(f"seed639_duplicate_identity:{index}")
        canonical_seen.add(canonical_id)
        float_seen.add(float_id)

        game = KalahGame.from_state(state)
        legal = [int(action) for action in game.possible_moves()]
        if legal != record.get("legal_actions") or legal != source.get("legal_actions"):
            raise ValueError(f"seed639_legal_actions_mismatch:{index}")
        for model in ("B512", "S615"):
            vector = record.get(f"{model}_legal_probabilities")
            if (
                not isinstance(vector, list)
                or len(vector) != 6
                or any(
                    not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0
                    for p in vector
                )
                or any(
                    vector[action] != 0.0 for action in range(6) if action not in legal
                )
                or not math.isclose(math.fsum(vector), 1.0, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(
                    f"seed639_invalid_masked_probabilities:{model}:{index}"
                )
        b = record["B512_legal_probabilities"]
        s = record["S615_legal_probabilities"]
        recorded_tv = float(record.get("total_variation", math.nan))
        recorded_b_entropy = float(record.get("B512_entropy_bits", math.nan))
        recorded_s_entropy = float(record.get("S615_entropy_bits", math.nan))
        recorded_entropy_delta = float(
            record.get("entropy_difference_S615_minus_B512_bits", math.nan)
        )
        b_top = max(range(6), key=lambda action: (b[action], -action))
        s_top = max(range(6), key=lambda action: (s[action], -action))
        tv = math.fsum(abs(left - right) for left, right in zip(b, s, strict=True)) / 2
        if (
            b_top != record.get("B512_top_action")
            or s_top != record.get("S615_top_action")
            or (b_top != s_top) != record.get("top_action_disagreement")
            or not math.isclose(tv, recorded_tv, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(
                _entropy(b), recorded_b_entropy, rel_tol=0, abs_tol=1e-15
            )
            or not math.isclose(
                _entropy(s), recorded_s_entropy, rel_tol=0, abs_tol=1e-15
            )
            or not math.isclose(
                _entropy(s) - _entropy(b),
                recorded_entropy_delta,
                rel_tol=0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(f"seed639_policy_metric_mismatch:{index}")
        for provenance_key in (
            "active_pit_stones",
            "game_id",
            "opening_identity",
            "opening_index",
            "continuation_ply",
        ):
            if record.get(provenance_key) != source.get(provenance_key):
                raise ValueError(
                    f"seed639_provenance_mismatch:{index}:{provenance_key}"
                )
        stones = sum(game.pits)
        if stones != source.get("active_pit_stones"):
            raise ValueError(f"seed639_stone_bin_mismatch:{index}")
        by_bin[stones] += 1
    if dict(by_bin) != targets:
        raise ValueError("seed639_stone_bin_quotas_mismatch")
    return {
        "passed": True,
        "states_verified": 512,
        "canonical_identities": len(canonical_seen),
        "production_float32_identities": len(float_seen),
        "stone_bin_counts": {str(key): by_bin[key] for key in sorted(targets)},
    }


def verify_subset_selection(
    records: list[dict[str, Any]], selected_state_ids: list[str]
) -> dict[str, Any]:
    """Independently reproduce the registered greedy TV-tail selection."""
    if len(records) != 512:
        raise ValueError("seed639_subset_source_count_not_512")
    ranked = sorted(
        records,
        key=lambda row: (
            -float(row["total_variation"]),
            _sha256(str(row["canonical_identity"]).encode("utf-8")),
            str(row["canonical_identity"]),
        ),
    )
    chosen: list[str] = []
    games: set[str] = set()
    openings: Counter[str] = Counter()
    for row in ranked:
        game_id = str(row["game_id"])
        opening_id = str(row["opening_identity"])
        if game_id in games or openings[opening_id] >= 4:
            continue
        chosen.append(str(row["canonical_identity"]))
        games.add(game_id)
        openings[opening_id] += 1
        if len(chosen) == 64:
            break
    available = len(chosen) == 64 and len(games) >= 32
    expected_ids = chosen if available else []
    if selected_state_ids != expected_ids:
        raise ValueError("seed639_secondary_subset_mismatch")
    return {
        "passed": True,
        "available": available,
        "selected_count": len(expected_ids),
        "distinct_continuation_games": len(games) if available else 0,
        "distinct_opening_clusters": len(openings) if available else 0,
    }
