"""Merge completed frozen dual-trajectory diagnostics into localization evidence."""

from __future__ import annotations

import json
from collections import Counter

from ml.alphazero_lite.arena_conflict_state_localization import (
    ARMS,
    ROOT,
    active_bucket,
    phase,
)

DATA_DIR = ROOT / "docs/data/alphazero-lite-replay-source-attribution"


def main() -> int:
    localization_path = DATA_DIR / "arena_conflict_state_localization.json"
    payload = json.loads(localization_path.read_text(encoding="utf-8"))
    pair_delta = {
        (row["arm"], row["opening_id"]): row["pair_score_delta"]
        for row in payload["opening_pair_contributions"]
    }
    rows_by_arm: dict[str, list[dict]] = {}
    for arm in ARMS:
        path = DATA_DIR / f"dual-trajectory-divergences-{arm}.json"
        rows_by_arm[arm] = json.loads(path.read_text(encoding="utf-8"))
    summary = {}
    for arm, arm_rows in rows_by_arm.items():
        summary[arm] = {
            "divergence_rows": len(arm_rows),
            "by_trajectory": dict(Counter(row["trajectory"] for row in arm_rows)),
            "exact_root_rows": sum(row["exact_root_handoff"] for row in arm_rows),
            "outcome_associated_rows": sum(
                pair_delta[(arm, row["opening_id"])] != 0.0 for row in arm_rows
            ),
            "active_stone_buckets": dict(
                Counter(
                    active_bucket(
                        sum(row["state"]["player_pits"])
                        + sum(row["state"]["opponent_pits"])
                    )
                    for row in arm_rows
                )
            ),
            "phases": dict(
                Counter(
                    phase(
                        int(row["ply"]),
                        sum(row["state"]["player_pits"])
                        + sum(row["state"]["opponent_pits"]),
                    )
                    for row in arm_rows
                )
            ),
        }
    payload["dual_counterfactual_trajectory_divergences"] = {
        "scope": "Both F and F-X trajectories were advanced independently from every frozen opening/seat under the fixed 384 PUCT/exact-root16 policy.",
        "artifacts": {arm: f"dual-trajectory-divergences-{arm}.json" for arm in ARMS},
        "summary": summary,
    }
    localization_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (DATA_DIR / "dual-trajectory-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
