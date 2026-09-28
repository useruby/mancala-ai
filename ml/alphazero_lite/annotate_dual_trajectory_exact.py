"""Attach model-independent exact action margins to solved dual-trajectory rows."""

from __future__ import annotations

import json

from ml.alphazero_lite.arena_conflict_state_localization import (
    ARMS,
    ROOT,
    evaluator,
    probe,
)

DATA_DIR = ROOT / "docs/data/alphazero-lite-replay-source-attribution"


def main() -> int:
    reference = evaluator("F")
    for arm in ARMS:
        path = DATA_DIR / f"dual-trajectory-divergences-{arm}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            if not row["exact_root_handoff"]:
                continue
            result = probe(reference, row["state"], seed=0)
            exact = result["exact_root_decision"]
            row["exact_action_margins"] = exact["action_margins"]
            row["exact_optimal_moves"] = exact["optimal_moves"]
            f_margin = exact["action_margins"][str(row["F_move"])]
            tx_margin = exact["action_margins"][str(row["treatment_move"])]
            row["exact_comparison"] = (
                "exact equivalent"
                if f_margin == tx_margin
                else "F exact-better"
                if f_margin > tx_margin
                else f"{arm} exact-better"
            )
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
