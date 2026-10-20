"""Independent NumPy recomputation from the committed seed508 outcome ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    publication = args.root / "docs/data/seed508-policy-value-crossover"
    rows = [
        json.loads(line)
        for line in (publication / "outcomes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    score = {"challenger": 1.0, "draw": 0.5, "current": 0.0}
    per_arm: dict[str, dict[int, list[dict]]] = {"P": {}, "V": {}}
    for row in rows:
        per_arm[row["arm"]].setdefault(int(row["opening_index"]), []).append(row)
    p_opening = np.array(
        [np.mean([score[r["winner"]] for r in per_arm["P"][i]]) for i in range(512)]
    )
    v_opening = np.array(
        [np.mean([score[r["winner"]] for r in per_arm["V"][i]]) for i in range(512)]
    )
    differences = p_opening - v_opening
    bootstrap = differences[
        np.random.default_rng(508).integers(0, 512, size=(20_000, 512))
    ].mean(axis=1)
    result = {
        "method": "independent recomputation from outcomes.jsonl; opening-cluster percentile bootstrap",
        "games": len(rows),
        "opening_differences": len(differences),
        "mean_P_minus_V": float(differences.mean()),
        "bootstrap_resamples": 20_000,
        "seed": 508,
        "two_sided_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_interval_decision": "P supported"
        if np.quantile(bootstrap, 0.025) > 0
        else "V supported"
        if np.quantile(bootstrap, 0.975) < 0
        else "inconclusive",
    }
    (publication / "independent-analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
