"""Deterministic read-only synthesis of published strength-gap evidence."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/strength-gap-audit-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def paired_summary(
    rows: list[tuple[int, float]], *, seed: int, draws: int = 20000
) -> dict:
    """Opening-cluster bootstrap interval for paired opening-level contrasts."""
    grouped: dict[int, list[float]] = {}
    for opening, value in rows:
        grouped.setdefault(opening, []).append(value)
    values = [statistics.fmean(v) for v in grouped.values()]
    if not values:
        raise ValueError("no complete opening pairs")
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws)
    )
    return {
        "n_openings": len(values),
        "mean": statistics.fmean(values),
        "bootstrap_95_percentile": [
            samples[int(0.025 * draws)],
            samples[int(0.975 * draws) - 1],
        ],
        "resamples": draws,
        "seed": seed,
    }


def exact_regret(margins: dict[int, int], action: int) -> tuple[int, bool]:
    if action not in margins or not margins:
        raise ValueError("action missing from exact legal-action oracle")
    best = max(margins.values())
    score = margins[action]

    def wdl(value: int) -> int:
        return (value > 0) - (value < 0)

    return best - score, wdl(score) < wdl(best)


def validate_oracle(
    margins: dict[int, int], legal: set[int], coverage_complete: bool
) -> None:
    if not coverage_complete or set(margins) != legal or not legal:
        raise ValueError("incomplete or corrupted exact legal-action oracle")


def validate_pair_keys(left: dict, right: dict) -> None:
    if set(left) != set(right):
        raise ValueError("paired evidence root identities differ")
    if any(left[key]["opening_index"] != right[key]["opening_index"] for key in left):
        raise ValueError("paired evidence opening identity differs")


def paired_roots(
    rows: list[dict], left: str, right: str, metric: str, seed: int
) -> dict:
    groups: dict[str, dict] = {}
    for row in rows:
        groups.setdefault(row["state_hash"], {})[row["policy"]] = row
    pairs = []
    for group in groups.values():
        if left in group and right in group:
            pairs.append(
                (
                    int(group[left]["opening_index"]),
                    float(group[left][metric]) - float(group[right][metric]),
                )
            )
    return paired_summary(pairs, seed=seed)


def analyze() -> tuple[dict, dict, dict]:
    data = ROOT / "docs/data"
    paths = [
        data / "seed557-retrospective-diagnostic" / x
        for x in (
            "cohort.json",
            "search-records.jsonl",
            "oracle-records.jsonl",
            "execution-freeze.json",
            "oracle-identity-verification.json",
        )
    ]
    paths += [
        data / "seed558-value-only-swap" / x
        for x in (
            "root-metrics.jsonl",
            "analysis.json",
            "execution-freeze.json",
            "oracle-records.jsonl",
        )
    ]
    paths += [
        data / "seed512-policy-strength-evaluation" / x
        for x in (
            "qualification-evidence.json",
            "freeze.json",
            "qualification-authorization.json",
        )
    ]
    paths += [data / "seed542-training-budget-432" / "development-metrics.json"]
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}

    d557 = data / "seed557-retrospective-diagnostic"
    cohort = json.loads((d557 / "cohort.json").read_text())["states"]
    searches = jsonl(d557 / "search-records.jsonl")
    oracles = {r["state_hash"]: r for r in jsonl(d557 / "oracle-records.jsonl")}
    if len(oracles) != len(cohort) or len(searches) != 3 * len(cohort):
        raise ValueError("incomplete #557 evidence pairing/oracle coverage")
    rows = []
    for item in cohort:
        oracle = oracles.get(item["state_hash"])
        if oracle is None:
            raise ValueError("missing exact oracle root")
        margins = {int(k): int(v) for k, v in oracle["action_margins"].items()}
        legal = set(oracle["legal_moves"])
        validate_oracle(margins, legal, oracle.get("coverage_complete") is True)
        for s in (x for x in searches if x["state_hash"] == item["state_hash"]):
            prior, prior_wdl = exact_regret(margins, int(s["highest_prior_action"]))
            post, post_wdl = exact_regret(margins, int(s["puct_action"]))
            mass = sum(
                float(s["priors"][a])
                for a in legal
                if margins[a] == max(margins.values())
            )
            rows.append(
                {
                    "cohort": "557",
                    "state_hash": item["state_hash"],
                    "opening_index": item["opening_index"],
                    "active_pit_stones": item["active_pit_stones"],
                    "legal_move_count": len(legal),
                    "policy": s["policy"],
                    "prior_action": s["highest_prior_action"],
                    "puct_action": s["puct_action"],
                    "optimal_prior_mass": mass,
                    "prior_regret": prior,
                    "puct_regret": post,
                    "puct_delta_regret": post - prior,
                    "prior_wdl_downgrade": prior_wdl,
                    "puct_wdl_downgrade": post_wdl,
                }
            )
    a557 = json.loads((d557 / "analysis.json").read_text())
    # Validate that independently recomputed estimands reproduce the published aggregates.
    for pol in ("T512", "C128", "P"):
        subset = [r for r in rows if r["policy"] == pol]
        if (
            abs(
                statistics.fmean(r["puct_regret"] for r in subset)
                - a557["metrics"][pol]["mean_final_margin_regret_after_puct"]
            )
            > 1e-12
        ):
            raise ValueError("#557 aggregate reconstruction mismatch")
    cmp557 = {
        "reproduced_published": {k: a557["metrics"][k] for k in ("T512", "P")},
        "prior_regret_T512_minus_P": paired_roots(
            rows, "T512", "P", "prior_regret", 557101
        ),
        "post_puct_regret_T512_minus_P": paired_roots(
            rows, "T512", "P", "puct_regret", 557001
        ),
        "raw_optimal_prior_mass_T512_minus_P": paired_roots(
            rows, "T512", "P", "optimal_prior_mass", 557003
        ),
    }

    d558 = data / "seed558-value-only-swap"
    roots558 = jsonl(d558 / "root-metrics.jsonl")
    if len(roots558) != 256 or len({r["state_hash"] for r in roots558}) != 256:
        raise ValueError("#558 incomplete or duplicate root pairing")
    diffs = [
        (
            int(r["opening_index"]),
            r["D432_value"]["regret"] - r["initializer_value"]["regret"],
        )
        for r in roots558
    ]
    change = sum(
        r["D432_value"]["action"] != r["initializer_value"]["action"] for r in roots558
    )
    a558 = json.loads((d558 / "analysis.json").read_text())
    c558 = paired_summary(diffs, seed=558)
    if (
        change != a558["selected_action_changed_roots"]
        or abs(
            c558["mean"] - a558["primary_contrast_D432_minus_initializer_mean_regret"]
        )
        > 1e-12
    ):
        raise ValueError("#558 published aggregate mismatch")
    cmp558 = {
        "n_roots": len(roots558),
        "changed_decisions": change,
        "changed_rate": change / len(roots558),
        "regret_contrast": c558,
        "published": a558,
    }

    evidence = json.loads(
        (
            data / "seed512-policy-strength-evaluation/qualification-evidence.json"
        ).read_text()
    )
    qrows = []
    control, treatment = evidence["pairs"]["control"], evidence["pairs"]["treatment"]
    validate_pair_keys(control, treatment)
    for root in sorted(control):
        c, t = control[root], treatment[root]
        qrows.append(
            {
                "cohort": "577",
                "state_hash": root,
                "opening_index": c["opening_index"],
                "control_regret": c["regret"],
                "treatment_regret": t["regret"],
                "control_wdl_downgrade": c["wdl"] < c["oracle_wdl"],
                "treatment_wdl_downgrade": t["wdl"] < t["oracle_wdl"],
                "control_action": c["selected_action"],
                "treatment_action": t["selected_action"],
                "changed_decision": c["selected_action"] != t["selected_action"],
            }
        )
    c577 = paired_summary(
        [
            (r["opening_index"], r["treatment_regret"] - r["control_regret"])
            for r in qrows
        ],
        seed=570,
    )
    if (
        abs(c577["mean"] - evidence["gate"]["mean_treatment_minus_control_regret"])
        > 1e-12
    ):
        raise ValueError("#577 published aggregate reconstruction mismatch")
    cmp577 = {
        "n_roots": len(qrows),
        "regret_contrast": c577,
        "decision_changes": sum(r["changed_decision"] for r in qrows),
        "control_wdl_downgrades": sum(r["control_wdl_downgrade"] for r in qrows),
        "treatment_wdl_downgrades": sum(r["treatment_wdl_downgrade"] for r in qrows),
        "published_gate": evidence["gate"],
    }
    return (
        {
            "source_sha256": source_hashes,
            "cohort_557": cmp557,
            "cohort_558": cmp558,
            "cohort_577": cmp577,
        },
        {"seed557": rows, "seed558": roots558, "seed577": qrows},
        {
            "seed542_metrics": json.loads(
                (
                    data / "seed542-training-budget-432/development-metrics.json"
                ).read_text()
            )
        },
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aggregate, roots, _ = analyze()
    (OUT / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    for cohort, rows in roots.items():
        (OUT / f"{cohort}-roots.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        )


if __name__ == "__main__":
    main()
