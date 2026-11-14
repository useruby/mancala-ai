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
TRUSTED_INPUT_SHA256 = {
    "docs/data/seed557-retrospective-diagnostic/cohort.json": "f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243",
    "docs/data/seed557-retrospective-diagnostic/search-records.jsonl": "6b336dbd73f0ca90620f91737f9fd9a8600e22cf178efb62fd63ec04e5e916e8",
    "docs/data/seed557-retrospective-diagnostic/oracle-records.jsonl": "75453e744a86b9dc8471628e0d133fa7d478550af9c8089ed8661426436a8480",
    "docs/data/seed557-retrospective-diagnostic/execution-freeze.json": "b0892a24872012486ff164d291913e1799737632a816a40c2547210a458bf36d",
    "docs/data/seed557-retrospective-diagnostic/oracle-identity-verification.json": "45f93ce95bf2bc0c58b8a44bdf072324eddfb2c943dd9678e6895613a5a4f60d",
    "docs/data/seed558-value-only-swap/root-metrics.jsonl": "11b91a7b06587ed5cc9a8054ba60abe54730e0afa33f7fd39c90e583178f8ed0",
    "docs/data/seed558-value-only-swap/analysis.json": "54c84ef477c379f0d002bbbf74d211ce715878d5f7954bb80bc53eb78ecc7f73",
    "docs/data/seed558-value-only-swap/execution-freeze.json": "4ab48b0e651ac86442dcbdeb4e2d75be0436750318197c8e5811a00e5b540dcc",
    "docs/data/seed558-value-only-swap/oracle-records.jsonl": "75453e744a86b9dc8471628e0d133fa7d478550af9c8089ed8661426436a8480",
    "docs/data/seed512-policy-strength-evaluation/qualification-evidence.json": "c7c71adb38b65f9372668d589a1a15fd36acd86f7ccaa26b82b6486f732472bc",
    "docs/data/seed512-policy-strength-evaluation/freeze.json": "3631361ba748511d2f4b532df808982da9c0559b0a71bf5b8627d68916f56096",
    "docs/data/seed512-policy-strength-evaluation/qualification-authorization.json": "cbd127dee72d0500ad5010dd8faeaa9f5a5dc7586c706e78a8af3c8e02314559",
    "docs/data/seed570-policy-prior-distillation/qualification-membership.json": "6447d6ae98a750de4046992fd84e53998843dd1f2f20706375de4c91fa798efc",
    "ml/alphazero_lite/seed570_qualification_cohort.py": "4a37b4ea4d89bc7fe00469d712c3ed4fa2cfe1762f54b8d98ff8edafeaf75a17",
    "docs/data/seed542-training-budget-432/development-metrics.json": "e8d9af03ab3eacb3e4641b33e4bd0de5ba97ea67d7f34f580278009819151dfb",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def paired_summary(
    rows: list[tuple[int, float]],
    *,
    seed: int,
    draws: int = 20000,
    quantile: str = "order_statistic",
) -> dict:
    """Opening-cluster bootstrap; legacy order statistic or NumPy linear quantiles."""
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
    if quantile == "order_statistic":
        interval = [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]]
    elif quantile == "linear":

        def linear(q: float) -> float:
            index = (len(samples) - 1) * q
            lower = int(index)
            fraction = index - lower
            upper = min(lower + 1, len(samples) - 1)
            return samples[lower] * (1.0 - fraction) + samples[upper] * fraction

        interval = [linear(0.025), linear(0.975)]
    else:
        raise ValueError(f"unknown bootstrap quantile convention:{quantile}")
    return {
        "n_openings": len(values),
        "mean": statistics.fmean(values),
        "bootstrap_95_percentile": interval,
        "quantile_convention": quantile,
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


def validate_trusted_digest(path: Path, expected: str) -> str:
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"trusted input hash mismatch:{path}")
    return actual


def paired_roots(
    rows: list[dict],
    left: str,
    right: str,
    metric: str,
    seed: int,
    expected_roots: set[str] | None = None,
) -> dict:
    groups: dict[str, dict] = {}
    for row in rows:
        if row["policy"] not in {left, right}:
            continue
        group = groups.setdefault(row["state_hash"], {})
        if row["policy"] in group:
            raise ValueError(
                f"duplicate policy record for root:{row['state_hash']}:{row['policy']}"
            )
        group[row["policy"]] = row
    if any(set(group) != {left, right} for group in groups.values()):
        raise ValueError(f"incomplete paired root policy records:{left}:{right}")
    if expected_roots is not None and set(groups) != expected_roots:
        raise ValueError(
            f"paired root identities differ from expected cohort:{left}:{right}"
        )
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


def verify_trusted_inputs() -> dict[str, str]:
    hashes = {}
    for relative, expected in TRUSTED_INPUT_SHA256.items():
        path = ROOT / relative
        actual = validate_trusted_digest(path, expected)
        hashes[relative] = actual
    # Authenticate verifier source before importing and executing it.
    from ml.alphazero_lite.seed570_qualification_cohort import (
        SOURCE_SHA256 as seed570_source_sha256,
        verify as verify_seed570_membership,
    )

    for relative, expected in seed570_source_sha256.items():
        actual = digest(ROOT / relative)
        if actual != expected:
            raise ValueError(f"#570 authenticated source mismatch:{relative}")
    membership = json.loads(
        (
            ROOT
            / "docs/data/seed570-policy-prior-distillation/qualification-membership.json"
        ).read_text()
    )
    verify_seed570_membership(membership)
    hashes.update(
        {relative: digest(ROOT / relative) for relative in seed570_source_sha256}
    )
    return hashes


def _subgroups(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {
        key: {
            "n": len(group),
            "decision_changes": sum(row["changed_decision"] for row in group),
            "change_rate": statistics.fmean(row["changed_decision"] for row in group),
            "treatment_minus_control_regret": statistics.fmean(
                row["treatment_regret"] - row["control_regret"] for row in group
            ),
            "treatment_minus_control_optimal_prior_mass": statistics.fmean(
                row["treatment_optimal_prior_mass"] - row["control_optimal_prior_mass"]
                for row in group
            ),
        }
        for key, group in sorted(groups.items(), key=lambda item: int(item[0]))
    }


def _value_subgroups(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for field in (
        "active_pit_stones",
        "legal_move_count",
        "best_second_best_margin_gap",
    ):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row[field]), []).append(row)
        for key, group in sorted(groups.items(), key=lambda item: int(item[0])):
            result[f"{field}={key}"] = {
                "n": len(group),
                "decision_changes": sum(
                    row["D432_value"]["action"] != row["initializer_value"]["action"]
                    for row in group
                ),
                "change_rate": statistics.fmean(
                    row["D432_value"]["action"] != row["initializer_value"]["action"]
                    for row in group
                ),
                "D432_minus_initializer_mean_regret": statistics.fmean(
                    row["D432_value"]["regret"] - row["initializer_value"]["regret"]
                    for row in group
                ),
            }
    return result


def analyze() -> tuple[dict, dict, dict]:
    trusted_hashes = verify_trusted_inputs()
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
    expected_keys = {
        (item["state_hash"], policy)
        for item in cohort
        for policy in ("T512", "C128", "P")
    }
    seen_keys = [(row["state_hash"], row["policy"]) for row in searches]
    if len(set(seen_keys)) != len(seen_keys) or set(seen_keys) != expected_keys:
        raise ValueError("#557 missing, duplicate, or unexpected root/policy record")
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
            rows,
            "T512",
            "P",
            "prior_regret",
            557101,
            expected_roots={item["state_hash"] for item in cohort},
        ),
        "post_puct_regret_T512_minus_P": paired_roots(
            rows,
            "T512",
            "P",
            "puct_regret",
            557001,
            expected_roots={item["state_hash"] for item in cohort},
        ),
        "raw_optimal_prior_mass_T512_minus_P": paired_roots(
            rows,
            "T512",
            "P",
            "optimal_prior_mass",
            557003,
            expected_roots={item["state_hash"] for item in cohort},
        ),
    }

    d558 = data / "seed558-value-only-swap"
    roots558 = jsonl(d558 / "root-metrics.jsonl")
    if len(roots558) != 256 or len({r["state_hash"] for r in roots558}) != 256:
        raise ValueError("#558 incomplete or duplicate root pairing")
    descriptors = {item["state_hash"]: item for item in cohort}
    margins557 = {
        row["state_hash"]: row for row in jsonl(d557 / "oracle-records.jsonl")
    }
    if any(row["state_hash"] not in descriptors for row in roots558):
        raise ValueError("#558 root does not join authenticated #557 descriptors")
    if len({int(row["opening_index"]) for row in roots558}) != len(roots558):
        raise ValueError("#558 opening-cluster pairing is not unique")
    if any(
        int(row["opening_index"])
        != int(descriptors[row["state_hash"]]["opening_index"])
        for row in roots558
    ):
        raise ValueError("#558 opening identity does not match #557 descriptor")
    if any(
        set(row) != {"state_hash", "opening_index", "initializer_value", "D432_value"}
        for row in roots558
    ):
        raise ValueError("#558 root arm pairing is incomplete or unexpected")
    if any(
        set(row["initializer_value"])
        != {"action", "optimal", "regret", "wdl_downgrade"}
        or set(row["D432_value"]) != {"action", "optimal", "regret", "wdl_downgrade"}
        for row in roots558
    ):
        raise ValueError("#558 baseline/treatment root record is incomplete")
    joined558 = []
    for row in roots558:
        oracle_margins = {
            int(action): int(value)
            for action, value in margins557[row["state_hash"]]["action_margins"].items()
        }
        scores = sorted(oracle_margins.values(), reverse=True)
        joined558.append(
            {
                **row,
                "active_pit_stones": descriptors[row["state_hash"]][
                    "active_pit_stones"
                ],
                "legal_move_count": len(
                    descriptors[row["state_hash"]]["legal_absolute_actions"]
                ),
                "best_second_best_margin_gap": scores[0] - scores[1],
            }
        )
    roots558 = joined558
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
        "by_strategic_descriptor": _value_subgroups(roots558),
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
    membership = json.loads(
        (
            data / "seed570-policy-prior-distillation/qualification-membership.json"
        ).read_text()
    )
    members = {row["state_hash"]: row for row in membership["roots"]}
    if set(members) != set(control) or len(members) != 423:
        raise ValueError("#577 root set does not match authenticated #570 membership")
    oracle_rows = jsonl(
        data / "seed559-fixed-position-feasibility/run-0003/oracle-records.jsonl"
    )
    oracle_by_hash = {row["state_hash"]: row for row in oracle_rows}
    if len(oracle_rows) != 512 or len(oracle_by_hash) != 512:
        raise ValueError("#559 oracle root count or identity mismatch")
    oracle_cohort = json.loads(
        (
            data / "seed559-fixed-position-feasibility/run-0003/cohort-manifest.json"
        ).read_text()
    )["states"]
    oracle_descriptors = {row["state_hash"]: row for row in oracle_cohort}
    if set(oracle_by_hash) != set(oracle_descriptors):
        raise ValueError(
            "#559 exact-action oracle and state manifest membership differ"
        )
    for root in sorted(control):
        c, t = control[root], treatment[root]
        member = members[root]
        oracle = oracle_by_hash.get(root)
        if oracle is None:
            raise ValueError(
                f"#577 qualification root lacks authenticated #559 oracle:{root}"
            )
        margins = {int(k): int(v) for k, v in oracle["action_margins"].items()}
        legal = set(int(action) for action in member["legal_moves"])
        validate_oracle(margins, legal, oracle.get("coverage_complete") is True)
        if int(member["opening_index"]) != int(c["opening_index"]):
            raise ValueError("#577/#570 opening identity mismatch")
        if state_hash(member["state"]) != root:
            raise ValueError("#577 membership state hash mismatch")
        source_descriptor = oracle_descriptors[root]
        if source_descriptor["state"] != member["state"]:
            raise ValueError("#577/#559 source state mismatch")
        if legal != set(int(action) for action in oracle["legal_moves"]):
            raise ValueError("#577/#559 legal action identity mismatch")
        ordered_scores = sorted(margins.values(), reverse=True)
        top = ordered_scores[0]
        runner_up = ordered_scores[1] if len(ordered_scores) > 1 else top
        cw, tw = c["search_witness"], t["search_witness"]
        for witness, row in ((cw, c), (tw, t)):
            if set(int(action) for action in witness["legal_moves"]) != legal:
                raise ValueError("#577 search witness legal moves mismatch")
            if int(witness["selected_action"]) != int(row["selected_action"]):
                raise ValueError("#577 search witness selected action mismatch")
        cp, tp = (
            [float(value) for value in cw["network_priors"]],
            [float(value) for value in tw["network_priors"]],
        )
        prior_c = max(legal, key=lambda action: (cp[action], -action))
        prior_t = max(legal, key=lambda action: (tp[action], -action))
        prior_c_regret, prior_c_wdl = exact_regret(margins, prior_c)
        prior_t_regret, prior_t_wdl = exact_regret(margins, prior_t)
        control_selected_regret, _ = exact_regret(margins, int(c["selected_action"]))
        treatment_selected_regret, _ = exact_regret(margins, int(t["selected_action"]))
        if (
            int(c["regret"]) != control_selected_regret
            or int(t["regret"]) != treatment_selected_regret
        ):
            raise ValueError("#577 regret disagrees with authenticated #559 oracle")
        qrows.append(
            {
                "cohort": "577",
                "state_hash": root,
                "opening_index": c["opening_index"],
                "active_pit_stones": sum(member["state"]["player_pits"])
                + sum(member["state"]["opponent_pits"]),
                "legal_move_count": len(legal),
                "best_second_best_margin_gap": top - runner_up,
                "control_prior_action": prior_c,
                "treatment_prior_action": prior_t,
                "control_prior_regret": prior_c_regret,
                "treatment_prior_regret": prior_t_regret,
                "control_optimal_prior_mass": sum(
                    cp[action] for action in legal if margins[action] == top
                ),
                "treatment_optimal_prior_mass": sum(
                    tp[action] for action in legal if margins[action] == top
                ),
                "control_prior_wdl_downgrade": prior_c_wdl,
                "treatment_prior_wdl_downgrade": prior_t_wdl,
                "control_puct_delta_regret": control_selected_regret - prior_c_regret,
                "treatment_puct_delta_regret": treatment_selected_regret
                - prior_t_regret,
                "control_regret": c["regret"],
                "treatment_regret": t["regret"],
                "control_wdl_downgrade": c["wdl"] < c["oracle_wdl"],
                "treatment_wdl_downgrade": t["wdl"] < t["oracle_wdl"],
                "control_action": c["selected_action"],
                "treatment_action": t["selected_action"],
                "changed_decision": c["selected_action"] != t["selected_action"],
                "changed_decision_better": treatment_selected_regret
                < control_selected_regret,
                "changed_decision_worse": treatment_selected_regret
                > control_selected_regret,
            }
        )
    c577 = paired_summary(
        [
            (r["opening_index"], r["treatment_regret"] - r["control_regret"])
            for r in qrows
        ],
        seed=570,
        quantile="linear",
    )
    if (
        abs(c577["mean"] - evidence["gate"]["mean_treatment_minus_control_regret"])
        > 1e-12
    ):
        raise ValueError("#577 published aggregate reconstruction mismatch")
    prior_regret_contrast = paired_summary(
        [
            (
                row["opening_index"],
                row["treatment_prior_regret"] - row["control_prior_regret"],
            )
            for row in qrows
        ],
        seed=570101,
        quantile="linear",
    )
    prior_mass_contrast = paired_summary(
        [
            (
                row["opening_index"],
                row["treatment_optimal_prior_mass"] - row["control_optimal_prior_mass"],
            )
            for row in qrows
        ],
        seed=570102,
        quantile="linear",
    )
    if (
        c577["bootstrap_95_percentile"] != evidence["gate"]["opening_bootstrap_95"]
        or sum(row["control_wdl_downgrade"] for row in qrows)
        != round(evidence["gate"]["control_wdl_downgrade_rate"] * len(qrows))
        or sum(row["treatment_wdl_downgrade"] for row in qrows)
        != round(evidence["gate"]["treatment_wdl_downgrade_rate"] * len(qrows))
    ):
        raise ValueError("#577 complete published gate reconstruction mismatch")
    cmp577 = {
        "n_roots": len(qrows),
        "regret_contrast": c577,
        "raw_prior_regret_treatment_minus_control": prior_regret_contrast,
        "optimal_prior_mass_treatment_minus_control": prior_mass_contrast,
        "decision_changes": sum(r["changed_decision"] for r in qrows),
        "puct_improved_prior": {
            "control": sum(r["control_puct_delta_regret"] < 0 for r in qrows),
            "treatment": sum(r["treatment_puct_delta_regret"] < 0 for r in qrows),
        },
        "puct_worsened_prior": {
            "control": sum(r["control_puct_delta_regret"] > 0 for r in qrows),
            "treatment": sum(r["treatment_puct_delta_regret"] > 0 for r in qrows),
        },
        "control_wdl_downgrades": sum(r["control_wdl_downgrade"] for r in qrows),
        "treatment_wdl_downgrades": sum(r["treatment_wdl_downgrade"] for r in qrows),
        "changed_decision_outcomes": {
            "better_exact_regret": sum(r["changed_decision_better"] for r in qrows),
            "worse_exact_regret": sum(r["changed_decision_worse"] for r in qrows),
            "equal_exact_regret": sum(
                r["changed_decision"]
                and not r["changed_decision_better"]
                and not r["changed_decision_worse"]
                for r in qrows
            ),
        },
        "by_active_pit_stones": _subgroups(qrows, "active_pit_stones"),
        "by_legal_move_count": _subgroups(qrows, "legal_move_count"),
        "by_best_second_best_margin_gap": _subgroups(
            qrows, "best_second_best_margin_gap"
        ),
        "published_gate": evidence["gate"],
    }
    source_hashes.update(trusted_hashes)
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
