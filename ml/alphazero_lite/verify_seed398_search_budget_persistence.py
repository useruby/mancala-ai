"""Portable, model-free verifier for the seed398 search-budget probe bundle.

Run from the repository root with:
    python -m ml.alphazero_lite.verify_seed398_search_budget_persistence
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-search-budget-persistence"
TREATMENTS = ("SS", "FS", "SF", "FF")
CHECKPOINTS = (384, 768, 1536)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify() -> dict[str, Any]:
    registration_path = DATA / "registration.json"
    raw_path = DATA / "raw-probes.json"
    registration, raw = read(registration_path), read(raw_path)
    require(
        registration["schema"] == "seed398-search-budget-persistence-registration-v1",
        "registration_schema_mismatch",
    )
    require(
        raw["schema"] == "seed398-search-budget-persistence-probes-v1",
        "raw_schema_mismatch",
    )
    require(
        raw["registration_sha256"] == digest(registration_path),
        "raw_registration_hash_mismatch",
    )
    require(
        registration["ledger_sha256"]
        == digest(
            ROOT
            / "docs/data/seed398-policy-value-composition/validated-outcome-ledger.jsonl"
        ),
        "ledger_hash_mismatch",
    )
    require(
        registration["suite_sha256"]
        == digest(
            ROOT
            / "docs/data/seed398-policy-value-composition/seed398-openings-v2.jsonl"
        ),
        "suite_hash_mismatch",
    )
    selected = registration["states"]
    require(
        len(selected) == 64 and len({row["opening_index"] for row in selected}) == 64,
        "selection_count_or_opening_uniqueness_mismatch",
    )
    hashes = [row["state_hash"] for row in selected]
    require(len(set(hashes)) == 64, "selected_state_hash_not_unique")
    require(
        all(
            row["ss_action_384"] in row["legal_actions"]
            and row["ff_action_384"] in row["legal_actions"]
            for row in selected
        ),
        "selected_action_illegal",
    )

    def ordering_key(row: dict[str, Any]) -> tuple[str, int]:
        text = (
            f"seed398-budget-persistence-v1:{row['opening_index']}:{row['state_hash']}"
        )
        return hashlib.sha256(text.encode()).hexdigest(), row["opening_index"]

    require(
        selected == sorted(selected, key=ordering_key), "selection_hash_order_mismatch"
    )
    grouped = {(row["opening_index"], row["treatment"]): row for row in raw["probes"]}
    require(
        len(raw["probes"]) == 256 and len(grouped) == 256, "probe_accounting_mismatch"
    )
    state_results = []
    for selection in selected:
        opening = selection["opening_index"]
        probes = {name: grouped[(opening, name)] for name in TREATMENTS}
        for treatment, probe in probes.items():
            snapshots = probe["snapshots"]
            require(
                [row["simulations"] for row in snapshots] == list(CHECKPOINTS),
                f"snapshot_accounting_mismatch:{opening}:{treatment}",
            )
            for snap in snapshots:
                legal = set(selection["legal_actions"])
                require(
                    set(map(int, snap["visits"])) == legal,
                    f"visit_actions_mismatch:{opening}:{treatment}",
                )
                require(
                    set(map(int, snap["q_values"])) == legal
                    and set(map(int, snap["priors"])) == legal,
                    f"telemetry_actions_mismatch:{opening}:{treatment}",
                )
                require(
                    sum(snap["visits"].values()) == snap["simulations"],
                    f"visit_total_mismatch:{opening}:{treatment}:{snap['simulations']}",
                )
                ordered = sorted(snap["visits"].values(), reverse=True)
                expected_gap = (
                    ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
                ) / sum(ordered)
                require(
                    abs(expected_gap - snap["normalized_top_two_visit_gap"]) < 1e-12,
                    f"gap_mismatch:{opening}:{treatment}",
                )
            require(
                probe["final_action"] == snapshots[-1]["selected_action"],
                f"final_action_mismatch:{opening}:{treatment}",
            )
        ss, ff = probes["SS"], probes["FF"]
        state_results.append(
            {
                "same_1536": ss["final_action"] == ff["final_action"],
                "ss_retains": all(
                    next(s for s in ss["snapshots"] if s["simulations"] == budget)[
                        "selected_action"
                    ]
                    == selection["ss_action_384"]
                    for budget in (768, 1536)
                ),
                "ff_retains": all(
                    next(s for s in ff["snapshots"] if s["simulations"] == budget)[
                        "selected_action"
                    ]
                    == selection["ff_action_384"]
                    for budget in (768, 1536)
                ),
                "ss_gap": ss["snapshots"][-1]["normalized_top_two_visit_gap"],
                "ff_gap": ff["snapshots"][-1]["normalized_top_two_visit_gap"],
            }
        )
    same = sum(row["same_1536"] for row in state_results)
    persistent = sum(
        row["ss_retains"]
        and row["ff_retains"]
        and row["ss_gap"] >= 0.10
        and row["ff_gap"] >= 0.10
        for row in state_results
    )
    classification = (
        "budget_sensitive_disagreement"
        if same >= 32
        else "persistent_model_disagreement"
        if persistent >= 48
        else "mixed_or_unresolved"
    )
    summary = read(DATA / "summary.json")
    require(
        summary["classification"] == classification, "summary_classification_mismatch"
    )
    require(
        summary["ss_ff_same_at_1536"] == same
        and summary["persistent_with_margin"] == persistent,
        "summary_count_mismatch",
    )
    return {
        "status": "verified",
        "classification": classification,
        "sampled_states": 64,
        "ss_ff_same_at_1536": same,
        "persistent_with_margin": persistent,
        "registration_sha256": digest(registration_path),
        "raw_probes_sha256": digest(raw_path),
        "summary_sha256": digest(DATA / "summary.json"),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
