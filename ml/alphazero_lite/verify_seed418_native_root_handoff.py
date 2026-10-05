"""Read-only verifier for published seed418 diagnostic evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed418_native_root_handoff import (
    ROOT,
    digest,
    replay_cohort,
)
from ml.alphazero_lite.seed418_analysis import analyze, choose_exact_action
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_seed416_policy_target_softening import (
    verify as verify_seed416,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_bound_files(data: Path, binding: dict[str, str]) -> None:
    for name, expected_hash in binding.items():
        if digest(data / name) != expected_hash:
            raise ValueError(f"published_evidence_hash_mismatch:{name}")


def verify(root: Path = ROOT) -> dict[str, Any]:
    data = root / "docs/data/seed418-native-root-handoff"
    prereg = json.loads((data / "preregistration.json").read_text())
    manifest_path = data / "cohort-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = root / "docs/data/seed416-policy-target-softening"
    suite_path, ledger_path = (
        source / "openings-v3.jsonl",
        source / "outcome-ledger.jsonl",
    )
    if (
        digest(suite_path) != prereg["source_suite_sha256"]
        or digest(ledger_path) != prereg["source_ledger_sha256"]
    ):
        raise ValueError("source_evidence_hash_mismatch")
    for relative, expected in prereg["sources"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"execution_source_hash_mismatch:{relative}")
    for key, relative in (
        ("native_probe_sha256", "model-artifact/runtime/kalah_v1_tablebase"),
        ("tablebase_sha256", "model-artifact/runtime/kalah_v1_21.kvtb"),
        (
            "tablebase_generation_metadata_sha256",
            "docs/data/alphazero-lite-kvtb21-generation.json",
        ),
    ):
        if digest(root / relative) != prereg["artifacts"][key]:
            raise ValueError(f"frozen_artifact_hash_mismatch:{key}")
    if digest(manifest_path) != prereg["manifest_sha256"]:
        raise ValueError("cohort_manifest_hash_mismatch")
    tablebase_metadata = json.loads(
        (root / "docs/data/alphazero-lite-kvtb21-generation.json").read_text()
    )
    if (
        tablebase_metadata.get("header", {}).get("magic") != "KVTB1"
        or tablebase_metadata.get("header", {}).get("tier") != 21
        or tablebase_metadata.get("header", {}).get("declared_max_tier") != 21
        or prereg["native"].get("tablebase_declared_max_tier") != 21
    ):
        raise ValueError("native_tablebase_domain_identity_invalid")
    native_source = (
        root / "native/kalah_v1_tablebase/kalah_v1_tablebase.cc"
    ).read_text()
    compact_native_source = re.sub(r"\s+", "", native_source)
    if (
        "constexpruint64_tkSupportedMaxTier=21;" not in compact_native_source
        or "tier>kSupportedMaxTier" not in compact_native_source
        or "parsed.tier>kSupportedMaxTier" not in compact_native_source
    ):
        raise ValueError("native_tablebase_implementation_domain_invalid")
    public_seed416 = verify_seed416(root)
    if not public_seed416["valid"]:
        raise ValueError("seed416_public_validation_failed")
    binding416 = json.loads((source / "evaluation-binding.json").read_text())
    if (
        prereg["artifacts"]["runtime_policy_sha256"]
        != binding416["runtime_policy_sha256"]
    ):
        raise ValueError("seed416_runtime_policy_identity_mismatch")
    suite = read_jsonl(suite_path)
    source_rows = read_jsonl(ledger_path)
    expected = replay_cohort(suite, source_rows)
    if manifest["states"] != expected:
        raise ValueError("cohort_replay_mismatch")
    search = read_jsonl(data / "search-records.jsonl")
    oracle = read_jsonl(data / "oracle-records.jsonl")
    if len(search) != 128 or len(oracle) != 128:
        raise ValueError("case_count_invalid")
    rows = []
    for item, srow, orow in zip(manifest["states"], search, oracle, strict=True):
        if (
            item["state_hash"] != srow["state_hash"]
            or item["state_hash"] != orow["state_hash"]
        ):
            raise ValueError("case_state_binding_mismatch")
        legal = sorted(int(move) for move in orow["legal_moves"])
        margins = {
            int(move): int(value) for move, value in orow["action_margins"].items()
        }
        if legal != sorted(margins) or orow.get("coverage_complete") is not True:
            raise ValueError("exact_legal_action_coverage_invalid")
        game = KalahGame.from_state(item["state"])
        if (
            game.over()
            or sum(game.pits) != int(item["active_pit_stones"])
            or sorted(game.possible_moves()) != legal
            or int(orow["root_player"]) != game.current_player
            or int(srow["search_selected_move"]) not in legal
        ):
            raise ValueError("case_state_or_action_invalid")
        seed, seed_context_hash = derive_search_seed(**srow["seed_context"])
        if (
            int(srow["seed"]) != seed
            or srow["seed_context_hash"] != seed_context_hash
            or srow["seed_context"]["opening_state_hash"]
            != item["source_opening_state_hash"]
            or srow["seed_context"]["canonical_current_state_hash"]
            != item["state_hash"]
        ):
            raise ValueError("search_seed_context_invalid")
        if int(orow["selected_move"]) != choose_exact_action(
            margins, [float(value) for value in orow["network_priors"]], legal
        ):
            raise ValueError("exact_action_tie_rule_invalid")
        rows.append(
            {
                **item,
                **srow,
                "legal_moves": legal,
                "action_margins": orow["action_margins"],
                "native_decision_latency_ms": orow["native_decision_latency_ms"],
                "coverage_complete": True,
            }
        )
    analysis = analyze(rows)
    published = json.loads((data / "analysis.json").read_text())
    if analysis != published:
        raise ValueError("analysis_recomputation_mismatch")
    latency = json.loads((data / "latency-accounting.json").read_text())
    if (
        latency["warm_native_decision_mean_ms"]
        != analysis["native_decision_latency_ms"]["mean"]
        or latency["warm_native_decision_p95_ms"]
        != analysis["native_decision_latency_ms"]["p95_nearest_rank"]
    ):
        raise ValueError("latency_accounting_mismatch")
    binding = json.loads((data / "evidence-binding.json").read_text())
    verify_bound_files(data, binding)
    return {
        "valid": True,
        "cases": 128,
        "decision": analysis["decision"],
        "mean_margin_regret": analysis["mean_final_margin_regret"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
