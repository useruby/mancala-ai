#!/usr/bin/env python3
"""Validate the committed, dependency-free A16 lineage closeout."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = "docs/data/alphazero-lite-a16-lineage-closeout.json"
FROZEN_HASHES = {
    "head_sha": "91823644a205af601234287e66773d6fae0bb25e",
    "merge_commit": "d09d6b03101e07a3723835ea3ce26bdf89a70d26",
    "a16_artifact_hash": "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34",
    "a16_weights_hash": "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789",
    "candidate_aggregate_hash": "60ad77daddb1fa01645068867892ec5525381b0b304d85235a989631c3c340ce",
    "suite_manifest_hash": "076b8d031b01ffe5dc710ab2ae271c2d018a9e8174ee1ea95989aeff40fb85f0",
}
CLASSIFICATIONS = {
    "PR #264": "joint_full_network_does_not_beat_incumbent",
    "PR #265/#266": "scale_degrades_strength",
    "generation-2-selfplay-anchor": "second_iteration_regression",
    "control-ep2-multi-iteration": "control_ep2_still_best+extra_data_not_helpful+destructive_iteration",
    "promoted-current-puct-iter2-smoke": "iter2_mixed_signal",
    "residual-v4-classic-mcts-strength": "architecture_only_not_strength",
    "exact-tablebase-direct-patch": "exact_tablebase_control_regression_intrinsic",
    "tablebase-value-overlay-1600": "tablebase_value_overlay_not_competitive_as_single_phase_bootstrap",
}
EXPECTED_SUITES = {
    "canonical": (
        None,
        "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04",
    ),
    "AK": (37042, "c7df7293b641cac5b28a18424ab6489707dc1be15e65796392fea19258d6c57c"),
    "AL": (38042, "15e450006cf9299831d57147db2fe4ab9e9ec5182d8eff5cc921d17ad7c6267d"),
    "AM": (39042, "228bb3bda7be11675d8845fbb55b6e88badbde9b4c2f6b04a9e83d5df69b0bc2"),
    "AN": (40042, "5d31671aa6b5b86beb066b74848ce9fffcf05630a7d735ac845ae0f380dd0f0f"),
    "AO": (41042, "dec338722f5a8b5442cb6163e19dfa39ee1dae9c9d94b3a1088eeb2f27c6fc00"),
    "AP": (42042, "a91361b16ffb885080bc6d2e8ff7d8be35aa8213abcda28542dc0e63453bfba6"),
}


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def registry_suites(repo_root: Path) -> dict[str, tuple[int | None, str]]:
    """Extract sealed suite identities without importing training dependencies."""
    source = (repo_root / "ml/alphazero_lite/consumed_suite_registry.py").read_text()
    numeric_matches = re.findall(
        r'ConsumedSuite\(\s*"([A-Z]+)",.*?\s+(\d+),\s*"([0-9a-f]{64})",?',
        source,
        flags=re.DOTALL,
    )
    canonical_match = re.search(
        r'ConsumedSuite\(\s*"canonical",.*?\s+None,\s*"([0-9a-f]{64})",?',
        source,
        flags=re.DOTALL,
    )
    result: dict[str, tuple[int | None, str]] = {
        label: (int(seed), sha256) for label, seed, sha256 in numeric_matches
    }
    if canonical_match:
        result["canonical"] = (None, canonical_match.group(1))
    return result


def validate(repo_root: Path = REPO_ROOT, ledger_path: Path | None = None) -> None:
    """Reject changes that reopen A16 or disconnect the ledger from its evidence."""
    ledger_file = ledger_path or repo_root / LEDGER_PATH
    with ledger_file.open() as handle:
        ledger: dict[str, Any] = json.load(handle)
    if ledger.get("final_classification") != "a16_lineage_closed":
        fail("final classification altered")
    if ledger.get("frozen_pr266") != FROZEN_HASHES:
        fail("frozen PR #266 hashes mismatch")

    experiments = ledger.get("experiments", [])
    identities = [experiment.get("identity") for experiment in experiments]
    if len(identities) != len(set(identities)):
        fail("duplicate experiment identity")
    if set(identities) != set(CLASSIFICATIONS):
        fail("experiment identities altered")
    for experiment in experiments:
        identity = experiment["identity"]
        source = repo_root / experiment["source_document"]
        if not source.is_file():
            fail(f"missing source document: {experiment['source_document']}")
        if experiment.get("published_classification") != CLASSIFICATIONS[identity]:
            fail(f"classification altered: {identity}")
        if not all(
            classification in source.read_text()
            for classification in CLASSIFICATIONS[identity].split("+")
        ):
            fail(f"source classification disagreement: {identity}")
        if experiment.get("a16_descendant") and experiment.get("lifecycle") in {
            "proposed",
            "promoted",
        }:
            fail(f"A16 descendant is {experiment['lifecycle']}: {identity}")

    exhausted = set(ledger.get("exhausted_branches", []))
    eligibility = ledger.get("branch_eligibility", {})
    if exhausted != set(eligibility):
        fail("exhausted branch set mismatch")
    if any(eligibility.values()):
        fail("exhausted branch marked eligible for continuation")

    registered = registry_suites(repo_root)
    if {label: registered.get(label) for label in EXPECTED_SUITES} != EXPECTED_SUITES:
        fail("consumed-suite registry disagreement")
    for experiment in experiments:
        suites = experiment["consumed_suites"]
        if suites["status"] == "consumed":
            labels = suites["labels"]
            if any(label not in registered for label in labels):
                fail(f"consumed-suite registry disagreement: {experiment['identity']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    validate(ledger_path=args.ledger)
    print("a16_lineage_closed")


if __name__ == "__main__":
    main()
