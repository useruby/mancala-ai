"""Publish and hash-bind the completed seed-414 diagnostic evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed414_root_budget_analysis as analysis
from ml.alphazero_lite import seed414_root_budget_confirmation as runner


def publish() -> dict[str, Any]:
    reg = runner.read_json(runner.REGISTRATION)
    probes = runner._load_jsonl(runner.PROBES)
    trajectories = runner._load_jsonl(runner.TRAJECTORIES)
    aliases = runner._load_jsonl(runner.ALIASES)
    report, matrix = analysis.calculate(reg, probes, trajectories, aliases)
    analysis_path = runner.OUT / "analysis.json"
    matrix_path = runner.OUT / "paired-matrix.json"
    results_path = runner.OUT / "results.md"
    _write_json(analysis_path, report)
    _write_json(matrix_path, matrix)
    primary = report["primary"]
    results = (
        "# Seed 414 FF384 vs FF1536 first-action confirmation\n\n"
        "Fresh, preregistered 128-state diagnostic of first-action quality under two frozen continuation references. It is not an overall-strength or promotion result. No historical outcomes were reused.\n\n"
        f"- Decision: **{report['decision']}**.\n"
        f"- Mean paired gain: **{primary['mean_gain']:.6f}**.\n"
        f"- Paired bootstrap 95% interval (10,000 state-level resamples, seed 414): **[{primary['bootstrap_95_interval'][0]:.6f}, {primary['bootstrap_95_interval'][1]:.6f}]**.\n"
        f"- Reference gains: seed455 **{report['reference_mean_gains']['seed455']:.6f}**; original O0 E4 **{report['reference_mean_gains']['original_O0_E4']:.6f}**.\n"
        f"- First-action change rate: **{report['action_change_rate']:.3%}** ({report['action_changes']}/128).\n"
        f"- Mean root-search time: **{report['search_timing']['root_mean_seconds']:.3f}s**; physical continuation count: **{report['search_timing']['physical_continuations']}**.\n"
        f"- Mean store-margin changes: {json.dumps(report['reference_mean_margin_changes'], sort_keys=True)}.\n\n"
        "Evidence files include the registration, source snapshots, extended exclusion proof, 128 E4 root probes, complete physical trajectories, 512 logical aliases/cases, and paired matrix.\n\n"
        "Verify without loading model artifacts or invoking searches:\n\n"
        "```sh\nPYTHONPATH=. python -m ml.alphazero_lite.verify_seed414_root_budget_confirmation\n```\n"
    )
    _write_text(results_path, results)
    snapshots = runner.OUT / "execution-source-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot_hashes = {}
    source_paths = _source_paths()
    for name, digest in reg["source_sha256"].items():
        source = source_paths[name]
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"registered_source_changed:{name}")
        target = snapshots / f"{name}.py"
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"source_snapshot_conflict:{name}")
        target.write_bytes(data)
        snapshot_hashes[name] = digest
    published = {
        "suite": runner.SUITE,
        "registration": runner.REGISTRATION,
        "proof": runner.PROOF,
        "probes": runner.PROBES,
        "trajectories": runner.TRAJECTORIES,
        "aliases": runner.ALIASES,
        "analysis": analysis_path,
        "matrix": matrix_path,
        "results": results_path,
    }
    binding = {
        "schema": "seed414-root-budget-confirmation-publication-binding-v1",
        "hashes": {name: runner.sha(path) for name, path in published.items()},
        "source_snapshot_hashes": snapshot_hashes,
        "verifier_sha256": runner.sha(
            Path(__file__).resolve().parent
            / "verify_seed414_root_budget_confirmation.py"
        ),
        "analysis_source_sha256": runner.sha(
            Path(__file__).resolve().parent / "seed414_root_budget_analysis.py"
        ),
        "verification_command": "PYTHONPATH=. python -m ml.alphazero_lite.verify_seed414_root_budget_confirmation",
    }
    _write_json(runner.OUT / "publication-binding.json", binding)
    return report


def _source_paths() -> dict[str, Path]:
    root = runner.ROOT
    base = root / "ml/alphazero_lite"
    return {
        "runner": base / "seed414_root_budget_confirmation.py",
        "analysis": base / "seed414_root_budget_analysis.py",
        "verifier": base / "verify_seed414_root_budget_confirmation.py",
        "exclusion": base / "seed414_exclusion_proof.py",
        "publisher": base / "publish_seed414_root_budget_confirmation.py",
        "arena": base / "arena.py",
        "rules": base / "kalah_rules.py",
        "seed_contract": base / "evaluation_seed_contract.py",
        "native_adapter": base / "native_exact_root_tablebase.py",
        "runtime_search_policy": base / "runtime_search_policy.py",
        "opening_suite": base / "build_opening_suite.py",
        "population": base / "seed461_order_population.py",
    }


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    if path.exists() and path.read_text() != value:
        raise ValueError(f"published_artifact_conflict:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(value)


if __name__ == "__main__":
    print(json.dumps(publish(), indent=2))
