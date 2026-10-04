"""Create publication outputs after the frozen continuations are complete."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed398_ff1536_analysis as analysis
from ml.alphazero_lite import seed398_ff1536_confirmation as runner


def publish() -> dict[str, Any]:
    registration_path = runner.OUT / "registration.json"
    outcomes_path = runner.OUT / "new-outcomes.jsonl"
    registration = json.loads(registration_path.read_text())
    new_rows = [
        json.loads(line)
        for line in outcomes_path.read_text().splitlines()
        if line.strip()
    ]
    baseline = {
        reference: [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        for reference, path in runner.REFERENCES.items()
    }
    report, matrix = analysis.calculate(
        registration,
        new_rows,
        baseline["seed455"],
        baseline["original_O0_E4"],
    )
    (runner.OUT / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (runner.OUT / "provenance-matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    )
    primary = report["primary"]
    results = (
        "# FF 1,536-search first-action exploratory comparison\n\n"
        "This retrospective experiment uses the same ordered 64 states from #405 and the frozen FF 1,536-simulation action snapshot. It reuses 120 matching #406/#408 outcomes bound to source-ledger hashes and case identities, and adds exactly eight continuations. The design uses previously observed outcomes; it provides no promotion or overall-strength evidence.\n\n"
        f"- Primary mean gain over FF384: **{primary['mean_gain']:.6f}**.\n"
        f"- Paired bootstrap 95% interval (10,000 resamples, seed 411): **[{primary['bootstrap_95_interval'][0]:.6f}, {primary['bootstrap_95_interval'][1]:.6f}]**.\n"
        f"- Reference means: seed455 **{report['reference_mean_gains']['seed455']:.6f}**; original O0 E4 **{report['reference_mean_gains']['original_O0_E4']:.6f}**.\n"
        f"- Decision: **{report['decision']}**.\n\n"
        "Store-margin changes versus FF384 and descriptive differences versus SS384 are included by reference in `analysis.json` and case-by-case in `provenance-matrix.json`. `new-outcomes.jsonl` contains all eight complete trajectories. The 128-case matrix includes each trajectory and source provenance.\n\n"
        "Reproduce the read-only publication check with `PYTHONPATH=. python -m ml.alphazero_lite.verify_seed398_ff1536_confirmation`.\n"
    )
    results_path = runner.OUT / "results.md"
    results_path.write_text(results)
    published = {
        "registration": registration_path,
        "new_outcomes": outcomes_path,
        "analysis": runner.OUT / "analysis.json",
        "matrix": runner.OUT / "provenance-matrix.json",
        "results": results_path,
    }
    binding = {
        "schema": "seed398-ff1536-confirmation-publication-binding-v1",
        "hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in published.items()
        },
        "verifier_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parent
                / "verify_seed398_ff1536_confirmation.py"
            ).read_bytes()
        ).hexdigest(),
        "analysis_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parent / "seed398_ff1536_analysis.py"
            ).read_bytes()
        ).hexdigest(),
    }
    (runner.OUT / "publication-binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(publish(), indent=2))
