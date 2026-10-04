"""Create the immutable publication binding and concise results page."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-e4-reference-diagnostic"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish() -> dict[str, str]:
    registration = DATA / "registration.json"
    outcomes = DATA / "outcomes.jsonl"
    analysis = DATA / "analysis.json"
    verifier = ROOT / "ml/alphazero_lite/verify_seed398_e4_reference_diagnostic.py"
    record = {
        "schema": "seed398-e4-reference-diagnostic-publication-binding-v1",
        "registration_sha256": digest(registration),
        "outcomes_sha256": digest(outcomes),
        "analysis_sha256": digest(analysis),
        "verifier_sha256": digest(verifier),
        "verification_command": "PYTHONPATH=. python -m ml.alphazero_lite.verify_seed398_e4_reference_diagnostic",
        "published_files": [
            "registration.json",
            "outcomes.jsonl",
            "analysis.json",
            "results.md",
        ],
    }
    binding = DATA / "publication-binding.json"
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if binding.exists() and binding.read_text() != payload:
        raise ValueError("publication_binding_conflict")
    binding.write_text(payload)
    result = json.loads(analysis.read_text())
    text = f"""# Seed398 first-action diagnostic under original O0 E4

Retrospective reference-dependence diagnostic. It uses the exact 64 registered #405 states and #406's recorded SS/FF 384-search first actions. After the forced move, both players use the original O0 E4 runtime artifact. There are 256 games: 64 cases × two first actions × two continuation budgets.

## Frozen configuration

- E4 checkpoint SHA256: `{json.loads(registration.read_text())["reference"]["checkpoint_sha256"]}`.
- Continuation budgets: 1,536 primary and 384 secondary; `c_puct=1.25`; #406 search options and seed contexts; base seed 406.
- Root-16 native solver; final-score-margin objective; prior-based tie rule; exact-leaf solving disabled.
- Complete trajectories and the state-wise paired matrix are in `outcomes.jsonl` and `analysis.json`.

## Results

- Primary 1,536 mean FF-minus-SS score: **{result["primary"]["mean_delta"]:.8f}**; paired bootstrap 95% interval **[{result["primary"]["paired_bootstrap_95_interval"][0]:.8f}, {result["primary"]["paired_bootstrap_95_interval"][1]:.8f}]**.
- Secondary 384 mean: **{result["secondary"]["mean_delta"]:.8f}**.
- Descriptive E4 delta minus #406 seed455 delta: **{result["interaction_E4_minus_seed455"]["mean_delta"]:.8f}**, paired bootstrap interval **[{result["interaction_E4_minus_seed455"]["paired_bootstrap_95_interval"][0]:.8f}, {result["interaction_E4_minus_seed455"]["paired_bootstrap_95_interval"][1]:.8f}]**.
- Preregistered conclusion: **{result["decision"]}**.

The interaction is descriptive. This retrospective diagnostic supports no training, state reselection, promotion, minimax-quality, or overall-strength claim.

Verify the public ledger without model artifacts or native searches:

```sh
PYTHONPATH=. python -m ml.alphazero_lite.verify_seed398_e4_reference_diagnostic
```
"""
    results = DATA / "results.md"
    if results.exists() and results.read_text() != text:
        raise ValueError("published_results_conflict")
    results.write_text(text)
    return record


if __name__ == "__main__":
    print(json.dumps(publish(), indent=2))
