"""Validate and analyze the frozen constant-LR E3 versus E4 arena."""

from __future__ import annotations

from pathlib import Path

from ml.alphazero_lite import analyze_seed461_e2_e4_average as analysis

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"


def main() -> None:
    analysis.REG = DATA / "seed461-e3-e4-registration.json"
    analysis.SUITE = DATA / "seed461-e3-e4-openings.jsonl"
    analysis.CANDIDATES = DATA / "seed461-e3-e4-candidate-binding.json"
    analysis.BINDING = DATA / "seed461-e3-e4-evaluation-binding.json"
    analysis.RESULT = DATA / "seed461-e3-e4-results.json"
    analysis.MATRIX = DATA / "seed461-e3-e4-opening-score-matrix.json"
    analysis.RESULT_MD = DATA / "seed461-e3-e4-results.md"
    analysis.main()


if __name__ == "__main__":
    main()
