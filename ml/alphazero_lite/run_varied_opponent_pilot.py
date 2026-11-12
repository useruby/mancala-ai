"""Execute or resume the frozen varied-opponent pilot by atomic pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import varied_opponent_pilot as pilot


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text())
    digest = freeze.pop("freeze_sha256", None)
    actual = hashlib.sha256(canonical_json(freeze)).hexdigest()
    freeze["freeze_sha256"] = digest
    if digest != actual:
        raise ValueError("pilot_freeze_digest_mismatch")
    if (
        len(freeze.get("assignments", [])) != 256
        or len(freeze.get("openings", [])) != 64
    ):
        raise ValueError("pilot_freeze_assignment_count_mismatch")
    return freeze


def pair_jobs(freeze: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in freeze["assignments"]:
        grouped.setdefault(str(spec["pair_id"]), []).append(spec)
    return [
        {"game_specs": sorted(specs, key=lambda spec: int(spec["designated_p_seat"]))}
        for _pair, specs in sorted(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(
            "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"
        ),
    )
    args = parser.parse_args()
    freeze = load_freeze(args.freeze)
    jobs = pair_jobs(freeze)
    if len(jobs) != 128:
        raise ValueError("pilot_freeze_pair_count_mismatch")
    artifacts = args.artifacts
    pilot.verify_components(artifacts)
    # Runtime sidecars and assets are checked in the no-games preflight. Recheck
    # them immediately before any work is dispatched.
    pilot.effective_runtime_policy(artifacts)
    if args.resume:
        pilot.load_pair_checkpoints(args.checkpoint_dir, jobs)
    completed = pilot.run_checkpointed_pairs(
        jobs,
        artifacts,
        args.checkpoint_dir,
        attempt_dir=args.checkpoint_dir.parent / "attempts",
    )
    summary = pilot.summarize_pair_accounting(completed)
    print(
        json.dumps(
            {"pairs": len(completed), "accounting": summary}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
