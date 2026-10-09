"""Portable, read-only source reconstruction and seed449 publication proof."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
SOURCES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
COMPRESSED_SHA256 = {
    "fresh": "5f5062416ae620e4e3491989f19c166d26840b3fead8074323f23eb01211746b",
    "generic_bootstrap": "51132b09c453712b8f48ab748e5025b25338681f9f89dd65a1fe2598ca266330",
    "random_teacher": "fa6f83e3fc6bb3c93f87e9b640b8d58213f526115e3fdf582c96d9b1b7b7d1d7",
    "opening_disagreement": "5d73b65f57cf3e4199608c5cfbd80fa9656d5f267b5a823b12c7940a80824bdc",
    "stability": "77de3df0a93f928f8e131566c1f18f83b7d0100874248e2d29a6cb37f6e45b36",
}
COMPRESSED_BASE = Path("docs/data/seed426-canonical-overlap/sources")
SEED416 = Path("docs/data/seed416-policy-target-softening/registration-v3.json")
SEED449 = Path("docs/data/seed449-policy-gain-localization")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reconstruct_derivatives(
    root: Path, registration: dict[str, Any], directory: Path
) -> dict[str, Path]:
    from ml.alphazero_lite.seed416_policy_target_softening import transform_row

    if tuple(item["name"] for item in registration["replays"]) != SOURCES:
        raise ValueError("seed450_source_order_invalid")
    derivative_paths = {}
    for name in SOURCES:
        compressed = root / COMPRESSED_BASE / f"{name}.jsonl.gz"
        if sha(compressed) != COMPRESSED_SHA256[name]:
            raise ValueError(f"seed450_compressed_snapshot_hash_invalid:{name}")
        output = directory / f"{name}.jsonl"
        digest = hashlib.sha256()
        source_digest = hashlib.sha256()
        source_rows = 0
        with (
            gzip.open(compressed, "rt", encoding="utf-8") as source,
            output.open("x", encoding="utf-8", newline="") as dest,
        ):
            for line_number, line in enumerate(source, 1):
                source_digest.update(line.encode("utf-8"))
                source_rows = line_number
                row = json.loads(line)
                lane_a, changed = transform_row(row, "A")
                if changed or lane_a != row:
                    raise ValueError(
                        f"seed450_lane_a_transform_changed:{name}:{line_number}"
                    )
                encoded = (
                    json.dumps(lane_a, separators=(",", ":"), ensure_ascii=True) + "\n"
                )
                dest.write(encoded)
                digest.update(encoded.encode("utf-8"))
        spec = registration["derivatives"][name]["A"]
        if (
            source_digest.hexdigest() != spec["source_sha256"]
            or source_rows != spec["rows"]
        ):
            raise ValueError(f"seed450_decompressed_source_identity_invalid:{name}")
        if digest.hexdigest() != spec["derivative_sha256"]:
            raise ValueError(
                f"seed450_derivative_serialization_or_identity_invalid:{name}"
            )
        derivative_paths[name] = output
    return derivative_paths


def _row24_proof(root: Path) -> dict[str, Any]:
    with gzip.open(
        root / COMPRESSED_BASE / "fresh.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number == 24:
                row = json.loads(line)
                return {
                    "raw_source_line": 24,
                    "policy_target_actual_mode": row.get("policy_target_actual_mode"),
                    "policy_target_mode": row.get("policy_target_mode"),
                    "teacher_source": row.get("teacher_source"),
                    "exact_selected_action": row.get("exact_selected_action"),
                    "exact_optimal_actions": row.get("exact_optimal_actions"),
                    "exact_action_margins": row.get("exact_action_margins"),
                    "exact_root_metadata_present": all(
                        key in row
                        for key in (
                            "exact_selected_action",
                            "exact_optimal_actions",
                            "exact_action_margins",
                        )
                    ),
                    "stored_policy": row["policy"],
                    "lane_a_stored_policy_unchanged": True,
                    "production_validator_branch": "exact_root_one_hot: validates normalized target has a single unit-mass action, then returns before ordinary requested-mode matching; legality is checked first",
                    "loader_mode_precedence": "policy_target_actual_mode takes precedence over policy_target_mode",
                }
    raise ValueError("seed450_fresh_row24_missing")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    seed416_path = root / SEED416
    seed416 = json.loads(seed416_path.read_text())
    seed449_reg = json.loads((root / SEED449 / "registration.json").read_text())
    if sha(seed416_path) != seed449_reg["bound_input_sha256"][str(SEED416)]:
        raise ValueError("seed450_seed416_registration_hash_invalid")
    for relative, digest in seed449_reg["source_sha256"].items():
        if sha(root / relative) != digest:
            raise ValueError(f"seed450_historical_source_hash_invalid:{relative}")
    transform_source = root / "ml/alphazero_lite/seed416_policy_target_softening.py"
    if (
        sha(transform_source)
        != seed416["source_hashes"][
            "ml/alphazero_lite/seed416_policy_target_softening.py"
        ]
    ):
        raise ValueError("seed450_transform_source_hash_invalid")

    # Use a transient directory rooted inside the supplied checkout. The absolute paths
    # retained in registration-v3 are never opened or passed to a file-opening helper.
    with tempfile.TemporaryDirectory(
        prefix="seed450-derivatives-", dir=root
    ) as temporary:
        derivatives = _reconstruct_derivatives(root, seed416, Path(temporary))
        from ml.alphazero_lite import train
        from ml.alphazero_lite import verify_seed449_policy_gain_localization as frozen

        original_loader = train.load_jsonl_replay

        def rooted_loader(paths, weights=None, **kwargs):
            mapped = [derivatives[item["name"]] for item in seed416["replays"]]
            return original_loader(mapped, weights, **kwargs)

        train.load_jsonl_replay = rooted_loader
        try:
            report = frozen.verify(root)
        finally:
            train.load_jsonl_replay = original_loader
    if report["exposures"] != 2607 or report["identities"] != 1242:
        raise ValueError("seed450_population_semantics_invalid")
    if report["classification"] != "frequency_concentrated_policy_gain":
        raise ValueError("seed450_classification_invalid")
    with (root / SEED449 / "ordered-row-ledger.jsonl").open(encoding="utf-8") as stream:
        validated_rows = [json.loads(line) for line in stream]
    exposure_frequency: dict[str, int] = {}
    for row in validated_rows:
        identity = row["input_identity"]
        exposure_frequency[identity] = exposure_frequency.get(identity, 0) + 1
    source_singletons = {
        source: sum(
            row["source_ref"]["source"] == source
            and exposure_frequency[row["input_identity"]] == 1
            for row in validated_rows
        )
        for source in ("fresh", "random_teacher")
    }
    return {
        **report,
        "source_reconstruction": "valid",
        "lane_a_derivative_serialization": "exact_sha256_match",
        "historical_absolute_derivatives_opened": False,
        "fresh_source_row_24": _row24_proof(root),
        "source_confound": {
            "fresh_singleton_exposures": source_singletons["fresh"],
            "random_teacher_singleton_exposures": source_singletons["random_teacher"],
        },
        "historical_seed447_classification": "close_joint_output_cap_branch",
        "interpretation": "Frequency and source are confounded. The descriptive frequency classification is preserved; replay duplication is not established as causal and these data alone do not justify changing training weights.",
        "correction_timing": "post-execution correction; seed449 freeze and publication are unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
