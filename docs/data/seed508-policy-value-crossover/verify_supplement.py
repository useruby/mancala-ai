"""Read-only audit of seed508 commands, runtime reports, and published results."""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path


def _flag(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"command_flag:{name}") from exc


def verify(root: Path, out: Path | None = None) -> dict:
    out = (out or root / "docs/data/seed508-policy-value-crossover").resolve()
    root = root.resolve()
    fixture_receipt = out / "fixture-receipt.json"
    if fixture_receipt.exists():
        receipt = json.loads(fixture_receipt.read_text())
        for relative, digest in receipt["files"].items():
            path = out / relative
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"fixture_receipt_mismatch:{relative}")
    registration = json.loads((out / "registration.json").read_text())
    commands = [
        json.loads(line) for line in (out / "commands.jsonl").read_text().splitlines()
    ]
    outcomes = [
        json.loads(line) for line in (out / "outcomes.jsonl").read_text().splitlines()
    ]
    if len(commands) != 1024 or len(outcomes) != 2048:
        raise ValueError("supplement_record_count")
    search = registration["runtime"]["search_options"]
    command_search = {
        key: value for key, value in search.items() if key != "random_opening_plies"
    }
    for record in commands:
        arm, i, argv = record["arm"], int(record["opening_index"]), record["argv"]
        expected = registration["arms"][arm]

        def artifact(source: str) -> str:
            return registration["components"][source]["artifact"]

        checks = {
            "--challenger": artifact(expected["policy_source"]),
            "--current": artifact("initializer"),
            "--challenger-policy-artifact": artifact(expected["policy_source"]),
            "--challenger-value-artifact": artifact(expected["value_source"]),
            "--games": "2",
            "--games-per-opening": "2",
            "--workers": "1",
            "--challenger-simulations": "128",
            "--current-simulations": "128",
            "--c-puct": "1.25",
            "--seed-contract": "azlite_eval_seed_v2",
            "--random-opening-plies": "0",
            "--max-moves": "200",
            "--seed": str(488 + i),
            "--opening-prefixes-jsonl": f"batches/{arm}-opening-{i:04d}.jsonl",
            "--game-jsonl": f"batches/{arm}-games-{i:04d}.jsonl",
            "--out": f"batches/{arm}-report-{i:04d}.json",
        }
        for flag, value in checks.items():
            actual = _flag(argv, flag)
            if flag in (
                "--challenger",
                "--current",
                "--challenger-policy-artifact",
                "--challenger-value-artifact",
            ):
                if Path(actual).parts[-len(Path(value).parts) :] != Path(value).parts:
                    raise ValueError(f"command_contract:{arm}:{i}:{flag}")
            elif flag in ("--opening-prefixes-jsonl", "--game-jsonl", "--out"):
                if not actual.endswith(value):
                    raise ValueError(f"command_contract:{arm}:{i}:{flag}")
            elif actual != value:
                raise ValueError(f"command_contract:{arm}:{i}:{flag}")
        opening_path = out / "batches" / f"{arm}-opening-{i:04d}.jsonl"
        if hashlib.sha256(opening_path.read_bytes()).hexdigest() != _flag(
            argv, "--suite-sha256"
        ):
            raise ValueError(f"command_contract:{arm}:{i}:--suite-sha256")
        for flag in (
            "--challenger-search-options-json",
            "--current-search-options-json",
        ):
            if json.loads(_flag(argv, flag)) != command_search:
                raise ValueError(f"command_contract:{arm}:{i}:{flag}")
        report = json.loads(
            (out / "batches" / f"{arm}-report-{i:04d}.json").read_text()
        )
        notes = report["notes"]
        runtime = registration["runtime"]
        if (
            Path(notes["current_path"]).parts[-2:]
            != Path(_flag(argv, "--current")).parts[-2:]
        ):
            raise ValueError(f"report_current_binding:{arm}:{i}")
        if (
            Path(notes["challenger_path"]).parts[-2:]
            != Path(_flag(argv, "--challenger")).parts[-2:]
        ):
            raise ValueError(f"report_challenger_binding:{arm}:{i}")
        if report["games_played"] != 2 or notes["base_seed"] != 488 + i:
            raise ValueError(f"report_totals_or_seed:{arm}:{i}")
        arm_rows = [
            row
            for row in outcomes
            if row["arm"] == arm and int(row["opening_index"]) == i
        ]
        totals = {
            "wins": sum(row["winner"] == "challenger" for row in arm_rows),
            "draws": sum(row["winner"] == "draw" for row in arm_rows),
            "losses": sum(row["winner"] == "current" for row in arm_rows),
        }
        if any(report[key] != value for key, value in totals.items()):
            raise ValueError(f"report_outcome_totals:{arm}:{i}")
        profile = notes["search_profile"]
        for key, value in (
            ("c_puct", 1.25),
            ("simulations", 128),
            ("exact_root_solve_enabled", True),
            ("exact_root_solve_threshold", 16),
            ("exact_root_objective", "final_score_margin"),
            (
                "exact_root_tie_rule",
                "highest_legal_network_prior_then_lowest_move_index",
            ),
        ):
            if profile.get(key) != value:
                raise ValueError(f"report_runtime:{arm}:{i}:{key}")
        if (
            profile.get("exact_root_native_probe_sha256")
            != registration["runtime"]["runtime_identities_sha256"][
                "model-artifact/runtime/kalah_v1_tablebase"
            ]
            or profile.get("exact_root_tablebase_sha256")
            != registration["runtime"]["runtime_identities_sha256"][
                "model-artifact/runtime/kalah_v1_21.kvtb"
            ]
        ):
            raise ValueError(f"report_runtime:{arm}:{i}:exact_root_identity")
        if (
            notes["random_opening_plies"]
            != runtime["search_options"]["random_opening_plies"]
        ):
            raise ValueError(f"report_runtime:{arm}:{i}:random_opening_plies")
    published = json.loads((out / "results.json").read_text())
    if published.get("primary", {}).get("mean_difference") != 0.029296875:
        raise ValueError("published_result_mismatch")
    if published.get("standalone", {}).get("P", {}).get("wins") != 493:
        raise ValueError("published_result_mismatch")
    if published.get("standalone", {}).get("V", {}).get("wins") != 453:
        raise ValueError("published_result_mismatch")
    return {"commands": len(commands), "outcomes": len(outcomes), "status": "verified"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.out), indent=2, sort_keys=True))
