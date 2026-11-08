"""Seed557 retrospective fixed-position diagnostic from the #556 archive."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed557-retrospective-diagnostic"
COMMIT = "f38838e6ffe20a44c1e8c161a9daed0f1243f0ae"
ARCHIVE = "docs/data/seed556-exploratory-arena/screen-0005/"
SEED = 557
POLICIES = ("T512", "C128", "P")
OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "root_temperature": 0.0,
    "tactical_root_bias": 0.0,
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(state: dict[str, Any]) -> str:
    return sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def search_seed(state_hash: str) -> int:
    return int(sha256(f"seed557:{state_hash}".encode())[:8], 16) % (2**31)


def git_blob(path: str, commit: str = COMMIT) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def load_authenticated_archive() -> tuple[list[dict], list[dict], dict]:
    receipt = json.loads(git_blob(ARCHIVE + "verification-receipt.json"))
    openings_bytes = git_blob(ARCHIVE + "openings.jsonl")
    expected = receipt["evidence_sha256"]
    if sha256(openings_bytes) != expected["openings.jsonl"]:
        raise ValueError("#556 openings receipt hash mismatch")
    openings = [json.loads(line) for line in openings_bytes.splitlines() if line]
    if len(openings) != 1024:
        raise ValueError("#556 registered opening count mismatch")
    records: list[dict] = []
    for index in range(1024):
        rel = f"batches/games-{index:04d}.jsonl"
        content = git_blob(ARCHIVE + rel)
        if sha256(content) != expected.get(rel):
            raise ValueError(f"#556 evidence hash mismatch:{ARCHIVE + rel}")
        rows = [json.loads(line) for line in content.splitlines() if line]
        if len(rows) != 2:
            raise ValueError(f"#556 pair row count mismatch:{ARCHIVE + rel}")
        for row_number, game_row in enumerate(rows):
            records.append(
                {
                    "opening_index": index,
                    "game_row": game_row,
                    "source_path": ARCHIVE + rel,
                    "source_line": row_number + 1,
                    "source_record_sha256": sha256(
                        json.dumps(
                            game_row, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ),
                    "batch_sha256": expected[rel],
                }
            )
    return openings, records, receipt


def reconstruct_cohort(
    openings: list[dict], records: list[dict], limit: int = 256
) -> tuple[list[dict], dict]:
    eligible_by_opening: dict[int, list[dict]] = {}
    eligible_counts = {"games": 0, "positions": 0}
    for opening in openings:
        prefix_game = KalahGame([4] * 12, [0, 0], 0)
        for move in opening["source_prefix_moves_absolute"]:
            if not prefix_game.move(int(move)):
                raise ValueError("registered absolute opening prefix is illegal")
        if prefix_game.to_state() != opening["state"]:
            raise ValueError("registered opening prefix does not reconstruct its state")
        if canonical_hash(prefix_game.to_state()) != opening["state_hash"]:
            raise ValueError("registered opening prefix hash mismatch")
    for source in records:
        opening_index = source["opening_index"]
        opening = openings[opening_index]
        game_row = source["game_row"]
        game = KalahGame.from_state(opening["state"])
        absolute_moves = [
            int(move) for move in game_row["trajectory"].split(",") if move
        ]
        first_eligible = None
        for ply, absolute_move in enumerate(absolute_moves):
            if game.over():
                raise ValueError("source trajectory contains a move after terminal")
            legal_absolute = [game.pit_index(move) for move in game.possible_moves()]
            if 17 <= sum(game.pits) <= 21 and len(legal_absolute) >= 2:
                first_eligible = {
                    "opening_index": opening_index,
                    "game_within_opening": int(game_row["game_within_opening"]),
                    "source_game_index_local": int(game_row["game_index"]),
                    "challenger_player": int(game_row["challenger_player"]),
                    "source_side": (
                        "T512"
                        if game.current_player == int(game_row["challenger_player"])
                        else "P"
                    ),
                    "source_player_to_move": game.current_player,
                    "ply": ply,
                    "state": game.to_state(),
                    "state_hash": canonical_hash(game.to_state()),
                    "active_pit_stones": sum(game.pits),
                    "legal_absolute_actions": legal_absolute,
                    "opening_state_hash": opening["state_hash"],
                    "opening_prefix_moves": opening["prefix_moves"],
                    "trajectory_prefix_absolute": absolute_moves[:ply],
                    "source_path": source["source_path"],
                    "source_line": source["source_line"],
                    "source_record_sha256": source["source_record_sha256"],
                    "source_batch_sha256": source["batch_sha256"],
                    "decision_absolute_action": absolute_move,
                }
                break
            if absolute_move not in legal_absolute or not game.move(absolute_move):
                raise ValueError("source trajectory contains an illegal absolute move")
        if first_eligible is not None:
            eligible_counts["games"] += 1
            eligible_counts["positions"] += 1
            eligible_by_opening.setdefault(opening_index, []).append(first_eligible)

    unique: dict[str, dict] = {}
    for choices in eligible_by_opening.values():
        for item in choices:
            unique.setdefault(item["state_hash"], item)
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            sha256(f"seed557:{item['state_hash']}".encode()),
            item["state_hash"],
        ),
    )
    chosen: list[dict] = []
    used_openings: set[int] = set()
    for side in ("T512", "P") * limit:
        candidate = next(
            (
                item
                for item in ranked
                if item["source_side"] == side
                and item["opening_index"] not in used_openings
            ),
            None,
        )
        if candidate is not None:
            chosen.append(candidate)
            used_openings.add(candidate["opening_index"])
        if len(chosen) == limit:
            break
    # The second side pass is sorted by the same preregistered hash; canonical
    # final ordering is independent of which pass selected a candidate.
    chosen.sort(
        key=lambda item: (
            sha256(f"seed557:{item['state_hash']}".encode()),
            item["state_hash"],
        )
    )
    return chosen, {
        "eligible_game_positions": eligible_counts["games"],
        "eligible_openings": len(eligible_by_opening),
        "unique_states": len(unique),
        "selected": len(chosen),
        "shortfall": max(0, limit - len(chosen)),
        "selected_by_source_side": {
            side: sum(item["source_side"] == side for item in chosen)
            for side in ("T512", "P")
        },
    }


def action_result(margins: dict[int, int], move: int) -> tuple[int, bool]:
    best = max(margins.values())
    margin = margins[move]

    def wdl(value: int) -> int:
        return 1 if value > 0 else -1 if value < 0 else 0

    return best - margin, wdl(margin) < wdl(best)


def run_searches(
    cohort: list[dict], artifact_dirs: dict[str, Path], native: Path, tablebase: Path
) -> tuple[list[dict], list[dict]]:
    evaluators = {
        name: arena.ArtifactEvaluator(path) for name, path in artifact_dirs.items()
    }
    initializer = evaluators["initializer"]
    composed = {
        policy: arena.ComposedArtifactEvaluator(
            evaluators[policy],
            initializer,
            policy_source="current",
            value_source="candidate",
        )
        for policy in POLICIES
    }
    searches: list[dict] = []
    oracles: list[dict] = []
    with NativeExactRootTablebase(native, tablebase) as oracle:
        for item in cohort:
            game = KalahGame.from_state(item["state"])
            legal = game.possible_moves()
            if len(legal) < 2 or not 17 <= sum(game.pits) <= 21:
                raise ValueError("frozen cohort root no longer eligible")
            margins = oracle.root_action_margins(game, game.current_player)
            if set(margins) != set(legal):
                raise ValueError(
                    f"exact legal-action coverage failure:{item['state_hash']}"
                )
            oracles.append(
                {
                    "state_hash": item["state_hash"],
                    "legal_moves": legal,
                    "action_margins": {str(k): v for k, v in sorted(margins.items())},
                    "coverage_complete": True,
                }
            )
            for policy in POLICIES:
                priors, _ = composed[policy].evaluate(game)
                prior_action = max(legal, key=lambda move: (float(priors[move]), -move))
                seed = search_seed(item["state_hash"])
                result = arena.evaluate_artifact_position(
                    evaluator=composed[policy],
                    state=item["state"],
                    simulations=128,
                    seed=seed,
                    c_puct=1.25,
                    search_options=OPTIONS,
                    exact_root_solve_threshold=None,
                )
                searches.append(
                    {
                        "state_hash": item["state_hash"],
                        "policy": policy,
                        "seed": seed,
                        "highest_prior_action": prior_action,
                        "puct_action": int(result["selected_move"]),
                        "legal_moves": legal,
                        "priors": [float(value) for value in priors],
                        "visits": result["visits"],
                    }
                )
    return searches, oracles


def analyze(cohort: list[dict], searches: list[dict], oracles: list[dict]) -> dict:
    oracle_map = {row["state_hash"]: row for row in oracles}
    grouped = {(row["state_hash"], row["policy"]): row for row in searches}
    rows = []
    for item in cohort:
        oracle = oracle_map[item["state_hash"]]
        margins = {int(k): int(v) for k, v in oracle["action_margins"].items()}
        best = max(margins.values())
        for policy in POLICIES:
            search = grouped[(item["state_hash"], policy)]
            prior_regret, prior_wdl = action_result(
                margins, search["highest_prior_action"]
            )
            puct_regret, puct_wdl = action_result(margins, search["puct_action"])
            rows.append(
                {
                    **item,
                    "policy": policy,
                    "prior_regret": prior_regret,
                    "puct_regret": puct_regret,
                    "prior_wdl_downgrade": prior_wdl,
                    "puct_wdl_downgrade": puct_wdl,
                    "prior_optimal": prior_regret == 0,
                    "puct_optimal": puct_regret == 0,
                    "puct_improved_prior": puct_regret < prior_regret,
                    "puct_worsened_prior": puct_regret > prior_regret,
                    "exact_optimal_margin": best,
                }
            )
    metrics = {}
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        metrics[policy] = {
            "n": len(group),
            "prior_exact_optimal_rate": statistics.fmean(
                row["prior_optimal"] for row in group
            ),
            "puct_exact_optimal_rate": statistics.fmean(
                row["puct_optimal"] for row in group
            ),
            "prior_wdl_downgrade_rate": statistics.fmean(
                row["prior_wdl_downgrade"] for row in group
            ),
            "puct_wdl_downgrade_rate": statistics.fmean(
                row["puct_wdl_downgrade"] for row in group
            ),
            "puct_improved_prior_rate": statistics.fmean(
                row["puct_improved_prior"] for row in group
            ),
            "puct_worsened_prior_rate": statistics.fmean(
                row["puct_worsened_prior"] for row in group
            ),
            "mean_final_margin_regret_after_puct": statistics.fmean(
                row["puct_regret"] for row in group
            ),
        }
    stone_breakdown = _breakdown(rows, "active_pit_stones")
    side_breakdown = _breakdown(rows, "source_side")
    return {
        "primary_contrast": "paired T512 minus P mean final-margin regret after PUCT; positive means T512 leaves more seeds on the table",
        "secondary_contrast": "paired T512 minus C128 mean final-margin regret after PUCT",
        "paired_uncertainty": {
            "method": "opening-cluster bootstrap with replacement",
            "resamples": 20000,
            "interval": "percentile 95%",
            "primary_seed": 557001,
            "secondary_seed": 557002,
        },
        "metrics": metrics,
        "paired_T512_minus_P_regret": _paired(rows, "T512", "P", 557001),
        "paired_T512_minus_C128_regret": _paired(rows, "T512", "C128", 557002),
        "by_stone_count": stone_breakdown,
        "by_source_side": side_breakdown,
        "interpretation": "Retrospective diagnostic of positions reached in #556 only; it cannot establish why the arena result occurred or compare the policies across the full game.",
        "gap_localization": _gap_conclusion(metrics),
        "root_records": rows,
    }


def _paired(rows: list[dict], left: str, right: str, seed: int) -> dict:
    grouped: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row["policy"] in {left, right}:
            grouped.setdefault(row["opening_index"], {}).setdefault(row["policy"], {})[
                "after_search"
            ] = row["puct_regret"]
            grouped.setdefault(row["opening_index"], {}).setdefault(row["policy"], {})[
                "prior"
            ] = row["prior_regret"]
    pairs = [pair for pair in grouped.values() if left in pair and right in pair]
    after = [pair[left]["after_search"] - pair[right]["after_search"] for pair in pairs]
    prior = [pair[left]["prior"] - pair[right]["prior"] for pair in pairs]
    return {
        "n_openings": len(after),
        "mean_final_margin_regret_after_puct": statistics.fmean(after)
        if after
        else None,
        "bootstrap_95_percent_interval": _bootstrap_interval(after, seed),
        "mean_prior_final_margin_regret_difference": statistics.fmean(prior)
        if prior
        else None,
        "prior_bootstrap_95_percent_interval": _bootstrap_interval(prior, seed + 100),
    }


def _bootstrap_interval(values: list[float], seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(20000)
    )
    return [means[499], means[19499]]


def _breakdown(rows: list[dict], field: str) -> dict:
    values: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        values.setdefault(str(row[field]), {}).setdefault(row["policy"], []).append(row)
    return {
        category: {
            policy: {
                "n": len(group),
                "prior_exact_optimal_rate": statistics.fmean(
                    item["prior_optimal"] for item in group
                ),
                "prior_wdl_downgrade_rate": statistics.fmean(
                    item["prior_wdl_downgrade"] for item in group
                ),
                "puct_exact_optimal_rate": statistics.fmean(
                    item["puct_optimal"] for item in group
                ),
                "puct_wdl_downgrade_rate": statistics.fmean(
                    item["puct_wdl_downgrade"] for item in group
                ),
                "mean_final_margin_regret_after_puct": statistics.fmean(
                    item["puct_regret"] for item in group
                ),
                "puct_improved_prior_rate": statistics.fmean(
                    item["puct_improved_prior"] for item in group
                ),
                "puct_worsened_prior_rate": statistics.fmean(
                    item["puct_worsened_prior"] for item in group
                ),
            }
            for policy, group in by_policy.items()
        }
        for category, by_policy in values.items()
    }


def _gap_conclusion(metrics: dict) -> str:
    prior_gap = (
        metrics["T512"]["prior_exact_optimal_rate"]
        - metrics["P"]["prior_exact_optimal_rate"]
    )
    search_gap = (
        metrics["T512"]["mean_final_margin_regret_after_puct"]
        - metrics["P"]["mean_final_margin_regret_after_puct"]
    )
    if prior_gap < 0 and search_gap > 0:
        return "Gap is visible after search but not in raw exact-optimal prior rate; direction is mixed and attribution unresolved."
    if prior_gap > 0 and search_gap > 0:
        return "T512 gap appears in raw priors and after search on this cohort."
    if prior_gap > 0 and search_gap <= 0:
        return "T512 gap appears in raw priors but not after search; search closes or reverses it."
    return "No consistent T512 disadvantage is resolved by raw priors or post-search mean regret on this cohort."


def authenticate_runtime() -> dict:
    receipt556 = json.loads(git_blob(ARCHIVE + "verification-receipt.json"))
    receipt555 = json.loads(
        git_blob(
            "docs/data/seed555-policy-strength-screen/verification-receipt.json",
            "47a7cb39f0733c338daccceab54f8aea614c9464",
        )
    )
    specs = {
        "T512": (
            ROOT
            / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/T512",
            receipt556["evidence_sha256"],
            "runtime-artifacts/T512/",
        ),
        "P": (
            ROOT
            / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/P-E4",
            receipt556["evidence_sha256"],
            "runtime-artifacts/P-E4/",
        ),
        "initializer": (
            ROOT
            / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/initializer",
            receipt556["evidence_sha256"],
            "runtime-artifacts/initializer/",
        ),
        "C128": (
            ROOT / "docs/data/seed555-policy-strength-screen/runtime-artifacts/C128",
            receipt555["evidence_sha256"],
            "runtime-artifacts/C128/",
        ),
    }
    authenticated: dict[str, dict[str, str]] = {}
    for name, (directory, digests, prefix) in specs.items():
        authenticated[name] = {}
        for filename in (
            "metadata.json",
            "model.npz",
            "search_policy.json",
            "weights.json",
        ):
            path = directory / filename
            relative = prefix + filename
            if not path.is_file() or sha256(path.read_bytes()) != digests.get(relative):
                raise ValueError(f"runtime artifact authentication failed:{path}")
            authenticated[name][filename] = sha256(path.read_bytes())
    return {name: rows for name, rows in authenticated.items()}


def authentication_receipts() -> dict[str, str]:
    return {
        "seed555_verification_receipt_sha256": sha256(
            git_blob(
                "docs/data/seed555-policy-strength-screen/verification-receipt.json",
                "47a7cb39f0733c338daccceab54f8aea614c9464",
            )
        ),
        "seed556_verification_receipt_sha256": sha256(
            git_blob(ARCHIVE + "verification-receipt.json")
        ),
    }


def expected_oracle_identities() -> tuple[dict[str, Any], dict[str, str]]:
    receipt = json.loads(git_blob(ARCHIVE + "verification-receipt.json"))
    sources = {
        "registration.json": git_blob(ARCHIVE + "registration.json"),
        "runtime-policy-preflight.json": git_blob(
            ARCHIVE + "runtime-policy-preflight.json"
        ),
    }
    source_hashes: dict[str, str] = {}
    for name, contents in sources.items():
        digest = sha256(contents)
        if digest != receipt["evidence_sha256"].get(name):
            raise ValueError(f"#556 oracle identity source hash mismatch:{name}")
        source_hashes[name] = digest
    registration = json.loads(sources["registration.json"])
    preflight = json.loads(sources["runtime-policy-preflight.json"])
    expected = {
        "native_probe_sha256": preflight["exact_root_native_probe_sha256"],
        "tablebase_sha256": registration["tablebase_materialized_sha256"],
        "tablebase_size_bytes": int(registration["tablebase_materialized_size"]),
    }
    return expected, source_hashes


def observe_oracle_identities(native: Path, tablebase: Path) -> dict[str, Any]:
    if not native.is_file() or not tablebase.is_file():
        raise FileNotFoundError(
            "native exact-root probe or materialized tablebase missing"
        )
    return {
        "native_probe_sha256": sha256(native.read_bytes()),
        "tablebase_sha256": sha256(tablebase.read_bytes()),
        "tablebase_size_bytes": tablebase.stat().st_size,
    }


def authenticate_oracle_assets(native: Path, tablebase: Path) -> dict[str, Any]:
    expected, _source_hashes = expected_oracle_identities()
    observed = observe_oracle_identities(native, tablebase)
    if observed != expected:
        raise ValueError(
            f"native oracle assets do not match authenticated #556 identities: "
            f"observed={observed}, expected={expected}"
        )
    return observed


def validate_existing_freeze(existing: dict, proposed: dict) -> None:
    if existing != proposed:
        raise ValueError("conflicting existing execution freeze")


def oracle_identity_evidence(freeze_bytes: bytes, frozen_oracle: dict) -> dict:
    expected, source_hashes = expected_oracle_identities()
    frozen_identities = {
        "native_probe_sha256": frozen_oracle["native_probe_sha256"],
        "tablebase_sha256": frozen_oracle["tablebase_sha256"],
        "tablebase_size_bytes": expected["tablebase_size_bytes"],
    }
    if frozen_identities != expected:
        raise ValueError(
            "existing execution freeze conflicts with authenticated #556 oracle"
        )
    return {
        "schema": "seed557-authenticated-oracle-identities-v1",
        "source_commit": COMMIT,
        "authenticated_source_sha256": source_hashes,
        "expected_from_seed556": expected,
        "oracle_identities_in_frozen_execution": frozen_identities,
        "execution_freeze_sha256": sha256(freeze_bytes),
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "bind-preserved-freeze":
        freeze_path = DATA / "execution-freeze.json"
        identity_path = DATA / "oracle-identity-verification.json"
        freeze_bytes = freeze_path.read_bytes()
        freeze = json.loads(freeze_bytes)
        evidence = oracle_identity_evidence(freeze_bytes, freeze["oracle"])
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != evidence:
                raise ValueError(
                    "conflicting existing authenticated oracle identity record"
                )
        else:
            identity_path.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n"
            )
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return
    openings, records, receipt = load_authenticated_archive()
    cohort, selection = reconstruct_cohort(openings, records)
    cohort_path = DATA / "cohort.json"
    if cohort_path.exists():
        existing = json.loads(cohort_path.read_text())
        if existing.get("states") != cohort:
            raise ValueError(
                "existing frozen cohort differs from deterministic reconstruction"
            )
    else:
        cohort_path.write_text(
            json.dumps(
                {
                    "schema": "seed557-frozen-cohort-v1",
                    "selection": "first eligible position/game; 17-21 active pit stones; at least two legal actions; state-hash deduplication; seeded SHA256 ranking; at most one/opening; source side balanced where possible",
                    "seed": SEED,
                    "selection_summary": selection,
                    "states": cohort,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    authenticated = authenticate_runtime()
    native_path = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
    tablebase_path = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
    oracle_identities = authenticate_oracle_assets(native_path, tablebase_path)
    freeze = {
        "schema": "seed557-execution-freeze-v1",
        "source_commit": COMMIT,
        "verification_receipt_sha256": sha256(
            git_blob(ARCHIVE + "verification-receipt.json")
        ),
        "verified_game_batch_count": 1024,
        "verified_game_batch_sha256": {
            f"games-{i:04d}.jsonl": receipt["evidence_sha256"][
                f"batches/games-{i:04d}.jsonl"
            ]
            for i in range(1024)
        },
        "cohort_sha256": sha256(cohort_path.read_bytes()),
        "selection": selection,
        "authenticated_policies": authenticated,
        "authentication_receipts": authentication_receipts(),
        "search": {
            "simulations": 128,
            "c_puct": 1.25,
            "workers": 1,
            "fresh_tree_per_policy_root": True,
            "same_seed_all_policies_per_root": True,
            "exact_search_handoff": False,
            "search_options": OPTIONS,
        },
        "oracle": {
            "native_probe_sha256": oracle_identities["native_probe_sha256"],
            "tablebase_sha256": oracle_identities["tablebase_sha256"],
            "coverage": "every legal action at every frozen root",
        },
    }
    freeze_path = DATA / "execution-freeze.json"
    identity_path = DATA / "oracle-identity-verification.json"
    freeze_bytes = (json.dumps(freeze, indent=2, sort_keys=True) + "\n").encode()
    if freeze_path.exists():
        freeze_bytes = freeze_path.read_bytes()
        validate_existing_freeze(json.loads(freeze_bytes), freeze)
    identity_evidence = oracle_identity_evidence(freeze_bytes, freeze["oracle"])
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity_evidence:
            raise ValueError(
                "conflicting existing authenticated oracle identity record"
            )
    else:
        identity_path.write_text(
            json.dumps(identity_evidence, indent=2, sort_keys=True) + "\n"
        )
    if not freeze_path.exists():
        freeze_path.write_bytes(freeze_bytes)
    artifact_dirs = {
        "T512": ROOT
        / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/T512",
        "C128": ROOT
        / "docs/data/seed555-policy-strength-screen/runtime-artifacts/C128",
        "P": ROOT
        / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/P-E4",
        "initializer": ROOT
        / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/initializer",
    }
    if len(sys.argv) > 1 and sys.argv[1] == "freeze-only":
        print(json.dumps(selection, indent=2))
        return
    searches, oracles = run_searches(
        cohort,
        artifact_dirs,
        native_path,
        tablebase_path,
    )
    analysis = analyze(cohort, searches, oracles)
    (DATA / "search-records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in searches)
    )
    (DATA / "oracle-records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in oracles)
    )
    (DATA / "root-metrics.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in analysis.pop("root_records")
        )
    )
    (DATA / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(analysis["metrics"], indent=2))


if __name__ == "__main__":
    main()
