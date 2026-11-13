"""Freeze and execute the matched broader-opening P-vs-P coverage pilot.

The module's preparation and analysis paths do not launch games. Game execution
is exposed only through the explicitly registered ``run`` subcommand.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ml.alphazero_lite import run_varied_opponent_pilot_v2 as v2
from ml.alphazero_lite import varied_opponent_pilot as pilot
from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/broader-opening-coverage-pilot"
CONTROL_PATH = ROOT / "docs/data/varied-opponent-coverage-pilot/openings.json"
POOL_PATH = ROOT / "docs/data/seed556-exploratory-arena/screen-0005/openings.jsonl"
INVENTORY_PATH = (
    ROOT / "docs/data/seed556-560-exclusion-inventory/identity-inventory.json"
)
INVENTORY_SOURCE_PATH = (
    ROOT / "docs/data/seed556-560-exclusion-inventory/source-manifest.json"
)
ARTIFACT_ROOT = (
    ROOT / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"
)
TABLEBASE_SHA256 = "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
SOURCE_PATHS = (
    "ml/alphazero_lite/broader_opening_coverage_pilot.py",
    "ml/alphazero_lite/run_varied_opponent_pilot_v2.py",
    "ml/alphazero_lite/varied_opponent_pilot.py",
    "ml/alphazero_lite/run_varied_opponent_pilot.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/exact_root_policy_targets.py",
    "ml/alphazero_lite/exact_root_decision.py",
    "ml/alphazero_lite/native_exact_root_tablebase.py",
    "ml/alphazero_lite/classic_mcts.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/eval_cache.py",
    "ml/alphazero_lite/endgame_tablebase.py",
    "ml/alphazero_lite/opening_cache.py",
    "ml/alphazero_lite/opponent_pool.py",
    "ml/alphazero_lite/search_ablation.py",
    "ml/alphazero_lite/value_transforms.py",
    "ml/alphazero_lite/evaluation_seed_contract.py",
    "ml/alphazero_lite/report_validation.py",
    "ml/alphazero_lite/root_prior_transforms.py",
    "ml/alphazero_lite/policy_prior_localization.py",
    "ml/alphazero_lite/shadow_root_q.py",
    "ml/alphazero_lite/seed551_controlled_policy_strength.py",
)
SEED = 823611
PAIR_CONCURRENCY = 1
IDENTITIES = ("canonical-state", "production-kalah_v3-float32")


def parse_cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus


def live_resource_snapshot(
    *,
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    affinity: set[int] | None = None,
    nice: int | None = None,
) -> dict[str, Any]:
    """Read effective cgroup-v2 limits and process CPU scheduling controls."""
    relative = None
    for line in proc_cgroup.read_text().splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            relative = path.lstrip("/")
            break
    if relative is None:
        raise ValueError("broader_pilot_cgroup_v2_membership_missing")
    cgroup = cgroup_root / relative
    cpu_max = (cgroup / "cpu.max").read_text().split()
    if len(cpu_max) != 2 or cpu_max[0] == "max":
        quota_percent = None
    else:
        quota_percent = int(cpu_max[0]) / int(cpu_max[1]) * 100.0
    cpuset_path = cgroup / "cpuset.cpus.effective"
    if not cpuset_path.is_file():
        cpuset_path = cgroup / "cpuset.cpus"
    actual_affinity = set(os.sched_getaffinity(0)) if affinity is None else affinity
    actual_nice = os.getpriority(os.PRIO_PROCESS, 0) if nice is None else nice
    return {
        "cpu_quota_percent": quota_percent,
        "cpu_weight": int((cgroup / "cpu.weight").read_text().strip()),
        "memory_max_bytes": int((cgroup / "memory.max").read_text().strip()),
        "allowed_cpus": parse_cpu_set(cpuset_path.read_text().strip()),
        "process_affinity": set(actual_affinity),
        "nice": int(actual_nice),
    }


def enforce_live_resources(
    limits: dict[str, Any], *, observed: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Fail closed unless the invocation is already inside its frozen scope."""
    actual = live_resource_snapshot() if observed is None else observed
    expected = {
        "cpu_quota_percent": float(str(limits["cpu_quota"]).rstrip("%")),
        "cpu_weight": int(limits["cpu_weight"]),
        "memory_max_bytes": int(limits["memory_max_bytes"]),
        "allowed_cpus": parse_cpu_set(str(limits["allowed_cpus"])),
        "process_affinity": parse_cpu_set(str(limits["allowed_cpus"])),
        "nice": int(limits["nice"]),
    }
    mismatches = {
        key: {"expected": value, "observed": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"broader_pilot_live_resource_scope_mismatch:{mismatches}")
    return actual


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def identity_pair(state: dict[str, Any]) -> tuple[str, str]:
    return canonical_identity(state), production_float32_identity(state)


def signature(opening: dict[str, Any]) -> tuple[int, int, int]:
    state = opening["state"]
    active = sum(state["player_pits"]) + sum(state["opponent_pits"])
    return len(opening["prefix_moves"]), active, int(state["current_player"])


def select_matched_openings(
    controls: list[dict[str, Any]], pool: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select earliest eligible pool opening for each control in source order."""
    control_ids = [identity_pair(item["state"]) for item in controls]
    if len(set(x[0] for x in control_ids)) != len(controls):
        raise ValueError("duplicate_control_canonical_identity")
    if len(set(x[1] for x in control_ids)) != len(controls):
        raise ValueError("duplicate_control_float32_identity")
    blocked_canonical = {pair[0] for pair in control_ids}
    blocked_float32 = {pair[1] for pair in control_ids}
    eligible = []
    for row in pool:
        canonical, f32 = identity_pair(row["state"])
        if canonical in blocked_canonical or f32 in blocked_float32:
            continue
        eligible.append((int(row["opening_index"]), canonical, f32, row))
    eligible.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    by_signature: dict[
        tuple[int, int, int], list[tuple[int, str, str, dict[str, Any]]]
    ] = defaultdict(list)
    for item in eligible:
        by_signature[signature(item[3])].append(item)
    selected: list[dict[str, Any]] = []
    matched_controls: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    used_canonical: set[str] = set()
    used_float32: set[str] = set()
    for control_index, control in enumerate(controls):
        match = next(
            (
                row
                for _, canonical, f32, row in by_signature[signature(control)]
                if canonical not in used_canonical and f32 not in used_float32
            ),
            None,
        )
        if match is None:
            unmatched.append(
                {
                    "control_id": str(
                        control.get("opening_id", f"opening-{control_index:04d}")
                    ),
                    "source_opening_index": int(control["source_opening_index"]),
                    "prefix_length": signature(control)[0],
                    "active_stones": signature(control)[1],
                    "side_to_move": signature(control)[2],
                    "canonical_identity": identity_pair(control["state"])[0],
                    "production_float32_identity": identity_pair(control["state"])[1],
                }
            )
            continue
        canonical, f32 = identity_pair(match["state"])
        used_canonical.add(canonical)
        used_float32.add(f32)
        matched_controls.append(control)
        selected.append(match)
    audit = {
        "rule": (
            "Visit controls in #580 manifest order. For each control, choose the eligible public-pool row with the lowest opening_index, then canonical identity, that exactly matches prefix length, active stones, and current_player. Exclude any candidate whose canonical or production kalah_v3 float32 identity appears among controls. Consume each treatment opening at most once under both identity schemes. No criteria are relaxed."
        ),
        "pool_count": len(pool),
        "control_count": len(controls),
        "matched_count": len(selected),
        "unmatched_count": len(unmatched),
        "unmatched_controls": unmatched,
        "control_canonical_identity_count": len(blocked_canonical),
        "control_production_float32_identity_count": len(blocked_float32),
        "treatment_canonical_identity_count": len(used_canonical),
        "treatment_production_float32_identity_count": len(used_float32),
        "identity_disjointness": {
            "canonical": not bool(blocked_canonical & used_canonical),
            "production_kalah_v3_float32": not bool(blocked_float32 & used_float32),
        },
        "identity_set_sha256": {
            "control_canonical": digest_bytes(
                "\n".join(sorted(blocked_canonical)).encode()
            ),
            "control_production_kalah_v3_float32": digest_bytes(
                "\n".join(sorted(blocked_float32)).encode()
            ),
            "treatment_canonical": digest_bytes(
                "\n".join(sorted(used_canonical)).encode()
            ),
            "treatment_production_kalah_v3_float32": digest_bytes(
                "\n".join(sorted(used_float32)).encode()
            ),
        },
        "signature_counts": {
            "controls": {
                json.dumps(k): v
                for k, v in sorted(Counter(map(signature, controls)).items())
            },
            "eligible_pool": {
                json.dumps(k): len(v) for k, v in sorted(by_signature.items())
            },
        },
    }
    return matched_controls, selected, audit


def load_inventory(
    path: Path = INVENTORY_PATH, *, expected_sha256: str | None = None
) -> dict[str, set[str]]:
    actual_sha256 = sha(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("broader_pilot_novelty_inventory_hash_mismatch")
    inventory: dict[str, set[str]] = {"canonical": set(), "kalah_v3_float32": set()}
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row["scheme"] in inventory:
            inventory[row["scheme"]].add(row["identity"])
    return inventory


def build() -> dict[str, Any]:
    controls_source = json.loads(CONTROL_PATH.read_text())
    pool = [json.loads(line) for line in POOL_PATH.read_text().splitlines() if line]
    controls, treatment, matching = select_matched_openings(controls_source, pool)
    if len(treatment) != 62 or len(matching["unmatched_controls"]) != 2:
        raise ValueError("matched_opening_count_drift")
    assignments: list[dict[str, Any]] = []
    rng = random.Random(SEED)
    control_rows = []
    treatment_rows = []
    for index, (control, treatment_row) in enumerate(
        zip(controls, treatment, strict=False)
    ):
        opening_id = f"matched-{index:04d}"
        seed = rng.randrange(0, 2**31)
        for arm, opening, source_index, destination in (
            ("control", control, int(control["source_opening_index"]), control_rows),
            (
                "treatment",
                treatment_row,
                int(treatment_row["opening_index"]),
                treatment_rows,
            ),
        ):
            arm_opening_id = f"{opening_id}-{arm}"
            record = {
                "match_id": opening_id,
                "opening_id": arm_opening_id,
                "source_opening_index": source_index,
                "prefix_moves": opening["prefix_moves"],
                "state": opening["state"],
                "canonical_identity": identity_pair(opening["state"])[0],
                "production_float32_identity": identity_pair(opening["state"])[1],
                "signature": {
                    "prefix_length": signature(opening)[0],
                    "active_stones": signature(opening)[1],
                    "side_to_move": signature(opening)[2],
                },
            }
            destination.append(record)
            pair_id = f"{arm}-{opening_id}"
            for seat in (0, 1):
                assignments.append(
                    {
                        "game_id": f"{pair_id}-P{seat}",
                        "pair_id": pair_id,
                        "match_id": opening_id,
                        "arm": arm,
                        "opening_id": arm_opening_id,
                        "opening_moves": opening["prefix_moves"],
                        "seed": seed,
                        "designated_p_seat": seat,
                        "opponent": "P",
                        "search_workers": 1,
                    }
                )
    validate_pilot_specs(assignments)
    if len(assignments) != 248 or len(control_rows) != 62 or len(treatment_rows) != 62:
        raise ValueError("assignment_count_mismatch")
    for index, (control, treatment_row) in enumerate(
        zip(control_rows, treatment_rows, strict=True)
    ):
        if control["signature"] != treatment_row["signature"]:
            raise ValueError(f"matched_signature_mismatch:{index}")
    components = pilot.verify_components(ARTIFACT_ROOT)
    runtime = pilot.effective_runtime_policy(ARTIFACT_ROOT)
    inventory = load_inventory()
    starts = {scheme: set() for scheme in inventory}
    for opening in control_rows + treatment_rows:
        starts["canonical"].add(opening["canonical_identity"])
        starts["kalah_v3_float32"].add(opening["production_float32_identity"])
    return {
        "schema": "broader_opening_coverage_pilot_freeze_v1",
        "source_commit": "3ba9e71bf0135c2284234bc4e4b2140aabea8100",
        "seed": SEED,
        "design": "62 one-to-one matched public openings in each arm; one P-vs-P seat-swapped pair per arm and opening; identical seed within matched pair; 124 pairs/248 games; retain designated-P rows only.",
        "matching": matching,
        "control_manifest": control_rows,
        "treatment_manifest": treatment_rows,
        "source_hashes": {
            "control_manifest": sha(CONTROL_PATH),
            "public_pool": sha(POOL_PATH),
            "published_exclusion_inventory": sha(INVENTORY_PATH),
            "published_exclusion_source_manifest": sha(INVENTORY_SOURCE_PATH),
        },
        "exclusion": {
            "supplied_opening_positions": {
                key: sorted(value) for key, value in starts.items()
            },
            "supplied_opening_union_counts": {
                key: len(value) for key, value in starts.items()
            },
            "published_inventory_identity_counts": {
                key: len(value) for key, value in inventory.items()
            },
            "coverage_rule": "For each identity scheme independently, remove any retained designated-P row whose identity appears among the union of all 124 supplied control and treatment opening positions before distinct-above-32 counting.",
            "published_inventory_scope": "verified #556-#560 only; report overlap/novelty separately and do not treat this inventory as a full historical exclusion census",
        },
        "assignments": assignments,
        "artifact_components": components,
        "runtime_assets": {
            name: {"sha256": asset["sha256"]}
            for name, asset in runtime["runtime_assets"].items()
        },
        "execution_source_hashes": {
            relative: sha(ROOT / relative) for relative in SOURCE_PATHS
        },
        "search_settings": pilot.SEARCH_SETTINGS,
        "target_settings": pilot.TARGET_SETTINGS,
        "runtime_policy": "Artifact sidecars: PUCT/exact-root hybrid, exact-root threshold 16, final-score-margin objective, highest legal prior then lowest move-index tie rule, exact-leaf solving disabled.",
        "runtime_limits": {
            "pair_concurrency": PAIR_CONCURRENCY,
            "workers_per_game": 1,
            "cpu_quota": "1800%",
            "cpu_weight": "100",
            "memory_max_bytes": 38654705664,
            "allowed_cpus": "14-31",
            "nice": 19,
            "torch_intraop_threads": 1,
            "torch_interop_threads": 1,
        },
        "metric": {
            "name": "distinct retained designated-P positions with more than 32 active stones per CPU-hour",
            "identities": list(IDENTITIES),
            "supplied_opening_exclusion": "exclude union of all supplied openings under the identity being counted, before distinct counts",
            "cpu_denominator": "sum of pair CPU including evaluator/model setup and reaped native-probe CPU; capped/incomplete games contribute CPU and zero retained rows",
        },
        "decision_rule": "For the 62 matchable openings only, propose a larger data-generation run only if treatment yield is at least 1.15 times control yield under both canonical-state and production kalah_v3 float32 identities. Zero control yield or unreconciled CPU/attempt accounting is inconclusive. The 24-active-stone unmatched control cannot contribute >32-stone positions; the 33-active-stone unmatched control could, and is outside this 62-opening estimand.",
        "resume_contract": "Reuse the validated v2 pair-owned warmed probe worker and atomic checkpoint/attempt reconciliation. Resume validates existing commits and attempt records, skips verified pairs, preserves dispatch/start evidence, and stops on unreconciled started attempts.",
        "commands": {
            "preflight": "/home/alex/azlite-venv/bin/python -m ml.alphazero_lite.broader_opening_coverage_pilot preflight --freeze docs/data/broader-opening-coverage-pilot/freeze.json --expected-freeze-sha256 $PILOT_FREEZE_SHA256 --artifacts docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts --tablebase .tmp/broader-opening-coverage/kalah_v1_21.kvtb",
            "launch": 'PILOT_FREEZE_SHA256=<externally-reviewed-digest> AZLITE_KALAH_V1_TABLEBASE=.tmp/broader-opening-coverage/kalah_v1_21.kvtb systemd-run --user --scope --quiet --property=CPUQuota=1800% --property=CPUWeight=100 --property=MemoryMax=38654705664 --property=AllowedCPUs=14-31 --nice=19 -- taskset -c 14-31 /home/alex/azlite-venv/bin/python -m ml.alphazero_lite.broader_opening_coverage_pilot run --freeze docs/data/broader-opening-coverage-pilot/freeze.json --expected-freeze-sha256 "$PILOT_FREEZE_SHA256" --checkpoint-dir docs/data/broader-opening-coverage-pilot/pairs',
            "resume": 'PILOT_FREEZE_SHA256=<externally-reviewed-digest> AZLITE_KALAH_V1_TABLEBASE=.tmp/broader-opening-coverage/kalah_v1_21.kvtb systemd-run --user --scope --quiet --property=CPUQuota=1800% --property=CPUWeight=100 --property=MemoryMax=38654705664 --property=AllowedCPUs=14-31 --nice=19 -- taskset -c 14-31 /home/alex/azlite-venv/bin/python -m ml.alphazero_lite.broader_opening_coverage_pilot run --freeze docs/data/broader-opening-coverage-pilot/freeze.json --expected-freeze-sha256 "$PILOT_FREEZE_SHA256" --checkpoint-dir docs/data/broader-opening-coverage-pilot/pairs --resume; preserve all existing checkpoints and attempts, and retain the frozen resource/concurrency settings.',
            "analysis": "python -m ml.alphazero_lite.broader_opening_coverage_pilot analyze --freeze docs/data/broader-opening-coverage-pilot/freeze.json --checkpoint-dir docs/data/broader-opening-coverage-pilot/pairs --attempt-dir docs/data/broader-opening-coverage-pilot/attempts",
        },
        "freeze_sha256": "",
    }


def write_freeze() -> str:
    manifest = build()
    manifest["freeze_sha256"] = digest_bytes(
        canonical_json({k: v for k, v in manifest.items() if k != "freeze_sha256"})
    )
    DATA.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("freeze.json", manifest),
        ("game-assignments.json", manifest["assignments"]),
        ("control-openings.json", manifest["control_manifest"]),
        ("treatment-openings.json", manifest["treatment_manifest"]),
    ):
        (DATA / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return manifest["freeze_sha256"]


def load_freeze(path: Path, expected_sha256: str) -> dict[str, Any]:
    freeze = json.loads(path.read_text())
    body = {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    actual = digest_bytes(canonical_json(body))
    if actual != freeze.get("freeze_sha256") or actual != expected_sha256:
        raise ValueError("broader_pilot_freeze_digest_mismatch")
    if freeze.get("schema") != "broader_opening_coverage_pilot_freeze_v1":
        raise ValueError("broader_pilot_freeze_schema_mismatch")
    for relative, expected in freeze["execution_source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"broader_pilot_source_hash_mismatch:{relative}")
    return freeze


def jobs_from_freeze(freeze: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in freeze["assignments"]:
        grouped[spec["pair_id"]].append(spec)
    jobs = [{"game_specs": rows} for _, rows in sorted(grouped.items())]
    if len(jobs) != 124 or sum(len(job["game_specs"]) for job in jobs) != 248:
        raise ValueError("broader_pilot_jobs_count_mismatch")
    validate_pilot_specs([spec for job in jobs for spec in job["game_specs"]])
    return jobs


def validate_pilot_specs(specs: list[dict[str, Any]]) -> None:
    """Validate P-vs-P assignments without changing the shared #580 validator."""
    if any(spec.get("opponent") != "P" for spec in specs):
        raise ValueError("broader_pilot_opponent_must_be_P")
    normalized = [
        {
            **spec,
            "opponent": "P" if spec["arm"] == "control" else "initializer",
        }
        for spec in specs
    ]
    pilot.validate_game_specs(normalized)


def verify_pilot_pair_result(job: dict[str, Any], result: dict[str, Any]) -> None:
    """Replay and reconcile a P-vs-P pair under the unmodified shared verifier."""
    specs = job.get("game_specs", [])
    games = result.get("games")
    validate_pilot_specs(specs)
    if len(specs) != 2 or not isinstance(games, list) or len(games) != 2:
        raise ValueError("pair_checkpoint_game_count_mismatch")
    expected = {str(spec["game_id"]): spec for spec in specs}
    observed: dict[str, dict[str, Any]] = {}
    adapted_games = []
    for game in games:
        game_copy = json.loads(json.dumps(game))
        spec = game_copy.get("spec")
        game_id = str(spec.get("game_id", "")) if isinstance(spec, dict) else ""
        if game_id in observed or game_id not in expected:
            raise ValueError("pair_checkpoint_game_assignment_mismatch")
        if spec != expected[game_id]:
            raise ValueError("pair_checkpoint_game_assignment_mismatch")
        observed[game_id] = spec
        if spec["arm"] == "treatment":
            # The shared verifier only knows historical opponent labels. Adapt
            # verifier copies after exact P assignment equality; persisted rows
            # and checkpoints continue to record opponent=P.
            spec["opponent"] = "initializer"
            for row in game_copy.get("rows", []):
                row["opponent"] = "initializer"
        adapted_games.append(game_copy)
    if observed.keys() != expected.keys():
        raise ValueError("pair_checkpoint_game_assignment_mismatch")
    adapted_job = {
        "game_specs": [
            {
                **spec,
                "opponent": "P" if spec["arm"] == "control" else "initializer",
            }
            for spec in specs
        ]
    }
    adapted_result = {**result, "games": adapted_games}
    pilot.verify_pair_result(adapted_job, adapted_result)


def load_pilot_pair_checkpoints(
    checkpoint_dir: Path, jobs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {str(job["game_specs"][0]["pair_id"]): job for job in jobs}
    if len(expected) != len(jobs):
        raise ValueError("pair_assignment_duplicate_id")
    completed = []
    for path in (
        sorted(checkpoint_dir.glob("*.json")) if checkpoint_dir.exists() else []
    ):
        payload = json.loads(path.read_text())
        if payload.get("schema") != "varied_opponent_pair_checkpoint_v1":
            raise ValueError("pair_checkpoint_schema_mismatch")
        job, result = payload.get("assignment"), payload.get("result")
        if not isinstance(job, dict) or not isinstance(result, dict):
            raise ValueError("pair_checkpoint_payload_invalid")
        producing_attempt = payload.get("producing_attempt")
        if not isinstance(producing_attempt, int) or producing_attempt < 1:
            raise ValueError("pair_checkpoint_attempt_number_invalid")
        pair_id = str(job.get("game_specs", [{}])[0].get("pair_id", ""))
        if pair_id not in expected or job != expected[pair_id]:
            raise ValueError("pair_checkpoint_assignment_conflict")
        if path != pilot.pair_checkpoint_path(checkpoint_dir, pair_id):
            raise ValueError("pair_checkpoint_path_mismatch")
        verify_pilot_pair_result(job, result)
        completed.append(result)
    done = {str(result["pair_id"]) for result in completed}
    return completed, [job for pair_id, job in expected.items() if pair_id not in done]


def commit_pilot_pair(
    checkpoint_dir: Path,
    job: dict[str, Any],
    result: dict[str, Any],
    *,
    identity: dict[str, Any],
    attempt_number: int,
) -> Path:
    if not isinstance(attempt_number, int) or attempt_number < 1:
        raise ValueError("pair_checkpoint_attempt_number_invalid")
    verify_pilot_pair_result(job, result)
    if result.get("execution_identity") != identity:
        raise ValueError("pilot_v2_result_identity_mismatch")
    payload = {
        "schema": "varied_opponent_pair_checkpoint_v1",
        "assignment": json.loads(json.dumps(job, sort_keys=True)),
        "producing_attempt": attempt_number,
        "execution_identity": identity,
        "result": result,
    }
    path = pilot.pair_checkpoint_path(
        checkpoint_dir, str(job["game_specs"][0]["pair_id"])
    )
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("pair_checkpoint_conflict")
    else:
        pilot.write_json_atomic(path, payload)
    return path


def play_pair_with_probe(
    job: dict[str, Any],
    evaluators: Any,
    *,
    accounting_start: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Run a P-vs-P matched pair using v2's warmed probe lifecycle."""
    specs = job["game_specs"]
    validate_pilot_specs(specs)
    policy = evaluators.runtime_policy
    assets = policy["runtime_assets"]
    threshold = int(policy["effective_search_policy"]["exact_root_threshold"])
    process_start, child_start = accounting_start or (
        time.process_time_ns(),
        pilot.reaped_child_cpu_ns(),
    )
    games = []
    with v2.warmed_probe(
        assets["native_probe"]["path"], assets["tablebase"]["path"]
    ) as probe:
        exact_root = v2.exact_root_callback(probe, threshold)
        for spec in specs:
            games.append(
                pilot.play_game(
                    spec,
                    p_evaluator=evaluators.p,
                    opponent_evaluator=evaluators.p,
                    search_turn=lambda game, evaluator, rng, ply, retain: (
                        pilot.production_search_turn(
                            game,
                            evaluator,
                            rng,
                            ply,
                            retain,
                            exact_root=exact_root,
                        )
                    ),
                    max_moves=pilot.MAX_MOVES,
                )
            )
    child_cpu = pilot.reaped_child_cpu_ns() - child_start
    if child_cpu < 0:
        raise ValueError("reaped_child_cpu_counter_regressed")
    process_cpu = time.process_time_ns() - process_start
    result = {
        "pair_id": str(specs[0]["pair_id"]),
        "arm": str(specs[0]["arm"]),
        "games": games,
        "pair_process_cpu_time_ns": process_cpu,
        "native_child_cpu_time_ns": child_cpu,
        "pair_cpu_time_ns": process_cpu + child_cpu,
    }
    verify_pilot_pair_result(job, result)
    return result


def _run_pilot_attempt(
    job: dict[str, Any],
    artifacts: str,
    attempt_path: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    pilot.write_json_atomic(
        attempt_path,
        {
            "pair_id": str(job["game_specs"][0]["pair_id"]),
            "status": "started",
            "assignment": job,
            "execution_identity": identity,
        },
    )
    accounting_start = (time.process_time_ns(), pilot.reaped_child_cpu_ns())
    pilot._initialize_pair_process(artifacts)
    if pilot._PROCESS_EVALUATORS is None:
        raise RuntimeError("pair_process_evaluators_not_initialized")
    result = play_pair_with_probe(
        job, pilot._PROCESS_EVALUATORS, accounting_start=accounting_start
    )
    result["execution_identity"] = identity
    return result


def run_checkpointed_pairs(
    jobs: list[dict[str, Any]],
    artifacts: Path,
    checkpoint_dir: Path,
    attempt_dir: Path,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Use one worker process and the established v2 attempt/checkpoint protocol."""
    v2.bind_run_directory(checkpoint_dir, attempt_dir, identity)
    committed, pending = load_pilot_pair_checkpoints(checkpoint_dir, jobs)
    committed_by_id = {str(result["pair_id"]): result for result in committed}
    attempt_dir.mkdir(parents=True, exist_ok=True)
    numbers = pilot.reconcile_pair_attempts(
        attempt_dir, jobs, committed_by_id, checkpoint_dir
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=PAIR_CONCURRENCY, max_tasks_per_child=1
    ) as pool:
        future_jobs = {}
        for job in pending:
            pair_id = str(job["game_specs"][0]["pair_id"])
            number = numbers[pair_id]
            attempt_path = pilot.pair_attempt_path(attempt_dir, pair_id, number)
            pilot.write_json_atomic(
                attempt_path,
                {
                    "pair_id": pair_id,
                    "status": "dispatched",
                    "assignment": job,
                    "execution_identity": identity,
                },
            )
            future_jobs[
                pool.submit(
                    _run_pilot_attempt, job, str(artifacts), attempt_path, identity
                )
            ] = (job, attempt_path, number)
        for future in concurrent.futures.as_completed(future_jobs):
            job, attempt_path, number = future_jobs[future]
            result = future.result()
            checkpoint = commit_pilot_pair(
                checkpoint_dir,
                job,
                result,
                identity=identity,
                attempt_number=number,
            )
            pilot.write_json_atomic(
                attempt_path,
                {
                    "pair_id": result["pair_id"],
                    "status": "completed",
                    "assignment": job,
                    "execution_identity": identity,
                    "checkpoint_sha256": pilot.sha256(checkpoint),
                    "pair_cpu_time_ns": result["pair_cpu_time_ns"],
                },
            )
            committed.append(result)
    return committed


def coverage_identity_set(
    rows: list[dict[str, Any]], scheme: str, supplied_opening_ids: set[str]
) -> set[str]:
    """Filter to >32-stone rows and remove supplied starts for one identity."""
    if scheme not in {"canonical", "kalah_v3_float32"}:
        raise ValueError("unknown_coverage_identity_scheme")
    found: set[str] = set()
    for row in rows:
        state = row["canonical_state"]
        identity = (
            canonical_identity(state)
            if scheme == "canonical"
            else production_float32_identity(state)
        )
        if int(row["stone_count"]) > 32 and identity not in supplied_opening_ids:
            found.add(identity)
    return found


def preflight(
    freeze: dict[str, Any], tablebase: Path, artifacts: Path
) -> dict[str, Any]:
    if sha(tablebase) != TABLEBASE_SHA256:
        raise ValueError("broader_pilot_tablebase_hash_mismatch")
    assets = pilot.verify_components(artifacts)
    runtime = pilot.effective_runtime_policy(artifacts)
    if runtime["runtime_assets"]["tablebase"]["sha256"] != TABLEBASE_SHA256:
        raise ValueError("broader_pilot_runtime_tablebase_hash_mismatch")
    observed_assets = {
        name: {"sha256": asset["sha256"]}
        for name, asset in runtime["runtime_assets"].items()
    }
    if observed_assets != freeze["runtime_assets"]:
        raise ValueError("broader_pilot_runtime_asset_binding_mismatch")
    jobs = jobs_from_freeze(freeze)
    return {
        "schema": "broader_opening_coverage_no_games_preflight_v1",
        "passed": True,
        "games_launched": 0,
        "pairs": len(jobs),
        "games": sum(len(job["game_specs"]) for job in jobs),
        "pair_concurrency": PAIR_CONCURRENCY,
        "search_workers_per_game": 1,
        "freeze_sha256": freeze["freeze_sha256"],
        "components": assets,
        "tablebase_sha256": sha(tablebase),
        "runtime_assets": runtime["runtime_assets"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    for command in ("preflight", "run", "analyze"):
        child = sub.add_parser(command)
        child.add_argument("--freeze", type=Path, required=True)
        child.add_argument("--expected-freeze-sha256", required=True)
        if command == "preflight":
            child.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
            child.add_argument("--tablebase", type=Path, required=True)
        elif command == "run":
            child.add_argument("--checkpoint-dir", type=Path, required=True)
            child.add_argument("--resume", action="store_true")
            child.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
        else:
            child.add_argument("--checkpoint-dir", type=Path, required=True)
            child.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(write_freeze())
        return
    freeze = load_freeze(args.freeze, args.expected_freeze_sha256)
    if args.command == "run":
        enforce_live_resources(freeze["runtime_limits"])
    if args.command == "preflight":
        result = preflight(freeze, args.tablebase, args.artifacts)
        (DATA / "preflight.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "run":
        jobs = jobs_from_freeze(freeze)
        pilot.verify_components(args.artifacts)
        runtime = pilot.effective_runtime_policy(args.artifacts)
        observed_assets = {
            name: {"sha256": asset["sha256"]}
            for name, asset in runtime["runtime_assets"].items()
        }
        if observed_assets != freeze["runtime_assets"]:
            raise ValueError("broader_pilot_runtime_asset_binding_mismatch")
        identity = {
            "schema": "broader_opening_pilot_v2_run_identity_v1",
            "freeze_sha256": freeze["freeze_sha256"],
            "runner_source": "ml/alphazero_lite/run_varied_opponent_pilot_v2.py",
            "runner_sha256": freeze["execution_source_hashes"][
                "ml/alphazero_lite/run_varied_opponent_pilot_v2.py"
            ],
            "probe_lifetime": v2.PAIR_LIFETIME,
            "checkpoint_schema": "varied_opponent_pair_checkpoint_v1",
        }
        results = run_checkpointed_pairs(
            jobs,
            args.artifacts,
            args.checkpoint_dir,
            args.checkpoint_dir.parent / "attempts",
            identity,
        )
        print(
            json.dumps(
                pilot.summarize_pair_accounting(results), indent=2, sort_keys=True
            )
        )
        return
    result = analyze(freeze, args.checkpoint_dir, args.attempt_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def analyze(
    freeze: dict[str, Any], checkpoint_dir: Path, attempt_dir: Path
) -> dict[str, Any]:
    """Compute the frozen two-identity yield and attempt/CPU reconciliation."""
    jobs = jobs_from_freeze(freeze)
    identity = {
        "schema": "broader_opening_pilot_v2_run_identity_v1",
        "freeze_sha256": freeze["freeze_sha256"],
        "runner_source": "ml/alphazero_lite/run_varied_opponent_pilot_v2.py",
        "runner_sha256": freeze["execution_source_hashes"][
            "ml/alphazero_lite/run_varied_opponent_pilot_v2.py"
        ],
        "probe_lifetime": v2.PAIR_LIFETIME,
        "checkpoint_schema": "varied_opponent_pair_checkpoint_v1",
    }
    v2.bind_run_directory(checkpoint_dir, attempt_dir, identity)
    committed, pending = load_pilot_pair_checkpoints(checkpoint_dir, jobs)
    if pending:
        raise ValueError("broader_pilot_checkpoints_incomplete")
    by_id = {str(pair["pair_id"]): pair for pair in committed}
    pilot.reconcile_pair_attempts(attempt_dir, jobs, by_id, checkpoint_dir)
    supplied = {
        "canonical": {
            row["canonical_identity"]
            for row in freeze["control_manifest"] + freeze["treatment_manifest"]
        },
        "kalah_v3_float32": {
            row["production_float32_identity"]
            for row in freeze["control_manifest"] + freeze["treatment_manifest"]
        },
    }
    inventory_hash = freeze["source_hashes"]["published_exclusion_inventory"]
    inventory = load_inventory(expected_sha256=inventory_hash)
    totals = pilot.summarize_pair_accounting(committed)
    arms = {}
    for arm in ("control", "treatment"):
        pairs = [pair for pair in committed if pair["arm"] == arm]
        cpu_ns = sum(int(pair["pair_cpu_time_ns"]) for pair in pairs)
        rows = [row for pair in pairs for game in pair["games"] for row in game["rows"]]
        arm_data: dict[str, Any] = {
            "pairs": len(pairs),
            "cpu_time_ns": cpu_ns,
            "cpu_hours": cpu_ns / 3_600_000_000_000,
            "accepted_designated_p_rows": len(rows),
        }
        for scheme, key in (
            ("canonical", "canonical_identity"),
            ("kalah_v3_float32", "production_float32_identity"),
        ):
            eligible = [row for row in rows if int(row["stone_count"]) > 32]
            eligible_ids = coverage_identity_set(rows, scheme, supplied[scheme])
            novel_ids = eligible_ids - inventory[scheme]
            arm_data[scheme] = {
                "eligible_rows_above_32_before_opening_exclusion": len(eligible),
                "opening_identity_rows_excluded": sum(
                    (
                        canonical_identity(row["canonical_state"])
                        if scheme == "canonical"
                        else production_float32_identity(row["canonical_state"])
                    )
                    in supplied[scheme]
                    for row in eligible
                ),
                "distinct_after_opening_exclusion": len(eligible_ids),
                "per_cpu_hour": len(eligible_ids) / arm_data["cpu_hours"]
                if arm_data["cpu_hours"]
                else None,
                "published_556_560_inventory_overlap": len(
                    eligible_ids & inventory[scheme]
                ),
                "novel_to_published_556_560_inventory": len(novel_ids),
                "inventory_scope": "verified #556-#560 only",
                "identity_key_field": key,
            }
        arms[arm] = arm_data
    ratios = {}
    for scheme in ("canonical", "kalah_v3_float32"):
        control = arms["control"][scheme]["per_cpu_hour"]
        treatment = arms["treatment"][scheme]["per_cpu_hour"]
        ratios[scheme] = treatment / control if control else None
    return {
        "schema": "broader_opening_coverage_analysis_v1",
        "freeze_sha256": freeze["freeze_sha256"],
        "scope": "62 matchable openings per arm only",
        "published_inventory_sha256": inventory_hash,
        "pairs_reconciled": len(committed),
        "attempt_accounting": "reconciled by v2 runner",
        "accounting": totals,
        "arms": arms,
        "treatment_control_yield_ratio": ratios,
        "larger_run_gate_met": all(
            value is not None and value >= 1.15 for value in ratios.values()
        ),
        "decision_rule": freeze["decision_rule"],
    }


if __name__ == "__main__":
    main()
