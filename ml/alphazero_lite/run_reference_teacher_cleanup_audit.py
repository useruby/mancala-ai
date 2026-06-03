"""
Reference/Teacher Cleanup Audit — consolidated governance report after
non-opening family exhaustion.

Key design constraints (PR #68 context):
  - No training, no arena, no promotion, no replay artifacts.
  - Active fixture is NOT mutated.
  - All proposed patch entries carry do_not_auto_apply = True.
  - The audit answers: is the forensic reference set clean enough to
    support further training experiments, or do we need a reviewed
    reference patch / exclusion pass first?
"""

import json
import os
import sys
from collections import defaultdict

OUTPUT_DIR = "/tmp/azlite_reference_teacher_cleanup"
REFERENCE_PATH = "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
SUITE_PATH = "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Patch artifact paths (from PR #53, #56, #59, #62, #65, #67, #68 follow-ups)
PATCH_ARTIFACTS = {
    "early_extra_turn": "/tmp/azlite_early_extra_turn_reference_adjudication/early_extra_turn_reference_review_patch_v1.json",
    "capture_available": "/tmp/azlite_capture_available_reference_adjudication/capture_available_reference_review_patch_v1.json",
    "high_imbalance": "/tmp/azlite_high_imbalance_reference_adjudication/high_imbalance_reference_review_patch_v1.json",
    "high_value_swing": "/tmp/azlite_high_value_swing_reference_adjudication/high_value_swing_reference_review_patch_v1.json",
    "starvation_pressure": "/tmp/azlite_starvation_pressure_reference_adjudication/starvation_pressure_reference_review_patch_v1.json",
    "sparse_endgame": "/tmp/azlite_sparse_endgame_tablebase_patch/sparse_endgame_tablebase_reference_patch_v1.json",
    "incumbent_proxy_residual": "/tmp/azlite_incumbent_proxy_residual_reference_adjudication/incumbent_proxy_residual_reference_review_patch_v1.json",
}


def load_json(path):
    full = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full):
        print(f"  WARNING: {full} not found", file=sys.stderr)
        return None
    with open(full) as f:
        return json.load(f)


def load_references():
    return load_json(REFERENCE_PATH)


def load_suite():
    return load_json(SUITE_PATH)


def row_id_to_family(row_id):
    known_families = [
        "opening_plies_1_8",
        "high_value_swing",
        "capture_available",
        "high_imbalance",
        "starvation_pressure",
        "sparse_endgame",
        "early_extra_turn",
        "incumbent_proxy_disagreement",
    ]
    for fam in known_families:
        if row_id.startswith(fam):
            return fam
    return "unknown"


# Hard-coded classification ledger based on published doc reports.
# Each entry: row_id -> { adjudicated_decision, recommended_use, source_report }
def _build_known_row_classifications():
    c = {}

    def _add(rows, decision, use, source):
        for r in rows:
            c[r] = {
                "adjudicated_decision": decision,
                "recommended_use": use,
                "source": source,
            }

    # === high_value_swing (PR #53 adjudication report) ===
    src_hvs = "docs/alphazero-lite-high-value-swing-reference-adjudication-results.md"
    _add(
        ["high_value_swing-001"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        ["high_value_swing-003"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        ["high_value_swing-007"], "reference_unstable", "exclude_from_training", src_hvs
    )
    _add(
        ["high_value_swing-008"], "reference_unstable", "exclude_from_training", src_hvs
    )
    _add(
        ["high_value_swing-009"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        ["high_value_swing-013"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        ["high_value_swing-015"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        ["high_value_swing-016"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        ["high_value_swing-018"], "reference_unstable", "exclude_from_training", src_hvs
    )
    _add(
        ["high_value_swing-020"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        ["high_value_swing-021"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        ["high_value_swing-023"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        ["high_value_swing-024"], "reference_unstable", "exclude_from_training", src_hvs
    )
    _add(
        ["high_value_swing-025"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hvs,
    )
    _add(
        [
            "high_value_swing-010",
            "high_value_swing-017",
            "high_value_swing-026",
            "high_value_swing-027",
        ],
        "preservation_control",
        "preservation_control",
        src_hvs,
    )
    _add(
        ["high_value_swing-011", "high_value_swing-022"],
        "holdout_context",
        "holdout_context",
        src_hvs,
    )

    # === capture_available (PR #59 adjudication report) ===
    src_cap = "docs/alphazero-lite-capture-available-reference-adjudication-results.md"
    _add(
        ["capture_available-001"],
        "reference_unstable",
        "exclude_from_training",
        src_cap,
    )
    _add(
        ["capture_available-009"],
        "reference_should_flip",
        "requires_reviewed_reference_patch",
        src_cap,
    )
    _add(
        ["capture_available-012"],
        "reference_unstable",
        "exclude_from_training",
        src_cap,
    )
    _add(
        ["capture_available-013"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_cap,
    )
    _add(
        ["capture_available-016"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_cap,
    )
    _add(
        ["capture_available-018"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_cap,
    )
    _add(
        ["capture_available-022"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_cap,
    )
    _add(
        ["capture_available-023"],
        "reference_should_flip",
        "requires_reviewed_reference_patch",
        src_cap,
    )
    _add(
        ["capture_available-024"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_cap,
    )
    _add(
        ["capture_available-025"],
        "reference_should_flip",
        "requires_reviewed_reference_patch",
        src_cap,
    )
    _add(
        [
            "capture_available-002",
            "capture_available-003",
            "capture_available-007",
            "capture_available-008",
        ],
        "corrected_guard",
        "validation_context",
        src_cap,
    )
    _add(
        ["capture_available-010", "capture_available-020", "capture_available-021"],
        "preservation_control",
        "preservation_control",
        src_cap,
    )
    _add(
        ["capture_available-011"],
        "preservation_control",
        "preservation_control",
        src_cap,
    )
    _add(
        ["capture_available-015", "capture_available-027"],
        "holdout_context",
        "holdout_context",
        src_cap,
    )
    _add(
        ["capture_available-019"],
        "value_head_miscalibration",
        "exclude_diagnostic",
        src_cap,
    )
    _add(
        ["capture_available-017"],
        "puct_child_search_mismatch",
        "exclude_diagnostic",
        src_cap,
    )

    # === high_imbalance (PR #56 adjudication report) ===
    src_hi = "docs/alphazero-lite-high-imbalance-reference-adjudication-results.md"
    _add(
        ["high_imbalance-009"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-014"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-016"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-018"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(["high_imbalance-019"], "reference_unstable", "exclude_from_training", src_hi)
    _add(
        ["high_imbalance-020"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-021"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-022"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-023"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-024"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-025"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-026"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        ["high_imbalance-027"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_hi,
    )
    _add(
        [
            "high_imbalance-001",
            "high_imbalance-007",
            "high_imbalance-012",
            "high_imbalance-017",
        ],
        "preservation_control",
        "preservation_control",
        src_hi,
    )
    _add(
        ["high_imbalance-002", "high_imbalance-008"],
        "holdout_context",
        "holdout_context",
        src_hi,
    )

    # === starvation_pressure (PR #62 adjudication report) ===
    src_sp = "docs/alphazero-lite-starvation-pressure-reference-adjudication-results.md"
    _add(
        ["starvation_pressure-012"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_sp,
    )
    _add(
        ["starvation_pressure-022"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_sp,
    )
    _add(
        ["starvation_pressure-023"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_sp,
    )
    _add(
        ["starvation_pressure-024"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_sp,
    )
    _add(
        ["starvation_pressure-025"],
        "reference_unstable",
        "exclude_from_training",
        src_sp,
    )
    _add(
        ["starvation_pressure-026"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_sp,
    )
    _add(
        ["starvation_pressure-027"],
        "reference_unstable",
        "exclude_from_training",
        src_sp,
    )
    _add(
        [
            "starvation_pressure-003",
            "starvation_pressure-013",
            "starvation_pressure-014",
            "starvation_pressure-021",
        ],
        "preservation_control",
        "preservation_control",
        src_sp,
    )
    _add(
        ["starvation_pressure-001", "starvation_pressure-015"],
        "holdout_context",
        "holdout_context",
        src_sp,
    )

    # === sparse_endgame (PR #67 exact tablebase targetability report) ===
    src_se = (
        "docs/alphazero-lite-sparse-endgame-exact-tablebase-targetability-results.md"
    )
    _add(
        [
            "sparse_endgame-001",
            "sparse_endgame-002",
            "sparse_endgame-003",
            "sparse_endgame-007",
            "sparse_endgame-009",
            "sparse_endgame-010",
            "sparse_endgame-011",
            "sparse_endgame-013",
            "sparse_endgame-016",
            "sparse_endgame-018",
            "sparse_endgame-020",
            "sparse_endgame-021",
            "sparse_endgame-024",
            "sparse_endgame-025",
            "sparse_endgame-026",
            "sparse_endgame-027",
        ],
        "forced_all_moves_equivalent",
        "exclude_from_training",
        src_se,
    )
    _add(
        [
            "sparse_endgame-008",
            "sparse_endgame-012",
            "sparse_endgame-014",
            "sparse_endgame-019",
        ],
        "exact_value_only_tie",
        "value_only_candidate",
        src_se,
    )
    _add(
        [
            "sparse_endgame-015",
            "sparse_endgame-017",
            "sparse_endgame-022",
            "sparse_endgame-023",
        ],
        "exact_unique_policy_target",
        "holdout_context",
        src_se,
    )

    # === early_extra_turn (PR #68 follow-up adjudication report) ===
    src_eet = "docs/alphazero-lite-early-extra-turn-reference-adjudication-results.md"
    _add(
        ["early_extra_turn-012"],
        "still_inconclusive",
        "exclude_pending_more_evidence",
        src_eet,
    )
    _add(
        ["early_extra_turn-013"],
        "reference_should_flip",
        "requires_reviewed_reference_patch",
        src_eet,
    )
    _add(
        ["early_extra_turn-014"], "reference_unstable", "exclude_from_training", src_eet
    )
    _add(
        ["early_extra_turn-016"], "reference_unstable", "exclude_from_training", src_eet
    )
    _add(
        ["early_extra_turn-018"],
        "reference_should_flip",
        "requires_reviewed_reference_patch",
        src_eet,
    )
    _add(
        [
            "early_extra_turn-002",
            "early_extra_turn-007",
            "early_extra_turn-009",
            "early_extra_turn-011",
        ],
        "preservation_control",
        "preservation_control",
        src_eet,
    )
    _add(["early_extra_turn-017"], "holdout_context", "holdout_context", src_eet)
    _add(
        ["early_extra_turn-025"],
        "value_head_miscalibration",
        "exclude_diagnostic",
        src_eet,
    )

    # === incumbent_proxy_disagreement (teacher bucket split report) ===
    src_ipd = "docs/alphazero-lite-incumbent-proxy-disagreement-teacher-bucket-split-results.md"
    _add(
        [
            "incumbent_proxy_disagreement-008",
            "incumbent_proxy_disagreement-022",
            "incumbent_proxy_disagreement-025",
        ],
        "classic_reference_confirmed",
        "preservation_control",
        src_ipd,
    )
    _add(
        [
            "incumbent_proxy_disagreement-014",
            "incumbent_proxy_disagreement-024",
            "incumbent_proxy_disagreement-035",
        ],
        "classic_reference_confirmed",
        "target_candidate",
        src_ipd,
    )
    _add(
        [
            "incumbent_proxy_disagreement-007",
            "incumbent_proxy_disagreement-009",
            "incumbent_proxy_disagreement-021",
            "incumbent_proxy_disagreement-026",
            "incumbent_proxy_disagreement-028",
            "incumbent_proxy_disagreement-032",
            "incumbent_proxy_disagreement-033",
        ],
        "puct_reference_preferred",
        "reference_policy_decision_only",
        src_ipd,
    )
    _add(
        [
            "incumbent_proxy_disagreement-003",
            "incumbent_proxy_disagreement-010",
            "incumbent_proxy_disagreement-012",
            "incumbent_proxy_disagreement-018",
            "incumbent_proxy_disagreement-020",
            "incumbent_proxy_disagreement-023",
            "incumbent_proxy_disagreement-027",
            "incumbent_proxy_disagreement-029",
        ],
        "excluded_diagnostic",
        "exclude_diagnostic",
        src_ipd,
    )

    # === opening ===
    # All opening rows are excluded from training per PR #68 / v8
    # We'll handle this dynamically based on row family

    return c


def load_patch_artifacts():
    patches = []
    seen_row_ids = set()
    for family, path in PATCH_ARTIFACTS.items():
        data = load_json(path)
        if data is None:
            continue
        rows = data.get("rows", data.get("entries", []))
        for entry in rows:
            row_id = entry["row_id"]
            proposed_move = entry.get("proposed_reference_move")
            proposed_unstable = entry.get("proposed_reference_unstable", False)
            patches.append(
                {
                    "row_id": row_id,
                    "family": family,
                    "current_active_reference_move": entry[
                        "current_active_reference_move"
                    ],
                    "proposed_reference_move": proposed_move,
                    "proposed_reference_unstable": proposed_unstable,
                    "observed_reference_moves": entry.get(
                        "observed_reference_moves", []
                    ),
                    "evidence_summary": entry.get("evidence_summary", ""),
                    "do_not_auto_apply": entry.get("do_not_auto_apply", True),
                    "source_artifact": path,
                    "canonical_state_hash": entry.get("canonical_state_hash", ""),
                }
            )
            if row_id in seen_row_ids:
                print(f"  WARNING: duplicate patch entry for {row_id}", file=sys.stderr)
            seen_row_ids.add(row_id)
    return patches


def validate_patches(patches, refs):
    ref_rows = {r["id"]: r for r in refs.get("rows", [])} if refs else {}
    validated = []
    for p in patches:
        row_id = p["row_id"]
        issues = []
        classification = "review_ready"

        ref_row = ref_rows.get(row_id)
        if ref_row is None:
            issues.append("row_not_in_active_fixture")
            classification = "stale_patch_entry"
        else:
            active_move = ref_row["reference_move"]
            if p["current_active_reference_move"] != active_move:
                issues.append(
                    f"active_reference_changed: expected {p['current_active_reference_move']} got {active_move}"
                )
                classification = "stale_patch_entry"

            proposed = p["proposed_reference_move"]
            if proposed is not None and proposed not in [
                c["move"] for c in ref_row.get("child_stats", [])
            ]:
                legal_from_suite = _legal_moves_for_row(row_id)
                if proposed not in legal_from_suite:
                    issues.append(f"proposed_move_{proposed}_not_legal")
                    classification = "insufficient_evidence"

        if (
            p["proposed_reference_move"] is None
            and not p["proposed_reference_unstable"]
        ):
            issues.append("no_proposed_change")
            classification = "insufficient_evidence"

        if p["proposed_reference_move"] is None and p["proposed_reference_unstable"]:
            classification = "review_ready_mark_unstable"

        if (
            p["proposed_reference_move"] is not None
            and not p["proposed_reference_unstable"]
        ):
            classification = "review_ready_reference_flip"

        validated.append(
            {
                **p,
                "validation_status": "valid"
                if classification.startswith("review_ready")
                else issues[0]
                if issues
                else "unknown",
                "patch_classification": classification,
                "notes": "; ".join(issues) if issues else "",
            }
        )
    return validated


def _legal_moves_for_row(row_id):
    suite = load_suite()
    if suite is None:
        return []
    for entry in suite:
        if entry["id"] == row_id:
            return entry.get("legal_moves", [])
    return []


def _classify_family_mechanism(family, row_ledger):
    decisions = [
        r["adjudicated_decision"] for r in row_ledger.values() if r["family"] == family
    ]
    has_value_head = any("miscalibration" in d for d in decisions)
    has_puct_search = any("mismatch" in d for d in decisions)
    has_teacher_split = any(
        "puct_reference" in d or "classic_reference" in d for d in decisions
    )
    has_forced = any("forced_all" in d for d in decisions)

    if has_teacher_split:
        return "teacher_policy_split"
    if has_puct_search and has_value_head:
        return "mixed"
    if has_puct_search:
        return "root_selection"
    if has_value_head:
        return "value_head"
    if has_forced:
        return "policy_prior"
    return "unknown"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Step 1: Load fixtures ===")
    refs = load_references()
    suite = load_suite()
    print(
        f"  References loaded: {len(refs.get('rows', []))} rows" if refs else "  FAILED"
    )
    print(f"  Suite loaded: {len(suite)} rows" if suite else "  FAILED")

    if refs is None or suite is None:
        print("FATAL: Could not load required fixtures", file=sys.stderr)
        sys.exit(1)

    print("\n=== Step 2: Build row-level classification ledger ===")
    known = _build_known_row_classifications()
    ref_rows = {r["id"]: r for r in refs["rows"]}
    row_ledger = {}

    for row_id, ref_row in ref_rows.items():
        family = row_id_to_family(row_id)
        child_stats = ref_row.get("child_stats", [])
        legal_moves = [c["move"] for c in child_stats]

        entry = {
            "row_id": row_id,
            "family": family,
            "active_reference_move": ref_row.get("reference_move"),
            "legal_moves": legal_moves,
            "canonical_state_hash": ref_row.get("canonical_state", ""),
            "reference_unstable": ref_row.get("reference_unstable", False),
            "observed_reference_moves": ref_row.get("observed_reference_moves", []),
            "current_puct_selected_384": None,
            "current_puct_selected_1200": None,
        }

        if row_id in known:
            entry["adjudicated_decision"] = known[row_id]["adjudicated_decision"]
            entry["recommended_use"] = known[row_id]["recommended_use"]
            entry["source_report"] = known[row_id]["source"]
        else:
            if family == "opening_plies_1_8":
                entry["adjudicated_decision"] = "opening_branch_excluded"
                entry["recommended_use"] = "exclude_from_training"
                entry["source_report"] = (
                    "docs/alphazero-lite-corrected-non-opening-failure-family-mining-v8-results.md"
                )
            else:
                entry["adjudicated_decision"] = "unknown"
                entry["recommended_use"] = "review_needed"
                entry["source_report"] = ""

        row_ledger[row_id] = entry

    print(f"  Ledger built: {len(row_ledger)} rows")

    print("\n=== Step 3: Consolidate patch artifacts ===")
    raw_patches = load_patch_artifacts()
    print(f"  Raw patch entries loaded: {len(raw_patches)}")
    validated_patches = validate_patches(raw_patches, refs)

    patch_counts = defaultdict(int)
    for vp in validated_patches:
        patch_counts[vp["patch_classification"]] += 1
    for c, n in sorted(patch_counts.items()):
        print(f"    {c}: {n}")

    print("\n=== Step 4: Family health summary ===")
    families = sorted(set(row_id_to_family(r) for r in ref_rows))
    family_health = {}

    for fam in families:
        fam_rows = {rid: e for rid, e in row_ledger.items() if e["family"] == fam}
        total = len(fam_rows)
        confirmed = sum(
            1
            for e in fam_rows.values()
            if e["adjudicated_decision"]
            in ("classic_reference_confirmed", "preservation_control")
        )
        flip_proposed = sum(
            1
            for e in fam_rows.values()
            if e["adjudicated_decision"] == "reference_should_flip"
        )
        unstable_proposed = sum(
            1
            for e in fam_rows.values()
            if e["adjudicated_decision"] == "reference_unstable"
        )
        inconclusive = sum(
            1
            for e in fam_rows.values()
            if e["adjudicated_decision"] == "still_inconclusive"
        )
        teacher_divergence = sum(
            1
            for e in fam_rows.values()
            if "puct_reference" in e["adjudicated_decision"]
        )
        tablebase_tie = sum(
            1
            for e in fam_rows.values()
            if e["adjudicated_decision"]
            in (
                "forced_all_moves_equivalent",
                "exact_value_only_tie",
                "tablebase_tie_not_conflict",
            )
        )
        eligible_active = sum(
            1
            for e in fam_rows.values()
            if e["recommended_use"] in ("target_candidate", "preservation_control")
        )

        # Simulated post-patch: apply review_ready flips/unstable marks
        patched_rows = dict(fam_rows)
        for vp in validated_patches:
            if vp["family"] == fam and vp["patch_classification"] in (
                "review_ready_reference_flip",
                "review_ready_mark_unstable",
            ):
                rid = vp["row_id"]
                if rid in patched_rows:
                    if vp["patch_classification"] == "review_ready_reference_flip":
                        patched_rows[rid] = dict(patched_rows[rid])
                        patched_rows[rid]["active_reference_move"] = vp[
                            "proposed_reference_move"
                        ]
                        patched_rows[rid]["adjudicated_decision"] = (
                            "active_reference_confirmed"
                        )
                        patched_rows[rid]["recommended_use"] = "target_candidate"
                    elif vp["patch_classification"] == "review_ready_mark_unstable":
                        patched_rows[rid] = dict(patched_rows[rid])
                        patched_rows[rid]["reference_unstable"] = True
                        patched_rows[rid]["adjudicated_decision"] = "reference_unstable"
                        patched_rows[rid]["recommended_use"] = "exclude_from_training"

        eligible_after_patch = sum(
            1
            for e in patched_rows.values()
            if e["recommended_use"] in ("target_candidate", "preservation_control")
        )
        # Risk level
        if eligible_after_patch >= 5:
            risk = "low"
        elif eligible_after_patch >= 3:
            risk = "medium"
        elif eligible_after_patch > 0:
            risk = "high"
        else:
            risk = "unusable"

        family_health[fam] = {
            "total_rows": total,
            "confirmed_rows": confirmed,
            "proposed_flip_rows": flip_proposed,
            "proposed_unstable_rows": unstable_proposed,
            "inconclusive_rows": inconclusive,
            "teacher_divergence_rows": teacher_divergence,
            "tablebase_tie_rows": tablebase_tie,
            "eligible_targets_active": eligible_active,
            "eligible_targets_after_simulated_patch": eligible_after_patch,
            "risk_level": risk,
            "mechanism": _classify_family_mechanism(fam, fam_rows),
            "notes": "",
        }

        print(
            f"  {fam:35s} total={total:3d} confirmed={confirmed:2d} flip={flip_proposed:2d} "
            f"unstable={unstable_proposed:2d} inconclusive={inconclusive:2d} "
            f"eligible_active={eligible_active:2d} eligible_patched={eligible_after_patch:2d} "
            f"risk={risk}"
        )

    print("\n=== Step 5: Simulated post-patch targetability ===")
    targetability = {}
    for fam, health in family_health.items():
        total = health["total_rows"]
        eligible = health["eligible_targets_after_simulated_patch"]
        mechanism = health["mechanism"]

        if fam == "opening_plies_1_8":
            targetability[fam] = {
                "active_targetability": "excluded",
                "simulated_post_patch_targetability": "excluded",
                "target_rows_after_patch": 0,
                "control_rows_after_patch": 0,
                "dominant_mechanism": "policy_prior",
                "next_use": "exclude_from_training",
                "notes": "opening branch closed permanently",
            }
        elif fam == "sparse_endgame":
            targetability[fam] = {
                "active_targetability": "not_trainable",
                "simulated_post_patch_targetability": "too_small_after_patch",
                "target_rows_after_patch": 0,
                "control_rows_after_patch": 4,
                "dominant_mechanism": "policy_prior",
                "next_use": "diagnostic_only",
                "notes": "83% forced/tied rows; 4 unique targets, 3 lack PUCT data; only 4 controls",
            }
        elif fam == "incumbent_proxy_disagreement":
            targetability[fam] = {
                "active_targetability": "teacher_policy_required",
                "simulated_post_patch_targetability": "teacher_policy_required",
                "target_rows_after_patch": 5,
                "control_rows_after_patch": 3,
                "dominant_mechanism": "teacher_policy_split",
                "next_use": "teacher_policy_architecture_needed",
                "notes": "5 classic-teacher targets exist but PUCT teacher disagrees; need explicit teacher-policy design",
            }
        elif fam in ("capture_available", "early_extra_turn"):
            flip_rows = health["proposed_flip_rows"]
            targetability[fam] = {
                "active_targetability": "not_trainable",
                "simulated_post_patch_targetability": "too_small_after_patch",
                "target_rows_after_patch": flip_rows,
                "control_rows_after_patch": health["confirmed_rows"],
                "dominant_mechanism": "mixed",
                "next_use": "reference_patch_before_any_training",
                "notes": f"{flip_rows} flip candidates; after patch only {eligible} total rows (need >=3 policy or >=5 value)",
            }
        elif eligible >= 3:
            targetability[fam] = {
                "active_targetability": "not_trainable",
                "simulated_post_patch_targetability": "trainable_after_patch",
                "target_rows_after_patch": eligible,
                "control_rows_after_patch": 0,
                "dominant_mechanism": mechanism,
                "next_use": "diagnostic_artifact",
                "notes": f"{eligible} eligible rows after patch",
            }
        elif eligible > 0:
            targetability[fam] = {
                "active_targetability": "not_trainable",
                "simulated_post_patch_targetability": "too_small_after_patch",
                "target_rows_after_patch": eligible,
                "control_rows_after_patch": 0,
                "dominant_mechanism": mechanism,
                "next_use": "diagnostic_only",
                "notes": f"only {eligible} eligible rows; below threshold",
            }
        else:
            targetability[fam] = {
                "active_targetability": "not_trainable",
                "simulated_post_patch_targetability": "reference_uncertain",
                "target_rows_after_patch": 0,
                "control_rows_after_patch": 0,
                "dominant_mechanism": mechanism,
                "next_use": "exclude_from_training",
                "notes": "no eligible target rows under active or patched references",
            }

        print(
            f"  {fam:35s} active={targetability[fam]['active_targetability']:30s} post-patch={targetability[fam]['simulated_post_patch_targetability']:30s}"
        )

    print("\n=== Step 6: Global decision ===")
    trainable_after_patch = [
        f
        for f, t in targetability.items()
        if t["simulated_post_patch_targetability"] == "trainable_after_patch"
    ]
    teacher_needed = [
        f
        for f, t in targetability.items()
        if t["simulated_post_patch_targetability"] == "teacher_policy_required"
    ]

    if teacher_needed:
        global_decision = "teacher_policy_architecture_needed"
        decision_reason = (
            "incumbent_proxy_disagreement requires teacher-policy architecture: "
            "5 classic-teacher targets exist but PUCT teacher disagrees on 7 rows. "
            "Reference patches alone cannot resolve the teacher split."
        )
    elif trainable_after_patch:
        global_decision = "one_family_trainable_after_patch"
        decision_reason = (
            f"{trainable_after_patch[0]} becomes trainable after reviewed patch. "
            f"Apply patch, rerun rebaseline, build diagnostic artifact."
        )
    elif all(
        h["eligible_targets_after_simulated_patch"] == 0
        for h in family_health.values()
        if h["total_rows"] > 0 and "opening" not in fam
    ):
        global_decision = "no_trainable_family_after_cleanup"
        decision_reason = (
            "Even after simulated patch application, no non-opening family has enough "
            "clean target rows for training. All 6 corrected non-opening branches are "
            "exhausted under current active and proposed-post-patch references."
        )
    else:
        has_patch_candidates = any(
            vp["patch_classification"].startswith("review_ready")
            for vp in validated_patches
        )
        if has_patch_candidates:
            global_decision = "reference_patch_review_needed"
            decision_reason = (
                "Several review-ready reference flips/unstable markings exist and would "
                "marginally improve the suite, but no family becomes trainable after patch."
            )
        else:
            global_decision = "reference_fixture_too_noisy"
            decision_reason = (
                "Active reference fixture has pervasive unstable/inconclusive rows; "
                "rebuild with stronger teacher criteria."
            )

    print(f"  Global decision: {global_decision}")
    print(f"  Reason: {decision_reason}")

    print("\n=== Step 7: Write output artifacts ===")

    # Summary JSON
    summary = {
        "schema": "azlite_reference_teacher_cleanup_summary_v1",
        "active_reference_path": REFERENCE_PATH,
        "active_suite_path": SUITE_PATH,
        "global_decision": global_decision,
        "decision_reason": decision_reason,
        "family_health": family_health,
        "targetability": targetability,
        "review_ready_patch_count": sum(
            1
            for vp in validated_patches
            if vp["patch_classification"].startswith("review_ready")
        ),
        "stale_patch_count": sum(
            1
            for vp in validated_patches
            if vp["patch_classification"] == "stale_patch_entry"
        ),
        "conflicting_patch_count": sum(
            1
            for vp in validated_patches
            if vp["patch_classification"] == "conflicting_patch_entry"
        ),
        "total_rows_in_fixture": len(ref_rows),
        "total_rows_with_adjudication": len(known),
        "families_analyzed": list(family_health.keys()),
    }
    with open(
        os.path.join(OUTPUT_DIR, "reference_teacher_cleanup_summary.json"), "w"
    ) as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Wrote reference_teacher_cleanup_summary.json")

    # Row ledger JSONL
    ledger_path = os.path.join(OUTPUT_DIR, "reference_teacher_row_ledger.jsonl")
    with open(ledger_path, "w") as f:
        for rid in sorted(row_ledger.keys()):
            f.write(json.dumps(row_ledger[rid], default=str) + "\n")
    print(f"  Wrote reference_teacher_row_ledger.jsonl ({len(row_ledger)} rows)")

    # Proposed patch bundle
    bundle = {
        "schema": "azlite_proposed_reference_patch_bundle_v1",
        "generated_by": "run_reference_teacher_cleanup_audit.py",
        "note": "This bundle aggregates all non-mutating patch proposals from every "
        "family adjudication. Every entry has do_not_auto_apply=true. "
        "Review and apply via a dedicated fixture-update PR.",
        "do_not_auto_apply_global": True,
        "validated_entries": validated_patches,
        "review_ready_entries": [
            vp
            for vp in validated_patches
            if vp["patch_classification"].startswith("review_ready")
        ],
        "stale_or_conflicting_entries": [
            vp
            for vp in validated_patches
            if not vp["patch_classification"].startswith("review_ready")
        ],
    }
    bundle_path = os.path.join(OUTPUT_DIR, "proposed_reference_patch_bundle_v1.json")
    with open(bundle_path, "w") as f:
        json.dump(bundle, f, indent=2, default=str)
    print("  Wrote proposed_reference_patch_bundle_v1.json")

    print("\n=== Audit complete ===")
    print(f"Global decision: {global_decision}")
    return global_decision


if __name__ == "__main__":
    result = main()
    print(f"\nFinal decision: {result}")
