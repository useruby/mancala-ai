# AlphaZero-lite Incumbent Proxy Disagreement Reference Review Note

## Context

- This is a repo-side, non-mutating review artifact for the focused adjudication follow-up.
- The live corrected fixture remains unchanged: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Source adjudication report: `docs/alphazero-lite-incumbent-proxy-disagreement-reference-adjudication-results.md`.
- Source review patch proposal: `/tmp/azlite_incumbent_proxy_reference_adjudication/incumbent_proxy_reference_adjudication_patch.json`.

## Proposed Review Artifact

- Candidate review patch: `ml/alphazero_lite/fixtures/incumbent_proxy_disagreement_reference_review_patch_v1.json`
- Scope: changed rows only
- Changed row count: `1`

## Proposed Change

| row_id | current_reference_move | proposed_reference_move | evidence | note |
| --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-021 | 3 | 2 | 5/5 seeds at 5000, 10000, and 20000 selected move 2; highest-budget majority fraction 1.0 | overturn corrected reference candidate |

## Non-change Checks

- `incumbent_proxy_disagreement-007`: upheld
- `incumbent_proxy_disagreement-009`: upheld
- `incumbent_proxy_disagreement-023`: upheld
- `incumbent_proxy_disagreement-024`: upheld

## Review Guidance

- Treat the patch file as a review artifact only.
- If approved, apply only the overturned row into a future deliberate fixture update.
- Do not start training from this proposal alone.

## Exactly One Recommended Next Action

Recommendation: **review `ml/alphazero_lite/fixtures/incumbent_proxy_disagreement_reference_review_patch_v1.json`, and if approved, carry only `incumbent_proxy_disagreement-021` into a deliberate fixture update before any training.**
