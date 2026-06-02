# AlphaZero-lite Incumbent Proxy Disagreement Reference Review Note

## Context

- This is the repo-side review artifact that supported the focused adjudication follow-up.
- At review-artifact creation time, the live corrected fixture had not yet been updated.
- This PR now applies the approved fixture change for `incumbent_proxy_disagreement-021` in `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Source adjudication report: `docs/alphazero-lite-incumbent-proxy-disagreement-reference-adjudication-results.md`.
- Source review patch proposal: `/tmp/azlite_incumbent_proxy_reference_adjudication/incumbent_proxy_reference_adjudication_patch.json`.

## Proposed Review Artifact

- Candidate review patch: `ml/alphazero_lite/fixtures/incumbent_proxy_disagreement_reference_review_patch_v1.json`
- Scope: changed rows only
- Changed row count: `1`

## Proposed Change

| row_id | current_reference_move | proposed_reference_move | evidence | note |
| --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-021 | 3 at review time; now 2 in this PR | 2 | 5/5 seeds at 5000, 10000, and 20000 selected move 2; highest-budget majority fraction 1.0 | overturned during review and applied in this PR |

## Non-change Checks

- `incumbent_proxy_disagreement-007`: upheld
- `incumbent_proxy_disagreement-009`: upheld
- `incumbent_proxy_disagreement-023`: upheld
- `incumbent_proxy_disagreement-024`: upheld

## Review Guidance

- Treat the patch file as the preserved review artifact for the approved change.
- This PR already applies only the overturned row into the live fixture.
- Do not start training from this proposal alone.

## Exactly One Recommended Next Action

Recommendation: **treat `ml/alphazero_lite/fixtures/incumbent_proxy_disagreement_reference_review_patch_v1.json` as provenance for the approved `incumbent_proxy_disagreement-021` fixture update, and do not start training until the remaining mechanism-specific audit follow-up is chosen.**
