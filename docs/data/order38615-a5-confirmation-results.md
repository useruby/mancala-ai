# Order 38615 A E4 fresh confirmation

| Suite | Score | Wins | Draws | Losses | 95% opening-cluster interval |
|---:|---:|---:|---:|---:|
| 391 | 0.6362 | 597 | 109 | 318 | 0.6201–0.6523 |
| 392 | 0.6328 | 597 | 102 | 325 | 0.6167–0.6494 |

Equally weighted pooled score: **0.6345** (stratified opening bootstrap 95% interval 0.6230–0.6458).

Decision: **advance_to_promotion_review**.

The candidate was selected after inspecting #386, #388, #389, and #390 results; those results are selection evidence only and are excluded from these estimates. No training or model export was performed. Rejections in #386, #388, #389, and #390 remain preserved.

Registration SHA-256: `ef02fd57f638e7cf4f02030fae5c7da84b824f4d2c3192ee51b1ac629671c06f`. Candidate binding SHA-256: `31037f07e5f50aa0c945a6a37e1ec3ef17bf74746f1d531d5275cc32c261970a`. Evaluation binding SHA-256: `ebed6050dcf73d76fdf14438a74c3587a2a3a9a619de58ac0d7dc0da8f9f0239`.

Fresh suite SHA-256 values: seed 391 `3ee182e45ce262971f2e57b0e77bc49e5a323d2d68015e75ed95ebe57a91465f`; seed 392 `b961da2cf1c751d8f501203d9496b1a1d5d884da63b075f0e8f3d2c68f25efc7`.

this frozen order_38615_A E4 candidate versus this exact frozen seed455 opponent on the registered opening distribution; does not establish order generalization or validate rejected averaging treatments

Raw evidence is hash-bound by `order38615-a5-confirmation-opening-score-matrix.json` (SHA-256 `17d8fe40ed83327b1f6ed2f93d6e149c74272212b9e097446546d2412d0df036`).

Reproduce the point estimates and intervals from the registered opening-score matrix:
```sh
.venv/bin/python -m ml.alphazero_lite.reproduce_order38615_a5_confirmation docs/data/order38615-a5-confirmation-opening-score-matrix.json
```
