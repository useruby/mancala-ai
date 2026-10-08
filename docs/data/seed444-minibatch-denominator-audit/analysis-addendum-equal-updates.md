# seed444 analysis addendum — equal-update average

This appendix reports the equal-update minibatch average directly from the published float64 aggregate vectors. Each of the 263 batches gets weight 1/263, including the final 358-exposure batch. The exposure-weighted batch vectors use n_b/N. Since the independently enumerated policy coefficients are all exactly one, the denominator choice itself has no effect; the values below isolate equal-update/final-partial-batch weighting.

| Checkpoint | Equal-update policy norm | Equal-update total norm | Cosine(equal-update total, G_global) | ||equal-update total − exposure-weighted total||₂ | Relative final-partial weighting effect |
|---|---:|---:|---:|---:|---:|
| initializer | 1.14409753773 | 1.16970955572 | 0.999999241122 | 0.00144551266245 | 0.00123566871667 |
| adam_a16 | 1.27174670644 | 1.28613118441 | 0.999999365070 | 0.00146062742336 | 0.00113551584567 |

The equal-update vectors are stored as `initializer_equal_update_policy`, `initializer_equal_update_total`, `adam_a16_equal_update_policy`, and `adam_a16_equal_update_total` in `corrected-aggregate-vectors.npz`. The exposure-weighted reference vectors are included in that archive as well.
