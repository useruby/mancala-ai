# Diagnostic Vs Canonical Arena Alignment Audit

Classification: `diagnostic_canonical_search_contract_mismatch`.

The H4 artifact identity is correct in both evaluations: checkpoint
`ced2eb4b35ed4593408baaf6176832735e6d1667ee33b4f0d769ec91d80de3b9`
exports `3629ba489409baf7c7f194d104101dcffde4ac7ccbfd20409cd112214095b323`.
Both evaluations use parent `f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`.

The committed/frozen game rows also account correctly:

| Arena | W | D | L | Games | Score | P0 | P1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PR #378 diagnostic | 256 | 0 | 0 | 256 | 1.000000 | 1.000000 | 1.000000 |
| PR #377 canonical | 227 | 48 | 237 | 512 | 0.490234 | 0.566406 | 0.414062 |

All diagnostic opening pairs have score `1.0` (128 pairs). Canonical pair
accounting has 256 complete seat-swapped pairs. The scores are therefore not an
accounting or identity artifact.

The search contracts are not equivalent. The diagnostic report identifies its
exact-root solver as `ml.alphazero_lite.endgame_tablebase.EndgameTablebase` and
does not bind a native-probe or tablebase SHA. The canonical report identifies
`native_kvtb_root_action_probe_v1`, native probe
`d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48`, and
tablebase `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.
Both otherwise record 384 simulations, `c_puct=1.25`, deterministic roots,
zero FPU, no subtree reuse, no value normalization, and exact-root threshold 16.

Consequently PR #378's perfect result cannot be compared as a score from the
canonical `puct_exact_root_hybrid` contract. Per the audit stop rule, this audit
does not infer a suite-distribution effect, run root probes, replay games, or
alter PR #377's rejection or PR #378's conditional anchor comparison.

The machine-readable evidence is
`docs/data/diagnostic-vs-canonical-arena-alignment-audit.json`.
