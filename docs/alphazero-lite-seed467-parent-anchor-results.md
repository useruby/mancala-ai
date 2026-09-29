# Seed467 Parent-Anchor Prospective Result

## Registration

- Generation: `seed455-nextgen-s467-parent-anchor-w010-iter1`.
- Parent: promoted `seed48-nextgen-s455-default-value-iter1`.
- Parent weights SHA256: `f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`.
- Parent runtime-policy SHA256: `b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d`.
- Training seed: `467`; self-play seed sweep: `466,467,468`.
- The sole recipe change was a policy-only behavior anchor: seed455 raw legal policy on primary-training occurrences with active stones greater than 32, at weight `0.10`.

## Generation

- Self-play completed 1,600 games and wrote 66,448 rows.
- Fresh self-play SHA256: `6f7038dccf282fb9365a031a06f8c5a8af4a2b8a6eda32725fe125b06d654a76`.
- Exact-root handoffs / PUCT rows: `19,370 / 47,078`.
- Anchor SHA256: `172fb057207fb1669286e11caa4da5e6241179212b6dc1a0752800b10f03f663`.
- Anchor occurrences / unique canonical states / validation overlap: `33,042 / 13,052 / 0`.
- Training completed four epochs and 1,052 optimizer updates. `best_validation` selected E4; no arena result selected an epoch.
- Candidate weights SHA256: `3629ba489409baf7c7f194d104101dcffde4ac7ccbfd20409cd112214095b323`.

## Gate Result

- The frozen exact-root16 canonical prefilter suite SHA matched `811feb7d208467cdfa9a42244eb75adcc8877db5bc03f87dacbd3379e08bd4bf`.
- The candidate scored `0.490234375` (`227W/48D/237L`) against seed455, below the unchanged `0.55` threshold.
- The independent hard arena and downstream gate components were not run, as required after a prefilter failure.
- Classification: `seed467_parent_anchor_prefilter_failed`.
- Promotion status: rejected; no model artifact was promoted.

This is a valid single-seed prospective reject. It does not by itself identify whether the missed gate is due to forward-generation variance or non-transfer of the retrospective anchor effect.
