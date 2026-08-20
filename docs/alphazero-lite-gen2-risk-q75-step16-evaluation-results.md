# Gen-2 risk_q75 Step-16 Arena Evaluation

**Classification:** `cumulative_lineage_regression`

**Recommended next experiment:** make cumulative lineage benchmarks a promotion requirement

## Lineage

- p0_weights_sha256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- p1_weights_sha256: `77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12`
- p1_state_hash: `a86acb54b97c860289530fcb7ca64194724d43667580a52e2948359bfa3ebdf4`
- p2e_weights_sha256: `93f3a32f42979fbd72ae3e895e5a8749d7ae3eadf594ac310dcf5bcad0faa975`
- p2e_checkpoint_sha256: `a08724caf3690c7273628c88a2478a707733e3820595cd83de615bc567ff049a`
- p2e_state_hash: `77eb867796d3346292ebd18701534b52b37400325a1629e01ef1de8bd26ca45c`
- p2_state_hash: `336496d5fb33331240178c4b834b8faf9548e3915b45c9b5f7e4b7aad6626870`
- gen2_replay_sha256: `2cee30547f8bc5d7cad6f02f859ee5e8644386e9b59c8a054ef74548c72ce84b`
- batch_plan_sha256: `d9323a8df88f390b8f3a73459f1d0ef6d5939f467a7f43af5bacb40f287628bd`
- train_source_sha256: `82d9633957ca43d3cc388c4cffdd61c90f9a8d1f0e538f5cf66c0d5389b9a971`
- frozen_q75_mask_sha256: `cb36a22880c7fb5a1676c96e6e2a5f2b42d8a7562ca98083642862b6fed47d08`

## Canonical Arena Matrix

| Match | Budget | Effect | 95% CI | P0 seat | P1 seat | W/D/L | Safe |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| p1_vs_p0 | 384:256 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | 392/38/82 | True |
| p1_vs_p0 | 1200:1200 | +0.0234 | [+0.0078, +0.0430] | +0.0273 | +0.0195 | 204/128/180 | True |
| p2e_vs_p1 | 384:256 | -0.0098 | [-0.0195, -0.0020] | -0.0195 | +0.0000 | 390/48/74 | True |
| p2e_vs_p1 | 1200:1200 | +0.0410 | [+0.0137, +0.0664] | -0.0391 | +0.1211 | 242/70/200 | True |
| p2e_vs_p0 | 384:256 | -0.0234 | [-0.0410, -0.0039] | -0.0352 | -0.0117 | 382/48/82 | False |
| p2e_vs_p0 | 1200:1200 | +0.0645 | [+0.0469, +0.0840] | +0.0078 | +0.1211 | 246/86/180 | True |

## Training Context

| Step | Fit fraction | CE(search) | CE(P1) | Policy L1 vs P1 | Protected L1 | Unprotected L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.3643 | 1.034855 | 0.934214 | 0.002882 | 0.004193 | 0.002445 |
| 46 | 0.6935 | 1.034088 | 0.934222 | 0.004038 | 0.006533 | 0.003206 |

- Extra fit after safe checkpoint: `0.3292`
- Low-budget effect change (step46 - step16): `-0.0176`

## Search Diagnostics

| Comparison | Budget | Move change | Visit JS | Q-rank change | Root-value delta | Visit-margin change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| p2e_vs_p1 | 384:256 | 0.0039 | 0.000283 | +0.0000 | +0.000151 | -0.6602 |
| p2e_vs_p1 | 1200:1200 | 0.0000 | 0.000789 | +0.0000 | +0.000130 | +4.2930 |
| p2e_vs_p0 | 384:256 | 0.0156 | 0.000605 | +0.0039 | +0.000023 | -0.7852 |
| p2e_vs_p0 | 1200:1200 | 0.0117 | 0.001281 | -0.0273 | +0.000053 | -2.4648 |

Per-depth legal-policy L1/JS telemetry is retained in the JSON summary for both P2e-vs-P1 and P2e-vs-P0 at 384:256.
