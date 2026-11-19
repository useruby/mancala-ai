# Seed 615 training execution report

## Execution

- Authorized execution freeze SHA-256: `6ac9d4b4e52a6f4789ce406d9c88badd82e9c8ab7cf4b4d4d8e9c8676cd9910d`
- Owner authorization record SHA-256: `828a3dc94d1a01496806f9b649254a471dce435cbf482cfc43173b06c3ea25b8`
- Authorized by: Alexey Artamonov (i@useruby.com)
- Runtime: CPython 3.14.7; NumPy 2.5.2; PyTorch 2.13.0+cu130; Linux x86-64; CPU only; one PyTorch thread; deterministic algorithms enabled; `PYTHONHASHSEED=599`
- Training membership SHA-256: `5ed236266ed3804a9c0e9f75b7e10b7c81ea6e290f8c72d3cea06dd225a15010` (2,048 unique states; 512 / 1,024 / 512 quotas)
- Batch schedule SHA-256: `62edb14805d20af1cc7988d73aba95dcdf9e67842b6ea1d61a12c98a87c28599`
- Optimizer updates: exactly 512; all 18 trainable parameter Adam step counters verified at 512
- Frozen value parameters: unchanged from P-E4

## Evidence

- Final checkpoint SHA-256: `cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780`
- Training trace SHA-256: `8ee2252f6831ea0589a19dff2c1ebc79a0f3c3447764feb9c53086ba599a8b01`
- Training receipt SHA-256: `0dc4d04a6be2b638abf77bb959e93d82c60355f3f9a449a49d0fef0a8bbb81f3`
- Training integrity report SHA-256: `c383411a9ce9b896ef49431e502e0047b26de0e8600042d8b2873ae07c796266`
- Policy-loss diagnostics: update 1 `0.7192390561`; update 512 `0.6653289795`
- Final gradient norm: `6.3861565590`
- B512 tensor-equality diagnostic: the four value tensors match; the 18 policy/trunk tensors do not; overall exact equality is false. This diagnostic did not affect checkpoint selection.
- Independent result verifier: **passed**; confirmed the freeze and authorization bindings, 512 updates, 22 checkpoint tensors, trace hash, and structural/artifact integrity.

The independent verifier does not replay the complete optimizer computation; no independent numerical reproduction is claimed. Losses are training diagnostics only and are not evidence of improved playing strength. No evaluation games were run.
