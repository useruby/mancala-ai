# Seed477 counterfactual accounting correction

Post-execution semantic correction to seed476 supplement evidence. The prior report copied the recorded-Adam endpoint loss change into `full_negative` and `minibatch_negative` records and subtracted their first-order terms. Those counterfactual endpoints were never evaluated; the observed-change and remainder fields are therefore removed. No counterfactual forward evaluations, gradients, training, searches, or games were performed.

All counterfactual norms, dot products, cosines, signs, and first-order contributions are retained. Only recorded-Adam displacement records retain an evaluated observed change, first-order contribution, and remainder. Original source and evidence files are preserved byte-for-byte.

Original seed476 registration SHA-256: `321cc9f95468bba54e0ae8ea02a72bbf4d1fb4a65dca2147033ddf2f7561d86f`
Original seed476 receipt SHA-256: `2b8a0ff7354dfbd6895ff8d4e3055d28df1735cfaab97c56532ef24e7a521772`
Original seed476 evidence SHA-256: `7ab7714d9bed39b5d8fde024f3a86ee4e267eb6b31ca331e950aac766cdafd23`
Prior supplement registration/receipt/evidence SHA-256: `3760ad90383d37aafe0e724a6518189b16f7ec61b652fc6ea5027b64bef22713` / `5ff943885ae50615931e795f96d4a746c9943219ae1d46a73d305905de08966f` / `8f451175a78b9d72c46128b44baee105e2324b29fdd00e3d1278e5a074e4fcba`

## Recorded Adam endpoint reconciliation

- equal_input: observed total 0.00308844438381; first-order total -0.00394414642924; remainder total 0.00703259081305; endpoint change 0.00308844438381; residual 0.
  Category counts: {'adam_coordinate_transport_failure': 2, 'both_descent': 5, 'minibatch_transport_failure': 4, 'mixed_or_ambiguous': 5}
- exposure: observed total 0.00268130838802; first-order total -0.00433774157949; remainder total 0.00701904996751; endpoint change 0.00268130838802; residual 0.
  Category counts: {'adam_coordinate_transport_failure': 1, 'both_descent': 6, 'minibatch_transport_failure': 4, 'mixed_or_ambiguous': 5}
- Training: observed total -0.0316364490935; first-order total -0.0372173092673; remainder total 0.00558086017377; endpoint change -0.0316364490935; residual 0.

First-order direction categories and their counts are unchanged from the prior supplement; the correction only removes unsupported counterfactual finite-step fields. Retrospective descriptive evidence only; no causal, promotion, or strength claim.
