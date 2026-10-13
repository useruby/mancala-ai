# Seed468 first-step optimizer-direction transport

Classification: **first_order_descent_consistent**.

The fixed-initializer diagnostic produced 128 minibatch/objective/direction/
weighting comparisons. Under both exposure-weighted and equal-input fresh
strict-unseen `>32` policy gradients, each of P negative-gradient, P Adam,
J negative-gradient, and J Adam has at least 12 of 16 batches with cosine at
most `-1e-6`. Ambiguous and zero-norm cases are counted separately in
`diagnostic.json` (none occurred).

For each batch P is fresh weighted legal-action policy CE; J adds the registered
`0.3 * fresh Huber / 512` term. Both prospective directions start at the same
authenticated initializer. Adam uses zero moments and gradient clipping; the
negative-gradient vector is norm-matched to its objective's Adam displacement.
All underlying full-parameter gradients and displacement vectors are archived
in `vectors.npz`.

This is a first-order local diagnostic only. It does not establish a cause for
an entire training trajectory and is not a model-strength decision. No
trajectory training, searches, games, export, or promotion was performed.

Verify with:

```bash
python -m ml.alphazero_lite.verify_seed468_transport
```
