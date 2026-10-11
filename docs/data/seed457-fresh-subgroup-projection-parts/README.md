# Seed457 T tensor archive parts

`step-tensors-T.npz` exceeds GitHub's single-file limit. The two files in this
directory are consecutive byte ranges of the exact archive referenced by the
seed457 receipt; they are outside the receipt inventory until reassembled.

From the repository root, run:

```bash
cat docs/data/seed457-fresh-subgroup-projection-parts/step-tensors-T.npz.part-00 \
    docs/data/seed457-fresh-subgroup-projection-parts/step-tensors-T.npz.part-01 \
    > docs/data/seed457-fresh-subgroup-projection/step-tensors-T.npz
```

The reconstructed file must have SHA256
`d7a09d9c0d611f4921ffdce76b1967dc9393e987553d66b476a5cd4bd8365283`, as
recorded in `docs/data/seed457-fresh-subgroup-projection/receipt.json`. Its
length is 107,888,076 bytes. Once reassembled, the ordinary publication
inventory and independent verifier can read the original registered filename.
