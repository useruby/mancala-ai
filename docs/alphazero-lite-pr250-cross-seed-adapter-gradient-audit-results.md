# PR #250 Cross-Seed Adapter And Gradient Audit

This analysis used the four frozen candidates from the fresh-suite experiment. It performed backward passes only: no self-play, optimizer step, training, target modification, or promotion occurred.

At the shared A16 initialization, the full beta=.95 policy-adapter gradient was averaged over every row in each lane's frozen training split. Adapter deltas are the A16-to-step-16 displacements.

| Geometry | Successful cross-seed cosine | Matched-negative cross-seed cosine |
| --- | ---: | ---: |
| Adapter delta | .953130 | .933739 |
| Full-batch gradient | .972289 | .962428 |

Within each seed, positive and negative directions are also very close: delta cosine `.993834` / `.992487` and gradient cosine `.993730` / `.993121` for seed-45 / seed-46. The successful lanes therefore share a more consistent cross-seed direction, but the evidence is a small discriminative component of otherwise strongly aligned updates, not a distinct mechanism.

| Lane | Frozen rows | Adapter delta norm | Full-batch gradient norm |
| --- | ---: | ---: | ---: |
| seed45 fixed768 positive | 27,858 | .00224844 | .00513403 |
| seed45 fixed1024 negative | 27,858 | .00218880 | .00499063 |
| seed46 fresh1024 positive | 28,217 | .00223081 | .00466767 |
| seed46 fresh768 negative | 28,217 | .00225782 | .00491923 |

The detailed reproducible output is `/tmp/azlite_pr250_cross_seed_adapter_gradient_audit/summary.json`.
