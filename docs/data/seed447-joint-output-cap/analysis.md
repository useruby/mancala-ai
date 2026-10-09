# Seed447: joint policy-and-value output movement screen

## Registration and validity

This was a prospective 16-update, training-only acceptance screen. It is not
evidence of playing strength and does not authorize runtime export or promotion.
The value movement cap is a fixed per-step RMS limit of `0.01`; it provides no
guarantee about endpoint MSE.

- Registration SHA256: `614bd7b158b6590d5d0114534abcdec3bc7edee977220bfcd5e38d58901021cb`
- Seed442 initializer SHA256: `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`
- Seed442 B checkpoint SHA256: `4302216db4e359d3045ca00a01ab81f04b9ea85195432ca8cd6aaabd50cad2db`
- Training guard: 2,048 unique training inputs, reconstructed from seed442's registered procedure and excluded by exact and canonical identity from the strict unseen evaluation population.
- Evaluation population: 2,607 exposures representing 1,242 exact input identities; it was not used for proposal acceptance.
- Arm C exactly reproduced seed442 B's sixteen accepted scales and every archived post-step parameter tensor.

Arm C applies seed442's policy-only KL cap. Arm T adds mean squared value-output movement `<= 0.0001` on both the current minibatch and fixed guard, evaluated from float32 predictions using float64 differences, squares, and means. Both caps use all eight scales and select the largest feasible value. Each arm advanced Adam exactly once per proposal.

## Results

| Metric | Initializer | Seed442 ordinary Adam A | C: policy cap | T: joint cap |
| --- | ---: | ---: | ---: | ---: |
| Full training objective | 1.0214983345 | 1.0195943288 | 1.0019462621 | 1.0040609659 |
| Strict unseen policy CE, exposure weighted | 1.1706991593 | 1.1877460084 | 1.1558827694 | 1.1472948612 |
| Strict unseen policy CE, equal input | 1.1981604519 | 1.2212629525 | 1.1965274705 | 1.1968499526 |
| Strict unseen value MSE, exposure weighted | 0.4764106647 | 0.4848726256 | 0.4809968564 | 0.4755667934 |
| Strict unseen value MSE, equal input | 0.6169980428 | 0.6210176897 | 0.6208880654 | 0.6120165019 |

T−C effects:

| Metric | Exposure weighted | Equal input |
| --- | ---: | ---: |
| Policy CE | -0.0085879083 | +0.0003224821 |
| Value MSE | -0.0054300631 | -0.0088715636 |

T's full training objective was `+0.0021147038` relative to C and decreased by `0.0174373686` relative to the initializer. The value cap activated on 16 of 16 accepted nonzero proposals. There were no rejected nonzero proposals. C accepted scales were `[1/8, 1/8, 1/8, 1/4, 1/2, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2]`; T accepted scales were `[1/32, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/8, 1/8, 1/8, 1/8]`.

## Fixed decision

**Classification: `close_joint_output_cap_branch`.** T met the policy CE gate versus seed442 A under both weightings, the value MSE guards, training-objective decrease, cap-activation requirement, and non-rejection requirement. It missed the initializer policy CE requirement under equal-input weighting: T was `0.0013104993` below the initializer, short of the required `0.005` improvement. No follow-up is authorized by this screen.

## Verification amendment

The registered verifier had a post-execution defect: it passed NumPy arrays to
seed442's Torch-only trial setter. The frozen verifier and all bound experiment
evidence were preserved unchanged. The separately receipted
`verify_seed447_joint_output_cap_amendment.py` adapts that call, after which the
full read-only reconstruction passed. Its amendment receipt records the source
and original frozen-receipt hashes. The amended verifier also passed from a
physically copied checkout with evidence copied separately and an unrelated
working directory.

Complete ordered predictions and source-reconstructed targets are in the
`*-predictions.npz` files and `ordered-targets.npz`. All per-step parameters,
gradients, Adam moments, proposals, and trial measurements are in
`step-tensors.npz` and `evidence.json`.
