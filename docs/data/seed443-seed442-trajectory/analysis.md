# Seed443: seed442 trajectory and optimizer archive clarification

## Verification result

The canonical seed442 publication verifier and the independent seed443 semantic validator both pass. Arms A and B each contain 16 indexed steps. The parameter-state chain starts at the authoritative initializer, links exactly through each accepted post-state, and terminates at its published checkpoint. Tensor coverage, ordered model parameters, metadata, finite values, recorded parameter hashes and norms, reconstructed Adam moments, and the registered position hashes/prefix pass. Arm A's accepted tensors equal its ordinary Adam proposals. Arm B's recorded descending trials stop at the first passing scale; rejection records exhaust all eight registered scales. The unchanged decision remains `close_kl_capped_step_branch`.

The position hashes are recomputed from the frozen split/permutation inputs; the 8,192-position prefix is read independently from the frozen first permutation and compared to registration.

## Optimizer archive inventory

`seed442_kl_capped_adam.py` clones both moment tensors into `moment_before` and `moment_proposed`, and records hashes over those cloned values. In contrast, archive entries `exp_avg` and `exp_avg_sq` are assigned using `.detach().cpu().numpy()` without `.copy()`. On CPU these arrays share storage with the live Adam state. Adam mutates that storage on subsequent steps, so all 16 raw entries for each of 22 parameters in both moment families and both arms now contain the final step-16 moment: **352/352 first-moment entries and 352/352 second-moment entries per arm** match the corresponding reconstructed step-16 arrays exactly. These are raw post-step archive arrays as committed; they are not step-specific historical snapshots.

The `exp_avg_pre` and `exp_avg_sq_pre` entries are distinct cloned pre-step values. Their continuity is verified against zero at step 1 and the prior reconstructed post-state thereafter. `optimizer-state-reconstruction.npz` is a supplemental reconstruction, not an independently logged runtime snapshot. Each reconstructed update is calculated from archived clipped gradients and cloned pre-step moments using the registered Adam arithmetic; reconstructed post-moment hashes match evidence and each next pre-step moment matches the preceding reconstructed post-state.

The original seed442 archive and receipt remain unchanged. The synthetic NumPy-view regression demonstrates that a view follows later in-place updates and that a `.copy()` is required for a historical snapshot.
