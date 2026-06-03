# AlphaZero-lite Start-State Fairness Diagnostic Results

## 1. Context: first-player advantage in Kalah(6,4)

- Standard Kalah(6,4) remains the production game and the promotion target.
- The default self-play start was left unchanged: `[4,4,4,4,4,4]` vs `[4,4,4,4,4,4]`, stores `0-0`, `current_player=0`.
- This diagnostic only tested training-curriculum variants for self-play target generation.
- No rules were changed.
- No model was promoted.
- No production-scale sweep was run.

## 2. Start-state modes

- Added `--start-state-mode standard_4x6|random_symmetric_total24|preset_pool` to `ml/alphazero_lite/self_play.py`.
- `standard_4x6` preserves the existing production behavior.
- `random_symmetric_total24` samples a symmetric 24-stone pit distribution per side and alternates the start player by game index.
- `preset_pool` loads JSON or JSONL preset states from `--start-state-pool` and carries preset metadata into self-play rows.
- New self-play metadata is additive only: `start_state_mode`, `start_player`, `start_distribution`, `start_state_hash`, and preset metadata when present.

## 3. Fairness scan

- Scanner added: `ml/alphazero_lite/run_start_state_fairness_scan.py`.
- Run used `classic_mcts`, `1200` simulations, seeds `11,23,37`, and `200` sampled symmetric distributions.
- Output: `/tmp/azlite_start_state_fairness/start_state_scan.jsonl`.
- Rows written: `400`.
- Strict accepted rows at `max_abs_margin=0.10`, `max_seed_std=0.08`: `0`.
- Best observed absolute first-player margin: `0.1656`.
- Median absolute first-player margin: `0.4424`.
- Median per-start seed std: `0.0108`.

Interpretation:

- The scan did not find genuinely near-zero symmetric total-24 starts under the tested teacher.
- Symmetric redistribution alone does not remove first-player bias.

## 4. Balanced preset pool

- Built `/tmp/azlite_start_state_fairness/balanced_start_pool.jsonl`.
- Built `/tmp/azlite_start_state_fairness/random_symmetric_pool.jsonl`.
- Balanced pool size: `32` states.
- Construction: `16` lowest-margin sampled distributions, each kept with both `current_player=0` and `current_player=1`.
- Balanced pool margin range: `0.1656` to `0.2503`.
- This is a best-available low-margin pool, not a truly neutral pool.

## 5. Diagnostic self-play statistics

- Standard sample: `/tmp/azlite_start_state_fairness/self_play_standard_sample.jsonl`
- Random sample: `/tmp/azlite_start_state_fairness/self_play_random_sample.jsonl`
- Balanced sample: `/tmp/azlite_start_state_fairness/self_play_balanced_sample.jsonl`
- Shared settings matched the strongest safe baseline self-play lane apart from start-state mode.

| mode | games | rows | avg game length | outcome by start player | value mean | value std | first-8 entropy | standard-opening rows | noise mode | target eps |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `standard_4x6` | 24 | 920 | 38.33 | `p0=0.0417` | `0.0366` | `0.4522` | `0.7859` | `2.61%` | `denoised` | `0.0` |
| `random_symmetric_total24` | 24 | 808 | 33.67 | `p0=0.0833`, `p1=0.0000` | `0.0204` | `0.4429` | `0.6239` | `0.00%` | `denoised` | `0.0` |
| `balanced_preset_pool` | 24 | 872 | 36.33 | `p0=-0.0833`, `p1=-0.2500` | `0.0453` | `0.4700` | `0.6284` | `0.00%` | `denoised` | `0.0` |

- No broken or terminal-at-start games were observed.
- Random and balanced starts reduced opening-state reuse as intended.
- They also lowered early target entropy relative to the standard start sample.

## 6. Tiny training trace

- Temporary exports: `/tmp/azlite_start_state_fairness/exports/`
- Init checkpoint: materialized from `storage/ai/alphazero_lite/current/weights.json`
- Traces run: `standard_4x6_small`, `random_symmetric_total24_small`, `balanced_preset_pool_small`
- Epochs run for each trace: `1`, `2`, `4`

Best tiny-trace checkpoints by trace were the epoch-4 exports.

| trace | overall regret | overall blunder | overall value MAE | opening regret | incumbent_proxy_disagreement regret | standard-start dataset MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard_4x6_small_e4` | `0.1138` | `0.5625` | `0.4348` | `0.0646` | `0.0410` | `0.3308` |
| `random_symmetric_total24_small_e4` | `0.0975` | `0.5491` | `0.4666` | `0.0604` | `0.0286` | `0.3690` |
| `balanced_preset_pool_small_e4` | `0.1021` | `0.5714` | `0.4522` | `0.0600` | `0.0340` | `0.3999` |

Tiny-trace readout:

- Both curriculum modes improved `incumbent_proxy_disagreement` regret versus the standard tiny trace.
- Both curriculum modes slightly improved opening-bucket regret versus the standard tiny trace.
- Neither curriculum mode beat the standard tiny trace on overall standard-suite value calibration.
- Both curriculum modes were worse than the standard tiny trace on standard-start dataset calibration.
- Random-start was the strongest of the two curriculum variants, but still not strong enough to advance.

Representative opening guard rows remained active in all three traces.

## 7. Production-scale run, if any

- No production-scale candidate was run.
- Reason: the tiny traces did not clear the bar for a promising standard-game candidate.

## 8. Standard-game evaluation

- Standard-game evaluation remained the criterion.
- No arena or promotion gate was run because no full candidate advanced past the tiny trace.
- The corrected forensic suite stayed standard-game based throughout the diagnostic.

## 9. Side-balance evaluation

- No side-balance arena report was run.
- Reason: no full candidate advanced to standard-game evaluation.

## 10. Interpretation

- The curriculum change successfully altered the self-play opening distribution.
- It did not produce a clearly fair symmetric-start pool under the tested teacher; the best scanned symmetric distributions still showed meaningful first-player margin.
- The tiny traces suggest start diversification can reduce some local disagreement and opening regret.
- The same traces also show weaker standard-start calibration than the standard tiny trace.
- This looks more like a partial perturbation of target distribution than a clean fix for the current bottleneck.
- Training-curriculum benefit and new-game-variant benefit remain separate here: the result does not show that a non-standard start variant is intrinsically useful for promotion on standard Kalah(6,4).

Final classification:

- `start_bias_not_primary_bottleneck`

## 11. Exactly one recommended next action

Recommendation: **return to teacher-interference or the next non-opening failure family instead of advancing a balanced-start curriculum lane.**
