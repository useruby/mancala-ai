# Seed554: fixed-state teacher-budget training ablation

Seed554 compares training on paired temperature-1.0 policy targets produced by
128- and 512-simulation denoised PUCT searches on the same control-training
states. It changes policy labels only for roots above 16 active stones; all
lower-stone targets, occurrence multiplicities, values, inputs, legal masks,
weights, and the control training schedule are retained.

C128 is a newly regenerated temperature-1.0 control. It is not required to
reproduce historical C108, whose policy targets used the generation temperature
schedule.

The source authentication contract binds the six #549 control training shards,
occurrence identities, published product receipts, schedule (108 updates and
54,896 exposures), shared 4,738-occurrence development membership, and the
initializer bytes. Historical #549's full verifier is not represented as
passing: its tablebase runtime limitation remains as disclosed by #553. No
tablebase or exact-leaf query is used by seed554; teacher searches enter
ordinary PUCT directly with exact-leaf solving disabled.

Run the recorded Python environment from the repository root:

```sh
python -m ml.alphazero_lite.seed554_fixed_state_teacher_budget authenticate
python -m ml.alphazero_lite.seed554_fixed_state_teacher_budget freeze
python -m ml.alphazero_lite.seed554_fixed_state_teacher_budget generate
python -m ml.alphazero_lite.verify_seed554_fixed_state_teacher_budget
python -m ml.alphazero_lite.seed554_fixed_state_teacher_budget bind
python -m ml.alphazero_lite.seed554_fixed_state_teacher_budget train
python -m ml.alphazero_lite.verify_seed554_fixed_state_teacher_budget
```

Searches append durable JSONL records. Restart validates each completed record
before skipping it. Independent verification is read-only. The separate
training-input binding is only created after derivative verification. C128's
production-update equality check runs on every update before the T512 arm.

Only update 108 is the treatment candidate. Development diagnostics reconcile
initializer and both final checkpoints against unchanged #549 development
targets, with exposure and equal-production-float32-input weighting; they are
descriptive and do not select a checkpoint or establish target quality or
playing strength.
