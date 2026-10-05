# Erratum 1 — timing instrumentation asymmetry

The original #420 execution and all of its evidence are preserved unchanged.
Inspection after publication found that its timed cache-off searches used
`RecordingEvaluator`, which serialized and hashed every state and retained full
policy/value traces. Timed cache-on searches did not perform that work. This
violated the registered uninstrumented timing scope and biased the comparison
against cache-off.

The original equivalence checks and call-count observations remain valid: the
instrumentation records outputs and requests without changing the evaluator
results. The published latency and speedup estimates, however, are unsuitable
for the intended uninstrumented comparison and must not be used for the
advancement decision. No estimated tracing cost is subtracted. Seed421 is a
protocol-correction rerun on the same bound cohort, artifact, seeds, budgets,
capacity, search options, repetition count, analysis method, and thresholds;
it is not a sample extension or capacity sweep.
