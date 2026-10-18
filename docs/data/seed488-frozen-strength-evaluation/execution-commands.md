# Seed488 execution and verification commands

Runtime: `.venv-azlite`, Python 3.14.7, NumPy 2.5.2.

The registered resume was invoked from the repository root:

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.seed488_strength_evaluation --execute
```

The frozen runner skipped receipt-bound opening 0 and resumed the remaining
paired openings in `resume-corrected/`, using one worker, 1,200 simulations per
side, `c_puct=1.25`, seed `488 + parent_opening_index`, exact-root threshold 16,
and the registered 200-ply cap. Each pair was validated before commit.

Independent publication verification:

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed488_strength_evaluation --root .
```

Focused tests and quality checks:

```bash
.venv-azlite/bin/python -m pytest -q ml/alphazero_lite/test_seed488_strength_evaluation.py ml/alphazero_lite/test_seed488_opening_replay.py
.venv-azlite/bin/python -m unittest ml.alphazero_lite.test_run_manifest -v
.venv-azlite/bin/python -m unittest ml.alphazero_lite.test_seed425_publication_portable -v
.venv-azlite/bin/python -m ml.alphazero_lite.validate_forensic_reference_artifact
probe=$(bash script/ai/setup_native_mtdf.sh .tmp/girving-kalah)
NATIVE_MTDF_PROBE="$probe" .venv-azlite/bin/python -m unittest ml.alphazero_lite.test_native_mtdf_probe -v
.venv-azlite/bin/ruff check ml/alphazero_lite script/ai
.venv-azlite/bin/ruff format --check ml/alphazero_lite script/ai
```

The initial focused-test invocation coincided with active run output and two
recovery-fixture tests saw the new pending batch files. The focused suite was
then run from a physically separate checkout and passed all 31 tests.
