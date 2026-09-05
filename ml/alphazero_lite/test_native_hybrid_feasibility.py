import unittest

from ml.alphazero_lite.run_exact_solver_feasibility_preflight import (
    DEFAULT_SEED,
    generate_feasibility_corpus,
)
from ml.alphazero_lite.run_native_hybrid_feasibility import verify_frozen_corpus


class NativeHybridFeasibilityTest(unittest.TestCase):
    def test_frozen_seed_271_corpus_identity(self) -> None:
        corpus = generate_feasibility_corpus(DEFAULT_SEED)
        self.assertEqual(
            "fe9c317f6dace7d5c77eb7214db4739643a5b7d3ac0b83a39be885119b459a59",
            verify_frozen_corpus(corpus),
        )

    def test_frozen_corpus_has_the_declared_buckets(self) -> None:
        corpus = generate_feasibility_corpus(DEFAULT_SEED)
        self.assertEqual(96, len(corpus))
        self.assertEqual(32, sum(17 <= row["stones_remaining"] <= 24 for row in corpus))
        self.assertEqual(32, sum(25 <= row["stones_remaining"] <= 32 for row in corpus))
        self.assertEqual(32, sum(33 <= row["stones_remaining"] <= 40 for row in corpus))


if __name__ == "__main__":
    unittest.main()
