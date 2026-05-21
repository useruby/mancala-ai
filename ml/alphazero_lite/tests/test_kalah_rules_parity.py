import json
import subprocess
import unittest
from pathlib import Path

from ml.alphazero_lite import self_play
from ml.alphazero_lite.kalah_rules import KalahGame


def load_kalah_v3_parity_states() -> list[dict]:
    fixture_path = Path("test/fixtures/ai/kalah_v3_parity_states.json")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


class KalahRulesParityTest(unittest.TestCase):
    def test_python_engine_matches_golden_vectors(self):
        fixture_path = Path("test/fixtures/ai/kalah_rule_vectors.json")
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))

        self.assertEqual("kalah_v1", payload["rules_version"])

        for vector in payload["vectors"]:
            game = KalahGame.from_state(vector["initial_state"])

            for step in vector["steps"]:
                absolute_move = game.pit_index(step["relative_move"])
                self.assertEqual(step["absolute_move"], absolute_move, vector["id"])

                ok = game.move(absolute_move)
                self.assertEqual(step["ok"], ok, vector["id"])
                self.assertEqual(step["state"], game.to_state(), vector["id"])
                self.assertEqual(step["winner"], game.winner, vector["id"])
                self.assertEqual(step["over"], game.over(), vector["id"])
                self.assertEqual(
                    step["possible_moves"], game.possible_moves(), vector["id"]
                )

    def test_python_and_ruby_encoders_match_for_supported_input_encodings(self):
        fixture_path = Path("test/fixtures/ai/kalah_rule_vectors.json")
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        states = [vector["initial_state"] for vector in payload["vectors"]]

        encoder = self_play.encode_state

        repo_root = Path(__file__).resolve().parents[3]
        runner = (
            "payload = JSON.parse(ARGV.fetch(0)); "
            "result = payload.fetch('states').map do |state| "
            "  AI::AlphaZeroLite::StateEncoder.encode(state, version: payload.fetch('input_encoding')) "
            "end; "
            "puts JSON.generate(result)"
        )

        for input_encoding in ("kalah_v1", "kalah_v2"):
            ruby_vectors = json.loads(
                subprocess.run(
                    [
                        "bin/rails",
                        "runner",
                        runner,
                        json.dumps(
                            {"input_encoding": input_encoding, "states": states}
                        ),
                    ],
                    cwd=repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            python_vectors = [
                encoder(state, input_encoding=input_encoding) for state in states
            ]
            self.assertEqual(ruby_vectors, python_vectors)

    def test_kalah_v3_python_and_ruby_encoders_match_for_tactical_and_endgame_states(
        self,
    ):
        encoder = self_play.encode_state

        repo_root = Path(__file__).resolve().parents[3]
        parity_cases = load_kalah_v3_parity_states()
        states = [case["state"] for case in parity_cases]
        runner = (
            "payload = JSON.parse(ARGV.fetch(0)); "
            "result = payload.fetch('states').map do |state| "
            "  AI::AlphaZeroLite::StateEncoder.encode(state, version: 'kalah_v3') "
            "end; "
            "puts JSON.generate(result)"
        )

        ruby_vectors = json.loads(
            subprocess.run(
                [
                    "bin/rails",
                    "runner",
                    runner,
                    json.dumps({"states": states}),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        python_vectors = [encoder(state, input_encoding="kalah_v3") for state in states]
        self.assertEqual(len(parity_cases), len(ruby_vectors))
        self.assertEqual(len(parity_cases), len(python_vectors))

        for parity_case, ruby_vector, python_vector in zip(
            parity_cases, ruby_vectors, python_vectors
        ):
            with self.subTest(state_id=parity_case["id"]):
                self.assertEqual(ruby_vector, python_vector)


if __name__ == "__main__":
    unittest.main()
