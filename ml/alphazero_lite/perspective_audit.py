#!/usr/bin/env python3
"""Audit perspective consistency for self-play training rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PITS_PER_PLAYER = 6
SIGNED_VALUE_TARGET_MODES = {"sharpened", "phase_aware_sharpened", "hybrid"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-errors", type=int, default=50)
    return parser.parse_args()


def decode_state(features: list[float]) -> tuple[list[int], list[int], int]:
    player_pits = [max(0, int(round(features[idx] * 48.0))) for idx in range(6)]
    opponent_pits = [max(0, int(round(features[6 + idx] * 48.0))) for idx in range(6)]
    current_player = int(round(features[14]))
    return player_pits, opponent_pits, current_player


def expected_value(winner: int | None, player: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def value_matches_perspective(value: float, *, winner: int | None, player: int, value_target_mode: str) -> bool:
    expected = expected_value(winner, player)
    if value_target_mode not in SIGNED_VALUE_TARGET_MODES:
        return abs(value - expected) <= 1e-6

    if expected == 0.0:
        return abs(value) <= 1e-6
    return value * expected > 0.0


def signed_value_requires_winner(*, value: float, winner: int | None, value_target_mode: str) -> bool:
    return value_target_mode in SIGNED_VALUE_TARGET_MODES and winner is None and abs(value) > 1e-6


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checks = {
        "rows": 0,
        "policy_sum": 0,
        "policy_non_negative": 0,
        "policy_legal_support": 0,
        "value_range": 0,
        "value_perspective": 0,
        "player_feature_consistency": 0,
    }
    errors: list[dict] = []

    with data_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            checks["rows"] += 1

            state = row["state"]
            policy = row["policy"]
            value = float(row["value"])
            player = int(row.get("player", round(state[14])))
            winner = row.get("winner", None)
            value_target_mode = str(row.get("value_target_mode", "default"))

            player_pits, opponent_pits, current_player = decode_state(state)
            active_pits = player_pits if current_player == 0 else opponent_pits
            legal_moves = [idx for idx in range(PITS_PER_PLAYER) if active_pits[idx] > 0]

            if abs(sum(policy) - 1.0) <= 1e-3:
                checks["policy_sum"] += 1
            else:
                errors.append({"line": line_no, "code": "policy_sum", "detail": sum(policy)})

            if all(prob >= -1e-8 for prob in policy):
                checks["policy_non_negative"] += 1
            else:
                errors.append({"line": line_no, "code": "policy_negative"})

            illegal_mass = sum(policy[idx] for idx in range(PITS_PER_PLAYER) if idx not in legal_moves)
            if illegal_mass <= 1e-3:
                checks["policy_legal_support"] += 1
            else:
                errors.append({"line": line_no, "code": "illegal_policy_mass", "detail": illegal_mass})

            if -1.0 <= value <= 1.0:
                checks["value_range"] += 1
            else:
                errors.append({"line": line_no, "code": "value_range", "detail": value})

            if winner is None:
                winner_int = None
            else:
                winner_int = int(winner)

            if signed_value_requires_winner(value=value, winner=winner_int, value_target_mode=value_target_mode):
                errors.append(
                    {
                        "line": line_no,
                        "code": "signed_value_requires_winner",
                        "detail": {
                            "value": value,
                            "winner": winner_int,
                            "player": player,
                            "value_target_mode": value_target_mode,
                        },
                    }
                )
            elif value_matches_perspective(value, winner=winner_int, player=player, value_target_mode=value_target_mode):
                checks["value_perspective"] += 1
            else:
                errors.append(
                    {
                        "line": line_no,
                        "code": "value_perspective",
                        "detail": {
                            "value": value,
                            "winner": winner_int,
                            "player": player,
                            "value_target_mode": value_target_mode,
                        },
                    }
                )

            if player == current_player:
                checks["player_feature_consistency"] += 1
            else:
                errors.append(
                    {
                        "line": line_no,
                        "code": "player_feature_consistency",
                        "detail": {"player": player, "state_current_player": current_player},
                    }
                )

            if len(errors) >= args.max_errors:
                break

    passed = all(count == checks["rows"] for key, count in checks.items() if key != "rows")
    report = {
        "schema": "perspective_audit_v1",
        "rows": checks["rows"],
        "checks": checks,
        "passed": passed,
        "errors": errors,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote perspective audit report to {out_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
