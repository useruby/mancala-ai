#!/usr/bin/env python3
"""Build margin-ordered value variants of frozen exact train rows.

Reads frozen exact train rows (which preserve ``exact_root_margin`` and
``player``) and rewrites only the ``value`` target, keeping ``state``,
``policy``, ordering, and all provenance identical:

* ``margin48``: root-perspective margin divided by 48 (the pits encoding
  divisor; range observed [-0.79, +0.79], always within [-1, 1]).
* ``margin_tanh``: tanh(root-perspective margin / 12), a compressed
  ordering-preserving alternative for sensitivity comparison.

The ±1 sign-value baseline is the input file itself. No new solver
labels are produced; this tests the value-ordering hypothesis with zero
label cost.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ml.alphazero_lite import exact_teacher_labeling as exact

VARIANTS = ("margin48", "margin_tanh")
MARGIN_DIVISOR = 48.0
TANH_SCALE = 12.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-train", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser.parse_args(argv)


def root_perspective_margin(exact_margin: int, player: int) -> int:
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")
    return int(exact_margin) if player == 0 else -int(exact_margin)


def variant_value(root_margin: int, variant: str) -> float:
    if variant == "margin48":
        return float(root_margin) / MARGIN_DIVISOR
    if variant == "margin_tanh":
        return math.tanh(float(root_margin) / TANH_SCALE)
    raise ValueError(f"unknown variant: {variant}")


def build_variant_rows(
    exact_rows: list[dict[str, Any]], variant: str
) -> list[dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    variant_rows: list[dict[str, Any]] = []
    for row in exact_rows:
        if "exact_root_margin" not in row or "player" not in row:
            raise ValueError(f"row {row.get('source_id')} lacks exact margin/player")
        root_margin = root_perspective_margin(row["exact_root_margin"], row["player"])
        value = variant_value(root_margin, variant)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"value out of range for {row.get('source_id')}: {value}")
        variant_row = dict(row)
        variant_row["value"] = value
        variant_row["value_target_mode"] = "default"
        variant_row["teacher_value_variant"] = variant
        variant_row["teacher_value_margin_divisor"] = (
            MARGIN_DIVISOR if variant == "margin48" else TANH_SCALE
        )
        variant_row["teacher_value_root_margin"] = root_margin
        variant_rows.append(variant_row)
    return variant_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exact_rows = exact.read_jsonl(args.exact_train)
    variant_rows = build_variant_rows(exact_rows, args.variant)
    exact.write_jsonl(args.out_train, variant_rows)
    summary = {
        "schema": "exact_value_variant_v1",
        "variant": args.variant,
        "train_rows": len(variant_rows),
        "train_sha256": exact.sha256_file(args.out_train),
        "exact_train_path": str(args.exact_train),
        "exact_train_sha256": exact.sha256_file(args.exact_train),
        "order_identical": [r["source_id"] for r in variant_rows]
        == [r["source_id"] for r in exact_rows],
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"variant={args.variant} rows={len(variant_rows)} sha={summary['train_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
