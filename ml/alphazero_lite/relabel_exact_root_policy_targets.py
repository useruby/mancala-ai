#!/usr/bin/env python3
"""Create and audit a uniform-optimal-set exact-root replay artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.alphazero_lite.exact_root_policy_targets import relabel_exact_root_policy_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = relabel_exact_root_policy_jsonl(Path(args.source), output)
    Path(args.audit_out).write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
