#!/usr/bin/env python3
"""Refresh any immutable replay's policy teachers under a runtime search policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.alphazero_lite.replay_policy_refresh import (
    audit_replay,
    refresh_replay_policy_targets,
    target_drift_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--runtime-policy", type=Path, required=True)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, required=True)
    args = parser.parse_args()
    original = audit_replay(args.source)
    refreshed = refresh_replay_policy_targets(
        args.source,
        args.out,
        parent_artifact=args.parent_artifact,
        runtime_policy_path=args.runtime_policy,
        native_probe=args.native_probe,
        tablebase_path=args.tablebase,
    )
    drift = target_drift_audit(
        args.source, args.out, parent_artifact=args.parent_artifact
    )
    args.audit_out.write_text(
        json.dumps(
            {"original": original, "refresh": refreshed, "target_drift": drift},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
