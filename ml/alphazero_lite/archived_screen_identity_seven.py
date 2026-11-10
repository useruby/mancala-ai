"""Seven-source, Torch-free archived screen replay and provenance census."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import archived_screen_identity_inventory as replay

ROOT = replay.ROOT
AUDIT = ROOT / replay.AUDIT_RELATIVE
MANIFEST_NAME = "seed564-seven-source-manifest.json"
MANIFEST_SHA256 = "5df74d7b49c75598c0ab57964ebc66913549571fab108b3703ae66a97ff06991"
BASELINE_IDENTITIES = 147_146
BASELINE_INDEX_BINDINGS = {
    "docs/data/seed556-560-exclusion-inventory/source-manifest.json": "ad691da4dc0272c6a37f077f961023443d05295321d54cb7c339c51a9391d5ce",
    "docs/data/seed556-560-exclusion-inventory/count-report.json": "b6eea0fe006da2164ca4ad75528a6b08d44b191b2eb0364e5b5c211d95dd7639",
    "docs/data/seed556-560-exclusion-inventory/identity-inventory.json": "8b5bf0ca78a9d370a91d9cb309cfa7f39c4fec8e2fba3576a8483727c0a6cb52",
    "docs/data/expanded-training-dev-exclusion-inventory/source-manifest.json": "5c0ae635f6e12c91701557124aab51ae259dd6b11e4fc3a7a40f185a16c5b6d2",
    "docs/data/expanded-training-dev-exclusion-inventory/count-report.json": "d018de15ff5a6a98879df6f4853a90c89cf97bf3ab9b4ef4deb436b2953a13b8",
    "docs/data/expanded-training-dev-exclusion-inventory/identity-inventory.jsonl": "3605078c7b14a14e23e1cb1c74e44660f5f5bc5316b3bae3d980d5d303f7f5db",
}

# Final publication/recovery artifacts are pinned independently of any local
# receipt regeneration. The merge commit IDs identify the publication parents.
SOURCES: tuple[dict[str, Any], ...] = (
    {
        "key": "511",
        "name": "seed511-fixed-composition-screen-v4",
        "archive": "docs/data/seed511-fixed-composition-screen-v4",
        "merge_commit": "193fc4d15cec8bc24a0c33abbcefb2a12ab345e7",
        "anchors": {
            "docs/data/seed511-fixed-composition-screen-v4/publication-receipt.json": "daf24729427f2b45d8ac19d83f2b6a367d35e006279a2a236585eba82d6b3006",
            "docs/data/seed511-fixed-composition-screen-v4/registration.json": "89f04c6e1f774d81a7866d3913dd5ffa990b9e04a58f9a7f066b8f19204445c4",
            "docs/data/seed511-fixed-composition-screen-v4/execution-freeze.json": "e6476fa3f587292e41ff97d99db87b942587c211fb861f8837e37077292eb633",
            "docs/data/seed511-fixed-composition-screen-v4/openings.jsonl": replay.SUITE_SHA256,
            "docs/data/seed511-fixed-composition-screen-v4/outcomes.jsonl": "6324ce385df6ee3e824630b84744947e96cda97748caf703cd6ecc60f5ad2cb8",
        },
        "receipt": "publication-receipt.json",
        "recovered_openings": [],
        "failed_attempts": [],
    },
    {
        "key": "512",
        "name": "seed512-fixed-composition-evaluation",
        "archive": "docs/data/seed512-fixed-composition-evaluation",
        "merge_commit": "320991da30563bde3b8f4993146581f82352d4b7",
        "anchors": {
            "docs/data/seed512-fixed-composition-evaluation/publication-receipt.json": "8e8e7acc08f85fde189c55d6d4d37652de902ecb1d72ce250f31719a2309bb3f",
            "docs/data/seed512-supplement-receipt.json": "656f9a782c446f71534f23c393d250419f2b737e75dbde674f0aa8979db78574",
            "docs/data/seed512-fixed-composition-evaluation-copy-verification.json": "ef3c9ea6d2cabfcd4886d0480b2550db8fcf370a68b6cf76c00e0025e759b143",
            "docs/data/seed512-fixed-composition-evaluation/registration.json": "44d41e9d5bc78654a8713d5e9027cb3071b4484a2042308be0591d8aafac5309",
            "docs/data/seed512-fixed-composition-evaluation/execution-freeze.json": "34ab5b882f4906bc5c0e641a1c59aebd2f387c0cf083d54c2a2a1146f2d52f25",
            "docs/data/seed512-fixed-composition-evaluation/openings.jsonl": replay.SUITE_SHA256,
            "docs/data/seed512-fixed-composition-evaluation/outcomes.jsonl": "b3c5679e8523e968f7b0f2e76ae477f677ca1688e5e513e28ea64cf845cf84bb",
        },
        "receipt": "publication-receipt.json",
        "recovered_openings": [],
        "failed_attempts": [],
    },
    {
        "key": "545",
        "name": "seed542-expanded-policy-strength-screen",
        "archive": "docs/data/seed542-expanded-policy-strength-screen",
        "merge_commit": "56c5ae4e24ee73500f4f4ad027974cd186bba1ab",
        "anchors": {
            "docs/data/seed564-archived-screen-identity-audit/seed545-source-manifest.json": "5664eb3ee58665de9370f105b2fd31d60448e5d0dd3174868ad7aead1f1a5c09",
            "docs/data/seed542-expanded-policy-strength-screen/POSTRUN-VERIFICATION-RESULTS.md": replay.CORRECTION_RESULTS_SHA256,
            "ml/alphazero_lite/verify_seed542_expanded_policy_strength_completeness.py": replay.CORRECTED_VERIFIER_SHA256,
            "docs/data/seed542-expanded-policy-strength-screen/registration.json": replay.REGISTRATION_SHA256,
            "docs/data/seed542-expanded-policy-strength-screen/execution-freeze.json": replay.EXECUTION_FREEZE_SHA256,
            "docs/data/seed542-expanded-policy-strength-screen/continuation-registration-v3.json": replay.CONTINUATION_REGISTRATION_SHA256,
            "docs/data/seed542-expanded-policy-strength-screen/continuation-execution-freeze-v3.json": replay.CONTINUATION_FREEZE_SHA256,
            "docs/data/seed542-expanded-policy-strength-screen/openings.jsonl": replay.SUITE_SHA256,
            "docs/data/seed542-expanded-policy-strength-screen/outcomes.jsonl": replay.LEDGER_SHA256,
            "docs/data/seed542-expanded-policy-strength-screen/opening0-recovery-binding.json": "8cc56b54db9b93cb0638fada4e93eceefdc38b1ba288915c65871a58a8d0dc15",
            "docs/data/seed542-expanded-policy-strength-screen/opening1-recovery-binding-v1.json": "04076cc725e721ceafa6e6430b14376764fe9a69b08b7ce017d997e39e0ba58c",
        },
        "receipt": "continuation-registration-v3.json",
        "recovered_openings": [0, 1],
        "failed_attempts": [
            "docs/data/seed542-expanded-policy-strength-screen/PRELAUNCH-FAILURE-0000.json"
        ],
    },
    {
        "key": "547",
        "name": "seed542-update-432-strength-screen-v3",
        "archive": "docs/data/seed542-update-432-strength-screen-v3",
        "merge_commit": "b245852f9d6eda84ef2d5c2efbe5fcc14f3b1c9d",
        "anchors": {
            "docs/data/seed542-update-432-strength-screen-v3/verification-correction-v1.json": "6c11255164b60eb83ae06492bbf9307f8e5b7fc9a47e3277de5f32832959ec4b",
            "docs/data/seed542-update-432-strength-screen-v3/attempt-disposition.json": "e4f58e6d042e2c14e2a8c454685aabc94417ee0b58b08b9437b991d586e400c0",
            "docs/data/seed542-update-432-strength-screen-v3/independent-verification.json": "7612269cbb4fc93aae4b7080bb7eac42787182973ee268d040a807b20651504e",
            "docs/data/seed542-update-432-strength-screen-v3/continuation-registration-v6.json": "62d51231c933241c6fb08083abd10f417fe30af66872c7e69118be07b86d99e9",
            "docs/data/seed542-update-432-strength-screen-v3/continuation-execution-freeze-v6.json": "240d92c0ee6db0ae01d292cb310e8db1dbc9223b1e81fdfbe0a8b3ec8f693a5c",
            "docs/data/seed542-update-432-strength-screen-v3/continuation-pair-bindings-v6.jsonl": "e8cdb731c25a50ea7d808ab7b838e2719ac25eddc8fd6c8125d38844069e3266",
            "docs/data/seed542-update-432-strength-screen-v3/registration.json": "8541fef485ac915cbf2b32ac7b46f49e43e9fa867fb8411822594eaa824059f6",
            "docs/data/seed542-update-432-strength-screen-v3/execution-freeze.json": "65286df075d662232a439e7288285fd31d34a9cfc5f4d5d5aa2b6b4006cd60ab",
            "docs/data/seed542-update-432-strength-screen-v3/openings.jsonl": replay.SUITE_SHA256,
            "docs/data/seed542-update-432-strength-screen-v3/outcomes.jsonl": "8cb6f4523c9832d6c79a15a363548ad989d9da4c145e3729017c9be29cd3ddaf",
        },
        "receipt": "independent-verification.json",
        "recovered_openings": [0, 1, 2],
        "failed_attempts": [
            "docs/data/seed542-update-432-strength-screen-v3/attempt-disposition.json"
        ],
    },
    {
        "key": "551",
        "name": "seed551-controlled-policy-strength-screen",
        "archive": "docs/data/seed551-controlled-policy-strength-screen",
        "merge_commit": "ff0805ea6572dcf297a2f13db66abab28ad915e8",
        "anchors": {
            "docs/data/seed551-controlled-policy-strength-screen/verification-receipt.json": "c0ae73dc1a64b573dd669af34e075eefbb1d247bb21c5df58138419d68835b5d",
            "docs/data/seed551-controlled-policy-strength-screen/verification-receipt-supersession-02.json": "e58ded417510e3b059be2c00574dfb13eab99ee76a7d8b2d45a85474b72458ca",
            "docs/data/seed551-physical-copy-verification.json": "c972f7d921d74b694fa96592544f7fec4ec557024a46f2258f70a6b1009ccbfc",
            "docs/data/seed551-controlled-policy-strength-screen/registration.json": "ea0a95409fee37bf91d967b1842c6dd65574c5cccb814f6ec623c3c4549d55a0",
            "docs/data/seed551-controlled-policy-strength-screen/execution-freeze.json": "fc8ab0a7a9a1441417b8ad83efd203ab79adc76a877bd9f388e394776e3fdbe9",
            "docs/data/seed551-controlled-policy-strength-screen/openings.jsonl": replay.SUITE_SHA256,
            "docs/data/seed551-controlled-policy-strength-screen/outcomes.jsonl": "109874fc29ae3f42303a44bf2335dfaf76fe68fdc04c4fb5d8f105236706b3ba",
            "docs/data/seed551-controlled-policy-strength-screen/recovery/recovery-0001.json": "585e44a4b09b8bd2d496fa9c7b8e699e1d671ad6650bcc9a33b570407bfdede6",
        },
        "receipt": "verification-receipt.json",
        "recovered_openings": [1],
        "failed_attempts": [
            "docs/data/seed551-controlled-policy-strength-screen/verification-failure-0001.json"
        ],
    },
    {
        "key": "552",
        "name": "seed552-c108-serving-p",
        "archive": "docs/data/seed552-c108-serving-p",
        "merge_commit": "ab1376c00e71627663ce474c1ea46b5b6e55c3e2",
        "anchors": {
            "docs/data/seed552-c108-serving-p/publication-receipt.json": "f86b6058c8784c4c7ac48c5979a35db6de833e08a140d29bbe56cdf65fea3bfa",
            "docs/data/seed552-c108-serving-p/verification-receipt.json": "e538490a04eafac5d34df0ccc479e7a44828117ad01e8bfc8654395cec168c3d",
            "docs/data/seed552-c108-serving-p/physical-copy-verification-supersession-08.json": "32969d9c0424f39b876b77dc1a0fe25eefec8be54a486752501a74c93a5fea9a",
            "docs/data/seed552-c108-serving-p/registration.json": "4fb54a802af1d0fc137d25c89a30d2450ee2274f89847c51a9d4923b5ebf8013",
            "docs/data/seed552-c108-serving-p/execution-freeze.json": "1b8d78f7c8adda0a526f7ddf3925c6f3e1b70a3f07d2c616675d887e64bebcaf",
            "docs/data/seed552-c108-serving-p/openings.jsonl": replay.SUITE_SHA256,
            "docs/data/seed552-c108-serving-p/outcomes.jsonl": "6b0e2bb6fcfaea4e81b1b5261ac72fbe6406ab878cc08d4da01a18c28b206e8a",
        },
        "receipt": "verification-receipt.json",
        "recovered_openings": [],
        "failed_attempts": [
            "docs/data/seed552-c108-serving-p-preparation-attempt-0001/PREPARATION-FAILURE.json",
            "docs/data/seed552-c108-serving-p-preparation-attempt-0002/PREPARATION-FAILURE.json",
            "docs/data/seed552-c108-serving-p-preparation-attempt-0003/PREPARATION-FAILURE.json",
            "docs/data/seed552-c108-serving-p-preparation-attempt-0004/PREPARATION-FAILURE.json",
            "docs/data/seed552-c108-serving-p-preparation-attempt-0005/PREPARATION-FAILURE.json",
            "docs/data/seed552-c108-serving-p-preparation-attempt-0006/PREPARATION-FAILURE.json",
            "docs/data/seed552-c108-serving-p-preparation-attempt-0007/PREPARATION-FAILURE.json",
            "docs/data/seed552-c108-serving-p-preparation-attempt-0008/PREPARATION-FAILURE.json",
            "docs/data/seed552-c108-serving-p-preparation-attempt-0009/PREPARATION-FAILURE.json",
        ],
    },
    {
        "key": "555",
        "name": "seed555-policy-strength-screen",
        "archive": "docs/data/seed555-policy-strength-screen",
        "merge_commit": "47a7cb39f0733c338daccceab54f8aea614c9464",
        "anchors": {
            "docs/data/seed555-policy-strength-screen/verification-receipt.json": "a12b3284acb72ab817e27e79a265097da3873e7570c93fdbcd3a47810028dc37",
            "docs/data/seed555-policy-strength-screen/attempt-verification-correction-02.json": "f352600f6d9221a30b9249c92d45e59207911b7f4b56953a6618b730409121c4",
            "docs/data/seed555-policy-strength-screen/physical-copy-verification-correction-05.json": "723d5bb1708adb38fa429a5778ef3c2bdbe575dc6b922fa4d2beef2adc025e00",
            "docs/data/seed555-policy-strength-screen/registration.json": "491476eee5650c07e643576da07ff4fcaf6adaf3196a1b33c5092b22d9f77dc7",
            "docs/data/seed555-policy-strength-screen/execution-freeze.json": "e559e5c4821f119e6559a2f80712cbd6a554db4438ee5d113b394bc8da845d65",
            "docs/data/seed555-policy-strength-screen/openings.jsonl": replay.SUITE_SHA256,
            "docs/data/seed555-policy-strength-screen/outcomes.jsonl": "10a64e24d7901a05c1edf6adf7c32451c281cf3328b73229bfee827f445deb0e",
        },
        "receipt": "verification-receipt.json",
        "recovered_openings": [],
        "failed_attempts": [],
    },
)


class SevenSourceError(ValueError):
    """Raised when a source publication or trajectory cannot be reconciled."""


def validate_recovered_openings(recovered_openings: list[int]) -> None:
    """Reject recovered opening pairs that are duplicated or out of range."""
    if len(set(recovered_openings)) != len(recovered_openings) or any(
        index not in range(1024) for index in recovered_openings
    ):
        raise SevenSourceError("recovery_pair_accounting_invalid")


def _sha(path: Path) -> str:
    return replay.sha256_file(path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return replay._load_rows(path)


def _receipt_file_map(receipt: dict[str, Any]) -> dict[str, str]:
    mapping = receipt.get("files") or receipt.get("evidence_sha256") or {}
    return {str(path): str(digest) for path, digest in mapping.items()}


def bootstrap_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Build the fixed seven-source manifest after checking publication anchors."""
    source_rows = []
    for definition in SOURCES:
        anchor_hashes: dict[str, str] = {}
        for repo_relative, expected in definition["anchors"].items():
            path = root / repo_relative
            if not path.is_file() or _sha(path) != expected:
                raise SevenSourceError(
                    f"final_publication_binding_mismatch:{repo_relative}"
                )
            anchor_hashes[repo_relative] = expected
        archive = root / definition["archive"]
        receipt_path = archive / definition["receipt"]
        receipt = json.loads(receipt_path.read_text())
        receipt_files = _receipt_file_map(receipt)
        suite_sha = _sha(archive / "openings.jsonl")
        ledger_path = archive / "outcomes.jsonl"
        ledger_sha = _sha(ledger_path)
        ledger = _read_rows(ledger_path)
        if len(ledger) != 2048:
            raise SevenSourceError(f"ledger_population_mismatch:{definition['key']}")
        game_batches: dict[str, str] = {}
        for opening_index in range(1024):
            relative = f"batches/games-{opening_index:04d}.jsonl"
            path = archive / relative
            if not path.is_file():
                raise SevenSourceError(
                    f"game_batch_missing:{definition['key']}:{relative}"
                )
            actual = _sha(path)
            receipt_digest = receipt_files.get(relative)
            if receipt_digest is not None and receipt_digest != actual:
                raise SevenSourceError(
                    f"receipt_game_binding_mismatch:{definition['key']}:{relative}"
                )
            game_batches[relative] = actual
        source_rows.append(
            {
                "key": definition["key"],
                "name": definition["name"],
                "archive": definition["archive"],
                "merge_commit": definition["merge_commit"],
                "receipt_path": f"{definition['archive']}/{definition['receipt']}",
                "receipt_sha256": _sha(receipt_path),
                "anchor_files": anchor_hashes,
                "registered_suite_sha256": suite_sha,
                "outcomes_sha256": ledger_sha,
                "game_batches": game_batches,
                "recovered_openings": definition["recovered_openings"],
                "failed_or_superseded_attempts": {
                    path: _sha(root / path)
                    for path in definition["failed_attempts"]
                    if (root / path).is_file()
                },
                "receipt_file_count": len(receipt_files),
            }
        )
    if len({row["key"] for row in source_rows}) != len(SOURCES):
        raise SevenSourceError("duplicate_source_key")
    return {"schema": "seed564_seven_source_manifest_v1", "sources": source_rows}


def _source_snapshot(root: Path, source: dict[str, Any]) -> set[Path]:
    archive = root / source["archive"]
    files = {root / path for path in source["anchor_files"]} | {
        archive / source["receipt_path"].split("/", 3)[-1],
        archive / "openings.jsonl",
        archive / "outcomes.jsonl",
        *(archive / relative for relative in source["game_batches"]),
        *(archive / f"batches/report-{index:04d}.json" for index in range(1024)),
    }
    files.update(root / path for path in source["failed_or_superseded_attempts"])
    return files


def _winner_label(row: dict[str, Any]) -> str | None:
    for key in ("winner", "outcome"):
        value = row.get(key)
        if value in ("challenger", "draw"):
            return value
        if value in ("current", "opponent", "initializer", "P"):
            return "current"
    return None


def _ledger_matches_game(
    source_key: str, ledger: dict[str, Any], row: dict[str, Any], margin: int
) -> None:
    if "challenger_margin" in ledger and int(ledger["challenger_margin"]) != margin:
        raise SevenSourceError(f"ledger_margin_mismatch:{source_key}")
    if "margin" in ledger and int(ledger["margin"]) != margin:
        raise SevenSourceError(f"ledger_margin_mismatch:{source_key}")
    if "winner" in ledger:
        ledger_winner = ledger["winner"]
        valid = {
            "challenger": margin > 0,
            "opponent": margin < 0,
            "current": margin < 0,
            "draw": margin == 0,
        }.get(ledger_winner, False)
        if not valid:
            raise SevenSourceError(f"ledger_winner_mismatch:{source_key}")
    if "outcome" in ledger:
        outcome = ledger["outcome"]
        if (margin == 0 and outcome != "draw") or (margin != 0 and outcome == "draw"):
            raise SevenSourceError(f"ledger_outcome_mismatch:{source_key}")
    raw_winner = _winner_label(row)
    if raw_winner is None:
        raise SevenSourceError(f"game_winner_label_unknown:{source_key}")
    if (
        (margin > 0 and raw_winner != "challenger")
        or (margin < 0 and raw_winner != "current")
        or (margin == 0 and raw_winner != "draw")
    ):
        raise SevenSourceError(f"game_winner_mismatch:{source_key}")


def load_verified_baseline(root: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Rebuild #562+#563 identity unions from their pinned published indexes."""
    for relative, expected in BASELINE_INDEX_BINDINGS.items():
        path = root / relative
        if not path.is_file() or _sha(path) != expected:
            raise SevenSourceError(
                f"baseline_published_index_binding_mismatch:{relative}"
            )
    source_manifest_562 = json.loads(
        (
            root / "docs/data/seed556-560-exclusion-inventory/source-manifest.json"
        ).read_text()
    )
    source_manifest_563 = json.loads(
        (
            root
            / "docs/data/expanded-training-dev-exclusion-inventory/source-manifest.json"
        ).read_text()
    )
    count_562 = json.loads(
        (
            root / "docs/data/seed556-560-exclusion-inventory/count-report.json"
        ).read_text()
    )
    count_563 = json.loads(
        (
            root
            / "docs/data/expanded-training-dev-exclusion-inventory/count-report.json"
        ).read_text()
    )
    identities = {"canonical": set(), "kalah_v3_float32": set()}
    index_562 = (
        root / "docs/data/seed556-560-exclusion-inventory/identity-inventory.json"
    )
    for row in _read_rows(index_562):
        scheme = str(row.get("scheme"))
        if scheme not in identities or not isinstance(row.get("identity"), str):
            raise SevenSourceError("baseline_562_index_row_invalid")
        identities[scheme].add(row["identity"])
    index_563 = (
        root
        / "docs/data/expanded-training-dev-exclusion-inventory/identity-inventory.jsonl"
    )
    for row in _read_rows(index_563):
        for scheme, field in (
            ("canonical", "canonical_state_sha256"),
            ("kalah_v3_float32", "kalah_v3_float32_sha256"),
        ):
            if not isinstance(row.get(field), str):
                raise SevenSourceError("baseline_563_index_row_invalid")
            identities[scheme].add(row[field])
    expected_report = count_563["families_vs_562"]["combined"]
    for scheme in identities:
        published_562 = count_562["overlap"][scheme]
        if int(published_562["union"]) != 58_101:
            raise SevenSourceError(f"baseline_562_count_report_mismatch:{scheme}")
        if len(identities[scheme]) != BASELINE_IDENTITIES:
            raise SevenSourceError(f"baseline_rebuilt_union_count_mismatch:{scheme}")
        if (
            int(expected_report[scheme]["562_distinct"]) != 58_101
            or int(expected_report[scheme]["distinct"]) != 91_082
            or int(expected_report[scheme]["union_with_562"]) != BASELINE_IDENTITIES
        ):
            raise SevenSourceError(f"baseline_count_report_mismatch:{scheme}")
        if (
            len(
                {
                    row["identity"]
                    for row in _read_rows(index_562)
                    if row["scheme"] == scheme
                }
            )
            != 58_101
        ):
            raise SevenSourceError(f"baseline_562_rebuilt_count_mismatch:{scheme}")
    if (
        source_manifest_562.get("anchor_commit")
        != "2e92bbfd3f90c9517c4c4449d3bebd750f22ee71"
    ):
        raise SevenSourceError("baseline_562_source_manifest_anchor_mismatch")
    if (
        source_manifest_563.get("anchor_commit")
        != "3e493173345cd3119d62bf7173db3a512ed80966"
    ):
        raise SevenSourceError("baseline_563_source_manifest_anchor_mismatch")
    summary = {
        "reconstructed_from_published_indexes": True,
        "#562_distinct": 58_101,
        "#563_distinct": 91_082,
        "#562_plus_#563_union": BASELINE_IDENTITIES,
        "index_sha256": BASELINE_INDEX_BINDINGS,
    }
    return identities, summary


def replay_one_source(
    root: Path, source: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], set[Path]]:
    """Authenticate then replay one source using its published outcomes ledger."""
    archive = root / source["archive"]
    if _sha(archive / "openings.jsonl") != source["registered_suite_sha256"]:
        raise SevenSourceError(f"suite_sha_mismatch:{source['key']}")
    if _sha(archive / "outcomes.jsonl") != source["outcomes_sha256"]:
        raise SevenSourceError(f"outcomes_sha_mismatch:{source['key']}")
    receipt_path = root / source["receipt_path"]
    if _sha(receipt_path) != source["receipt_sha256"]:
        raise SevenSourceError(f"final_receipt_sha_mismatch:{source['key']}")
    for repo_relative, expected in source["anchor_files"].items():
        if _sha(root / repo_relative) != expected:
            raise SevenSourceError(
                f"final_corrected_binding_mismatch:{source['key']}:{repo_relative}"
            )
    for relative, expected in source["game_batches"].items():
        if _sha(archive / relative) != expected:
            raise SevenSourceError(
                f"authenticated_batch_mismatch:{source['key']}:{relative}"
            )
    for relative, expected in source["failed_or_superseded_attempts"].items():
        if _sha(root / relative) != expected:
            raise SevenSourceError(
                f"noncommitted_attempt_binding_mismatch:{source['key']}:{relative}"
            )
    openings = _read_rows(archive / "openings.jsonl")
    ledger_rows = _read_rows(archive / "outcomes.jsonl")
    if len(openings) != 1024 or len(ledger_rows) != 2048:
        raise SevenSourceError(f"population_size_mismatch:{source['key']}")
    ledger: dict[tuple[int, int], dict[str, Any]] = {}
    for record in ledger_rows:
        key = (
            int(record["opening_index"]),
            int(record.get("game_position", record.get("batch_game_position", -1))),
        )
        if key[1] not in (0, 1) or key in ledger:
            raise SevenSourceError(
                f"duplicate_or_invalid_committed_game:{source['key']}:{key}"
            )
        ledger[key] = record
    occurrences: list[dict[str, Any]] = []
    for opening_index, opening in enumerate(openings):
        if int(opening.get("opening_index", -1)) != opening_index:
            raise SevenSourceError(
                f"opening_index_mismatch:{source['key']}:{opening_index}"
            )
        relative = f"batches/games-{opening_index:04d}.jsonl"
        batch_path = archive / relative
        batch_sha = _sha(batch_path)
        game_rows = _read_rows(batch_path)
        if len(game_rows) != 2:
            raise SevenSourceError(
                f"pair_size_mismatch:{source['key']}:{opening_index}"
            )
        report_path = archive / f"batches/report-{opening_index:04d}.json"
        report_sha = _sha(report_path)
        for position, row in enumerate(game_rows):
            record = ledger.get((opening_index, position))
            if record is None:
                raise SevenSourceError(
                    f"committed_game_missing:{source['key']}:{opening_index}:{position}"
                )
            seat = int(row.get("challenger_player", -1))
            if seat not in (0, 1):
                raise SevenSourceError(
                    f"invalid_seat:{source['key']}:{opening_index}:{position}"
                )
            ledger_batch_hash = record.get(
                "game_sha256", record.get("batch_game_sha256")
            )
            if ledger_batch_hash != batch_sha:
                raise SevenSourceError(
                    f"ledger_batch_hash_mismatch:{source['key']}:{opening_index}"
                )
            ledger_report_hash = record.get(
                "report_sha256", record.get("batch_report_sha256")
            )
            if ledger_report_hash is not None and ledger_report_hash != report_sha:
                raise SevenSourceError(
                    f"ledger_report_hash_mismatch:{source['key']}:{opening_index}"
                )
            if (
                "challenger_player" in record
                and int(record["challenger_player"]) != seat
            ):
                raise SevenSourceError(
                    f"ledger_seat_mismatch:{source['key']}:{opening_index}:{position}"
                )
            try:
                trace, margin = replay.replay_game_trace(
                    opening,
                    row,
                    seat,
                    source=f"PR #{source['key']}",
                    run=source["name"],
                    opening_index=opening_index,
                    game_position=position,
                    batch_path=relative,
                )
            except replay.ArchiveVerificationError as exc:
                raise SevenSourceError(
                    f"trajectory_or_association_failure:{source['key']}:{opening_index}:{position}:{exc}"
                ) from exc
            occurrences.extend(trace)
            _ledger_matches_game(source["key"], record, row, margin)
    if len(ledger) != 2048:
        raise SevenSourceError(f"unaccounted_ledger_games:{source['key']}")
    recovered = source["recovered_openings"]
    try:
        validate_recovered_openings(recovered)
    except SevenSourceError as exc:
        raise SevenSourceError(
            f"recovery_pair_accounting_invalid:{source['key']}"
        ) from exc
    expected_pairs = {(index, pos) for index in range(1024) for pos in (0, 1)}
    if set(ledger) != expected_pairs:
        raise SevenSourceError(f"ledger_pair_population_incomplete:{source['key']}")
    return (
        occurrences,
        {"games": len(ledger), "pairs": 1024, "recovered_openings": recovered},
        _source_snapshot(root, source),
    )


def verify_seven(
    root: Path, output: Path, *, write_manifest: bool = False
) -> dict[str, Any]:
    """Build or execute the public seven-screen verifier and provenance census."""
    root, output = root.resolve(), output.resolve()
    manifest_path = output / MANIFEST_NAME
    if write_manifest:
        manifest = bootstrap_manifest(root)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        return {
            "status": "seven_source_manifest_created",
            "manifest_sha256": _sha(manifest_path),
        }
    if not manifest_path.is_file() or _sha(manifest_path) != MANIFEST_SHA256:
        raise SevenSourceError("pinned_seven_source_manifest_mismatch")
    manifest = json.loads(manifest_path.read_text())
    if len(manifest.get("sources", [])) != 7:
        raise SevenSourceError("seven_source_coverage_mismatch")
    audit_evidence = {
        manifest_path,
        *(root / relative for relative in BASELINE_INDEX_BINDINGS),
    }
    audit_evidence_before = replay._snapshot(audit_evidence)
    baseline, baseline_report = load_verified_baseline(root)
    all_occurrences: list[dict[str, Any]] = []
    source_reports = []
    source_sets: dict[str, dict[str, set[str]]] = {}
    for source in manifest["sources"]:
        evidence = _source_snapshot(root, source)
        before = replay._snapshot(evidence)
        occurrences, replay_report, evidence = replay_one_source(root, source)
        canon_rows, float_rows, counts = replay.identity_indexes(occurrences)
        if before != replay._snapshot(evidence):
            raise SevenSourceError(f"source_evidence_changed:{source['key']}")
        source_sets[source["key"]] = {
            "canonical": {row["identity"] for row in canon_rows},
            "kalah_v3_float32": {row["identity"] for row in float_rows},
            "opening_canonical": {
                row["identity"] for row in canon_rows if row["ply"] == -1
            },
            "opening_float32": {
                row["identity"] for row in float_rows if row["ply"] == -1
            },
            "continuation_canonical": {
                row["identity"] for row in canon_rows if row["ply"] >= 0
            },
            "continuation_float32": {
                row["identity"] for row in float_rows if row["ply"] >= 0
            },
        }
        all_occurrences.extend(occurrences)
        source_reports.append(
            {
                "source": source["key"],
                "name": source["name"],
                "replay": replay_report,
                "identity_counts": counts,
                "failed_or_superseded_attempts_separate": source[
                    "failed_or_superseded_attempts"
                ],
                "final_receipt_sha256": source["receipt_sha256"],
            }
        )
    canonical_rows, float_rows, combined_counts = replay.identity_indexes(
        all_occurrences
    )
    unions = {}
    pairwise = []
    for scheme in ("canonical", "kalah_v3_float32"):
        identities = set().union(
            *(source_sets[source["key"]][scheme] for source in manifest["sources"])
        )
        opening_key = f"opening_{'canonical' if scheme == 'canonical' else 'float32'}"
        continuation_key = (
            f"continuation_{'canonical' if scheme == 'canonical' else 'float32'}"
        )
        openings = set().union(
            *(source_sets[source["key"]][opening_key] for source in manifest["sources"])
        )
        continuations = set().union(
            *(
                source_sets[source["key"]][continuation_key]
                for source in manifest["sources"]
            )
        )
        unions[scheme] = {
            "distinct_across_sources": len(identities),
            "opening_root_union_distinct": len(openings),
            "continuation_union_distinct": len(continuations),
            "opening_root_additions_beyond_562_563": len(openings - baseline[scheme]),
            "continuation_additions_beyond_562_563": len(
                continuations - baseline[scheme]
            ),
            "additions_beyond_562_563_baseline": len(identities - baseline[scheme]),
            "overlap_with_562_563_baseline": len(identities & baseline[scheme]),
        }
        for source, source_report in zip(
            manifest["sources"], source_reports, strict=True
        ):
            selected = source_sets[source["key"]]
            source_report.setdefault("baseline_comparison", {})[scheme] = {
                "distinct": len(selected[scheme]),
                "overlap_with_562_563": len(selected[scheme] & baseline[scheme]),
                "additions_beyond_562_563": len(selected[scheme] - baseline[scheme]),
                "opening_root_overlap_with_562_563": len(
                    selected[opening_key] & baseline[scheme]
                ),
                "opening_root_additions_beyond_562_563": len(
                    selected[opening_key] - baseline[scheme]
                ),
                "continuation_overlap_with_562_563": len(
                    selected[continuation_key] & baseline[scheme]
                ),
                "continuation_additions_beyond_562_563": len(
                    selected[continuation_key] - baseline[scheme]
                ),
            }
        for left_index, left in enumerate(manifest["sources"]):
            for right in manifest["sources"][left_index + 1 :]:
                left_sets, right_sets = (
                    source_sets[left["key"]],
                    source_sets[right["key"]],
                )
                pairwise.append(
                    {
                        "scheme": scheme,
                        "left": left["key"],
                        "right": right["key"],
                        "all_state_overlap": len(
                            left_sets[scheme] & right_sets[scheme]
                        ),
                        "opening_root_overlap": len(
                            left_sets[opening_key] & right_sets[opening_key]
                        ),
                        "continuation_overlap": len(
                            left_sets[continuation_key] & right_sets[continuation_key]
                        ),
                        "left_root_right_continuation": len(
                            left_sets[opening_key] & right_sets[continuation_key]
                        ),
                        "right_root_left_continuation": len(
                            right_sets[opening_key] & left_sets[continuation_key]
                        ),
                    }
                )
    report = {
        "schema": "seed564_seven_source_replay_report_v1",
        "baseline_verified": True,
        "evidence_bytes_and_mtimes_unchanged": audit_evidence_before
        == replay._snapshot(audit_evidence),
        "baseline_identities": BASELINE_IDENTITIES,
        "baseline_reconciliation": baseline_report,
        "sources": source_reports,
        "combined_identity_occurrences": combined_counts["occurrences"],
        "source_union": unions,
        "pairwise_overlaps": pairwise,
        "root_overlap_is_not_continuation_overlap": True,
        "recorded_seed461_continuations_unresolved": True,
        "fresh_confirmation_gate_483": "closed",
    }
    if not report["evidence_bytes_and_mtimes_unchanged"]:
        raise SevenSourceError(
            "baseline_or_source_manifest_changed_during_verification"
        )
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("canonical", canonical_rows), ("kalah-v3-float32", float_rows)):
        body = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ).encode()
        (output / f"seven-source-{name}-index.jsonl.gz").write_bytes(
            gzip.compress(body, mtime=0)
        )
    (output / "seven-source-count-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=AUDIT)
    parser.add_argument("--write-source-manifest", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_seven(
            args.root, args.out, write_manifest=args.write_source_manifest
        )
    except (
        SevenSourceError,
        replay.ArchiveVerificationError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        parser.exit(2, f"verification failed: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
