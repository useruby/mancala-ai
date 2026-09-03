"""Immutable consumed-opening registry for sealed AlphaZero evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import run_pr249_fresh_suite_generalization as pr249
from ml.alphazero_lite import run_pr251_cross_seed_strength_residual_transfer as pr251
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file


@dataclass(frozen=True)
class ConsumedSuite:
    label: str
    path: Path
    seed: int | None
    sha256: str
    status: str = "consumed"


_ROOT = Path("/tmp")
_SPECS = (
    ConsumedSuite(
        "canonical",
        pr249.CANONICAL_SUITE,
        None,
        "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04",
    ),
    ConsumedSuite(
        "A",
        _ROOT / "azlite_pr249_fresh_suite_generalization/suites/suite_A.jsonl",
        1042,
        "c8277e659c7a4e137140d83c187781f40e6b25c4b1dff5ec4da3f2e09fdcc6ab",
    ),
    ConsumedSuite(
        "B",
        _ROOT / "azlite_pr249_fresh_suite_generalization/suites/suite_B.jsonl",
        2042,
        "1f4c17eb7df21af75bc29c3274b3951899ae5fb2522f762d5270d58ddf93b37e",
    ),
    ConsumedSuite(
        "C",
        _ROOT / "azlite_pr249_fresh_suite_generalization/suites/suite_C.jsonl",
        3042,
        "b56783a2a2bbf63168cfb642f2b878badc80ace0279a9bbb7778757a4e4ba90d",
    ),
    ConsumedSuite(
        "D",
        _ROOT
        / "azlite_pr251_cross_seed_strength_residual_transfer/suites/suite_D.jsonl",
        4042,
        "e9d931031e75d39d188699d0b77ecc91d429c6998d06aada5f04e34b0384b1e2",
    ),
    ConsumedSuite(
        "E",
        _ROOT
        / "azlite_pr251_cross_seed_strength_residual_transfer/suites/suite_E.jsonl",
        5042,
        "3e78f5425370bb17ed2280e850a5f93b62cb86d94a0222263d481fcf866def37",
    ),
    ConsumedSuite(
        "F",
        _ROOT
        / "azlite_pr251_cross_seed_strength_residual_transfer/suites/suite_F.jsonl",
        6042,
        "8cba0ea407dde34696877d5ee2fe7110cd5610b2392f446df1bc02c1900d7c0e",
    ),
    ConsumedSuite(
        "G",
        _ROOT / "azlite_pr252_phase_target_delta_attribution/suites/suite_G.jsonl",
        7042,
        "ec331fc5672d0af95083443620ede6aac68755280ba375a9727bdd781918b216",
    ),
    ConsumedSuite(
        "H",
        _ROOT / "azlite_pr252_phase_target_delta_attribution/suites/suite_H.jsonl",
        8042,
        "24e4ae8cb9ab336959f243c1dedd903201497d94e04fbc99ddaeed8894c58682",
    ),
    ConsumedSuite(
        "I",
        _ROOT / "azlite_pr252_phase_target_delta_attribution/suites/suite_I.jsonl",
        9042,
        "5c8265a4c387c1a2f40ddc492da0435db548b98e02692582ccbf4b119f621b8e",
    ),
    ConsumedSuite(
        "J",
        _ROOT / "azlite_pr253_semantic_receiver_target_surgery/suites/suite_J.jsonl",
        10042,
        "7c0b097e18949a2d0ac5657116c565f3b9c0bb942ae69c9741dcd328915bf98b",
    ),
    ConsumedSuite(
        "K",
        _ROOT / "azlite_pr253_semantic_receiver_target_surgery/suites/suite_K.jsonl",
        11042,
        "64725ca85c6657eeefbbc329bc8a03f762bbbb7b615ff4fefeb4356fa158e1fd",
    ),
    ConsumedSuite(
        "L",
        _ROOT / "azlite_pr253_semantic_receiver_target_surgery/suites/suite_L.jsonl",
        12042,
        "147ff9d503975641da2ad366e1df93aa07a296190f850ed0514ecbf903b43ac0",
    ),
    ConsumedSuite(
        "M",
        _ROOT / "azlite_pr254_third_seed_budget_replay/suites/suite_M.jsonl",
        13042,
        "2d5552eaa68c2e5ad8397fb8a0c3eb1e9831a8a668991457dd9afe0c2b2960aa",
        "consumed_invalid_due_to_overlap",
    ),
    ConsumedSuite(
        "N",
        _ROOT / "azlite_pr254_third_seed_budget_replay/suites/suite_N.jsonl",
        14042,
        "5c37f39a1ea50478e3bbb17f5c0c02cb55068d8a9180c324e39b30cb83fc464e",
        "consumed_invalid_due_to_overlap",
    ),
    ConsumedSuite(
        "O",
        _ROOT / "azlite_pr254_third_seed_budget_replay/suites/suite_O.jsonl",
        15042,
        "0e066bbd3e6288efe4a7ba7bc26c29b27358bf8d85d686801a0a0521c8a4425b",
        "consumed_invalid_due_to_overlap",
    ),
    ConsumedSuite(
        "P",
        _ROOT / "azlite_pr254_third_seed_budget_repair/suites/suite_P.jsonl",
        16042,
        "25f8de0e48421e6b112b1bd16425d2f9d645a235694dcd2757d9ae0e9a68e8b6",
    ),
    ConsumedSuite(
        "Q",
        _ROOT / "azlite_pr254_third_seed_budget_repair/suites/suite_Q.jsonl",
        17042,
        "5ba80e5857c9f52f533c6a3e3b1548a77daebbe45d806d2c76c6b9b425767ef5",
    ),
    ConsumedSuite(
        "R",
        _ROOT / "azlite_pr254_third_seed_budget_repair/suites/suite_R.jsonl",
        18042,
        "56766034b7027cccce8492a4cb80ca23f2edec409f2d07b9cd39cb55bfdd106b",
    ),
    ConsumedSuite(
        "S",
        _ROOT / "azlite_pr256_fresh768_replay_replication/suites/suite_S.jsonl",
        19042,
        "59224ca54c9102363cc00e50d424497485e13372909e59e0454a2ce2ae31f619",
    ),
    ConsumedSuite(
        "T",
        _ROOT / "azlite_pr256_fresh768_replay_replication/suites/suite_T.jsonl",
        20042,
        "e73cb13277168cf0d449f942525d16dd032cdd8e396393f598ac35af612526dd",
    ),
    ConsumedSuite(
        "U",
        _ROOT / "azlite_pr256_fresh768_replay_replication/suites/suite_U.jsonl",
        21042,
        "efcab1d3aecff06c223245a1b03f5679c2038168de6b4c46ffa557385dcf349a",
    ),
    ConsumedSuite(
        "V",
        _ROOT / "azlite_pr258_two_replay_aggregation/suites/suite_V.jsonl",
        22042,
        "b08ef373f8a958b7279afa3dde481e582a465451cc520160b737f5b4b90195e2",
    ),
    ConsumedSuite(
        "W",
        _ROOT / "azlite_pr258_two_replay_aggregation/suites/suite_W.jsonl",
        23042,
        "f6c016c67e51f5e829ee70e93be5614dbbef312cd1db9f95908b3363520d5712",
    ),
    ConsumedSuite(
        "X",
        _ROOT / "azlite_pr258_two_replay_aggregation/suites/suite_X.jsonl",
        24042,
        "00a93262cbbf2ca9dc0b97b5275337c80badb1aae338f25854d98628527184a0",
    ),
    ConsumedSuite(
        "Y",
        _ROOT / "azlite_pr259_two_replay_second_epoch/suites/suite_Y.jsonl",
        25042,
        "f6247d0e59b78af776270f80ecd6a00fc84c8fbcb69a462cbf07d5ecee49a6a9",
    ),
    ConsumedSuite(
        "Z",
        _ROOT / "azlite_pr259_two_replay_second_epoch/suites/suite_Z.jsonl",
        26042,
        "2b004a0e1a52f9a2776d1dfc459181d1bf89b45c591ac0f9a4fd39245ee2e996",
    ),
    ConsumedSuite(
        "AA",
        _ROOT / "azlite_pr259_two_replay_second_epoch/suites/suite_AA.jsonl",
        27042,
        "629e997fadca2f220cab61684ca6bc77116154e42c7371ebaf91b83b09093375",
    ),
    ConsumedSuite(
        "AB",
        _ROOT / "azlite_pr260_value_head_refresh/suites/suite_AB.jsonl",
        28042,
        "527cc607a3dddbfcaf0203783b58f6b38b9df2fcea2ebc6cbb112d54d7727641",
    ),
    ConsumedSuite(
        "AC",
        _ROOT / "azlite_pr260_value_head_refresh/suites/suite_AC.jsonl",
        29042,
        "93786884529f4d75a22a82e5a87074db1f42b5bcb43aaecc42a0abcc28370bcf",
    ),
    ConsumedSuite(
        "AD",
        _ROOT / "azlite_pr260_value_head_refresh/suites/suite_AD.jsonl",
        30042,
        "28b631f74992e81f3227ac3355de6d3cca1ee9724199742039e8331389f98d87",
    ),
    ConsumedSuite(
        "AE",
        _ROOT / "azlite_pr261_policy_representation/suites/suite_AE.jsonl",
        31042,
        "21012b3a1eb54f1209de34468390a5f0e4ca123fe1b1b6676f6d3def404e2f05",
    ),
    ConsumedSuite(
        "AF",
        _ROOT / "azlite_pr261_policy_representation/suites/suite_AF.jsonl",
        32042,
        "f5b09723b807ea820c9338f5ae9c07bcbb82fceeaeaa8db2ae7bb509329eeddd",
    ),
    ConsumedSuite(
        "AG",
        _ROOT / "azlite_pr261_policy_representation/suites/suite_AG.jsonl",
        33042,
        "d16a01beaae8af6959523e715c5272e6e6c51fbe21a52d0ff86ef096f780fcb2",
    ),
    ConsumedSuite(
        "AH",
        _ROOT / "azlite_pr262_policy_hidden_capacity/suites/suite_AH.jsonl",
        34042,
        "c16148b43cb652f2dc28ca4b8e94c67f66da471f3cc36d318e73ce3258483784",
    ),
    ConsumedSuite(
        "AI",
        _ROOT / "azlite_pr262_policy_hidden_capacity/suites/suite_AI.jsonl",
        35042,
        "1c1de16cc5c4f16696858b054c07747575301e27ff9308f270dfdc4cfd13579b",
    ),
    ConsumedSuite(
        "AJ",
        _ROOT / "azlite_pr262_policy_hidden_capacity/suites/suite_AJ.jsonl",
        36042,
        "95b3c2dc333a5411562b1a1aeeccb0e093a1af9f7e6f4aa4b61362301416798d",
    ),
    ConsumedSuite(
        "AK",
        _ROOT / "azlite_pr264_joint_alphazero_iteration/suites/suite_AK.jsonl",
        37042,
        "c7df7293b641cac5b28a18424ab6489707dc1be15e65796392fea19258d6c57c",
    ),
    ConsumedSuite(
        "AL",
        _ROOT / "azlite_pr264_joint_alphazero_iteration/suites/suite_AL.jsonl",
        38042,
        "15e450006cf9299831d57147db2fe4ab9e9ec5182d8eff5cc921d17ad7c6267d",
    ),
    ConsumedSuite(
        "AM",
        _ROOT / "azlite_pr264_joint_alphazero_iteration/suites/suite_AM.jsonl",
        39042,
        "228bb3bda7be11675d8845fbb55b6e88badbde9b4c2f6b04a9e83d5df69b0bc2",
    ),
    ConsumedSuite(
        "AN",
        _ROOT / "azlite_pr265_unique_data_scale/suites/suite_AN.jsonl",
        40042,
        "5d31671aa6b5b86beb066b74848ce9fffcf05630a7d735ac845ae0f380dd0f0f",
    ),
    ConsumedSuite(
        "AO",
        _ROOT / "azlite_pr265_unique_data_scale/suites/suite_AO.jsonl",
        41042,
        "dec338722f5a8b5442cb6163e19dfa39ee1dae9c9d94b3a1088eeb2f27c6fc00",
    ),
    ConsumedSuite(
        "AP",
        _ROOT / "azlite_pr265_unique_data_scale/suites/suite_AP.jsonl",
        42042,
        "a91361b16ffb885080bc6d2e8ff7d8be35aa8213abcda28542dc0e63453bfba6",
    ),
)


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def _reconstruct(labels: tuple[str, ...], output: Path) -> dict[str, Path]:
    """Recreate J/L or M/O using their originally committed selector contract."""
    preceding = [
        spec
        for spec in _SPECS
        if spec.label in ("canonical", "A", "B", "C", "D", "E", "F", "G", "H", "I")
    ]
    entries = {
        spec.label: suites.load_suite_jsonl(str(spec.path)) for spec in preceding
    }
    used = set().union(*(pr249.suite_keys(value) for value in entries.values()))
    selected_paths: dict[str, Path] = {}
    for spec in (spec for spec in _SPECS if spec.label in labels):
        selected = suites.select_diverse(
            [
                entry
                for entry in pr249.all_openings()
                if suites.canonical_key(entry["state"]) not in used
            ],
            128,
            spec.seed,
        )
        path = output / f"suite_{spec.label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        if sha256_file(path) != spec.sha256:
            fail(f"{spec.label} deterministic reconstruction SHA mismatch")
        selected_paths[spec.label] = path
        used |= pr249.suite_keys(selected)
    return selected_paths


def load(workdir: Path) -> dict[str, ConsumedSuite]:
    """Load the authoritative registry, validating every consumed suite SHA."""
    if len({spec.label for spec in _SPECS}) != len(_SPECS):
        fail("duplicate consumed-suite label")
    missing_jkl = any(
        not spec.path.is_file() for spec in _SPECS if spec.label in ("J", "K", "L")
    )
    recovered = (
        _reconstruct(("J", "K", "L"), workdir / "recovered_consumed")
        if missing_jkl
        else {}
    )
    missing_mno = any(
        not spec.path.is_file() for spec in _SPECS if spec.label in ("M", "N", "O")
    )
    if missing_mno:
        recovered.update(_reconstruct(("M", "N", "O"), workdir / "recovered_consumed"))
    result = {}
    for spec in _SPECS:
        path = recovered.get(spec.label, spec.path)
        if not path.is_file() or sha256_file(path) != spec.sha256:
            fail(f"consumed suite SHA mismatch: {spec.label}")
        result[spec.label] = ConsumedSuite(
            spec.label, path, spec.seed, spec.sha256, spec.status
        )
    # M/N/O must retain the known defective overlap fingerprint even when their files exist.
    jkl = set().union(
        *(
            pr249.suite_keys(suites.load_suite_jsonl(str(result[label].path)))
            for label in ("J", "K", "L")
        )
    )
    if {
        label: len(
            pr249.suite_keys(suites.load_suite_jsonl(str(result[label].path))) & jkl
        )
        for label in ("M", "N", "O")
    } != {"M": 13, "N": 14, "O": 22}:
        fail("M/N/O historical overlap fingerprint mismatch")
    validate(result)
    return result


def validate(registry: dict[str, ConsumedSuite]) -> None:
    """Reject partial registries before either replay filtering or suite selection."""
    expected = {spec.label: spec.sha256 for spec in _SPECS}
    actual = {label: spec.sha256 for label, spec in registry.items()}
    if actual != expected:
        fail("consumed-suite registry mismatch")


def entries(registry: dict[str, ConsumedSuite]) -> dict[str, list[dict[str, Any]]]:
    return {
        label: suites.load_suite_jsonl(str(spec.path))
        for label, spec in registry.items()
    }


def final_keys(registry: dict[str, ConsumedSuite]) -> set[str]:
    return set().union(*(pr249.suite_keys(rows) for rows in entries(registry).values()))


def prefix_keys(registry: dict[str, ConsumedSuite]) -> set[tuple[int, ...]]:
    return set().union(
        *(pr251.prefix_keys(rows) for rows in entries(registry).values())
    )


def manifest(registry: dict[str, ConsumedSuite]) -> dict[str, Any]:
    return {
        label: {"seed": spec.seed, "sha256": spec.sha256, "status": spec.status}
        for label, spec in registry.items()
    }
