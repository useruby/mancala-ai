DEFAULT_INPUT_ENCODING = "kalah_v1"
BASE_FEATURE_ORDER = (
    "player_pits_0",
    "player_pits_1",
    "player_pits_2",
    "player_pits_3",
    "player_pits_4",
    "player_pits_5",
    "opponent_pits_0",
    "opponent_pits_1",
    "opponent_pits_2",
    "opponent_pits_3",
    "opponent_pits_4",
    "opponent_pits_5",
    "player_store",
    "opponent_store",
    "current_player",
)
KALAH_V3_EXTRA_FEATURE_ORDER = (
    "player_extra_turn_available",
    "player_capture_available",
    "player_empty_pit_ratio",
    "player_high_seed_ratio",
    "player_one_to_store_ratio",
    "player_remaining_stones",
    "opponent_extra_turn_available",
    "opponent_capture_available",
    "opponent_empty_pit_ratio",
    "opponent_high_seed_ratio",
    "opponent_one_to_store_ratio",
    "opponent_remaining_stones",
)
FEATURE_COUNTS = {
    "kalah_v1": 15,
    "kalah_v2": 15,
    "kalah_v3": 27,
}
SUPPORTED_INPUT_ENCODINGS = tuple(sorted(FEATURE_COUNTS))


def feature_count_for(input_encoding: str) -> int:
    try:
        return FEATURE_COUNTS[input_encoding]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_INPUT_ENCODINGS)
        raise ValueError(f"unsupported input_encoding {input_encoding!r}; expected one of: {supported}") from exc
