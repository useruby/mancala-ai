import numpy as np


MODES = {
    "classic_only": {
        "name": "classic_only",
        "use_policy": False,
        "use_value": False,
        "use_classic": True,
    },
    "policy_only": {
        "name": "policy_only",
        "use_policy": True,
        "use_value": False,
        "use_classic": False,
    },
    "value_only": {
        "name": "value_only",
        "use_policy": False,
        "use_value": True,
        "use_classic": False,
    },
    "full": {
        "name": "full",
        "use_policy": True,
        "use_value": True,
        "use_classic": False,
    },
}


def build_mode_config(mode_name):
    try:
        return dict(MODES[mode_name])
    except KeyError as exc:
        raise ValueError(f"unsupported ablation mode: {mode_name}") from exc


def neutral_value():
    return 0.0


def flat_legal_priors(legal_moves):
    priors = np.zeros(6, dtype=np.float32)
    if len(legal_moves) == 0:
        return priors

    weight = np.float32(1.0 / len(legal_moves))
    priors[legal_moves] = weight
    return priors
