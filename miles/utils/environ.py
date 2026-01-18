import os

_printed_experimental_rollout_refactor = False


def enable_experimental_rollout_refactor() -> bool:
    global _printed_experimental_rollout_refactor
    result = bool(int(os.environ.get("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "0")))
    if result and not _printed_experimental_rollout_refactor:
        print("[MILES] MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 is enabled (experimental feature)")
        _printed_experimental_rollout_refactor = True
    return result
