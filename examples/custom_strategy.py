"""Example injecting custom deterministic python rules into Strategy Agent."""

def custom_moving_average_crossover(fast: list, slow: list) -> int:
    """Basic integration demonstration returning pseudo-signals (1, 0, -1)."""
    if fast[-1] > slow[-1]:
        return 1
    return 0
