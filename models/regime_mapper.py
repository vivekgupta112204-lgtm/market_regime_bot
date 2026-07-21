"""Map HMM hidden states to human-readable regime labels.

After training, hidden-state indices are arbitrary.  This module
analyses the statistical properties of each state (average return,
average volatility) and assigns descriptive labels:

* **Bull Market** — highest average return.
* **Bear Market** — lowest (most negative) average return.
* **High Volatility** — highest average volatility.
* **Low Volatility** — lowest average volatility.
* **Sideways Market** — all remaining states.

When fewer than 5 states are trained, only the most relevant labels
are assigned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RegimeStats:
    """Statistics for a single hidden state.

    Attributes:
        state_id: Numeric state index (0-based).
        label: Human-readable regime label.
        observation_count: Number of bars assigned to this state.
        avg_return: Mean daily return in this state.
        avg_volatility: Mean rolling volatility in this state.
        avg_close: Mean closing price in this state.
    """

    state_id: int
    label: str
    observation_count: int
    avg_return: float
    avg_volatility: float
    avg_close: float


def map_regimes(
    df: pd.DataFrame,
    hidden_states: np.ndarray,
    n_states: int,
    return_column: str = "daily_return",
    volatility_column: str = "rolling_volatility",
) -> tuple[list[RegimeStats], dict[int, str]]:
    """Assign regime labels to hidden-state indices.

    The labelling heuristic works as follows:

    1. Compute per-state mean return and mean volatility.
    2. The state with the highest mean return → **Bull Market**.
    3. The state with the lowest mean return → **Bear Market**.
    4. Among remaining states, highest volatility → **High Volatility**.
    5. Among remaining states, lowest volatility → **Low Volatility**.
    6. Any leftover states → **Sideways Market**.

    Args:
        df: Feature DataFrame aligned with *hidden_states*.
        hidden_states: Array of state assignments (length = len(df)).
        n_states: Number of hidden states in the model.
        return_column: Column name for return feature.
        volatility_column: Column name for volatility feature.

    Returns:
        A 2-tuple of:
        - List of ``RegimeStats``, one per state.
        - Dictionary mapping ``state_id → label``.
    """
    state_stats: list[dict] = []

    for state_id in range(n_states):
        mask = hidden_states == state_id
        subset = df[mask]
        count = int(mask.sum())

        avg_ret = float(subset[return_column].mean()) if count > 0 else 0.0
        avg_vol = float(subset[volatility_column].mean()) if count > 0 else 0.0
        avg_close = float(subset["close"].mean()) if count > 0 and "close" in subset.columns else 0.0

        state_stats.append({
            "state_id": state_id,
            "count": count,
            "avg_return": avg_ret,
            "avg_volatility": avg_vol,
            "avg_close": avg_close,
        })

    # --- Label assignment ---------------------------------------------------
    label_map: dict[int, str] = {}
    assigned: set[int] = set()

    # Sort by average return to find bull / bear.
    by_return = sorted(state_stats, key=lambda s: s["avg_return"])

    # Bear: lowest return.
    bear_id = by_return[0]["state_id"]
    label_map[bear_id] = "Bear Market"
    assigned.add(bear_id)

    # Bull: highest return.
    bull_id = by_return[-1]["state_id"]
    if bull_id not in assigned:
        label_map[bull_id] = "Bull Market"
        assigned.add(bull_id)

    # From remaining, assign volatility labels.
    remaining = [s for s in state_stats if s["state_id"] not in assigned]
    if remaining:
        by_vol = sorted(remaining, key=lambda s: s["avg_volatility"])

        # High Volatility: highest vol among remaining.
        hv_id = by_vol[-1]["state_id"]
        label_map[hv_id] = "High Volatility"
        assigned.add(hv_id)

        # Low Volatility: lowest vol among remaining (if any left).
        if len(by_vol) >= 2:
            lv_id = by_vol[0]["state_id"]
            label_map[lv_id] = "Low Volatility"
            assigned.add(lv_id)

    # Everything else → Sideways.
    for s in state_stats:
        if s["state_id"] not in assigned:
            label_map[s["state_id"]] = "Sideways Market"

    # --- Build final output ------------------------------------------------
    regime_stats: list[RegimeStats] = []
    for s in state_stats:
        sid = s["state_id"]
        rs = RegimeStats(
            state_id=sid,
            label=label_map[sid],
            observation_count=s["count"],
            avg_return=s["avg_return"],
            avg_volatility=s["avg_volatility"],
            avg_close=s["avg_close"],
        )
        regime_stats.append(rs)

    # --- Log the mapping ---------------------------------------------------
    logger.info("Regime mapping ({} states):", n_states)
    for rs in regime_stats:
        logger.info(
            "  State {} → {:16s}  count={:>5}  avg_ret={:+.5f}  avg_vol={:.5f}",
            rs.state_id,
            rs.label,
            rs.observation_count,
            rs.avg_return,
            rs.avg_volatility,
        )

    return regime_stats, label_map


def format_regime_table(regime_stats: list[RegimeStats]) -> str:
    """Generate a human-readable regime statistics table.

    Args:
        regime_stats: List of ``RegimeStats`` objects.

    Returns:
        Multi-line formatted string.
    """
    header = (
        f"{'State':>5}  {'Label':>16}  {'Obs':>6}  "
        f"{'Avg Return':>12}  {'Avg Volatility':>14}"
    )
    sep = "─" * len(header)
    lines = [
        "",
        "═══════════════════════════════════════════════════════════════",
        "                    REGIME STATISTICS                         ",
        "═══════════════════════════════════════════════════════════════",
        header,
        sep,
    ]

    for rs in sorted(regime_stats, key=lambda r: r.state_id):
        lines.append(
            f"{rs.state_id:>5}  {rs.label:>16}  {rs.observation_count:>6}  "
            f"{rs.avg_return:>+12.6f}  {rs.avg_volatility:>14.6f}"
        )

    lines.append(sep)
    lines.append(
        "═══════════════════════════════════════════════════════════════"
    )
    return "\n".join(lines)
