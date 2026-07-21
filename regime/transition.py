"""Transition matrix analysis for hidden Markov model regimes.

Provides utilities to:

* Extract and format the learned transition matrix.
* Identify stable (self-loop dominant) regimes.
* Find the most-likely next transition from the current state.
* Compute regime persistence (expected duration in a state).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class TransitionInfo:
    """Structured analysis of the transition matrix from the current state.

    Attributes:
        current_state: Index of the current hidden state.
        current_label: Human-readable label.
        self_transition: Probability of staying in the same state.
        expected_duration: Expected number of bars before leaving.
        most_likely_next: Label of the most-likely *different* state.
        most_likely_next_prob: Probability of that transition.
        full_row: Full transition-probability row for the current state.
    """

    current_state: int
    current_label: str
    self_transition: float
    expected_duration: float
    most_likely_next: str
    most_likely_next_prob: float
    full_row: dict[str, float]


def analyse_transitions(
    transition_matrix: np.ndarray,
    current_state: int,
    label_map: dict[int, str],
) -> TransitionInfo:
    """Analyse transition dynamics from the current regime.

    Args:
        transition_matrix: The HMM transition matrix of shape
            ``(n_states, n_states)``.
        current_state: The index of the current hidden state.
        label_map: State-index → regime-label mapping.

    Returns:
        A ``TransitionInfo`` object with stability and next-state data.
    """
    row = transition_matrix[current_state]
    current_label = label_map.get(current_state, f"State {current_state}")

    self_prob = float(row[current_state])

    # Expected duration in current state (geometric distribution).
    expected_dur = 1.0 / (1.0 - self_prob) if self_prob < 1.0 else float("inf")

    # Most-likely *different* state.
    masked = row.copy()
    masked[current_state] = -1.0
    next_idx = int(np.argmax(masked))
    next_prob = float(row[next_idx])
    next_label = label_map.get(next_idx, f"State {next_idx}")

    # Full row as labelled dict.
    full_row: dict[str, float] = {}
    for sid in range(len(row)):
        label = label_map.get(sid, f"State {sid}")
        full_row[label] = round(float(row[sid]), 4)

    info = TransitionInfo(
        current_state=current_state,
        current_label=current_label,
        self_transition=round(self_prob, 4),
        expected_duration=round(expected_dur, 2),
        most_likely_next=next_label,
        most_likely_next_prob=round(next_prob, 4),
        full_row=full_row,
    )

    logger.debug(
        "Transition: {} → self={:.2%}, next={} ({:.2%}), E[dur]={:.1f}",
        current_label,
        self_prob,
        next_label,
        next_prob,
        expected_dur,
    )

    return info


def identify_stable_regimes(
    transition_matrix: np.ndarray,
    label_map: dict[int, str],
    threshold: float = 0.80,
) -> list[dict[str, float | str]]:
    """Identify regimes with high self-transition probability.

    Args:
        transition_matrix: The HMM transition matrix.
        label_map: State-index → label mapping.
        threshold: Minimum self-transition probability for "stable".

    Returns:
        List of dicts with ``label``, ``self_prob``, and
        ``expected_duration`` for each stable regime.
    """
    stable: list[dict[str, float | str]] = []
    n_states = transition_matrix.shape[0]

    for sid in range(n_states):
        self_prob = float(transition_matrix[sid, sid])
        if self_prob >= threshold:
            expected_dur = 1.0 / (1.0 - self_prob) if self_prob < 1.0 else float("inf")
            stable.append({
                "label": label_map.get(sid, f"State {sid}"),
                "self_prob": round(self_prob, 4),
                "expected_duration": round(expected_dur, 2),
            })

    if stable:
        logger.info("Stable regimes (self-prob ≥ {:.0%}):", threshold)
        for entry in stable:
            logger.info(
                "  {} — P(stay)={:.2%}, E[dur]={:.1f} bars",
                entry["label"],
                entry["self_prob"],
                entry["expected_duration"],
            )
    else:
        logger.info("No regimes exceed stability threshold {:.0%}", threshold)

    return stable


def format_transition_matrix(
    transition_matrix: np.ndarray,
    label_map: dict[int, str],
) -> str:
    """Render the transition matrix as a formatted text table.

    Args:
        transition_matrix: The HMM transition matrix.
        label_map: State-index → label mapping.

    Returns:
        Multi-line formatted string.
    """
    n = transition_matrix.shape[0]
    labels = [label_map.get(i, f"State {i}") for i in range(n)]
    col_width = max(len(l) for l in labels) + 2

    header = " " * col_width + "".join(f"{l:>{col_width}}" for l in labels)
    sep = "─" * len(header)

    lines = [
        "",
        "═" * len(header),
        "  STATE TRANSITION MATRIX",
        "═" * len(header),
        header,
        sep,
    ]

    for i in range(n):
        row_str = f"{labels[i]:>{col_width}}"
        for j in range(n):
            val = transition_matrix[i, j]
            marker = " *" if i == j else "  "
            row_str += f"{val:>{col_width - 2}.3f}{marker}"
        lines.append(row_str)

    lines.append(sep)
    lines.append("  (* = self-transition)")
    lines.append("═" * len(header))
    return "\n".join(lines)
