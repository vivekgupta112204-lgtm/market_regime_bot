"""Model selection based on information criteria.

Compares a list of ``HMMResult`` objects produced by ``HMMTrainer``
and selects the best model according to BIC (by default) or AIC.

Also generates a formatted comparison table for the training report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from models.hmm_trainer import HMMResult


@dataclass
class SelectionResult:
    """Outcome of the model-selection process.

    Attributes:
        best_result: The ``HMMResult`` with the lowest criterion score.
        criterion: The criterion used for selection (``"bic"`` or ``"aic"``).
        all_results: The full list of candidates for reporting.
    """

    best_result: HMMResult
    criterion: str
    all_results: list[HMMResult]


def select_best_model(
    results: list[HMMResult],
    criterion: str = "bic",
) -> SelectionResult:
    """Choose the best HMM from a list of trained candidates.

    Args:
        results: List of ``HMMResult`` objects (from ``HMMTrainer.train_range``).
        criterion: ``"bic"`` or ``"aic"`` — lower is better.

    Returns:
        A ``SelectionResult`` containing the winning model.

    Raises:
        ValueError: When *results* is empty or all models failed.
    """
    # Filter out failed results.
    valid = [r for r in results if r.model is not None]
    if not valid:
        raise ValueError("No valid HMM results to select from.")

    if criterion == "aic":
        best = min(valid, key=lambda r: r.aic)
    else:
        best = min(valid, key=lambda r: r.bic)

    logger.info(
        "Model selection ({}) → {} states  (LL={:.2f}, AIC={:.2f}, BIC={:.2f})",
        criterion.upper(),
        best.n_states,
        best.log_likelihood,
        best.aic,
        best.bic,
    )

    return SelectionResult(
        best_result=best,
        criterion=criterion,
        all_results=results,
    )


def format_comparison_table(results: list[HMMResult]) -> str:
    """Generate a formatted text table comparing all candidate models.

    Args:
        results: List of ``HMMResult`` objects.

    Returns:
        Multi-line string ready for logging or printing.
    """
    header = (
        f"{'States':>6}  {'Log-Lik':>12}  {'AIC':>12}  {'BIC':>12}  "
        f"{'Params':>7}  {'Converged':>9}  {'Time (s)':>9}"
    )
    separator = "─" * len(header)

    lines = [
        "",
        "═══════════════════════════════════════════════════════════════════════════",
        "                       HMM MODEL COMPARISON                              ",
        "═══════════════════════════════════════════════════════════════════════════",
        header,
        separator,
    ]

    for r in sorted(results, key=lambda x: x.n_states):
        if r.model is None:
            lines.append(
                f"{r.n_states:>6}  {'FAILED':>12}  {'—':>12}  {'—':>12}  "
                f"{'—':>7}  {'—':>9}  {'—':>9}"
            )
        else:
            lines.append(
                f"{r.n_states:>6}  {r.log_likelihood:>12.2f}  {r.aic:>12.2f}  "
                f"{r.bic:>12.2f}  {r.n_params:>7}  "
                f"{'Yes' if r.converged else 'No':>9}  "
                f"{r.train_time_seconds:>9.2f}"
            )

    lines.append(separator)
    lines.append(
        "═══════════════════════════════════════════════════════════════════════════"
    )
    return "\n".join(lines)
