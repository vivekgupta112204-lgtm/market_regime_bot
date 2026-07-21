"""Confidence scoring for regime predictions.

Wraps the raw posterior probabilities returned by the HMM predictor
into structured confidence objects that downstream consumers (alerts,
history, API) can use consistently.

Key metrics:
* **Dominant confidence** — P(most-likely state).
* **Entropy** — Shannon entropy of the posterior; high entropy signals
  the model is uncertain between multiple states.
* **Margin** — difference between the top-two state probabilities; a
  large margin indicates a decisive classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class ConfidenceScore:
    """Structured confidence output for a single prediction.

    Attributes:
        dominant_state: Index of the most-likely state.
        dominant_label: Human-readable label of the dominant state.
        dominant_confidence: Probability of the dominant state.
        entropy: Shannon entropy of the posterior distribution.
        margin: Probability gap between the top two states.
        probabilities: Full posterior dict ``{label: probability}``.
    """

    dominant_state: int
    dominant_label: str
    dominant_confidence: float
    entropy: float
    margin: float
    probabilities: dict[str, float]

    def is_high_confidence(self, threshold: float = 0.70) -> bool:
        """Return ``True`` when dominant confidence exceeds *threshold*.

        Args:
            threshold: Minimum probability to be considered high
                confidence.

        Returns:
            Boolean flag.
        """
        return self.dominant_confidence >= threshold


def compute_confidence(
    posterior: np.ndarray,
    label_map: dict[int, str],
) -> ConfidenceScore:
    """Build a ``ConfidenceScore`` from a posterior probability vector.

    Args:
        posterior: 1-D array of state probabilities summing to 1.
        label_map: State-index → regime-label mapping.

    Returns:
        A fully populated ``ConfidenceScore``.
    """
    posterior = np.asarray(posterior, dtype=np.float64)

    # Clamp to avoid log(0).
    safe_posterior = np.clip(posterior, 1e-12, 1.0)

    dominant_idx = int(np.argmax(posterior))
    dominant_prob = float(posterior[dominant_idx])
    dominant_label = label_map.get(dominant_idx, f"State {dominant_idx}")

    # Shannon entropy (base-e).
    entropy = float(-np.sum(safe_posterior * np.log(safe_posterior)))

    # Margin between top-two states.
    sorted_probs = np.sort(posterior)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 1.0

    # Full probability dict.
    prob_dict: dict[str, float] = {}
    for sid in range(len(posterior)):
        label = label_map.get(sid, f"State {sid}")
        prob_dict[label] = round(float(posterior[sid]), 4)

    score = ConfidenceScore(
        dominant_state=dominant_idx,
        dominant_label=dominant_label,
        dominant_confidence=round(dominant_prob, 4),
        entropy=round(entropy, 4),
        margin=round(margin, 4),
        probabilities=prob_dict,
    )

    logger.debug(
        "Confidence: {} ({:.2%}), entropy={:.3f}, margin={:.3f}",
        dominant_label,
        dominant_prob,
        entropy,
        margin,
    )

    return score


def format_confidence(score: ConfidenceScore) -> str:
    """Render a human-readable confidence summary.

    Args:
        score: The confidence result to format.

    Returns:
        Multi-line formatted string.
    """
    lines = [
        "┌──────────────────────────────────────┐",
        "│       REGIME CONFIDENCE SCORES       │",
        "├──────────────────────────────────────┤",
    ]
    for label, prob in sorted(score.probabilities.items(), key=lambda x: -x[1]):
        bar_len = int(prob * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        marker = " ◀" if label == score.dominant_label else ""
        lines.append(f"│ {label:>16}: {bar} {prob:>6.1%}{marker:>3} │")

    lines.append("├──────────────────────────────────────┤")
    lines.append(f"│ Entropy : {score.entropy:>6.3f}                    │")
    lines.append(f"│ Margin  : {score.margin:>6.3f}                    │")
    lines.append("└──────────────────────────────────────┘")
    return "\n".join(lines)
