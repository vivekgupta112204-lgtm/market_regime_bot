"""HMM regime predictor — wraps the trained GaussianHMM.

Provides a clean interface for:
* Predicting the most-likely hidden state for new observations.
* Computing posterior (state) probabilities per observation.
* Validating that the input feature matrix is compatible with the
  trained model.

This module operates on **feature matrices** (NumPy arrays) and is
intentionally agnostic to DataFrame column names — that mapping is
handled one layer up by the ``RegimeDetector``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hmmlearn.hmm import GaussianHMM
from loguru import logger


class RegimePredictor:
    """Predict market regimes from feature observations.

    Args:
        model: A fitted ``GaussianHMM`` instance.
        label_map: Dictionary mapping state indices to human-readable
            regime labels (e.g. ``{0: "Bull Market", 1: "Bear Market"}``).
    """

    def __init__(
        self,
        model: GaussianHMM,
        label_map: dict[int, str],
    ) -> None:
        self._model = model
        self._label_map = label_map
        self._n_states: int = model.n_components
        self._n_features: int = model.n_features

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(self, X: np.ndarray) -> None:
        """Ensure the observation matrix is compatible with the model.

        Args:
            X: Observation matrix ``(n_samples, n_features)``.

        Raises:
            ValueError: When the feature count does not match the model.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Feature dimension mismatch: model expects "
                f"{self._n_features} features, got {X.shape[1]}"
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_state(self, X: np.ndarray) -> np.ndarray:
        """Predict the most-likely hidden state for each observation.

        Args:
            X: Observation matrix ``(n_samples, n_features)``.

        Returns:
            Array of state indices, shape ``(n_samples,)``.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._validate_input(X)
        states = self._model.predict(X)
        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior probabilities for each state per observation.

        Args:
            X: Observation matrix ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, n_states)`` where each row
            sums to 1.0.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._validate_input(X)
        posteriors = self._model.predict_proba(X)
        return posteriors

    def predict_latest(
        self,
        X: np.ndarray,
    ) -> dict[str, Any]:
        """Predict the regime for the *last* observation in the sequence.

        This is the primary method used for real-time detection.  The
        full sequence is passed so that the HMM can leverage temporal
        dependencies, but only the final prediction is returned.

        Args:
            X: Observation matrix ``(n_samples, n_features)``.  The
               last row is the "current" candle.

        Returns:
            Dictionary with keys:
            ``state_id``, ``regime``, ``probabilities`` (dict),
            ``confidence`` (float, dominant-state probability).
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._validate_input(X)

        states = self._model.predict(X)
        posteriors = self._model.predict_proba(X)

        latest_state = int(states[-1])
        latest_proba = posteriors[-1]
        confidence = float(latest_proba[latest_state])

        regime_name = self._label_map.get(latest_state, f"State {latest_state}")

        prob_dict: dict[str, float] = {}
        for sid in range(self._n_states):
            label = self._label_map.get(sid, f"State {sid}")
            prob_dict[label] = round(float(latest_proba[sid]), 4)

        logger.debug(
            "Prediction → {} (state={}, confidence={:.2%})",
            regime_name,
            latest_state,
            confidence,
        )

        return {
            "state_id": latest_state,
            "regime": regime_name,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    def score(self, X: np.ndarray) -> float:
        """Compute the log-likelihood of the observation sequence.

        Args:
            X: Observation matrix ``(n_samples, n_features)``.

        Returns:
            Log-likelihood score.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._validate_input(X)
        return float(self._model.score(X))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        """Number of hidden states in the model."""
        return self._n_states

    @property
    def n_features(self) -> int:
        """Number of features expected by the model."""
        return self._n_features

    @property
    def label_map(self) -> dict[int, str]:
        """State-index to regime-label mapping."""
        return dict(self._label_map)

    @property
    def transition_matrix(self) -> np.ndarray:
        """The learned state-transition probability matrix."""
        return self._model.transmat_.copy()
