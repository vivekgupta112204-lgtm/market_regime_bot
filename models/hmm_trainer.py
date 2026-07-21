"""Gaussian HMM trainer.

Fits ``hmmlearn.GaussianHMM`` models with varying numbers of hidden
states and returns structured results that the ``HMMSelector`` uses for
model comparison.

Each training run records:

* The fitted model object.
* Log-likelihood on the training data.
* AIC / BIC information-criteria scores.
* Training wall-clock time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from hmmlearn.hmm import GaussianHMM
from loguru import logger


@dataclass
class HMMResult:
    """Container for a single trained HMM and its metrics.

    Attributes:
        n_states: Number of hidden states.
        model: The fitted ``GaussianHMM`` object.
        log_likelihood: Model log-likelihood on training data.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        n_params: Number of free parameters.
        train_time_seconds: Wall-clock training time.
        converged: Whether the EM algorithm converged.
    """

    n_states: int = 0
    model: GaussianHMM | None = None
    log_likelihood: float = -np.inf
    aic: float = np.inf
    bic: float = np.inf
    n_params: int = 0
    train_time_seconds: float = 0.0
    converged: bool = False


def _count_parameters(n_states: int, n_features: int, covariance_type: str) -> int:
    """Count the number of free parameters in a GaussianHMM.

    Args:
        n_states: Number of hidden states.
        n_features: Dimensionality of the observation space.
        covariance_type: One of ``full``, ``tied``, ``diag``, ``spherical``.

    Returns:
        Total number of free parameters.
    """
    # Transition matrix (each row sums to 1 → n_states-1 free per row).
    trans_params = n_states * (n_states - 1)
    # Start probabilities (sum to 1 → n_states - 1 free).
    start_params = n_states - 1
    # Means: n_states × n_features.
    mean_params = n_states * n_features

    if covariance_type == "full":
        cov_params = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "tied":
        cov_params = n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        cov_params = n_states * n_features
    elif covariance_type == "spherical":
        cov_params = n_states
    else:
        cov_params = n_states * n_features * (n_features + 1) // 2

    return trans_params + start_params + mean_params + cov_params


def _compute_aic(log_likelihood: float, n_params: int) -> float:
    """Compute the Akaike Information Criterion.

    Args:
        log_likelihood: Model log-likelihood.
        n_params: Number of free parameters.

    Returns:
        AIC score (lower is better).
    """
    return -2.0 * log_likelihood + 2.0 * n_params


def _compute_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Compute the Bayesian Information Criterion.

    Args:
        log_likelihood: Model log-likelihood.
        n_params: Number of free parameters.
        n_samples: Number of observations.

    Returns:
        BIC score (lower is better).
    """
    return -2.0 * log_likelihood + n_params * np.log(n_samples)


class HMMTrainer:
    """Train Gaussian HMMs over a range of state counts.

    Args:
        n_iter: Maximum EM iterations.
        n_init: Number of random restarts (best kept).
        covariance_type: Covariance matrix type.
        random_state: Random seed for reproducibility.
        tolerance: EM convergence tolerance.
    """

    def __init__(
        self,
        *,
        n_iter: int = 200,
        n_init: int = 10,
        covariance_type: str = "full",
        random_state: int = 42,
        tolerance: float = 0.01,
    ) -> None:
        self.n_iter = n_iter
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.tolerance = tolerance

    def train_single(
        self,
        X: np.ndarray,
        n_states: int,
    ) -> HMMResult:
        """Fit a single GaussianHMM with *n_states* hidden states.

        Multiple random initialisations are attempted (``n_init``) and
        the run with the highest log-likelihood is kept.

        Args:
            X: Observation matrix of shape ``(n_samples, n_features)``.
            n_states: Number of hidden states.

        Returns:
            An ``HMMResult`` containing the best model and its metrics.
        """
        n_samples, n_features = X.shape
        best_score = -np.inf
        best_model: GaussianHMM | None = None
        best_converged = False

        logger.info(
            "Training HMM with {} states (n_init={}, n_iter={}) …",
            n_states,
            self.n_init,
            self.n_iter,
        )

        t0 = time.perf_counter()

        for init_idx in range(self.n_init):
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state + init_idx,
                    tol=self.tolerance,
                )
                model.fit(X)
                score = model.score(X)

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_converged = model.monitor_.converged

            except Exception as exc:
                logger.warning(
                    "HMM init {}/{} for {} states failed: {}",
                    init_idx + 1,
                    self.n_init,
                    n_states,
                    exc,
                )
                continue

        elapsed = time.perf_counter() - t0

        if best_model is None:
            logger.error("All initialisations failed for {} states", n_states)
            return HMMResult(n_states=n_states)

        n_params = _count_parameters(n_states, n_features, self.covariance_type)
        aic = _compute_aic(best_score, n_params)
        bic = _compute_bic(best_score, n_params, n_samples)

        result = HMMResult(
            n_states=n_states,
            model=best_model,
            log_likelihood=best_score,
            aic=aic,
            bic=bic,
            n_params=n_params,
            train_time_seconds=elapsed,
            converged=best_converged,
        )

        logger.info(
            "  {} states → LL={:.2f}  AIC={:.2f}  BIC={:.2f}  "
            "converged={}  time={:.1f}s",
            n_states,
            best_score,
            aic,
            bic,
            best_converged,
            elapsed,
        )

        return result

    def train_range(
        self,
        X: np.ndarray,
        min_states: int = 2,
        max_states: int = 6,
    ) -> list[HMMResult]:
        """Train models for every state count in ``[min_states, max_states]``.

        Args:
            X: Observation matrix ``(n_samples, n_features)``.
            min_states: Minimum number of hidden states.
            max_states: Maximum number of hidden states.

        Returns:
            List of ``HMMResult`` objects, one per state count.
        """
        logger.info(
            "HMM grid search: {} → {} states on {} observations × {} features",
            min_states,
            max_states,
            X.shape[0],
            X.shape[1],
        )

        results: list[HMMResult] = []
        for n in range(min_states, max_states + 1):
            result = self.train_single(X, n)
            results.append(result)

        logger.info("Grid search complete — {} models trained", len(results))
        return results
