"""Unit tests for Phase 3 — Regime Detection Engine.

All tests run in-memory with synthetic data and a mock HMM model.
No network calls, API keys, or disk I/O beyond temp directories.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from regime.confidence import ConfidenceScore, compute_confidence, format_confidence
from regime.monitor import RegimeAlert, RegimeMonitor
from regime.predictor import RegimePredictor
from regime.regime_history import HistoryRecord, RegimeHistory
from regime.transition import (
    TransitionInfo,
    analyse_transitions,
    format_transition_matrix,
    identify_stable_regimes,
)


# ---------------------------------------------------------------------------
# Synthetic data / mock model helpers
# ---------------------------------------------------------------------------

def _make_label_map(n_states: int = 3) -> dict[int, str]:
    """Create a simple label map."""
    labels = ["Bull Market", "Bear Market", "Sideways Market", "High Volatility", "Low Volatility"]
    return {i: labels[i % len(labels)] for i in range(n_states)}


def _make_transition_matrix(n_states: int = 3) -> np.ndarray:
    """Create a synthetic transition matrix with high self-transition."""
    rng = np.random.default_rng(42)
    tm = rng.uniform(0.01, 0.1, (n_states, n_states))
    for i in range(n_states):
        tm[i, i] = rng.uniform(0.7, 0.95)
    # Normalise rows.
    tm = tm / tm.sum(axis=1, keepdims=True)
    return tm


class MockGaussianHMM:
    """Minimal mock of hmmlearn.GaussianHMM for testing.

    Provides just enough interface to satisfy ``RegimePredictor``.
    """

    def __init__(self, n_states: int = 3, n_features: int = 5) -> None:
        self.n_components = n_states
        self.n_features = n_features
        self.transmat_ = _make_transition_matrix(n_states)
        self._rng = np.random.default_rng(42)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return deterministic state assignments based on row index."""
        return np.array([i % self.n_components for i in range(len(X))])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return synthetic posterior probabilities."""
        n = len(X)
        proba = np.zeros((n, self.n_components))
        for i in range(n):
            dominant = i % self.n_components
            proba[i, dominant] = 0.85
            remaining = (1.0 - 0.85) / max(self.n_components - 1, 1)
            for j in range(self.n_components):
                if j != dominant:
                    proba[i, j] = remaining
        return proba

    def score(self, X: np.ndarray) -> float:
        return -500.0


# ---------------------------------------------------------------------------
# Predictor tests
# ---------------------------------------------------------------------------

class TestRegimePredictor:
    """Tests for the HMM prediction wrapper."""

    def test_predict_state(self) -> None:
        model = MockGaussianHMM(n_states=3, n_features=5)
        label_map = _make_label_map(3)
        predictor = RegimePredictor(model=model, label_map=label_map)

        X = np.random.randn(10, 5)
        states = predictor.predict_state(X)
        assert len(states) == 10
        assert all(0 <= s < 3 for s in states)

    def test_predict_proba(self) -> None:
        model = MockGaussianHMM(n_states=3, n_features=5)
        label_map = _make_label_map(3)
        predictor = RegimePredictor(model=model, label_map=label_map)

        X = np.random.randn(10, 5)
        proba = predictor.predict_proba(X)
        assert proba.shape == (10, 3)
        # Each row should sum to ~1.
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_latest(self) -> None:
        model = MockGaussianHMM(n_states=3, n_features=5)
        label_map = _make_label_map(3)
        predictor = RegimePredictor(model=model, label_map=label_map)

        X = np.random.randn(20, 5)
        result = predictor.predict_latest(X)

        assert "state_id" in result
        assert "regime" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_feature_dimension_mismatch(self) -> None:
        model = MockGaussianHMM(n_states=3, n_features=5)
        label_map = _make_label_map(3)
        predictor = RegimePredictor(model=model, label_map=label_map)

        X = np.random.randn(10, 7)  # Wrong feature count
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            predictor.predict_state(X)

    def test_single_row_input(self) -> None:
        model = MockGaussianHMM(n_states=2, n_features=4)
        label_map = _make_label_map(2)
        predictor = RegimePredictor(model=model, label_map=label_map)

        X = np.random.randn(4)  # 1-D input
        result = predictor.predict_latest(X)
        assert isinstance(result["state_id"], int)

    def test_properties(self) -> None:
        model = MockGaussianHMM(n_states=4, n_features=6)
        label_map = _make_label_map(4)
        predictor = RegimePredictor(model=model, label_map=label_map)

        assert predictor.n_states == 4
        assert predictor.n_features == 6
        assert len(predictor.label_map) == 4
        assert predictor.transition_matrix.shape == (4, 4)


# ---------------------------------------------------------------------------
# Confidence tests
# ---------------------------------------------------------------------------

class TestConfidence:
    """Tests for confidence scoring."""

    def test_compute_confidence(self) -> None:
        posterior = np.array([0.85, 0.10, 0.05])
        label_map = _make_label_map(3)
        score = compute_confidence(posterior, label_map)

        assert score.dominant_state == 0
        assert score.dominant_confidence == 0.85
        assert score.entropy > 0
        assert score.margin > 0
        assert len(score.probabilities) == 3

    def test_uniform_posterior(self) -> None:
        posterior = np.array([0.25, 0.25, 0.25, 0.25])
        label_map = _make_label_map(4)
        score = compute_confidence(posterior, label_map)

        assert score.dominant_confidence == 0.25
        assert score.margin == 0.0
        # Entropy should be high for uniform.
        assert score.entropy > 1.0

    def test_high_confidence_check(self) -> None:
        posterior = np.array([0.95, 0.03, 0.02])
        label_map = _make_label_map(3)
        score = compute_confidence(posterior, label_map)

        assert score.is_high_confidence(threshold=0.90) is True
        assert score.is_high_confidence(threshold=0.99) is False

    def test_format_confidence_output(self) -> None:
        posterior = np.array([0.70, 0.20, 0.10])
        label_map = _make_label_map(3)
        score = compute_confidence(posterior, label_map)
        output = format_confidence(score)
        assert "CONFIDENCE" in output
        assert "Bull Market" in output


# ---------------------------------------------------------------------------
# Transition tests
# ---------------------------------------------------------------------------

class TestTransition:
    """Tests for transition matrix analysis."""

    def test_analyse_transitions(self) -> None:
        tm = _make_transition_matrix(3)
        label_map = _make_label_map(3)
        info = analyse_transitions(tm, current_state=0, label_map=label_map)

        assert isinstance(info, TransitionInfo)
        assert info.current_state == 0
        assert 0.0 <= info.self_transition <= 1.0
        assert info.expected_duration >= 1.0
        assert info.most_likely_next in label_map.values()

    def test_stable_regime_identification(self) -> None:
        tm = np.array([
            [0.95, 0.03, 0.02],
            [0.05, 0.90, 0.05],
            [0.10, 0.10, 0.80],
        ])
        label_map = _make_label_map(3)
        stable = identify_stable_regimes(tm, label_map, threshold=0.85)

        assert len(stable) >= 1
        labels = [s["label"] for s in stable]
        assert "Bull Market" in labels  # State 0 has 0.95 self-transition

    def test_format_transition_matrix(self) -> None:
        tm = _make_transition_matrix(3)
        label_map = _make_label_map(3)
        output = format_transition_matrix(tm, label_map)
        assert "TRANSITION" in output


# ---------------------------------------------------------------------------
# History tests
# ---------------------------------------------------------------------------

class TestRegimeHistory:
    """Tests for the regime prediction history."""

    def test_add_and_retrieve(self) -> None:
        history = RegimeHistory(max_records=100)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

        history.add(
            timestamp=ts,
            state_id=0,
            regime="Bull Market",
            confidence=0.85,
            close_price=450.0,
            daily_return=0.01,
            entropy=0.3,
        )

        assert len(history) == 1
        assert history.latest is not None
        assert history.latest.regime == "Bull Market"

    def test_max_records_eviction(self) -> None:
        history = RegimeHistory(max_records=5)
        for i in range(10):
            history.add(
                timestamp=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
                state_id=i % 3,
                regime=f"State {i % 3}",
                confidence=0.80,
                close_price=100.0 + i,
                daily_return=0.01,
            )

        assert len(history) == 5

    def test_to_dataframe(self) -> None:
        history = RegimeHistory()
        for i in range(5):
            history.add(
                timestamp=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
                state_id=i % 2,
                regime=f"State {i % 2}",
                confidence=0.90,
                close_price=100.0,
                daily_return=0.005,
            )

        df = history.to_dataframe()
        assert len(df) == 5
        assert "regime" in df.columns

    def test_regime_durations(self) -> None:
        history = RegimeHistory()
        regimes = ["Bull", "Bull", "Bull", "Bear", "Bear", "Bull"]
        for i, regime in enumerate(regimes):
            history.add(
                timestamp=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
                state_id=0 if regime == "Bull" else 1,
                regime=regime,
                confidence=0.80,
                close_price=100.0,
                daily_return=0.01,
            )

        durations = history.regime_durations()
        assert "Bull" in durations
        assert "Bear" in durations
        assert durations["Bull"] == [3, 1]  # Two streaks: 3 and 1
        assert durations["Bear"] == [2]

    def test_save_csv(self) -> None:
        history = RegimeHistory()
        history.add(
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            state_id=0,
            regime="Bull",
            confidence=0.85,
            close_price=100.0,
            daily_return=0.01,
        )
        with TemporaryDirectory() as tmp:
            path = history.save_csv(Path(tmp) / "history.csv")
            assert path.exists()
            loaded = pd.read_csv(path)
            assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Monitor tests
# ---------------------------------------------------------------------------

class TestRegimeMonitor:
    """Tests for regime change alerting."""

    def _make_confidence(
        self, dominant: int, prob: float, n_states: int = 3
    ) -> ConfidenceScore:
        """Helper to create a ConfidenceScore."""
        label_map = _make_label_map(n_states)
        probs: dict[str, float] = {}
        remaining = (1.0 - prob) / max(n_states - 1, 1)
        for i in range(n_states):
            label = label_map[i]
            probs[label] = prob if i == dominant else remaining
        return ConfidenceScore(
            dominant_state=dominant,
            dominant_label=label_map[dominant],
            dominant_confidence=prob,
            entropy=0.5,
            margin=prob - remaining,
            probabilities=probs,
        )

    def test_no_alert_on_first_call(self) -> None:
        monitor = RegimeMonitor()
        conf = self._make_confidence(0, 0.85)
        alert = monitor.check(0, "Bull Market", conf)
        assert alert is None

    def test_no_alert_on_same_regime(self) -> None:
        monitor = RegimeMonitor()
        conf = self._make_confidence(0, 0.85)
        monitor.check(0, "Bull Market", conf)
        alert = monitor.check(0, "Bull Market", conf)
        assert alert is None

    def test_alert_on_regime_change(self) -> None:
        monitor = RegimeMonitor()
        conf0 = self._make_confidence(0, 0.85)
        conf1 = self._make_confidence(1, 0.80)

        monitor.check(0, "Bull Market", conf0)
        alert = monitor.check(1, "Bear Market", conf1)

        assert alert is not None
        assert isinstance(alert, RegimeAlert)
        assert alert.old_regime == "Bull Market"
        assert alert.new_regime == "Bear Market"
        assert monitor.alert_count == 1

    def test_alert_dict_serialization(self) -> None:
        monitor = RegimeMonitor()
        conf0 = self._make_confidence(0, 0.85)
        conf1 = self._make_confidence(1, 0.90)

        monitor.check(0, "Bull Market", conf0)
        alert = monitor.check(1, "Bear Market", conf1)

        alert_dict = alert.to_dict()
        assert "timestamp" in alert_dict
        assert "old_regime" in alert_dict
        assert "new_regime" in alert_dict
        assert "reason" in alert_dict

    def test_min_confidence_filtering(self) -> None:
        monitor = RegimeMonitor(min_confidence=0.90)
        conf0 = self._make_confidence(0, 0.85)
        conf1 = self._make_confidence(1, 0.50)  # Below threshold

        monitor.check(0, "Bull Market", conf0)
        alert = monitor.check(1, "Bear Market", conf1)

        # Alert is returned but not stored (below threshold).
        assert alert is not None
        assert monitor.alert_count == 0  # Not stored

    def test_clear_alerts(self) -> None:
        monitor = RegimeMonitor()
        conf0 = self._make_confidence(0, 0.85)
        conf1 = self._make_confidence(1, 0.80)

        monitor.check(0, "Bull", conf0)
        monitor.check(1, "Bear", conf1)
        assert monitor.alert_count == 1

        monitor.clear_alerts()
        assert monitor.alert_count == 0


# ---------------------------------------------------------------------------
# Integration-style test (all components together)
# ---------------------------------------------------------------------------

class TestDetectionPipeline:
    """End-to-end test of the detection pipeline without disk or network."""

    def test_full_pipeline(self) -> None:
        """Simulate a detection flow: predict → confidence → transition → history → monitor."""
        n_states = 3
        n_features = 5
        n_samples = 50

        model = MockGaussianHMM(n_states=n_states, n_features=n_features)
        label_map = _make_label_map(n_states)
        predictor = RegimePredictor(model=model, label_map=label_map)

        X = np.random.randn(n_samples, n_features)
        history = RegimeHistory()
        monitor = RegimeMonitor()

        states = predictor.predict_state(X)
        posteriors = predictor.predict_proba(X)

        alert_count = 0
        for i in range(n_samples):
            sid = int(states[i])
            regime = label_map[sid]
            conf = compute_confidence(posteriors[i], label_map)
            trans = analyse_transitions(predictor.transition_matrix, sid, label_map)

            ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
            history.add(
                timestamp=ts,
                state_id=sid,
                regime=regime,
                confidence=conf.dominant_confidence,
                close_price=100.0 + i,
                daily_return=0.01,
                entropy=conf.entropy,
            )

            alert = monitor.check(sid, regime, conf, timestamp=ts)
            if alert:
                alert_count += 1

        assert len(history) == n_samples
        assert alert_count > 0  # Should have transitions given cycling states
        assert history.latest is not None
