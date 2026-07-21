"""Market Regime Detection Engine — top-level façade.

This is the primary public API of Phase 3.  It exposes a single
high-level function:

    ``detect_market_regime()`` → dict (JSON-serializable)

The function:

1. Loads the trained HMM model bundle (once, then cached).
2. Fetches the latest OHLCV candles from the configured broker.
3. Runs the Phase 2 feature pipeline (engineer → scale).
4. Predicts the current hidden state and computes confidence.
5. Analyses transition probabilities.
6. Updates the regime history and checks for regime changes.
7. Generates Plotly visualisations.
8. Returns a structured JSON result.

Usage::

    from regime.detector import RegimeDetector

    detector = RegimeDetector.from_bundle("saved_models/SPY_1d_3states_…")
    result = detector.detect()
    print(result)

Or, as a standalone script::

    python -m regime.detector --bundle saved_models/SPY_1d_3states_…
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import Settings, get_settings
from features.feature_engineering import compute_all_features
from features.scaler import FeatureScaler
from models.model_loader import ModelLoader
from regime.confidence import ConfidenceScore, compute_confidence, format_confidence
from regime.monitor import RegimeAlert, RegimeMonitor
from regime.predictor import RegimePredictor
from regime.regime_history import RegimeHistory
from regime.transition import (
    TransitionInfo,
    analyse_transitions,
    format_transition_matrix,
    identify_stable_regimes,
)
from utils.logger import setup_logger


# Columns that are raw OHLCV and should not be fed to the model.
_NON_FEATURE_COLUMNS: set[str] = {
    "timestamp", "open", "high", "low", "close", "volume",
}


class RegimeDetector:
    """Unified regime-detection engine.

    Orchestrates model loading, feature transformation, prediction,
    confidence scoring, transition analysis, history tracking, and
    alerting.

    Args:
        predictor: A ``RegimePredictor`` wrapping the fitted HMM.
        scaler: The fitted ``FeatureScaler`` from Phase 2.
        feature_names: Ordered list of feature columns the model was
            trained on.
        label_map: State-index → regime-label mapping.
        settings: Application settings.
        metrics: Training metrics dict (AIC, BIC, etc.).
    """

    def __init__(
        self,
        predictor: RegimePredictor,
        scaler: FeatureScaler,
        feature_names: list[str],
        label_map: dict[int, str],
        settings: Settings,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self._predictor = predictor
        self._scaler = scaler
        self._feature_names = feature_names
        self._label_map = label_map
        self._settings = settings
        self._metrics = metrics or {}
        self._history = RegimeHistory()
        self._monitor = RegimeMonitor()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        settings: Settings | None = None,
    ) -> "RegimeDetector":
        """Construct a ``RegimeDetector`` from a saved model bundle.

        Args:
            bundle_path: Path to the model bundle directory created by
                Phase 2's ``ModelLoader.save_bundle()``.
            settings: Optional pre-built settings; loaded from disk
                when *None*.

        Returns:
            A ready-to-use ``RegimeDetector``.

        Raises:
            FileNotFoundError: When the bundle directory or required
                files are missing.
            ValueError: When the bundle is incomplete.
        """
        bundle_dir = Path(bundle_path)

        # Validate bundle structure.
        required_files = [
            "hmm_model.pkl",
            "feature_list.json",
            "regime_map.json",
        ]
        missing = [f for f in required_files if not (bundle_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Model bundle at {bundle_dir} is missing: {missing}"
            )

        logger.info("Loading model bundle from {}", bundle_dir)
        bundle = ModelLoader.load_bundle(bundle_dir)

        # Parse label_map (JSON stores keys as strings).
        raw_map = bundle.get("regime_map", {}).get("label_map", {})
        label_map: dict[int, str] = {int(k): v for k, v in raw_map.items()}

        # Build predictor.
        model = bundle["model"]
        predictor = RegimePredictor(model=model, label_map=label_map)

        # Load scaler.
        scaler_path = bundle_dir / "scaler.pkl"
        if scaler_path.exists():
            scaler = FeatureScaler.load(scaler_path)
        else:
            logger.warning("No scaler found in bundle — predictions may be unscaled")
            scaler = FeatureScaler()

        feature_names: list[str] = bundle.get("feature_names", [])
        metrics: dict[str, Any] = bundle.get("metrics", {})

        if settings is None:
            settings = get_settings()

        logger.success(
            "Detector ready: {} states, {} features, model={}",
            predictor.n_states,
            len(feature_names),
            bundle_dir.name,
        )

        return cls(
            predictor=predictor,
            scaler=scaler,
            feature_names=feature_names,
            label_map=label_map,
            settings=settings,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(
        self,
        df: pd.DataFrame | None = None,
        *,
        generate_charts: bool = False,
    ) -> dict[str, Any]:
        """Detect the current market regime.

        If *df* is not provided, the detector will fetch the latest
        candles using the real-time data service.

        Args:
            df: Optional OHLCV DataFrame.  When *None*, data is fetched
                from the configured broker.
            generate_charts: Whether to produce Plotly visualisations.

        Returns:
            JSON-serializable dictionary with keys:
            ``timestamp``, ``state``, ``regime``, ``confidence``,
            ``transition_probability``, ``probabilities``,
            ``transition``, ``alert``, ``processing_time_ms``.
        """
        t0 = time.perf_counter()

        # ── Step 1: Obtain data ───────────────────────────────────────
        if df is None:
            from services.realtime_data import fetch_latest_candles
            df = fetch_latest_candles(self._settings)

        # ── Step 2: Feature engineering ───────────────────────────────
        df_feat = compute_all_features(df.copy())

        # Ensure the feature columns used during training are present.
        missing_cols = [c for c in self._feature_names if c not in df_feat.columns]
        if missing_cols:
            raise ValueError(
                f"Feature columns missing from data: {missing_cols}"
            )

        # ── Step 3: Scale ─────────────────────────────────────────────
        if self._scaler.is_fitted:
            df_feat = self._scaler.transform(df_feat)

        # ── Step 4: Predict ───────────────────────────────────────────
        X = df_feat[self._feature_names].values.astype(np.float64)
        prediction = self._predictor.predict_latest(X)

        state_id: int = prediction["state_id"]
        regime: str = prediction["regime"]

        # ── Step 5: Confidence ────────────────────────────────────────
        posteriors = self._predictor.predict_proba(X)
        latest_posterior = posteriors[-1]
        confidence = compute_confidence(latest_posterior, self._label_map)

        # ── Step 6: Transition analysis ───────────────────────────────
        trans_matrix = self._predictor.transition_matrix
        trans_info = analyse_transitions(trans_matrix, state_id, self._label_map)

        # ── Step 7: History ───────────────────────────────────────────
        latest_ts = pd.to_datetime(df_feat["timestamp"].iloc[-1])
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.tz_localize("UTC")
        latest_close = float(df_feat["close"].iloc[-1])
        latest_return = float(df_feat["daily_return"].iloc[-1]) if "daily_return" in df_feat.columns else 0.0

        self._history.add(
            timestamp=latest_ts.to_pydatetime(),
            state_id=state_id,
            regime=regime,
            confidence=confidence.dominant_confidence,
            close_price=latest_close,
            daily_return=latest_return,
            entropy=confidence.entropy,
        )

        # ── Step 8: Monitor for change ────────────────────────────────
        alert = self._monitor.check(
            state_id=state_id,
            regime=regime,
            confidence_score=confidence,
            timestamp=latest_ts.to_pydatetime(),
        )

        # ── Step 9: Charts (optional) ─────────────────────────────────
        if generate_charts:
            self._generate_charts(df_feat, X)

        # ── Build result ──────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result: dict[str, Any] = {
            "timestamp": latest_ts.isoformat(),
            "state": state_id,
            "regime": regime,
            "confidence": confidence.dominant_confidence,
            "transition_probability": trans_info.self_transition,
            "expected_duration": trans_info.expected_duration,
            "most_likely_next": trans_info.most_likely_next,
            "most_likely_next_prob": trans_info.most_likely_next_prob,
            "probabilities": confidence.probabilities,
            "entropy": confidence.entropy,
            "margin": confidence.margin,
            "close_price": latest_close,
            "daily_return": round(latest_return, 6),
            "alert": alert.to_dict() if alert else None,
            "processing_time_ms": round(elapsed_ms, 2),
        }

        logger.info(
            "Detection complete in {:.0f} ms → {} (state={}, conf={:.2%})",
            elapsed_ms,
            regime,
            state_id,
            confidence.dominant_confidence,
        )

        return result

    def detect_batch(
        self,
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Run detection on every row of a historical DataFrame.

        Computes features once, then iterates the HMM predictions over
        the full sequence.  Populates the history and fires alerts.

        Args:
            df: OHLCV DataFrame (cleaned).

        Returns:
            List of JSON-serializable result dicts, one per bar.
        """
        t0 = time.perf_counter()

        df_feat = compute_all_features(df.copy())
        if self._scaler.is_fitted:
            df_feat = self._scaler.transform(df_feat)

        X = df_feat[self._feature_names].values.astype(np.float64)
        states = self._predictor.predict_state(X)
        posteriors = self._predictor.predict_proba(X)

        results: list[dict[str, Any]] = []
        trans_matrix = self._predictor.transition_matrix

        for i in range(len(df_feat)):
            sid = int(states[i])
            regime = self._label_map.get(sid, f"State {sid}")
            conf = compute_confidence(posteriors[i], self._label_map)
            trans = analyse_transitions(trans_matrix, sid, self._label_map)

            ts = pd.to_datetime(df_feat["timestamp"].iloc[i])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            close = float(df_feat["close"].iloc[i])
            ret = float(df_feat["daily_return"].iloc[i]) if "daily_return" in df_feat.columns else 0.0

            self._history.add(
                timestamp=ts.to_pydatetime(),
                state_id=sid,
                regime=regime,
                confidence=conf.dominant_confidence,
                close_price=close,
                daily_return=ret,
                entropy=conf.entropy,
            )

            alert = self._monitor.check(
                state_id=sid,
                regime=regime,
                confidence_score=conf,
                timestamp=ts.to_pydatetime(),
            )

            results.append({
                "timestamp": ts.isoformat(),
                "state": sid,
                "regime": regime,
                "confidence": conf.dominant_confidence,
                "transition_probability": trans.self_transition,
                "alert": alert.to_dict() if alert else None,
            })

        elapsed = time.perf_counter() - t0
        logger.info(
            "Batch detection: {} bars in {:.1f}s, {} alerts",
            len(results),
            elapsed,
            self._monitor.alert_count,
        )

        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _generate_charts(self, df_feat: pd.DataFrame, X: np.ndarray) -> None:
        """Generate and save Plotly charts.

        Args:
            df_feat: Feature DataFrame.
            X: Feature matrix.
        """
        chart_dir = self._settings.chart_dir / "detection"
        chart_dir.mkdir(parents=True, exist_ok=True)

        states = self._predictor.predict_state(X)
        posteriors = self._predictor.predict_proba(X)

        self._chart_regime_price(df_feat, states, chart_dir)
        self._chart_regime_timeline(df_feat, states, chart_dir)
        self._chart_transition_heatmap(chart_dir)
        self._chart_confidence_history(df_feat, posteriors, chart_dir)
        self._chart_duration_histogram(states, chart_dir)

    def _chart_regime_price(
        self, df: pd.DataFrame, states: np.ndarray, chart_dir: Path
    ) -> None:
        try:
            import plotly.graph_objects as go

            colours = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#1abc9c"]
            fig = go.Figure()

            for sid in sorted(self._label_map.keys()):
                mask = states == sid
                label = self._label_map[sid]
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, "timestamp"],
                    y=df.loc[mask, "close"],
                    mode="markers",
                    name=label,
                    marker=dict(color=colours[sid % len(colours)], size=4),
                ))

            fig.update_layout(
                title="Price vs Detected Regime",
                xaxis_title="Date",
                yaxis_title="Close Price",
                template="plotly_dark",
                height=600,
            )
            fig.write_html(str(chart_dir / "price_vs_regime.html"))
            logger.info("Chart → price_vs_regime.html")
        except ImportError:
            logger.warning("Plotly not available — skipping chart")

    def _chart_regime_timeline(
        self, df: pd.DataFrame, states: np.ndarray, chart_dir: Path
    ) -> None:
        try:
            import plotly.express as px

            tl_df = pd.DataFrame({
                "timestamp": df["timestamp"],
                "regime": [self._label_map.get(s, f"State {s}") for s in states],
            })
            fig = px.scatter(
                tl_df, x="timestamp", y="regime", color="regime",
                title="Regime Timeline", template="plotly_dark", height=400,
            )
            fig.update_traces(marker=dict(size=5, symbol="square"))
            fig.write_html(str(chart_dir / "regime_timeline.html"))
            logger.info("Chart → regime_timeline.html")
        except ImportError:
            pass

    def _chart_transition_heatmap(self, chart_dir: Path) -> None:
        try:
            import plotly.figure_factory as ff

            tm = np.round(self._predictor.transition_matrix, 3)
            labels = [self._label_map.get(i, f"State {i}") for i in range(tm.shape[0])]
            fig = ff.create_annotated_heatmap(
                z=tm.tolist(), x=labels, y=labels,
                colorscale="Blues", showscale=True,
            )
            fig.update_layout(
                title="Transition Heatmap", template="plotly_dark", height=500,
            )
            fig.write_html(str(chart_dir / "transition_heatmap.html"))
            logger.info("Chart → transition_heatmap.html")
        except ImportError:
            pass

    def _chart_confidence_history(
        self, df: pd.DataFrame, posteriors: np.ndarray, chart_dir: Path
    ) -> None:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            for sid in sorted(self._label_map.keys()):
                label = self._label_map[sid]
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=posteriors[:, sid],
                    mode="lines",
                    name=label,
                    line=dict(width=1.5),
                ))
            fig.update_layout(
                title="Confidence History",
                xaxis_title="Date",
                yaxis_title="Posterior Probability",
                template="plotly_dark",
                height=500,
            )
            fig.write_html(str(chart_dir / "confidence_history.html"))
            logger.info("Chart → confidence_history.html")
        except ImportError:
            pass

    def _chart_duration_histogram(
        self, states: np.ndarray, chart_dir: Path
    ) -> None:
        try:
            import plotly.express as px

            durations: list[dict] = []
            current = states[0]
            streak = 1
            for s in states[1:]:
                if s == current:
                    streak += 1
                else:
                    durations.append({
                        "regime": self._label_map.get(int(current), f"State {current}"),
                        "duration": streak,
                    })
                    current = s
                    streak = 1
            durations.append({
                "regime": self._label_map.get(int(current), f"State {current}"),
                "duration": streak,
            })

            dur_df = pd.DataFrame(durations)
            fig = px.histogram(
                dur_df, x="duration", color="regime",
                barmode="overlay", opacity=0.7,
                title="State Duration Histogram",
                template="plotly_dark", height=400,
            )
            fig.write_html(str(chart_dir / "duration_histogram.html"))
            logger.info("Chart → duration_histogram.html")
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def history(self) -> RegimeHistory:
        """The regime prediction history."""
        return self._history

    @property
    def monitor(self) -> RegimeMonitor:
        """The regime change monitor."""
        return self._monitor

    @property
    def predictor(self) -> RegimePredictor:
        """The underlying HMM predictor."""
        return self._predictor

    def print_transition_matrix(self) -> None:
        """Log the formatted transition matrix."""
        logger.info(
            format_transition_matrix(
                self._predictor.transition_matrix, self._label_map
            )
        )

    def print_stable_regimes(self) -> None:
        """Log regimes with high self-transition probability."""
        identify_stable_regimes(
            self._predictor.transition_matrix, self._label_map
        )

    def print_confidence(self, score: ConfidenceScore) -> None:
        """Log a formatted confidence display."""
        logger.info("\n{}", format_confidence(score))


# ===================================================================
# Convenience function — the top-level public API
# ===================================================================

def detect_market_regime(
    bundle_path: str | Path,
    df: pd.DataFrame | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Detect the current market regime — one-call convenience function.

    This is the primary public API of Phase 3.  It loads the model
    bundle, fetches (or accepts) data, and returns a JSON-serializable
    result.

    Args:
        bundle_path: Path to the saved model bundle directory.
        df: Optional OHLCV DataFrame.  When *None*, the latest
            candles are fetched from the configured broker.
        settings: Optional application settings.

    Returns:
        JSON-serializable dictionary::

            {
                "timestamp": "2025-07-19T12:00:00+00:00",
                "state": 2,
                "regime": "Bull Market",
                "confidence": 0.91,
                "transition_probability": 0.84,
                "probabilities": { ... },
                ...
            }
    """
    detector = RegimeDetector.from_bundle(bundle_path, settings=settings)
    return detector.detect(df=df)


# ===================================================================
# CLI entry point
# ===================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Market Regime Detection",
    )
    parser.add_argument(
        "--bundle",
        type=str,
        required=True,
        help="Path to the saved model bundle directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config.yaml file.",
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Generate Plotly visualisations.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Phase 3 detection."""
    args = _parse_args()

    settings = get_settings(config_path=args.config)
    setup_logger(settings.log_dir)

    logger.info("=" * 60)
    logger.info("  Phase 3 — Market Regime Detection Engine")
    logger.info("=" * 60)

    detector = RegimeDetector.from_bundle(args.bundle, settings=settings)
    result = detector.detect(generate_charts=args.charts)

    # Print the JSON result.
    import json
    print(json.dumps(result, indent=2, default=str))

    # Print transition matrix and stable regimes.
    detector.print_transition_matrix()
    detector.print_stable_regimes()

    logger.success("Phase 3 detection complete ✅")


if __name__ == "__main__":
    main()
