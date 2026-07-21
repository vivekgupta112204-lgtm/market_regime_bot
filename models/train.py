"""Phase 2 end-to-end training pipeline.

Orchestrates the full workflow::

    Load Phase 1 data
        → Feature engineering
        → Feature scaling
        → Feature selection
        → HMM grid search (2–6 states)
        → Model selection (BIC)
        → Regime mapping
        → Visualisation
        → Model saving
        → Training report

Usage::

    python -m models.train
    python -m models.train --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import Settings, get_settings
from data_loader.data_manager import DataManager
from features.feature_pipeline import FeaturePipeline
from models.hmm_selector import SelectionResult, format_comparison_table, select_best_model
from models.hmm_trainer import HMMResult, HMMTrainer
from models.model_loader import ModelLoader
from models.regime_mapper import RegimeStats, format_regime_table, map_regimes
from utils.logger import setup_logger


# ===================================================================
# Visualisation helpers (Plotly)
# ===================================================================

def _save_regime_price_chart(
    df: pd.DataFrame,
    hidden_states: np.ndarray,
    label_map: dict[int, str],
    chart_dir: Path,
) -> None:
    """Generate price chart coloured by detected regime.

    Args:
        df: Feature DataFrame with ``timestamp`` and ``close``.
        hidden_states: State assignments.
        label_map: State → label mapping.
        chart_dir: Output directory.
    """
    try:
        import plotly.graph_objects as go

        chart_dir.mkdir(parents=True, exist_ok=True)

        fig = go.Figure()

        # Colour palette for regimes.
        colours = [
            "#2ecc71", "#e74c3c", "#f39c12", "#3498db",
            "#9b59b6", "#1abc9c", "#e67e22",
        ]

        for state_id in sorted(label_map.keys()):
            mask = hidden_states == state_id
            label = label_map[state_id]
            colour = colours[state_id % len(colours)]

            fig.add_trace(go.Scatter(
                x=df.loc[mask, "timestamp"],
                y=df.loc[mask, "close"],
                mode="markers",
                name=label,
                marker=dict(color=colour, size=4),
            ))

        fig.update_layout(
            title="Price with Detected Regimes",
            xaxis_title="Date",
            yaxis_title="Close Price",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=600,
        )

        path = chart_dir / "regime_price_chart.html"
        fig.write_html(str(path))
        logger.info("Chart saved → {}", path.name)
    except ImportError:
        logger.warning("Plotly not installed — skipping regime price chart")


def _save_regime_timeline(
    df: pd.DataFrame,
    hidden_states: np.ndarray,
    label_map: dict[int, str],
    chart_dir: Path,
) -> None:
    """Generate a timeline heatmap of regime changes.

    Args:
        df: Feature DataFrame with ``timestamp``.
        hidden_states: State assignments.
        label_map: State → label mapping.
        chart_dir: Output directory.
    """
    try:
        import plotly.express as px

        chart_dir.mkdir(parents=True, exist_ok=True)

        timeline_df = pd.DataFrame({
            "timestamp": df["timestamp"],
            "regime": [label_map[s] for s in hidden_states],
            "state": hidden_states,
        })

        fig = px.scatter(
            timeline_df,
            x="timestamp",
            y="regime",
            color="regime",
            title="Regime Timeline",
            template="plotly_dark",
            height=400,
        )
        fig.update_traces(marker=dict(size=5, symbol="square"))
        fig.update_layout(
            yaxis_title="Regime",
            xaxis_title="Date",
            showlegend=False,
        )

        path = chart_dir / "regime_timeline.html"
        fig.write_html(str(path))
        logger.info("Chart saved → {}", path.name)
    except ImportError:
        logger.warning("Plotly not installed — skipping regime timeline")


def _save_transition_matrix(
    model: object,
    label_map: dict[int, str],
    chart_dir: Path,
) -> None:
    """Generate a heatmap of the HMM state-transition matrix.

    Args:
        model: Fitted GaussianHMM.
        label_map: State → label mapping.
        chart_dir: Output directory.
    """
    try:
        import plotly.figure_factory as ff

        chart_dir.mkdir(parents=True, exist_ok=True)
        transmat = np.round(model.transmat_, 3)  # type: ignore[union-attr]
        labels = [label_map.get(i, f"State {i}") for i in range(transmat.shape[0])]

        fig = ff.create_annotated_heatmap(
            z=transmat.tolist(),
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
        )
        fig.update_layout(
            title="Hidden State Transition Matrix",
            xaxis_title="To",
            yaxis_title="From",
            template="plotly_dark",
            height=500,
        )

        path = chart_dir / "transition_matrix.html"
        fig.write_html(str(path))
        logger.info("Chart saved → {}", path.name)
    except ImportError:
        logger.warning("Plotly not installed — skipping transition matrix chart")


def _save_feature_distributions(
    df: pd.DataFrame,
    feature_columns: list[str],
    hidden_states: np.ndarray,
    label_map: dict[int, str],
    chart_dir: Path,
) -> None:
    """Generate distribution plots for selected features per regime.

    Args:
        df: Feature DataFrame.
        feature_columns: Columns to plot.
        hidden_states: State assignments.
        label_map: State → label mapping.
        chart_dir: Output directory.
    """
    try:
        import plotly.express as px

        chart_dir.mkdir(parents=True, exist_ok=True)

        plot_df = df[feature_columns].copy()
        plot_df["regime"] = [label_map[s] for s in hidden_states]

        # Plot up to 6 most important features to keep the chart readable.
        cols_to_plot = feature_columns[:6]

        for col in cols_to_plot:
            fig = px.histogram(
                plot_df,
                x=col,
                color="regime",
                barmode="overlay",
                opacity=0.7,
                title=f"Distribution of {col} by Regime",
                template="plotly_dark",
                height=400,
            )
            path = chart_dir / f"dist_{col}.html"
            fig.write_html(str(path))

        logger.info("Feature distribution charts saved ({} features)", len(cols_to_plot))
    except ImportError:
        logger.warning("Plotly not installed — skipping feature distributions")


def _save_correlation_heatmap(
    df: pd.DataFrame,
    feature_columns: list[str],
    chart_dir: Path,
) -> None:
    """Generate a correlation heatmap for the feature matrix.

    Args:
        df: Feature DataFrame.
        feature_columns: Columns to include.
        chart_dir: Output directory.
    """
    try:
        import plotly.figure_factory as ff

        chart_dir.mkdir(parents=True, exist_ok=True)

        corr = df[feature_columns].corr()
        fig = ff.create_annotated_heatmap(
            z=np.round(corr.values, 2).tolist(),
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu_r",
            showscale=True,
        )
        fig.update_layout(
            title="Feature Correlation Heatmap",
            template="plotly_dark",
            height=700,
            width=900,
        )

        path = chart_dir / "correlation_heatmap.html"
        fig.write_html(str(path))
        logger.info("Chart saved → {}", path.name)
    except ImportError:
        logger.warning("Plotly not installed — skipping correlation heatmap")


# ===================================================================
# Training report
# ===================================================================

def _print_training_report(
    settings: Settings,
    n_observations: int,
    all_features: list[str],
    selected_features: list[str],
    selection: SelectionResult,
    regime_stats: list[RegimeStats],
    total_time: float,
) -> None:
    """Print a comprehensive training report to the logger.

    Args:
        settings: Application settings.
        n_observations: Number of rows used for training.
        all_features: Full feature list before selection.
        selected_features: Feature list after selection.
        selection: Model selection result.
        regime_stats: Per-state statistics.
        total_time: Total pipeline wall-clock time in seconds.
    """
    best = selection.best_result
    report_lines = [
        "",
        "═══════════════════════════════════════════════════════════════",
        "              PHASE 2 — TRAINING REPORT                      ",
        "═══════════════════════════════════════════════════════════════",
        f"  Symbol            : {settings.symbol}",
        f"  Timeframe         : {settings.timeframe.value}",
        f"  Date range        : {settings.start_date} → {settings.end_date}",
        f"  Broker            : {settings.broker.value}",
        f"  Observations      : {n_observations}",
        f"  Features (total)  : {len(all_features)}",
        f"  Features (used)   : {len(selected_features)}",
        f"  Scaler            : {settings.scaler_type.value}",
        "───────────────────────────────────────────────────────────────",
        f"  Selected model    : {best.n_states} states",
        f"  Covariance type   : {settings.hmm_covariance_type.value}",
        f"  Log-Likelihood    : {best.log_likelihood:.4f}",
        f"  AIC               : {best.aic:.4f}",
        f"  BIC               : {best.bic:.4f}",
        f"  Free parameters   : {best.n_params}",
        f"  Converged         : {best.converged}",
        f"  Training time     : {best.train_time_seconds:.2f} s",
        f"  Total pipeline    : {total_time:.2f} s",
        "═══════════════════════════════════════════════════════════════",
    ]
    logger.info("\n".join(report_lines))

    # Model comparison table
    logger.info(format_comparison_table(selection.all_results))

    # Regime statistics
    logger.info(format_regime_table(regime_stats))


# ===================================================================
# Main training function
# ===================================================================

def run_training(settings: Settings | None = None) -> dict:
    """Execute the complete Phase 2 training pipeline.

    Args:
        settings: Optional pre-built settings.  When *None*, settings
            are loaded from ``config.yaml`` / environment.

    Returns:
        Dictionary containing all training artefacts:
        ``df``, ``features``, ``model``, ``hidden_states``,
        ``regime_stats``, ``label_map``, ``selection``.
    """
    pipeline_start = time.perf_counter()

    if settings is None:
        settings = get_settings()

    setup_logger(settings.log_dir)

    logger.info("=" * 60)
    logger.info("  Phase 2 — Feature Engineering & HMM Training")
    logger.info("=" * 60)

    # ── Step 1: Load Phase 1 data ──────────────────────────────────────
    logger.info("Step 1/10: Loading Phase 1 data …")
    manager = DataManager(settings)
    df, validation_report = manager.get_data()
    logger.info("Loaded {} rows — validation: {}", len(df), "PASS" if validation_report.passed else "FAIL")

    # ── Step 2–4: Feature Pipeline (engineer → scale → select) ─────────
    logger.info("Steps 2–4: Feature engineering, scaling, selection …")
    pipeline = FeaturePipeline(settings)
    df, selected_features = pipeline.run(df)

    all_features = pipeline.feature_columns
    n_observations = len(df)

    # ── Step 5: HMM Training ──────────────────────────────────────────
    logger.info("Step 5: Training HMM models …")
    X = df[selected_features].values.astype(np.float64)

    trainer = HMMTrainer(
        n_iter=settings.hmm_n_iter,
        n_init=settings.hmm_n_init,
        covariance_type=settings.hmm_covariance_type.value,
        random_state=settings.hmm_random_state,
        tolerance=settings.hmm_tolerance,
    )
    results = trainer.train_range(
        X,
        min_states=settings.hmm_min_states,
        max_states=settings.hmm_max_states,
    )

    # ── Step 6: Model Selection ───────────────────────────────────────
    logger.info("Step 6: Selecting best model …")
    selection = select_best_model(results, criterion="bic")
    best = selection.best_result

    # ── Step 7: Regime Mapping ────────────────────────────────────────
    logger.info("Step 7: Mapping regimes …")
    hidden_states = best.model.predict(X)
    regime_stats, label_map = map_regimes(
        df, hidden_states, best.n_states,
    )

    # ── Step 8: Visualisation ─────────────────────────────────────────
    logger.info("Step 8: Generating visualisations …")
    chart_dir = settings.chart_dir

    _save_regime_price_chart(df, hidden_states, label_map, chart_dir)
    _save_regime_timeline(df, hidden_states, label_map, chart_dir)
    _save_transition_matrix(best.model, label_map, chart_dir)
    _save_feature_distributions(df, selected_features, hidden_states, label_map, chart_dir)
    _save_correlation_heatmap(df, selected_features, chart_dir)

    # ── Step 9: Model Saving ──────────────────────────────────────────
    logger.info("Step 9: Saving model bundle …")
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{settings.symbol}_{settings.timeframe.value}_{best.n_states}states_{timestamp_str}"
    bundle_dir = settings.model_dir / bundle_name

    scaler_path = pipeline.save_scaler(bundle_dir)

    config_snapshot = settings.model_dump()
    ModelLoader.save_bundle(
        directory=bundle_dir,
        result=best,
        feature_names=selected_features,
        regime_stats=regime_stats,
        label_map=label_map,
        config_snapshot=config_snapshot,
        scaler_path=scaler_path,
    )

    # ── Step 10: Report ───────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_start
    logger.info("Step 10: Generating training report …")
    _print_training_report(
        settings=settings,
        n_observations=n_observations,
        all_features=all_features,
        selected_features=selected_features,
        selection=selection,
        regime_stats=regime_stats,
        total_time=total_time,
    )

    logger.success("Phase 2 training complete ✅")

    return {
        "df": df,
        "features": selected_features,
        "model": best.model,
        "hidden_states": hidden_states,
        "regime_stats": regime_stats,
        "label_map": label_map,
        "selection": selection,
        "bundle_dir": bundle_dir,
    }


# ===================================================================
# CLI entry point
# ===================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2 — HMM Regime Training Pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config.yaml file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Phase 2 training."""
    args = _parse_args()

    from config.settings import get_settings

    try:
        settings = get_settings(config_path=args.config)
    except Exception as exc:
        print(f"[FATAL] Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    run_training(settings)


if __name__ == "__main__":
    main()
