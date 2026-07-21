"""Pydantic-powered settings for Market Regime Bot.

Settings are loaded in order of precedence (highest → lowest):
    1. Environment variables (loaded from ``.env`` automatically).
    2. ``config.yaml`` on disk.
    3. Pydantic field defaults defined here.

Usage::

    from config.settings import get_settings
    cfg = get_settings()
    print(cfg.symbol, cfg.broker)
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from config.constants import Broker, CovarianceType, ScalerType, Timeframe

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file → market_regime_bot/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env from the project root.
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """Read the YAML configuration file and return a flat dictionary.

    Args:
        path: Explicit path to ``config.yaml``.  When *None* the default
              location ``<PROJECT_ROOT>/config/config.yaml`` is used.

    Returns:
        Parsed YAML contents as a dictionary, or an empty dict when the
        file does not exist.
    """
    config_path = path or PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return data


# ---------------------------------------------------------------------------
# Phase 4 — Strategy Engine settings
# ---------------------------------------------------------------------------
class StrategySettings(BaseModel):
    """Trading strategy hyperparameters and risk settings."""

    atr_multiplier_stop_loss: float = Field(default=2.0, description="ATR multiplier for stop loss")
    atr_multiplier_take_profit: float = Field(default=4.0, description="ATR multiplier for take profit")
    risk_percentage: float = Field(default=1.0, ge=0.1, le=10.0, description="Risk % per trade")
    min_regime_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum regime confidence to trade")
    max_trades_per_day: int = Field(default=5, ge=1, description="Maximum number of trades per day")
    session_start_hour: int = Field(default=9, ge=0, le=23, description="Trading session start hour (UTC)")
    session_end_hour: int = Field(default=16, ge=0, le=23, description="Trading session end hour (UTC)")
    min_lot_size: float = Field(default=1.0, description="Minimum order size")
    max_position_size: float = Field(default=100000.0, description="Maximum position size in fiat")
    leverage: float = Field(default=1.0, ge=1.0, description="Leverage multiplier")


# ---------------------------------------------------------------------------
# Phase 5 — Advanced Risk Management settings
# ---------------------------------------------------------------------------
class RiskSettings(BaseModel):
    """Portfolio-level risk management settings."""

    max_daily_loss_fiat: float = Field(default=1000.0, ge=0.0, description="Maximum daily loss in fiat before trading stops")
    max_drawdown_pct: float = Field(default=0.10, ge=0.0, le=1.0, description="Max allowed portfolio drawdown (e.g. 0.10 = 10%)")
    max_open_positions: int = Field(default=3, ge=1, description="Max concurrent open positions across the portfolio")
    max_portfolio_exposure_pct: float = Field(default=0.50, ge=0.0, le=1.0, description="Max portion of capital allowed to be deployed")
    correlation_threshold: float = Field(default=0.75, ge=-1.0, le=1.0, description="Reject trades if correlation with existing > threshold")
    trailing_stop_activation_rr: float = Field(default=1.0, ge=0.0, description="Risk/Reward ratio at which trailing stop activates")
    trailing_stop_distance_atr: float = Field(default=1.5, ge=0.1, description="Trailing stop distance in ATR multiples")


# ---------------------------------------------------------------------------
# Phase 6 — Execution settings
# ---------------------------------------------------------------------------
class ExecutionSettings(BaseModel):
    """Trade execution settings."""

    paper_mode: bool = Field(default=True, description="Run in paper trading mode")
    default_commission: float = Field(default=1.0, ge=0.0, description="Default commission per trade (for paper)")
    default_slippage_pct: float = Field(default=0.0005, ge=0.0, description="Default slippage percentage (for paper)")
    execution_timeout_seconds: int = Field(default=30, ge=1, description="Order execution timeout")
    retry_attempts: int = Field(default=3, ge=0, description="Order retry attempts")


# ---------------------------------------------------------------------------
# Phase 7 — Backtesting settings
# ---------------------------------------------------------------------------
class BacktestSettings(BaseModel):
    """Backtesting, walk-forward, and optimisation settings."""

    commission_rate: float = Field(default=0.001, ge=0.0, description="Commission as fraction of notional")
    slippage_pct: float = Field(default=0.0005, ge=0.0, description="Slippage as fraction of price")
    spread_pct: float = Field(default=0.0002, ge=0.0, description="Bid-ask spread fraction")
    initial_capital: float = Field(default=100_000.0, ge=0.0, description="Backtest starting capital")
    benchmark_symbol: str = Field(default="SPY", description="Benchmark ticker for comparison")
    walk_forward_train_bars: int = Field(default=252, ge=20, description="Training window bars")
    walk_forward_test_bars: int = Field(default=63, ge=5, description="Testing window bars")
    walk_forward_step_bars: int = Field(default=63, ge=5, description="Step size between windows")
    optimization_method: str = Field(default="grid", description="Optimization method: grid or random")
    optimization_samples: int = Field(default=50, ge=1, description="Samples for random search")
    report_output_dir: str = Field(default="reports", description="Directory for report output")


# ---------------------------------------------------------------------------
# API-key settings sourced exclusively from environment variables
# ---------------------------------------------------------------------------
class APIKeys(BaseModel):
    """API credentials pulled from environment variables."""

    alpaca_api_key: str = Field(default="", description="Alpaca API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca secret key")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca REST endpoint",
    )
    binance_api_key: str = Field(default="", description="Binance API key")
    binance_secret_key: str = Field(default="", description="Binance secret key")
    bybit_api_key: str = Field(default="", description="Bybit API key")
    bybit_secret_key: str = Field(default="", description="Bybit secret key")

    @classmethod
    def from_env(cls) -> "APIKeys":
        """Construct an ``APIKeys`` instance from current environment variables."""
        return cls(
            alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            alpaca_base_url=os.getenv(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ),
            binance_api_key=os.getenv("BINANCE_API_KEY", ""),
            binance_secret_key=os.getenv("BINANCE_SECRET_KEY", ""),
            bybit_api_key=os.getenv("BYBIT_API_KEY", ""),
            bybit_secret_key=os.getenv("BYBIT_SECRET_KEY", ""),
        )


# ---------------------------------------------------------------------------
# Main application settings
# ---------------------------------------------------------------------------
class Settings(BaseModel):
    """Validated application-level configuration.

    Fields map 1-to-1 with keys in ``config.yaml`` and can be overridden
    via environment variables of the same name (case-insensitive).
    """

    broker: Broker = Field(default=Broker.YAHOO, description="Data source identifier")
    symbol: str = Field(default="SPY", description="Instrument ticker / symbol")
    timeframe: Timeframe = Field(default=Timeframe.D1, description="OHLCV bar size")
    start_date: date = Field(
        default=date(2020, 1, 1), description="Start of data window (UTC)"
    )
    end_date: date = Field(
        default=date(2025, 1, 1), description="End of data window (UTC)"
    )
    initial_capital: float = Field(
        default=100_000.0, ge=0.0, description="Starting portfolio value"
    )
    data_folder: Path = Field(
        default=Path("data"), description="Root folder for data artefacts"
    )
    log_folder: Path = Field(
        default=Path("logs"), description="Root folder for log files"
    )
    use_cache: bool = Field(default=True, description="Reload cached data when valid")
    cache_expiry_hours: int = Field(
        default=24, ge=1, description="Hours before cache is considered stale"
    )
    max_retries: int = Field(
        default=3, ge=1, description="Max retry attempts for downloads"
    )
    retry_delay_seconds: int = Field(
        default=5, ge=1, description="Seconds between retries"
    )
    max_missing_pct: float = Field(
        default=5.0, ge=0.0, le=100.0, description="Max tolerated missing-bar %"
    )
    max_duplicate_pct: float = Field(
        default=1.0, ge=0.0, le=100.0, description="Max tolerated duplicate-row %"
    )

    # ---- Phase 2: Feature Engineering & HMM Training ----------------------

    scaler_type: ScalerType = Field(
        default=ScalerType.ROBUST, description="Feature scaling method"
    )
    correlation_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Drop features with |correlation| above this",
    )
    hmm_min_states: int = Field(
        default=2, ge=2, description="Minimum HMM state count to test"
    )
    hmm_max_states: int = Field(
        default=6, ge=2, description="Maximum HMM state count to test"
    )
    hmm_covariance_type: CovarianceType = Field(
        default=CovarianceType.FULL, description="HMM covariance type"
    )
    hmm_n_iter: int = Field(
        default=200, ge=10, description="Max EM iterations per model"
    )
    hmm_n_init: int = Field(
        default=10, ge=1, description="Random restart count for HMM fitting"
    )
    hmm_random_state: int = Field(
        default=42, description="Random seed for reproducibility"
    )
    hmm_tolerance: float = Field(
        default=0.01, gt=0.0, description="EM convergence tolerance"
    )
    model_folder: Path = Field(
        default=Path("saved_models"), description="Folder for serialised models"
    )
    chart_folder: Path = Field(
        default=Path("charts"), description="Folder for generated charts"
    )

    # ---- Phase 4: Strategy ------------------------------------------------

    strategy: StrategySettings = Field(default_factory=StrategySettings)

    # ---- Phase 5: Risk ----------------------------------------------------

    risk: RiskSettings = Field(default_factory=RiskSettings)

    # ---- Phase 6: Execution -----------------------------------------------

    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)

    # ---- Phase 7: Backtesting ---------------------------------------------

    backtest: BacktestSettings = Field(default_factory=BacktestSettings)

    api_keys: APIKeys = Field(default_factory=APIKeys)

    # ---- Validators -------------------------------------------------------

    @field_validator("symbol")
    @classmethod
    def _symbol_uppercase(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _dates_in_order(self) -> "Settings":
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must precede "
                f"end_date ({self.end_date})"
            )
        return self

    # ---- Derived paths ----------------------------------------------------

    @property
    def raw_data_dir(self) -> Path:
        """Absolute path to the raw-data directory."""
        return PROJECT_ROOT / self.data_folder / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Absolute path to the processed-data directory."""
        return PROJECT_ROOT / self.data_folder / "processed"

    @property
    def cache_dir(self) -> Path:
        """Absolute path to the cache directory."""
        return PROJECT_ROOT / self.data_folder / "cache"

    @property
    def log_dir(self) -> Path:
        """Absolute path to the log directory."""
        return PROJECT_ROOT / self.log_folder

    @property
    def model_dir(self) -> Path:
        """Absolute path to the saved-models directory."""
        return PROJECT_ROOT / self.model_folder

    @property
    def chart_dir(self) -> Path:
        """Absolute path to the chart-output directory."""
        return PROJECT_ROOT / self.chart_folder

    def ensure_directories(self) -> None:
        """Create all required directories if they do not already exist."""
        for directory in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.cache_dir,
            self.log_dir,
            self.model_dir,
            self.chart_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Factory — cached singleton
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    """Build and cache a ``Settings`` instance.

    The function merges YAML file values with API keys pulled from the
    environment. The result is cached so repeated calls are free.

    Args:
        config_path: Optional override path for ``config.yaml``.

    Returns:
        A fully validated ``Settings`` object.
    """
    yaml_path = Path(config_path) if config_path else None
    yaml_data = _load_yaml_config(yaml_path)
    api_keys = APIKeys.from_env()
    yaml_data["api_keys"] = api_keys.model_dump()

    settings = Settings(**yaml_data)
    settings.ensure_directories()
    return settings
