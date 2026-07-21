# HMM Enterprise Regime-Based Trading Platform

![CI/CD](https://github.com/yourusername/market_regime_bot/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An institutional-grade, fully autonomous quantitative trading platform built on Python. The platform utilizes **Hidden Markov Models (HMM)** for market regime mathematical detection (Trend/Volatility), combined with **Reinforcement Learning (PPO)** and **FinBERT NLP Transformers** to allocate capital sequentially mimicking a real quant firm swarm architecture.

## 🚀 Key Features

*   **Mathematical Regime Detection**: Dynamic GaussianHMM decoding Bull/Bear/Sideways market states.
*   **Swarm AI Ensemble**: Coordinated autonomous agents (Research, News, Portfolio, Risk, Execution).
*   **Markowitz Risk Optimizations**: Scipy SLSQP boundary minimizations calculating continuous optimal Sharpe portfolios via dynamic Expected Returns & Covariance.
*   **Robust Risk Engines**: Hard limits on Historical VaR, Expected Shortfall (CVaR), and continuous running maximum drawdowns.
*   **Multi-Broker Adapters**: Production Alpaca API native integrations for high-frequency routing (TWAP, VWAP scaling based on ATR/Volume metrics).
*   **Event-Driven Resilient Deployments**: Watchdog services, MLOps Auto-Retrainer sequences, and Streamlit asynchronous dashboard integrations.

## 🏗 Architecture

```text
market_regime_bot/
├── ai/                  # PPO & LLM Strategy Ensembles and Multi-Agents
├── analytics/           # Math libraries (VaR, Drawdowns, Sharpe)
├── api/                 # FastAPI REST and WebSocket hooks
├── automation/          # APScheduler and Cron automated runtimes
├── backtesting/         # Walk-Forward simulated vector backtesting engine 
├── broker/              # Alpaca, Binance, CCXT Multi-broker interfaces
├── dashboard/           # Streamlit Live Monitor Front-End
├── mlops/               # Model Registry, Retrainers, and Statistical Drift Detection
└── optimization/        # Scipy Mean-Variance Institutional Capital Allocators
```

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/market_regime_bot.git
cd market_regime_bot

# 2. Setup Virtual Environment
make install

# 3. Environment Variables
cp .env.example .env
# Edit .env with your Alpaca API Keys and Encryption Keys.
```

## 🖥 Usage

### Local Deployment
```bash
make start
```
* Dashboard runs at `http://localhost:8501`
* API runs at `http://localhost:8000`

### Docker Production Deployment
```bash
make docker-build
make docker-up
```

## 📈 Paper vs Live Trading
The system natively guards LIVE endpoints. By default, booting initiates `PAPER_TRADING`. 
To run with real money:
```bash
export BOT_MODE=PRODUCTION
make start
```

## 🔒 Security
All broker secrets are stored locally. Production setups use the `config/.secrets.enc` symmetric Fernet encryption logic to prevent disk leaks. **Never commit your `.env` file**.

## 🤝 Contributing
Please see `CONTRIBUTING.md` for guidelines. We enforce `black`, `isort`, and `flake8` for CI checks.

---
*Disclaimer: This repository is highly experimental. The maintainers are not responsible for financial loss resulting from deploying these algorithms.*
